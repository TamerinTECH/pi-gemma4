"""Exercise the actual native loader with tiny files; never load a real model.

Run in the Linux build environment:
  CF_NATIVE_BINARY=/work/cf-engine/build-x86/bin/cf-run \
    python3 /work/cf-engine/test_native_artifact.py -v

The payload is 72 bytes, and the deliberately non-GGUF source is 128 bytes.
The positive --version control proves initialization completed without inference.
Current llama.cpp --version calls exit(0) inside argument parsing, before
cf_finish. Passing initialization is neither CF dispatch nor inference evidence.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class NativeArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = Path(os.environ.get("CF_NATIVE_BINARY", str(
            Path(__file__).with_name("build-x86") / "bin" / "cf-run"))).resolve()
        if sys.platform != "linux" or not cls.binary.is_file():
            raise unittest.SkipTest("native Linux cf-run unavailable; set CF_NATIVE_BINARY in the build container")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cf-native-artifact-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifact = self.root / "artifact"
        self.artifact.mkdir()
        self.source = self.root / "tiny-source.bin"
        # Construct original four Q4_0 blocks, then independently transpose one
        # four-row tile. Little-endian scales/codes are copied verbatim.
        raw = bytes((17 * i + 3) % 256 for i in range(72))
        packed = b"".join(raw[row * 18:row * 18 + 2] for row in range(4))
        packed += b"".join(raw[row * 18 + 2:(row + 1) * 18] for row in range(4))
        self.source.write_bytes(b"synthetic-non-model-source".ljust(32, b"\0") + raw + bytes(24))
        self.assertEqual(self.source.stat().st_size, 128)
        self.payload = self.artifact / "tensor.cfq4"
        self.payload.write_bytes(packed)
        self.manifest = self.artifact / "manifest.json"
        source = {"name": self.source.name, "bytes": 128,
                  "sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
                  "alignment": 32, "data_offset": 32, "gguf_version": 3}
        tensor = {"name": "blk.0.ffn_gate_up_exps.weight", "type": "Q4_0", "type_id": 2,
                  "shape": [32, 4, 1], "layout": "q4-tile4-v1", "file": self.payload.name,
                  "bytes": 72, "sha256": hashlib.sha256(packed).hexdigest(),
                  "source_sha256": hashlib.sha256(raw).hexdigest(),
                  "source_file": 0, "source_offset": 32, "kernel": "q4-dotprod-or-portable"}
        self.plan = {"format": "cf-engine-plan", "version": 1,
                     "source": {"architecture": "gemma4", "files": [source],
                                "alternative_file_identities": []},
                     "tensors": [tensor], "inventory": [copy.deepcopy(tensor)]}

    def invoke(self, kernel="portable", extra_args=()):
        self.manifest.write_text(json.dumps(self.plan), encoding="utf-8")
        # Do not inherit session configuration, logits output destinations,
        # validation modes or memory experiments from a benchmark shell.
        environment = {key: value for key, value in os.environ.items() if not key.startswith("CF_")}
        if isinstance(extra_args, str) or any(not isinstance(arg, str) for arg in extra_args):
            raise TypeError("extra_args must be a sequence of individual argument strings")
        command = [str(self.binary), "--engine", "cf", "--cf-plan", str(self.manifest),
                   "--cf-kernel", kernel, "--cf-memory", "resident", *extra_args,
                   "-m", str(self.source), "--version"]
        return subprocess.run(command, capture_output=True, text=True, timeout=15, env=environment)

    def reject(self, expected_message, kernel="portable", extra_args=()):
        result = self.invoke(kernel, extra_args)
        detail = f"exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
        self.assertEqual(result.returncode, 3, detail)
        self.assertIn("CF_INIT_ERROR", result.stderr, detail)
        self.assertIn(expected_message, result.stderr, detail)
        self.assertNotIn("pigemma-probe", result.stdout, detail)
        self.assertNotIn("CF_ENGINE mode=cf", result.stderr, detail)

    def accept_initialization(self):
        result = self.invoke()
        detail = f"exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
        # The pinned --version handler exits 0 directly, before cf_finish. This
        # positive control proves only initialization, never execution coverage.
        self.assertEqual(result.returncode, 0, detail)
        self.assertIn("CF_ENGINE mode=cf", result.stderr, detail)
        self.assertNotIn("CF_INIT_ERROR", result.stderr, detail)
        self.assertIn("pigemma-probe", result.stdout, detail)

    def mutate_matching_tensor_identity(self, key, value):
        self.plan["tensors"][0][key] = copy.deepcopy(value)
        self.plan["inventory"][0][key] = copy.deepcopy(value)

    def test_valid_artifact_initializes_without_model_loading(self):
        self.accept_initialization()

    def test_alternative_split_file_identity_is_checked(self):
        original = copy.deepcopy(self.plan["source"]["files"])
        self.plan["source"]["files"][0]["name"] = "desktop-original-not-present.gguf"
        self.plan["source"]["alternative_file_identities"] = original
        self.accept_initialization()

    def test_alternative_identity_with_wrong_digest_rejected(self):
        original = copy.deepcopy(self.plan["source"]["files"])
        original[0]["sha256"] = "0" * 64
        self.plan["source"]["files"][0]["name"] = "desktop-original-not-present.gguf"
        self.plan["source"]["alternative_file_identities"] = original
        self.reject("source SHA256 mismatch")

    def test_unsupported_version(self):
        self.plan["version"] = 2
        self.reject("unsupported artifact version")

    def test_float_version_not_accepted_as_integer_version(self):
        self.plan["version"] = 1.0
        self.reject("unsupported artifact version")

    def test_boolean_version_not_accepted_as_integer_version(self):
        self.plan["version"] = True
        self.reject("unsupported artifact version")

    def test_unknown_format(self):
        self.plan["format"] = "unrelated-plan"
        self.reject("unsupported artifact version")

    def test_existing_traversal_path_rejected(self):
        outside = self.root / "outside.cfq4"
        outside.write_bytes(self.payload.read_bytes())
        self.plan["tensors"][0]["file"] = "../outside.cfq4"
        self.reject("artifact path escapes plan")

    def test_absolute_path_rejected(self):
        self.plan["tensors"][0]["file"] = str(self.payload)
        self.reject("invalid artifact path")

    def test_symlink_escape_rejected(self):
        outside = self.root / "outside.cfq4"
        outside.write_bytes(self.payload.read_bytes())
        link = self.artifact / "linked.cfq4"
        link.symlink_to(outside)
        self.plan["tensors"][0]["file"] = link.name
        self.reject("artifact path escapes plan")

    def test_payload_shorter_than_declared(self):
        self.payload.write_bytes(self.payload.read_bytes()[:-1])
        self.reject("payload size mismatch")

    def test_declared_payload_size_disagrees_with_shape(self):
        self.mutate_matching_tensor_identity("bytes", 73)
        self.reject("payload bounds mismatch")

    def test_payload_changed_without_size_change(self):
        data = bytearray(self.payload.read_bytes())
        data[-1] ^= 1
        self.payload.write_bytes(data)
        self.reject("packed SHA256 mismatch")

    def test_wrong_payload_checksum(self):
        self.plan["tensors"][0]["sha256"] = "f" * 64
        self.reject("packed SHA256 mismatch")

    def test_source_changed_without_size_change(self):
        data = bytearray(self.source.read_bytes())
        data[0] ^= 1
        self.source.write_bytes(data)
        self.reject("source SHA256 mismatch")

    def test_source_size_identity_mismatch(self):
        self.plan["source"]["files"][0]["bytes"] += 1
        self.reject("source size mismatch")

    def test_source_name_identity_mismatch(self):
        self.plan["source"]["files"][0]["name"] = "other-model.gguf"
        self.reject("model filename absent from source identities")

    def test_shape_overflow_rejected_before_mapping(self):
        self.mutate_matching_tensor_identity("shape", [1 << 63, 4, 128])
        self.reject("unsupported shape")

    def test_negative_shape(self):
        self.mutate_matching_tensor_identity("shape", [-32, 4, 1])
        self.reject("negative or invalid shape")

    def test_zero_shape(self):
        self.mutate_matching_tensor_identity("shape", [32, 4, 0])
        self.reject("unsupported shape")

    def test_tail_dimension(self):
        self.mutate_matching_tensor_identity("shape", [32, 3, 1])
        self.reject("unsupported shape")

    def test_unsupported_quantization(self):
        self.mutate_matching_tensor_identity("type", "Q4_1")
        self.reject("unsupported packed type")

    def test_lut_kernel_cannot_read_tile4_payload(self):
        self.reject("layout/kernel mismatch", kernel="lut")

    def test_portable_kernel_cannot_read_bitplane_payload(self):
        self.plan["tensors"][0]["layout"] = "q4-bitplane4-v1"
        self.reject("layout/kernel mismatch")

    def test_unknown_kernel(self):
        self.reject("unknown kernel", kernel="assumed-sve2")

    def test_schedule_with_initial_lut_rejected_before_source_read(self):
        # The source deliberately does not exist: the precise configuration
        # error must occur before source identity I/O or model loading.
        self.source.unlink()
        self.reject("schedule supports tile4 layout only", kernel="lut",
                    extra_args=("--cf-schedule", "reference,dotprod"))

    def test_unknown_scheduler_rejected_before_source_read(self):
        self.source.unlink()
        self.reject("unknown decode scheduler",
                    extra_args=("--cf-scheduler", "unrecognized-scheduler"))

    def test_unsupported_schedule_entry_rejected_before_source_read(self):
        self.source.unlink()
        self.reject("unsupported schedule entry",
                    extra_args=("--cf-schedule", "reference,lut"))

    def test_explicit_assembly_unavailable_on_x86_rejected_before_source_read(self):
        if os.uname().machine.lower() not in ("x86_64", "amd64", "i386", "i686"):
            self.skipTest("this regression specifically verifies the x86 build cannot execute A76 assembly")
        self.source.unlink()
        for kernel in ("asm", "asm-unroll2", "asm-prefetch512"):
            with self.subTest(kernel=kernel):
                self.reject("requested assembly is not compiled or runtime dotprod unavailable", kernel=kernel)

    def test_duplicate_tensor_identity(self):
        self.plan["tensors"].append(copy.deepcopy(self.plan["tensors"][0]))
        self.reject("duplicate tensor")

    def test_duplicate_inventory_identity(self):
        self.plan["inventory"].append(copy.deepcopy(self.plan["inventory"][0]))
        self.reject("duplicate inventory tensor")

    def test_selected_tensor_missing_from_inventory(self):
        self.plan["inventory"] = []
        self.reject("selected tensor absent from inventory")

    def test_invalid_source_tensor_digest(self):
        self.mutate_matching_tensor_identity("source_sha256", "wrong")
        self.reject("invalid source tensor digest")

    def test_selected_tensor_identity_differs_from_inventory(self):
        original = copy.deepcopy(self.plan)
        for key, value in (("type", "Q4_1"), ("shape", [32, 8, 1]), ("bytes", 144),
                           ("source_sha256", "0" * 64), ("source_file", 1), ("source_offset", 64)):
            with self.subTest(key=key):
                self.plan = copy.deepcopy(original)
                self.plan["tensors"][0][key] = value
                self.reject(f"selected/inventory mismatch: {key}")

    def test_empty_plan(self):
        self.plan["tensors"] = []
        self.reject("empty CF plan")


if __name__ == "__main__":
    unittest.main()
