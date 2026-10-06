"""Synthetic, independent packing/origin checks; no real model loading required."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("cf_compiler", Path(__file__).with_name("compiler.py"))
cf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cf)


def string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def write_source(path: Path, shape=(64, 8, 3), extra_type=0) -> tuple[bytes, dict[str, int]]:
    # Every possible scale bit pattern is meaningful to a layout transformation,
    # including negative zero, infinities and NaNs. No fp16 conversion is allowed.
    rng = np.random.default_rng(618)
    raw = rng.integers(0, 256, np.prod(shape) // 32 * 18, dtype=np.uint8).tobytes()
    tensors = [("blk.0.ffn_gate_up_exps.weight", shape, 2, raw),
               ("blk.0.ffn_down_exps.weight", (32, 4, 3), 2, bytes(range(216))),
               ("blk.1.ffn_gate_up_exps.weight", (32, 4, 3), extra_type,
                b"\0" * (384 * (4 if extra_type == 0 else 2))),
               ("per_layer_token_embd.weight", (4, 8), 1, b"\0" * 64),
               ("token_embd.weight", (4, 8), 1, b"\0" * 64)]
    fields = [("general.architecture", 8, string("gemma4")),
              ("gemma4.expert_count", 4, struct.pack("<I", 3)),
              ("gemma4.expert_used_count", 4, struct.pack("<I", 2))]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(fields))
    for name, kind, value in fields:
        header += string(name) + struct.pack("<I", kind) + value
    relative = 0
    offsets = {}
    for name, dims, kind, payload in tensors:
        offsets[name] = relative
        header += string(name) + struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims)
        header += struct.pack("<IQ", kind, relative)
        relative = (relative + len(payload) + 31) // 32 * 32
    data_offset = (len(header) + 31) // 32 * 32
    data = bytearray(data_offset + relative)
    data[:len(header)] = header
    for name, _, _, payload in tensors:
        absolute = data_offset + offsets[name]
        data[absolute:absolute + len(payload)] = payload
        offsets[name] = absolute
    path.write_bytes(data)
    return raw, offsets


def scalar_pack(raw: bytes, k: int, layout: str) -> bytes:
    # Scalar oracle deliberately follows logical index definitions, independent
    # of the compiler's vectorized transposes and bit-plane implementation.
    blocks = k // 32
    output = bytearray()
    for tile in range(len(raw) // (blocks * 72)):
        for block in range(blocks):
            rows = [raw[((tile * 4 + row) * blocks + block) * 18:
                        ((tile * 4 + row) * blocks + block + 1) * 18] for row in range(4)]
            output.extend(b"".join(row[:2] for row in rows))
            for row in rows:
                if layout == "tile4":
                    output.extend(row[2:])
                else:
                    codes = [(row[2 + i % 16] >> (4 * (i // 16))) & 15 for i in range(32)]
                    planes = bytearray(16)
                    for group in range(8):
                        for bit in range(4):
                            mask = sum(((codes[group * 4 + j] >> bit) & 1) << j for j in range(4))
                            planes[group * 2 + bit // 2] |= mask << (4 * (bit % 2))
                    output.extend(planes)
    return bytes(output)


@pytest.mark.parametrize("k,groups", [(32, 1), (64, 6), (256, 7), (2816, 2), (704, 3)])
@pytest.mark.parametrize("layout", ["tile4", "bitplane"])
def test_roundtrip_matches_independent_oracle(k, groups, layout):
    raw = np.random.default_rng(812).integers(0, 256, k // 32 * 72 * groups, dtype=np.uint8).tobytes()
    packed = cf.pack_rows(raw, k, layout)
    assert packed == scalar_pack(raw, k, layout)
    assert len(packed) == len(raw)
    assert cf.unpack_rows(packed, k, layout) == raw


@pytest.mark.parametrize("layout", ["tile4", "bitplane"])
def test_all_scale_bits_preserved(layout):
    raw = bytearray(65536 * 18)
    for value in range(65536):
        struct.pack_into("<H", raw, value * 18, value)
        raw[value * 18 + 2:value * 18 + 18] = bytes(range(16))
    assert cf.unpack_rows(cf.pack_rows(bytes(raw), 32, layout), 32, layout) == raw


@pytest.mark.parametrize("layout", ["tile4", "bitplane"])
def test_compile_inventory_and_streamed_identity(tmp_path, layout):
    model = tmp_path / "source.gguf"
    raw, offsets = write_source(model)
    out = tmp_path / layout
    plan = cf.compile_model(model, out, layout=layout)
    assert plan["format"] == "cf-engine-plan" and plan["version"] == 1
    assert plan["source"]["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert len(plan["inventory"]) == 5
    assert {t["type"] for t in plan["inventory"]} == {"Q4_0", "F32", "F16"}
    assert len(plan["tensors"]) == 1
    tensor = plan["tensors"][0]
    assert tensor["shape"] == [64, 8, 3]
    assert tensor["source_offset"] == offsets[tensor["name"]]
    assert tensor["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert (out / tensor["file"]).read_bytes() == scalar_pack(raw, 64, layout)
    assert plan["memory"]["duplicate_source_and_packed_bytes"] == len(raw)
    assert next(t for t in plan["inventory"] if t["name"].startswith("per_layer"))["access"] == "token-indexed-per-layer-lookup"
    assert cf.verify_artifact(out / "manifest.json", [model]) == plan
    # Force chunks smaller than a single row tile; conversion must still retain
    # exact ordering over expert boundaries and the final partial chunk.
    tiny = tmp_path / "tiny-chunks.cfq4"
    cf.pack_tensor(model, tensor["source_offset"], tensor["shape"], tiny, layout, chunk_bytes=1)
    assert tiny.read_bytes() == (out / tensor["file"]).read_bytes()


def test_layers_projection_and_declared_fallback(tmp_path):
    model = tmp_path / "source.gguf"
    write_source(model)
    plan = cf.compile_model(model, tmp_path / "artifact", layers="0-1", projections="both")
    assert len(plan["tensors"]) == 2
    unsupported = next(t for t in plan["inventory"] if t["name"].startswith("blk.1"))
    assert unsupported["cf_supported"] is False
    assert "F32" in unsupported["fallback_reason"]
    with pytest.raises(cf.ArtifactError, match="no supported tensors"):
        cf.compile_model(model, tmp_path / "no-tensors", layers="1")
    with pytest.raises(cf.ArtifactError, match="must be empty"):
        cf.compile_model(model, tmp_path / "artifact")


@pytest.mark.parametrize("shape", [[0, 4, 1], [31, 4, 1], [32, 3, 1], [32, 4, 0], [32, 4],
                                    [32, 4, 1.0], [32, 4, True], [32, 1 << 60, 1 << 20]])
def test_invalid_shape_rejected(shape):
    with pytest.raises(cf.ArtifactError):
        cf.packed_size(shape)


@pytest.mark.parametrize("field,value", [("format", "unknown"), ("version", 2), ("version", True)])
def test_format_corruption(tmp_path, field, value):
    model = tmp_path / "source.gguf"
    write_source(model)
    out = tmp_path / "artifact"
    plan = cf.compile_model(model, out)
    plan[field] = value
    (out / "manifest.json").write_text(json.dumps(plan))
    with pytest.raises(cf.ArtifactError, match="format/version"):
        cf.verify_artifact(out / "manifest.json")


@pytest.mark.parametrize("mutation", ["truncate", "payload", "digest", "inverse", "bounds", "shape", "type",
                                      "duplicate", "path", "windows_path", "absolute", "inventory"])
def test_artifact_corruption_rejected(tmp_path, mutation):
    model = tmp_path / "source.gguf"
    write_source(model)
    out = tmp_path / "artifact"
    plan = cf.compile_model(model, out)
    tensor = plan["tensors"][0]
    payload = out / tensor["file"]
    manifest = out / "manifest.json"
    if mutation == "truncate":
        payload.write_bytes(payload.read_bytes()[:-1])
    elif mutation == "payload":
        raw = bytearray(payload.read_bytes()); raw[-1] ^= 0x80; payload.write_bytes(raw)
    elif mutation == "digest":
        tensor["sha256"] = "f" * 64
    elif mutation == "inverse":
        raw = bytearray(payload.read_bytes()); raw[0] ^= 1; payload.write_bytes(raw)
        tensor["sha256"] = hashlib.sha256(raw).hexdigest()
    elif mutation == "bounds":
        plan["inventory"][0]["source_offset"] = model.stat().st_size
    elif mutation == "shape":
        tensor["shape"] = [64, 8, 4]
    elif mutation == "type":
        tensor["type"] = "Q4_1"
    elif mutation == "duplicate":
        plan["tensors"].append(copy.deepcopy(tensor))
    elif mutation == "path":
        tensor["file"] = "../tensor.cfq4"
    elif mutation == "windows_path":
        tensor["file"] = "C:\\tensor.cfq4"
    elif mutation == "absolute":
        tensor["file"] = "/tmp/tensor.cfq4"
    elif mutation == "inventory":
        plan["inventory"][0]["source_sha256"] = "0" * 64
    manifest.write_text(json.dumps(plan))
    with pytest.raises(cf.ArtifactError):
        cf.verify_artifact(manifest)


def test_source_bounds_and_model_identity(tmp_path):
    model = tmp_path / "source.gguf"
    write_source(model)
    out = tmp_path / "artifact"
    plan = cf.compile_model(model, out)
    original = model.read_bytes()
    model.write_bytes(original[:-16])
    with pytest.raises(cf.ArtifactError, match="bounds"):
        cf.compile_model(model, tmp_path / "truncated")
    model.write_bytes(b"BAD!" + original[4:])
    with pytest.raises(cf.ArtifactError, match="identity"):
        cf.verify_artifact(out / "manifest.json", [model])
    with pytest.raises(cf.ArtifactError, match="bounds"):
        cf.pack_tensor(model, model.stat().st_size, [32, 4, 1], tmp_path / "bad.cfq4", "tile4")


def test_source_hash_mismatch_leaves_no_payload(tmp_path):
    model = tmp_path / "source.gguf"
    _, offsets = write_source(model)
    output = tmp_path / "bad.cfq4"
    with pytest.raises(cf.ArtifactError, match="changed"):
        cf.pack_tensor(model, offsets["blk.0.ffn_gate_up_exps.weight"], [64, 8, 3], output,
                       "bitplane", expected_source_sha256="0" * 64)
    assert not output.exists()


def test_alternative_sources_remain_declared_not_substitute_for_identity(tmp_path):
    model = tmp_path / "source.gguf"
    write_source(model)
    alternatives = tmp_path / "alternatives.json"
    identity = [{"name": "pi-split-00001-of-00005.gguf", "bytes": 1234, "sha256": "1" * 64}]
    alternatives.write_text(json.dumps(identity))
    plan = cf.compile_model(model, tmp_path / "artifact", alternative_source_identities=alternatives)
    assert plan["source"]["alternative_file_identities"] == identity
    assert "declared-only" in plan["source"]["alternative_identity_status"]
    with pytest.raises(cf.ArtifactError, match="identity"):
        cf.verify_artifact(tmp_path / "artifact" / "manifest.json", [alternatives])


def test_input_and_layout_rejection():
    for layout in ("tile4", "bitplane"):
        with pytest.raises(cf.ArtifactError):
            cf.pack_rows(bytes(71), 32, layout)
        with pytest.raises(cf.ArtifactError):
            cf.unpack_rows(bytes(71), 32, layout)
    with pytest.raises(cf.ArtifactError, match="layout"):
        cf.pack_rows(bytes(72), 32, "future-v3")
    assert cf.parse_layers("0,2-4") == {0, 2, 3, 4}
    with pytest.raises(cf.ArtifactError):
        cf.parse_layers("4-2")


@pytest.mark.integration
@pytest.mark.parametrize("projection", ["gate_up", "down"])
@pytest.mark.parametrize("layout", ["tile4", "bitplane"])
def test_real_google_26b_single_expert(tmp_path, projection, layout):
    model = Path(__file__).resolve().parents[1] / "models/google__gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf"
    if not model.is_file():
        pytest.skip("desktop Google QAT 26B source not present")
    header = cf.read_gguf_header(model, backend="stdlib")
    tensor = next(t for t in header.tensors if t.name == f"blk.0.ffn_{projection}_exps.weight")
    assert tensor.type == "Q4_0"
    shape = [*tensor.shape[:2], 1]
    with model.open("rb") as stream:
        stream.seek(tensor.offset)
        raw = stream.read(cf.packed_size(shape))
    output = tmp_path / "real-one-expert.cfq4"
    digest = hashlib.sha256(raw).hexdigest()
    record = cf.pack_tensor(model, tensor.offset, shape, output, layout, digest)
    assert record["source_sha256"] == digest
    assert cf.unpack_rows(output.read_bytes(), shape[0], layout) == raw
