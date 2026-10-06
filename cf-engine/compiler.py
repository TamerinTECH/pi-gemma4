"""CF deployment compiler v1: reversible Q4_0 layouts, never requantization.

Offline only. The existing pigemma GGUF reader owns GGUF parsing; NumPy only
transposes compact integer codes in bounded chunks. No model is instantiated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import BinaryIO

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pigemma.gguf_map import ArraySummary, GGML_TYPES, GgufMapError, read_gguf_header, classify_tensor

FORMAT = "cf-engine-plan"
VERSION = 1
LAYOUTS = {"tile4": "q4-tile4-v1", "bitplane": "q4-bitplane4-v1"}
EXPERT = re.compile(r"^blk\.(\d+)\.ffn_(gate_up|gate|up|down)_exps\.weight$")
CHUNK_BYTES = 4 * 1024 * 1024
MAX_BYTES = (1 << 63) - 1


class ArtifactError(ValueError):
    """Invalid or incompatible source/artifact; fail closed before execution."""


def require_int(value, label: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= MAX_BYTES:
        raise ArtifactError(f"invalid {label}: {value!r}")
    return value


def validate_shape(shape) -> tuple[int, int, int]:
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise ArtifactError("packed tensor shape must be [K,M,E]")
    k, m, e = (require_int(v, "shape dimension", 1) for v in shape)
    if k % 32 or m % 4:
        raise ArtifactError("packed tensor requires K divisible by 32 and M divisible by 4")
    if e * (m // 4) * (k // 32) * 72 > MAX_BYTES:
        raise ArtifactError("packed tensor size overflows signed 64-bit address range")
    return k, m, e


def packed_size(shape) -> int:
    k, m, e = validate_shape(shape)
    return e * (m // 4) * (k // 32) * 72


def _layout(layout: str) -> str:
    layout = LAYOUTS.get(layout, layout)
    if layout not in LAYOUTS.values():
        raise ArtifactError(f"unsupported packed layout {layout!r}")
    return layout


def pack_rows(raw: bytes, k: int, layout: str) -> bytes:
    """Pack one or more whole four-row groups, preserving every scale bit."""
    layout = _layout(layout)
    validate_shape([k, 4, 1])
    group_bytes = k // 32 * 72
    if not raw or len(raw) % group_bytes:
        raise ArtifactError("input does not contain whole four-row Q4_0 groups")
    source = np.frombuffer(raw, np.uint8).reshape(-1, 4, k // 32, 18)
    blocks = source.transpose(0, 2, 1, 3)
    output = np.empty((source.shape[0], k // 32, 72), dtype=np.uint8)
    output[..., :8] = blocks[..., :2].reshape(*output.shape[:2], 8)
    codes = blocks[..., 2:]
    if layout == LAYOUTS["bitplane"]:
        logical = np.concatenate((codes & 15, codes >> 4), axis=-1).reshape(*codes.shape[:-1], 8, 4)
        planes = np.zeros((*codes.shape[:-1], 8, 2), dtype=np.uint8)
        for bit in range(4):
            masks = np.zeros(logical.shape[:-1], dtype=np.uint8)
            for j in range(4):
                masks |= ((logical[..., j] >> bit) & 1) << j
            planes[..., bit // 2] |= masks << (4 * (bit % 2))
        codes = planes.reshape(codes.shape)
    output[..., 8:] = codes.reshape(*output.shape[:2], 64)
    return output.tobytes()


def unpack_rows(packed: bytes, k: int, layout: str) -> bytes:
    """Inverse layout transformation; fp16 values are never decoded/re-encoded."""
    layout = _layout(layout)
    validate_shape([k, 4, 1])
    group_bytes = k // 32 * 72
    if not packed or len(packed) % group_bytes:
        raise ArtifactError("payload does not contain whole four-row tiles")
    source = np.frombuffer(packed, np.uint8).reshape(-1, k // 32, 72)
    blocks = np.empty((*source.shape[:2], 4, 18), dtype=np.uint8)
    blocks[..., :2] = source[..., :8].reshape(*source.shape[:2], 4, 2)
    codes = source[..., 8:].reshape(*source.shape[:2], 4, 16)
    if layout == LAYOUTS["bitplane"]:
        planes = codes.reshape(*codes.shape[:-1], 8, 2)
        logical = np.zeros((*codes.shape[:-1], 8, 4), dtype=np.uint8)
        for bit in range(4):
            masks = (planes[..., bit // 2] >> (4 * (bit % 2))) & 15
            for j in range(4):
                logical[..., j] |= ((masks >> j) & 1) << bit
        logical = logical.reshape(*codes.shape[:-1], 32)
        codes = logical[..., :16] | (logical[..., 16:] << 4)
    blocks[..., 2:] = codes
    return blocks.transpose(0, 2, 1, 3).tobytes()


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ArtifactError(f"truncated input: need {size} bytes, received {len(data)}")
    return data


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(CHUNK_BYTES):
            digest.update(data)
    return digest.hexdigest()


def hash_source(path: Path, header) -> tuple[str, dict[str, str]]:
    """One sequential pass hashes the complete source and every tensor range."""
    full = hashlib.sha256()
    tensor_hashes = {}
    position = 0
    with path.open("rb") as stream:
        for tensor in sorted(header.tensors, key=lambda t: t.offset):
            if tensor.offset < position:
                raise ArtifactError(f"overlapping tensor range: {tensor.name}")
            for count, digest in ((tensor.offset - position, None), (tensor.n_bytes, hashlib.sha256())):
                remaining = count
                while remaining:
                    data = _read_exact(stream, min(CHUNK_BYTES, remaining))
                    full.update(data)
                    if digest is not None:
                        digest.update(data)
                    remaining -= len(data)
                position += count
                if digest is not None:
                    tensor_hashes[tensor.name] = digest.hexdigest()
        while data := stream.read(CHUNK_BYTES):
            full.update(data)
            position += len(data)
    if position != header.file_size:
        raise ArtifactError("source size changed during identity scan")
    return full.hexdigest(), tensor_hashes


def pack_tensor(source: Path, offset: int, shape, output: Path, layout: str,
                expected_source_sha256: str | None = None, chunk_bytes: int = CHUNK_BYTES) -> dict:
    """Stream conversion, verifying a byte-exact inverse for every output chunk."""
    k, m, e = validate_shape(shape)
    size = packed_size(shape)
    offset = require_int(offset, "source offset")
    if offset + size > source.stat().st_size:
        raise ArtifactError("source tensor exceeds source file bounds")
    tile_bytes = k // 32 * 72
    if tile_bytes > CHUNK_BYTES:
        raise ArtifactError("one four-row tile exceeds the compiler chunk-memory limit")
    take = max(1, min(CHUNK_BYTES, chunk_bytes) // tile_bytes) * tile_bytes
    source_digest, packed_digest = hashlib.sha256(), hashlib.sha256()
    if output.exists():
        raise ArtifactError(f"refusing to replace existing packed payload: {output}")
    try:
        with source.open("rb") as src, output.open("xb") as dst:
            src.seek(offset)
            remaining = size
            while remaining:
                raw = _read_exact(src, min(take, remaining))
                packed = pack_rows(raw, k, layout)
                if unpack_rows(packed, k, layout) != raw:
                    raise ArtifactError("internal error: packed layout roundtrip failed")
                source_digest.update(raw)
                packed_digest.update(packed)
                dst.write(packed)
                remaining -= len(raw)
        if expected_source_sha256 and source_digest.hexdigest() != expected_source_sha256:
            raise ArtifactError("source tensor changed after inventory identity scan")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {"bytes": size, "sha256": packed_digest.hexdigest(),
            "source_sha256": source_digest.hexdigest(), "roundtrip": "byte-identical"}


def discover_sources(model: Path) -> list[Path]:
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", model.name)
    if not match:
        return [model.resolve()]
    count = int(match[3])
    if not 1 <= count <= 1000:
        raise ArtifactError("invalid GGUF split count")
    paths = [model.with_name(f"{match[1]}-{i:05d}-of-{count:05d}.gguf").resolve()
             for i in range(1, count + 1)]
    for path in paths:
        if not path.is_file():
            raise ArtifactError(f"missing source split: {path}")
    return paths


def parse_layers(value: str) -> set[int] | None:
    if value == "all":
        return None
    layers = set()
    for item in value.split(","):
        if re.fullmatch(r"\d+", item):
            layers.add(int(item))
        elif re.fullmatch(r"\d+-\d+", item):
            first, last = map(int, item.split("-"))
            if last < first or last - first > 10000:
                raise ArtifactError("invalid layer range")
            layers.update(range(first, last + 1))
        else:
            raise ArtifactError("layers must be comma-separated indices/ranges, or all")
    return layers


def access_classification(name: str, separate_output: bool) -> str:
    # Lookup tables are not a full dense matrix read on every token. These are
    # logical access classes, deliberately not a DRAM-byte estimate.
    if "per_layer" in name and ("embd" in name or "embed" in name):
        return "token-indexed-per-layer-lookup"
    if name == "token_embd.weight":
        return "token-lookup" if separate_output else "token-lookup-and-tied-output-projection"
    kind, _ = classify_tensor(name)
    return {"expert": "router-selected-expert-projection", "output": "output-projection",
            "expert_scale": "router-selected-expert-scale", "norm": "elementwise",
            "attn": "dense-projection", "ffn_dense": "dense-projection",
            "router": "router-projection"}.get(kind, "reference-defined-access")


def _json_metadata(value):
    if isinstance(value, ArraySummary):
        return {"array_element_type": value.elem_type, "array_count": value.count,
                "contents": "covered-by-source-file-sha256"}
    if isinstance(value, list):
        return [_json_metadata(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_metadata(item) for key, item in value.items()}
    return value


def _supported(tensor) -> str | None:
    if tensor.type != "Q4_0":
        return f"unsupported type {tensor.type}; reference fallback"
    try:
        validate_shape(tensor.shape)
    except ArtifactError as error:
        return f"{error}; reference fallback"
    return None


def _validate_header(header, all_names: set[str]) -> None:
    previous_end = header.data_offset
    for tensor in sorted(header.tensors, key=lambda t: t.offset):
        if not tensor.name or tensor.name in all_names:
            raise ArtifactError(f"empty or duplicate tensor name {tensor.name!r}")
        all_names.add(tensor.name)
        for dim in tensor.shape:
            require_int(dim, "source shape dimension", 1)
        require_int(tensor.offset, "source tensor offset")
        require_int(tensor.n_bytes, "source tensor size", 1)
        if tensor.offset < previous_end or tensor.offset + tensor.n_bytes > header.file_size:
            raise ArtifactError(f"tensor overlaps or exceeds source bounds: {tensor.name}")
        if tensor.offset % header.alignment:
            raise ArtifactError(f"unaligned source tensor: {tensor.name}")
        previous_end = tensor.offset + tensor.n_bytes


def _identities(path: Path | None) -> list[dict]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ArtifactError("alternative source identities must be a JSON list of file records")
    for entry in data:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ArtifactError("alternative identity must contain name, bytes and sha256")
        require_int(entry.get("bytes"), "alternative source size", 1)
        _check_sha(entry.get("sha256"))
    return data


def compile_model(model: Path, out: Path, layers: str = "0", projections: str = "gate_up",
                  layout: str = "tile4", alternative_source_identities: Path | None = None) -> dict:
    layout = _layout(layout)
    selected_layers = parse_layers(layers)
    if projections not in ("gate_up", "down", "both"):
        raise ArtifactError("projections must be gate_up, down or both")
    if out.exists() and any(out.iterdir()):
        raise ArtifactError("output directory must be empty; existing evidence is never overwritten")
    alternatives = _identities(alternative_source_identities)
    paths = discover_sources(model)
    headers = [read_gguf_header(path, backend="stdlib") for path in paths]
    all_names: set[str] = set()
    for header in headers:
        _validate_header(header, all_names)
    architecture = headers[0].fields.get("general.architecture")
    if not isinstance(architecture, str) or not architecture:
        raise ArtifactError("source has no architecture identity")
    for index, header in enumerate(headers):
        if header.fields.get("general.architecture", architecture) != architecture:
            raise ArtifactError("split sources have different architecture identities")
        if len(headers) > 1 and header.fields.get("split.no") != index:
            raise ArtifactError("source split.no disagrees with source file order")
        if len(headers) > 1 and header.fields.get("split.count") != len(headers):
            raise ArtifactError("source split.count disagrees with supplied splits")
    out.mkdir(parents=True, exist_ok=True)
    inventory, tensors, sources = [], [], []
    selected_bytes = 0
    separate_output = "output.weight" in all_names
    for file_index, (path, header) in enumerate(zip(paths, headers)):
        before = path.stat()
        digest, tensor_hashes = hash_source(path, header)
        sources.append({"name": path.name, "path": str(path), "bytes": header.file_size,
                        "sha256": digest, "gguf_version": header.version, "alignment": header.alignment,
                        "data_offset": header.data_offset})
        for tensor in header.tensors:
            match = EXPERT.fullmatch(tensor.name)
            reason = _supported(tensor) if match else "operation outside CF v1 scope; reference fallback"
            selected = bool(match and (selected_layers is None or int(match[1]) in selected_layers)
                            and (projections == "both" or (match[2] == "down") == (projections == "down")))
            entry = {"name": tensor.name, "type": tensor.type, "type_id": tensor.type_id,
                     "shape": tensor.shape, "bytes": tensor.n_bytes, "source_file": file_index,
                     "source_offset": tensor.offset, "source_sha256": tensor_hashes[tensor.name],
                     "access": access_classification(tensor.name, separate_output),
                     "cf_supported": reason is None, "selected": selected and reason is None,
                     "fallback_reason": reason or (None if selected else "not selected by compiler options")}
            inventory.append(entry)
            if not selected or reason is not None:
                continue
            filename = f"tensor-{len(tensors):04d}.cfq4"
            print(f"pack {tensor.name} {tensor.shape} {tensor.n_bytes} bytes", file=sys.stderr, flush=True)
            packed = pack_tensor(path, tensor.offset, tensor.shape, out / filename, layout,
                                 tensor_hashes[tensor.name])
            record = {key: entry[key] for key in ("name", "type", "shape", "source_file", "source_offset")}
            record.update(packed)
            record.update({"layout": layout, "file": filename, "layer": int(match[1]), "projection": match[2],
                           "kernel": "q4-dotprod-or-portable" if layout == LAYOUTS["tile4"] else "q4-bitplane-lut",
                           "quantization": {"block_elements": 32, "source_block_bytes": 18,
                                            "scale": "unchanged-little-endian-fp16-bits",
                                            "integer": "unsigned-nibble-minus-8", "requantized": False}})
            tensors.append(record)
            selected_bytes += tensor.n_bytes
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ArtifactError("source changed during compilation; discard incomplete output")
    if not tensors:
        raise ArtifactError("no supported tensors selected; inventory was inspected but no executable CF plan exists")
    k_max = max(t["shape"][0] for t in tensors)
    q8_bytes = k_max // 32 * 34
    lut_bytes = k_max // 32 * 8 * 16 * 2 if layout == LAYOUTS["bitplane"] else 0
    total_source = sum(source["bytes"] for source in sources)
    source_identity = {"architecture": architecture, "files": sources,
                       "alternative_file_identities": alternatives,
                       "alternative_identity_status": "declared-only; runtime must verify selected tensor bytes",
                       "metadata": _json_metadata(headers[0].fields)}
    if len(sources) == 1:
        source_identity.update({key: sources[0][key] for key in ("name", "bytes", "sha256")})
    plan = {"format": FORMAT, "version": VERSION, "compiler": "cf-engine/compiler.py-v1",
            "execution_contract": "semantics-preserving-against-exact-source-quantized-gguf",
            "source": source_identity, "target": {"cpu": "Cortex-A76", "isa": "aarch64",
                "runtime_feature_verification_required": True,
                "candidate_features": ["asimd", "dotprod", "fp16"], "excluded_features": ["i8mm", "sve"],
                "per_core_l2_bytes": 512 * 1024, "shared_l3_bytes": 2 * 1024 * 1024},
            "options": {"layers": layers, "projections": projections, "layout": layout},
            "tensors": tensors, "inventory": inventory,
            "plans": {"decode": {"future_tokens": 0, "order": [t["name"] for t in tensors],
                "schedule": "selected-expert-row-tiles", "input_preparation": "once-per-distinct-input-vector",
                "down_inputs": "separate-per-selected-expert", "workers": "existing-persistent-ggml-pool"},
                "prefill": {"order": [t["name"] for t in tensors], "schedule": "known-token-expert-row-tiles",
                "token_grouping": "reference MUL_MAT_ID routing; no speculative future tokens",
                "input_preparation": "once-per-distinct-input-vector"}},
            "scratch": {"scope": "per simultaneously live input, multiplied by executor input capacity",
                "q8_activation_bytes": q8_bytes, "lut_bytes": lut_bytes, "accumulator_bytes_per_worker": 128,
                "integer_temporary_bytes_per_worker": 64 if layout == LAYOUTS["bitplane"] else 0,
                "arithmetic_state_accounting": "accumulators and integer temporaries are register/compiler-local state, not extra arena allocations; not a complete stack/register bound",
                "lifetimes": [{"buffer": "q8-activation", "begin": "operation-prepare", "end": "operation-complete"},
                              {"buffer": "lut", "begin": "operation-prepare", "end": "operation-complete"}],
                "allocation_owner": "Q8 activation and LUT arenas owned by CF adapter; worker pool reused from reference GGML"},
            "memory": {"accounting_kind": "logical-file-and-buffer-estimates-not-measured-DRAM-traffic",
                "source_files_bytes": total_source, "packed_files_bytes": selected_bytes,
                "duplicate_source_and_packed_bytes": selected_bytes,
                "maximum_weight_file_bytes_if_both_resident": total_source + selected_bytes,
                "canonical_selected_representation": "packed; reference mapping remains available for fallback",
                "kv_bytes": None, "kv_accounting": "runtime-reports-actual-allocation-dependent-on-context-and-kv-types",
                "runtime_metadata_bytes": None, "runtime_metadata_accounting": "runtime-measurement-required",
                "policies": {"resident": "no expert pinning or residency bookkeeping",
                             "constrained": "requires executor-wide accounted budget; compiler alone does not enforce it"},
                "page_ranges": [{"file": t["file"], "offset": 0, "bytes": t["bytes"],
                                 "expert_stride_bytes": t["bytes"] // t["shape"][2]} for t in tensors]}}
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps(plan, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    verify_artifact(manifest)
    return plan


def _check_sha(value) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ArtifactError("invalid SHA256 digest")


def _payload_path(root: Path, name) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ArtifactError("invalid packed payload path")
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in ("..", ".") for part in pure.parts):
        raise ArtifactError("packed payload path escapes artifact directory")
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ArtifactError("packed payload symlink escapes artifact directory")
    return candidate


def verify_artifact(manifest: Path, source_paths: list[Path] | None = None) -> dict:
    """Check versions, tensor bounds, payload hashes, inverse and optional source identity."""
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("format") != FORMAT or type(data.get("version")) is not int or data["version"] != VERSION:
        raise ArtifactError("unsupported artifact format/version")
    sources = data.get("source", {}).get("files")
    if not isinstance(sources, list) or not sources:
        raise ArtifactError("missing source file identity")
    for source in sources:
        require_int(source.get("bytes"), "source file size", 1)
        require_int(source.get("data_offset"), "source file data offset")
        require_int(source.get("alignment"), "source file alignment", 1)
        _check_sha(source.get("sha256"))
    inventory = data.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ArtifactError("missing source tensor inventory")
    names = {}
    ranges = [[] for _ in sources]
    for entry in inventory:
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ArtifactError("invalid or duplicate inventory tensor name")
        index = require_int(entry.get("source_file"), "source file index")
        if index >= len(sources):
            raise ArtifactError("inventory source file index out of bounds")
        offset = require_int(entry.get("source_offset"), "source tensor offset")
        size = require_int(entry.get("bytes"), "source tensor size", 1)
        if offset < sources[index]["data_offset"] or offset + size > sources[index]["bytes"]:
            raise ArtifactError("inventory tensor exceeds source bounds")
        if offset % sources[index]["alignment"]:
            raise ArtifactError("inventory tensor is not source aligned")
        type_id = require_int(entry.get("type_id"), "source tensor type")
        if type_id not in GGML_TYPES or GGML_TYPES[type_id][0] != entry.get("type"):
            raise ArtifactError("unknown or inconsistent inventory tensor type")
        shape = entry.get("shape")
        if not isinstance(shape, list) or not 1 <= len(shape) <= 8:
            raise ArtifactError("invalid inventory tensor shape")
        elements = 1
        for dimension in shape:
            elements *= require_int(dimension, "source shape dimension", 1)
        _, block_size, block_bytes = GGML_TYPES[type_id]
        if shape[0] % block_size or elements // block_size * block_bytes != size:
            raise ArtifactError("inventory type, shape and bytes are inconsistent")
        _check_sha(entry.get("source_sha256"))
        names[name] = entry
        ranges[index].append((offset, offset + size))
    for tensor_ranges in ranges:
        previous = 0
        for start, end in sorted(tensor_ranges):
            if start < previous:
                raise ArtifactError("inventory tensor ranges overlap")
            previous = end
    tensors = data.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ArtifactError("artifact selects no tensors")
    selected, files = set(), set()
    for tensor in tensors:
        name = tensor.get("name")
        if name in selected or name not in names:
            raise ArtifactError("selected tensor is duplicate or missing from inventory")
        selected.add(name)
        if tensor.get("type") != "Q4_0":
            raise ArtifactError("unsupported selected tensor type")
        layout = _layout(tensor.get("layout"))
        expected_size = packed_size(tensor.get("shape"))
        if tensor.get("bytes") != expected_size:
            raise ArtifactError("packed size does not match tensor shape")
        for key in ("type", "shape", "bytes", "source_sha256", "source_file", "source_offset"):
            if tensor.get(key) != names[name].get(key):
                raise ArtifactError(f"selected tensor differs from source inventory: {key}")
        _check_sha(tensor.get("sha256"))
        _check_sha(tensor.get("source_sha256"))
        path = _payload_path(manifest.parent, tensor.get("file"))
        if path in files:
            raise ArtifactError("duplicate packed payload file")
        files.add(path)
        if path.stat().st_size != expected_size:
            raise ArtifactError("packed payload file size mismatch")
        packed_digest, source_digest = hashlib.sha256(), hashlib.sha256()
        k = tensor["shape"][0]
        tile_bytes = k // 32 * 72
        if tile_bytes > CHUNK_BYTES:
            raise ArtifactError("packed tile exceeds verifier chunk-memory limit")
        take = max(1, CHUNK_BYTES // tile_bytes) * tile_bytes
        with path.open("rb") as stream:
            while packed := stream.read(take):
                packed_digest.update(packed)
                source_digest.update(unpack_rows(packed, k, layout))
        if packed_digest.hexdigest() != tensor["sha256"]:
            raise ArtifactError("packed payload SHA256 mismatch")
        if source_digest.hexdigest() != tensor["source_sha256"]:
            raise ArtifactError("packed inverse does not match source tensor SHA256")
    if source_paths is not None:
        if len(source_paths) != len(sources):
            raise ArtifactError("source identity file count mismatch")
        for path, source in zip(source_paths, sources):
            if path.stat().st_size != source["bytes"] or hash_file(path) != source["sha256"]:
                raise ArtifactError("source model identity mismatch")
        # Verify semantic identity against the actual GGUF header, not merely a
        # self-consistent manifest. File hashes alone do not validate a mislabeled
        # tensor range in an accidentally corrupted or untrusted manifest.
        seen = set()
        for index, path in enumerate(source_paths):
            header = read_gguf_header(path, backend="stdlib")
            if header.fields.get("general.architecture", data["source"]["architecture"]) != data["source"]["architecture"]:
                raise ArtifactError("source architecture identity mismatch")
            for actual in header.tensors:
                expected = names.get(actual.name)
                if expected is None or any(expected[key] != value for key, value in (
                    ("source_file", index), ("type", actual.type), ("type_id", actual.type_id),
                    ("shape", actual.shape), ("source_offset", actual.offset), ("bytes", actual.n_bytes))):
                    raise ArtifactError("inventory differs from actual source GGUF header")
                seen.add(actual.name)
                if actual.name in selected:
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        stream.seek(actual.offset)
                        remaining = actual.n_bytes
                        while remaining:
                            raw = _read_exact(stream, min(CHUNK_BYTES, remaining))
                            digest.update(raw)
                            remaining -= len(raw)
                    if digest.hexdigest() != expected["source_sha256"]:
                        raise ArtifactError("selected tensor source bytes do not match inventory identity")
        if seen != set(names):
            raise ArtifactError("inventory contains tensors absent from actual source")
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="source GGUF or any numbered split (all splits are included)")
    parser.add_argument("--out", type=Path, help="new or empty artifact directory")
    parser.add_argument("--layers", default="0", help="comma-separated layer indices/ranges or all (default 0)")
    parser.add_argument("--projections", choices=("gate_up", "down", "both"), default="gate_up")
    parser.add_argument("--layout", choices=tuple(LAYOUTS), default="tile4")
    parser.add_argument("--alternative-source-identities", type=Path,
                        help="optional JSON list of declared {name,bytes,sha256} alternate split files")
    parser.add_argument("--verify", type=Path, help="verify existing manifest and optional --model source identity")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            plan = verify_artifact(args.verify, discover_sources(args.model) if args.model else None)
        else:
            if not args.model or not args.out:
                parser.error("compilation requires --model and --out")
            plan = compile_model(args.model, args.out, args.layers, args.projections, args.layout,
                                 args.alternative_source_identities)
        print(json.dumps({"format": FORMAT, "version": VERSION, "selected_tensors": len(plan["tensors"]),
                          "inventory_tensors": len(plan["inventory"]),
                          "packed_bytes": sum(t["bytes"] for t in plan["tensors"]), "verified": True}))
        return 0
    except (ArtifactError, GgufMapError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"CF compiler: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
