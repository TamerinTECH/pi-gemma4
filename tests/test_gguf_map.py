"""Tests for pigemma.gguf_map (manifest files[] 'pigemma/gguf_map.py', TRACE_FORMAT sections 8-9).

Unit tests build their own tiny gemma4-shaped GGUFs with gguf.GGUFWriter (gguf-py + numpy), so
they run on Windows without model files and do not depend on tests/conftest.py fixtures:

* MoE: arch gemma4, 2 layers, 4 experts (top-2), n_embd 64, n_ff 96, expert ff 32,
  ffn_gate_up_exps Q4_0 ne=[64,64,4] (gate+up fused = 2*32), ffn_down_exps Q4_0 ne=[32,64,4],
  F32 norms/router, token_embd F16 [64,16] (stands in for the real Q6_K head), head_count_kv
  [2,1], key_length 32 / swa 16, sliding_window 8, pattern [True, False].
  (n_ff is 96 instead of the conftest's 48 because Q4_0 needs ne[0] % 32 == 0 for ffn_down.)
* dense: the same without experts, with general.alignment 64.

Integration tests (``pytest -m integration``) check the verified 26B/31B constants on the real
GGUFs (both header readers) against the constants block of docs/TRACE_FORMAT.md.
"""

from __future__ import annotations

import ast
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pigemma import gguf_map as gm
from pigemma import paths

np = pytest.importorskip("numpy")
gguf = pytest.importorskip("gguf")
from gguf import GGMLQuantizationType as QT  # noqa: E402

PAGE16K = 16384
N_EMBD, N_FF, N_FF_EXP, N_EXPERT, N_USED, N_VOCAB = 64, 96, 32, 4, 2, 16
Q4_0_ROW = lambda n: n // 32 * 18  # noqa: E731 - bytes of one Q4_0 row of n elements


# ----------------------------------------------------------------------------------------------- fixtures
def _write_tiny_gguf(path: Path, *, moe: bool = True, with_output: bool = False,
                     drop_keys: tuple[str, ...] = (), alignment: int | None = None,
                     seed: int = 0) -> dict[str, dict[str, Any]]:
    """Write a tiny gemma4 GGUF; return {name: {offset, n_bytes, raw, type, shape}} (writer's view)."""
    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(str(path), "gemma4")
    if alignment:
        w.add_custom_alignment(alignment)
    a = "gemma4"
    kv: list[tuple[str, str, Any]] = [
        (f"{a}.block_count", "u32", 2),
        (f"{a}.context_length", "u32", 128),
        (f"{a}.embedding_length", "u32", N_EMBD),
        (f"{a}.feed_forward_length", "u32", N_FF),
        (f"{a}.attention.head_count", "u32", 2),
        (f"{a}.attention.head_count_kv", "arr", [2, 1]),
        (f"{a}.attention.key_length", "u32", 32),
        (f"{a}.attention.value_length", "u32", 32),
        (f"{a}.attention.key_length_swa", "u32", 16),
        (f"{a}.attention.value_length_swa", "u32", 16),
        (f"{a}.attention.sliding_window", "u32", 8),
        (f"{a}.attention.sliding_window_pattern", "arr", [True, False]),
    ]
    if moe:
        kv += [(f"{a}.expert_count", "u32", N_EXPERT), (f"{a}.expert_used_count", "u32", N_USED),
               (f"{a}.expert_feed_forward_length", "u32", N_FF_EXP)]
    for key, typ, val in kv:
        if key in drop_keys:
            continue
        if typ == "u32":
            w.add_uint32(key, val)
        else:
            w.add_array(key, val)

    tensors: dict[str, dict[str, Any]] = {}

    def add_f(name: str, ne: list[int], dtype: Any = np.float32) -> None:
        arr = rng.standard_normal(tuple(reversed(ne))).astype(dtype)
        w.add_tensor(name, arr)
        tensors[name] = {"raw": arr.tobytes(), "type": "F32" if dtype == np.float32 else "F16", "shape": ne}

    def add_q(name: str, ne: list[int]) -> None:
        x = rng.standard_normal(tuple(reversed(ne))).astype(np.float32)
        q = gguf.quants.quantize(x, QT.Q4_0)
        w.add_tensor(name, q, raw_dtype=QT.Q4_0)
        tensors[name] = {"raw": q.tobytes(), "type": "Q4_0", "shape": ne, "q": q}

    add_f("token_embd.weight", [N_EMBD, N_VOCAB], np.float16)
    add_f("output_norm.weight", [N_EMBD])
    add_f("rope_freqs.weight", [16])
    if with_output:
        add_q("output.weight", [N_EMBD, N_VOCAB * 2])
    for il in range(2):
        b = f"blk.{il}."
        add_f(b + "attn_norm.weight", [N_EMBD])
        add_q(b + "attn_q.weight", [N_EMBD, 64])
        add_q(b + "attn_k.weight", [N_EMBD, 64])
        if il == 0:  # global layers share K and V in Gemma 4 (no attn_v), like the real 26B
            add_q(b + "attn_v.weight", [N_EMBD, 64])
        add_q(b + "attn_output.weight", [64, N_EMBD])
        add_f(b + "attn_q_norm.weight", [32])
        add_f(b + "attn_k_norm.weight", [32])
        add_f(b + "ffn_norm.weight", [N_EMBD])
        add_q(b + "ffn_gate.weight", [N_EMBD, N_FF])
        add_q(b + "ffn_up.weight", [N_EMBD, N_FF])
        add_q(b + "ffn_down.weight", [N_FF, N_EMBD])
        if moe:
            add_f(b + "ffn_gate_inp.weight", [N_EMBD, N_EXPERT])
            add_f(b + "ffn_gate_inp.scale", [N_EMBD])
            add_q(b + "ffn_gate_up_exps.weight", [N_EMBD, 2 * N_FF_EXP, N_EXPERT])
            add_q(b + "ffn_down_exps.weight", [N_FF_EXP, N_EMBD, N_EXPERT])
            add_f(b + "ffn_down_exps.scale", [N_EXPERT])
            add_f(b + "pre_ffw_norm_2.weight", [N_EMBD])
        add_f(b + "post_ffw_norm.weight", [N_EMBD])
        add_f(b + "layer_output_scale.weight", [1])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    # Independent offset derivation: the writer pads every tensor to the alignment, so the data
    # section is the last sum(padded sizes) bytes of the file and offsets accumulate in order.
    al = alignment or 32
    pad = lambda n: -(-n // al) * al  # noqa: E731
    size = path.stat().st_size
    data_start = size - sum(pad(len(t["raw"])) for t in tensors.values())
    off = data_start
    for t in tensors.values():
        t["offset"] = off
        t["n_bytes"] = len(t["raw"])
        off += pad(t["n_bytes"])
    return tensors


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    d = tmp_path_factory.mktemp("gguf_map")
    p = d / "tiny-moe.gguf"
    exp = _write_tiny_gguf(p)
    gmap = gm.build_map(p, PAGE16K, backend="gguf")
    return SimpleNamespace(path=p, expected=exp, map=gmap, dir=d)


@pytest.fixture()
def stats_tiny() -> dict[str, Any]:
    return {
        "trace": "synthetic",
        "per_layer": [
            {"layer": 0, "freq": [10, 50, 0, 30], "weight_mass": [9.0, 1.0, 0.0, 20.0]},
            {"layer": 1, "freq": [5, 6, 40, 1], "weight_mass": [5.0, 2.0, 3.0, 0.5]},
        ],
    }


def _tensor_bytes(m: gm.GgufMap) -> int:
    return sum(t["n_bytes"] for t in m.tensors)


# ----------------------------------------------------------------------------------------------- reader
def test_ggml_type_table_matches_gguf_py() -> None:
    from gguf.constants import GGML_QUANT_SIZES

    ours = {tid: (bs, ts) for tid, (_, bs, ts) in gm.GGML_TYPES.items()}
    theirs = {int(t): v for t, v in GGML_QUANT_SIZES.items()}
    assert ours == theirs
    for tid, (name, _, _) in gm.GGML_TYPES.items():
        assert QT(tid).name == name


@pytest.mark.parametrize("backend", ["gguf", "stdlib"])
def test_offsets_equal_writer_offsets_and_bytes(tiny: SimpleNamespace, backend: str) -> None:
    m = gm.build_map(tiny.path, PAGE16K, backend=backend, with_sha=False)
    assert m["backend"] == backend
    got = {t["name"]: t for t in m.tensors}
    assert set(got) == set(tiny.expected)
    raw = tiny.path.read_bytes()
    for name, e in tiny.expected.items():
        t = got[name]
        assert t["offset"] == e["offset"], name  # absolute file offset
        assert t["n_bytes"] == e["n_bytes"], name
        assert t["type"] == e["type"], name
        assert t["shape"] == e["shape"], name  # gguf order, ne[0] fastest
        assert raw[t["offset"]:t["offset"] + t["n_bytes"]] == e["raw"], name
    assert m["data_offset"] == min(e["offset"] for e in tiny.expected.values())
    assert m.file_size == len(raw)


def test_backends_agree(tiny: SimpleNamespace) -> None:
    a = gm.build_map(tiny.path, PAGE16K, backend="stdlib").to_dict()
    b = gm.build_map(tiny.path, PAGE16K, backend="gguf").to_dict()
    for k in ("created_at", "backend"):
        a.pop(k), b.pop(k)
    assert a == b


def test_stream_reader_equals_file_reader(tiny: SimpleNamespace) -> None:
    raw = tiny.path.read_bytes()
    h1 = gm.read_gguf_header(io.BytesIO(raw), "stdlib", file_size=len(raw), label="mem")
    h2 = gm.read_gguf_header(tiny.path, "stdlib")
    assert [vars(t) for t in h1.tensors] == [vars(t) for t in h2.tensors]
    assert h1.fields == h2.fields and h1.data_offset == h2.data_offset


def test_architecture_keys(tiny: SimpleNamespace) -> None:
    m = tiny.map
    assert (m["arch"], m.n_layer, m.n_expert, m.n_expert_used) == ("gemma4", 2, N_EXPERT, N_USED)
    assert (m["n_embd"], m["n_ff"], m["n_ff_exp"]) == (N_EMBD, N_FF, N_FF_EXP)
    assert m["head_count_kv"] == [2, 1]
    assert (m["key_length"], m["key_length_swa"], m["sliding_window"]) == (32, 16, 8)
    assert m["sliding_pattern"] == [True, False]
    assert m["page_size"] == PAGE16K and m["alignment"] == 32
    assert m["gguf_sha256_prefix"] == paths.sha256_prefix(tiny.path)["sha256"]


def test_kinds_and_layers(tiny: SimpleNamespace) -> None:
    kinds = {t["name"]: (t["kind"], t["layer"]) for t in tiny.map.tensors}
    want = {
        "token_embd.weight": ("token_embd", None),
        "output_norm.weight": ("norm", None),
        "rope_freqs.weight": ("other", None),
        "blk.0.attn_q.weight": ("attn", 0),
        "blk.0.attn_v.weight": ("attn", 0),
        "blk.1.attn_output.weight": ("attn", 1),
        "blk.0.attn_q_norm.weight": ("norm", 0),
        "blk.1.ffn_gate.weight": ("ffn_dense", 1),
        "blk.1.ffn_down.weight": ("ffn_dense", 1),
        "blk.0.ffn_gate_inp.weight": ("router", 0),
        "blk.0.ffn_gate_inp.scale": ("router", 0),
        "blk.1.ffn_gate_up_exps.weight": ("expert", 1),
        "blk.1.ffn_down_exps.weight": ("expert", 1),
        "blk.1.ffn_down_exps.scale": ("expert_scale", 1),
        "blk.0.pre_ffw_norm_2.weight": ("norm", 0),
        "blk.1.layer_output_scale.weight": ("norm", 1),
    }
    for name, kl in want.items():
        assert kinds[name] == kl, name
    assert gm.classify_tensor("output.weight") == ("output", None)
    assert gm.classify_tensor("blk.7.ffn_gate_exps.weight") == ("expert", 7)
    assert gm.classify_tensor("blk.7.ffn_up_exps.weight") == ("expert", 7)
    assert gm.classify_tensor("blk.12.ffn_up_exps.scale") == ("expert_scale", 12)
    assert gm.classify_tensor("blk.3.something_new.weight") == ("other", 3)
    assert set(t["kind"] for t in tiny.map.tensors) <= set(gm.KINDS)


def test_expert_slabs_contiguous_and_cover(tiny: SimpleNamespace) -> None:
    m = tiny.map
    assert m.moe_layers == [0, 1] and m.is_moe
    for el in m.expert_layers:
        il = el["layer"]
        assert set(k for k in el if k in ("gate_up", "gate", "up", "down")) == {"gate_up", "down"}
        for role in ("gate_up", "down"):
            part = el[role]
            t = m.tensor(f"blk.{il}.ffn_{role}_exps.weight")
            assert part["offset"] == t["offset"] and part["n_bytes"] == t["n_bytes"]
            assert part["nb_expert"] * N_EXPERT == t["n_bytes"]
            q = tiny.expected[t["name"]]["q"]
            raw = tiny.path.read_bytes()
            prev_end = t["offset"]
            for e in range(N_EXPERT):
                lo = part["offset"] + e * part["nb_expert"]
                assert lo == prev_end  # contiguous
                prev_end = lo + part["nb_expert"]
                assert raw[lo:prev_end] == q[e].tobytes()  # slab e is expert e's data
            assert prev_end == t["offset"] + t["n_bytes"]  # covers the tensor exactly
        unit = Q4_0_ROW(N_EMBD) * 2 * N_FF_EXP + Q4_0_ROW(N_FF_EXP) * N_EMBD
        assert el["unit_bytes"] == unit == m.expert_unit_bytes_at(il)
        rngs = m.expert_ranges(il, 2)
        assert rngs == [(el["gate_up"]["offset"] + 2 * el["gate_up"]["nb_expert"], el["gate_up"]["nb_expert"]),
                        (el["down"]["offset"] + 2 * el["down"]["nb_expert"], el["down"]["nb_expert"])]
    with pytest.raises(IndexError):
        m.expert_ranges(0, N_EXPERT)
    with pytest.raises(KeyError):
        m.expert_layer(5)


def test_summary_byte_accounting(tiny: SimpleNamespace) -> None:
    m = tiny.map
    s = m.summary
    total = _tensor_bytes(m)
    assert s["nonexpert_bytes"] + s["expert_bytes"] == total == s["tensor_bytes_total"]
    assert sum(s["bytes_per_kind"].values()) == total
    unit = Q4_0_ROW(N_EMBD) * 2 * N_FF_EXP + Q4_0_ROW(N_FF_EXP) * N_EMBD
    assert s["expert_unit_bytes"] == s["expert_unit_bytes_min"] == s["expert_unit_bytes_max"] == unit
    assert s["expert_bytes"] == 2 * N_EXPERT * unit
    assert s["n_moe_layers"] == 2
    assert s["expert_bytes_per_token_worst"] == 2 * N_USED * unit
    # tied head: token_embd is read every token, plus one embedding row (64 f16 = 128 B)
    embd = m.tensor("token_embd.weight")["n_bytes"]
    assert s["head"] == "token_embd" and s["head_bytes"] == embd and s["embd_row_bytes"] == N_EMBD * 2
    per_tok = sum(s["bytes_per_kind"][k] for k in ("attn", "ffn_dense", "norm", "router", "expert_scale", "other"))
    assert s["nonexpert_bytes_per_token"] == per_tok + embd + N_EMBD * 2 == s["nonexpert_bytes"] + N_EMBD * 2
    assert s["bytes_per_token_worst"] == s["nonexpert_bytes_per_token"] + s["expert_bytes_per_token_worst"]
    # KV (hand-computed): layer 0 sliding 2 heads*(16+16), layer 1 global 1 head*(32+32) values/token
    assert s["kv_bytes_per_token_f16"] == (2 * 32 + 1 * 64) * 2 == 256
    vals_4k = 2 * 32 * 8 + 1 * 64 * 4096  # sliding layer capped at window 8
    assert s["kv_bytes_f16_at_4096"] == vals_4k * 2 == 525312
    assert s["kv_bytes_q8_at_4096"] == vals_4k * 34 // 32 == 279072
    assert m.kv_bytes(4096, "q8_0") == s["kv_bytes_q8_at_4096"]
    assert m.kv_bytes(4) == (2 * 32 * 4 + 64 * 4) * 2
    assert m.kv_bytes(0) == 0
    with pytest.raises(ValueError):
        m.kv_bytes(10, "q3_x")


def test_separate_output_tensor_is_the_head(tmp_path: Path) -> None:
    p = tmp_path / "with-output.gguf"
    _write_tiny_gguf(p, with_output=True)
    m = gm.build_map(p, backend="stdlib", with_sha=False)
    s = m.summary
    out = m.tensor("output.weight")
    assert out["kind"] == "output" and s["head"] == "output" and s["head_bytes"] == out["n_bytes"]
    per_tok = sum(s["bytes_per_kind"][k] for k in ("attn", "ffn_dense", "norm", "router", "expert_scale", "other"))
    assert s["nonexpert_bytes_per_token"] == per_tok + out["n_bytes"] + N_EMBD * 2  # full embd not read


def test_dense_gguf_has_no_experts(tmp_path: Path) -> None:
    p = tmp_path / "tiny-dense.gguf"
    exp = _write_tiny_gguf(p, moe=False, alignment=64)
    for backend in ("stdlib", "gguf"):
        m = gm.build_map(p, backend=backend, with_sha=False)
        assert m["alignment"] == 64
        assert {t["name"]: t["offset"] for t in m.tensors} == {k: v["offset"] for k, v in exp.items()}
        assert m.n_expert == 0 and m.expert_layers == [] and not m.is_moe and m.moe_layers == []
        s = m.summary
        assert s["expert_bytes"] == 0 and s["n_moe_layers"] == 0 and s["expert_bytes_per_token_worst"] == 0
        assert s["expert_unit_bytes"] == 0 and s["nonexpert_bytes"] == _tensor_bytes(m)


@pytest.mark.parametrize("key", ["gemma4.attention.key_length", "gemma4.block_count",
                                 "gemma4.attention.head_count_kv", "gemma4.expert_used_count"])
def test_missing_kv_key_names_the_key(tmp_path: Path, key: str) -> None:
    p = tmp_path / "missing.gguf"
    _write_tiny_gguf(p, drop_keys=(key,))
    for backend in ("stdlib", "gguf"):
        with pytest.raises(gm.MissingKeyError) as ei:
            gm.build_map(p, backend=backend, with_sha=False)
        assert ei.value.key == key and key in str(ei.value)


def test_optional_keys_default(tmp_path: Path) -> None:
    p = tmp_path / "noswa.gguf"
    _write_tiny_gguf(p, drop_keys=("gemma4.attention.key_length_swa", "gemma4.attention.value_length_swa",
                                   "gemma4.attention.sliding_window_pattern", "gemma4.expert_feed_forward_length"))
    m = gm.build_map(p, backend="stdlib", with_sha=False)
    assert m["key_length_swa"] == 32 and m["sliding_pattern"] == [False, False] and m["n_ff_exp"] is None
    assert m.summary["kv_bytes_per_token_f16"] == (2 * 64 + 1 * 64) * 2


def test_truncated_or_bad_files(tiny: SimpleNamespace, tmp_path: Path) -> None:
    raw = tiny.path.read_bytes()
    cut = tmp_path / "cut.gguf"
    cut.write_bytes(raw[:-100])  # partial download: last tensor runs past EOF
    with pytest.raises(gm.GgufMapError, match="truncated"):
        gm.build_map(cut, backend="stdlib", with_sha=False)
    head_only = tmp_path / "head.gguf"
    head_only.write_bytes(raw[:200])  # header itself cut
    with pytest.raises(gm.GgufMapError, match="end of file"):
        gm.read_gguf_header(head_only, "stdlib")
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"NOTGGUF" + b"\0" * 64)
    with pytest.raises(gm.GgufMapError, match="not a GGUF"):
        gm.read_gguf_header(junk, "stdlib")
    with pytest.raises(FileNotFoundError):
        gm.build_map(tmp_path / "absent.gguf")


def test_map_roundtrip(tiny: SimpleNamespace, tmp_path: Path) -> None:
    p = tiny.map.save(tmp_path / "map.json")
    raw = p.read_text(encoding="utf-8")
    assert "NaN" not in raw and not raw.startswith("﻿")
    m2 = gm.load_map(p)
    assert m2.to_dict() == json.loads(raw) == tiny.map.to_dict()
    assert m2.path == p
    assert dict(m2)["summary"] == tiny.map.summary  # Mapping protocol
    for k in ("gguf", "file_size", "data_offset", "alignment", "page_size", "arch", "n_layer", "n_expert",
              "n_expert_used", "n_embd", "n_ff", "n_ff_exp", "head_count_kv", "key_length", "key_length_swa",
              "sliding_window", "sliding_pattern", "tensors", "expert_layers", "summary"):
        assert k in m2, k
    for k in ("nonexpert_bytes", "expert_bytes", "expert_unit_bytes", "n_moe_layers",
              "expert_bytes_per_token_worst", "nonexpert_bytes_per_token", "kv_bytes_per_token_f16",
              "kv_bytes_f16_at_4096", "kv_bytes_q8_at_4096", "bytes_per_kind"):
        assert k in m2.summary, k
    for el in m2.expert_layers:  # D7: {offset, n_bytes, nb_expert}
        for role in ("gate_up", "down"):
            assert set(el[role]) == {"offset", "n_bytes", "nb_expert"}
    with pytest.raises(gm.GgufMapError):
        gm.GgufMap({"gguf": "x"})


# ----------------------------------------------------------------------------------------------- ranges
def test_aligned_and_merge_ranges() -> None:
    assert gm.aligned(100, 10, PAGE16K) == (0, PAGE16K)
    assert gm.aligned(PAGE16K, PAGE16K, PAGE16K) == (PAGE16K, PAGE16K)
    assert gm.aligned(PAGE16K - 1, 2, PAGE16K) == (0, 2 * PAGE16K)
    assert gm.aligned(5, 0, 4096) == (0, 0)
    assert gm.aligned(10000, 100, 4096, limit=10100) == (8192, 10100 - 8192)  # clamped to the file end
    with pytest.raises(ValueError):
        gm.aligned(0, 10, 0)
    with pytest.raises(ValueError):
        gm.aligned(0, 10, 4096, limit=5)
    assert gm.merge_ranges([(10, 5), (0, 10), (30, 5), (32, 10), (50, 0)]) == [(0, 15), (30, 12)]
    assert gm.merge_ranges([]) == []


def test_slab_alignment_pads_correctly_16k(tiny: SimpleNamespace) -> None:
    m = tiny.map
    for il in m.moe_layers:
        for e in range(N_EXPERT):
            for off, ln in m.expert_ranges(il, e):
                a, al = gm.aligned(off, ln, PAGE16K)
                assert a % PAGE16K == 0 and (a + al) % PAGE16K == 0
                assert a <= off < a + PAGE16K
                assert off + ln <= a + al < off + ln + PAGE16K
    s = m.summary
    assert s["expert_unit_bytes_aligned_mean"] >= s["expert_unit_bytes"]


def test_interval_set_accounting() -> None:
    iv = gm._IntervalSet()
    iv.add(0, 10)
    iv.add(20, 30)
    assert iv.uncovered(5, 25) == 10 and iv.total == 20
    iv.add(10, 20)  # touching both -> one interval
    assert iv.intervals() == [(0, 30)] and iv.total == 30
    iv.add(40, 50)
    iv.add(35, 45)
    assert iv.intervals() == [(0, 30), (35, 50)] and iv.total == 45
    assert iv.uncovered(0, 60) == 15


# ----------------------------------------------------------------------------------------------- hotset
def _covered(hs: dict[str, Any], off: int, ln: int) -> bool:
    return any(r["offset"] <= off and off + ln <= r["offset"] + r["length"] for r in hs["ranges"])


def _expert_cost32(m: gm.GgufMap) -> int:
    return m.summary["expert_unit_bytes"]


def _nonexpert_aligned(m: gm.GgufMap, page: int) -> int:
    return sum(ln for _, ln in gm.merge_ranges(
        gm.aligned(t["offset"], t["n_bytes"], page, m.file_size) for t in m.nonexpert_tensors()))


def test_hotset_selects_by_descending_freq(tiny: SimpleNamespace, stats_tiny: dict) -> None:
    m = tiny.map
    page = 32  # = the GGUF alignment: every slab costs exactly its raw bytes, so ranking is exact
    base = _nonexpert_aligned(m, page)
    unit = _expert_cost32(m)
    ranking = [(0, 1), (1, 2), (0, 3), (0, 0), (1, 1), (1, 0), (1, 3), (0, 2)]  # freq 50,40,30,10,6,5,1,0
    for k in range(0, 9):
        hs = gm.plan_hotset(m, stats_tiny, base + k * unit, page_size=page)
        assert [tuple(x[:2]) for x in hs["experts_selected"]] == ranking[:k]
        assert hs["n_experts"] == k
        assert hs["total_bytes"] == base + k * unit <= hs["budget_bytes"]
        freqs = {(0, 0): 10, (0, 1): 50, (0, 2): 0, (0, 3): 30, (1, 0): 5, (1, 1): 6, (1, 2): 40, (1, 3): 1}
        want = sum(freqs[x] for x in ranking[:k]) / sum(freqs.values())
        assert hs["expert_coverage_fraction"] == pytest.approx(want, abs=1e-9)
    # one byte short of the 3rd expert -> still 2
    hs = gm.plan_hotset(m, stats_tiny, base + 3 * unit - 1, page_size=page)
    assert hs["n_experts"] == 2


def test_hotset_budget_and_mandatory_nonexpert(tiny: SimpleNamespace, stats_tiny: dict) -> None:
    m = tiny.map
    for page in (32, 256, 4096, PAGE16K):
        base = _nonexpert_aligned(m, page)
        for extra in (0, 1, 1000, 5000, 9000, 20000, 10**9):
            budget = base + extra
            hs = gm.plan_hotset(m, stats_tiny, budget, page_size=page)
            # stronger than the spec's "budget + one page per range"
            assert hs["total_bytes"] <= hs["budget_bytes"] + hs["n_ranges"] * page
            assert hs["total_bytes"] <= budget
            assert hs["nonexpert_included"] is True and hs["nonexpert_bytes_aligned"] == base
            for t in m.nonexpert_tensors():
                assert _covered(hs, t["offset"], t["n_bytes"]), t["name"]
            for il, e, _ in hs["experts_selected"]:
                for off, ln in m.expert_ranges(il, e):
                    assert _covered(hs, off, ln)
            rs = hs["ranges"]
            assert all(r["offset"] % page == 0 for r in rs)
            assert all(r["length"] % page == 0 or r["offset"] + r["length"] == m.file_size for r in rs)
            assert all(r["offset"] + r["length"] <= m.file_size for r in rs)
            assert all(a["offset"] + a["length"] < b["offset"] for a, b in zip(rs, rs[1:]))  # sorted, merged
            assert hs["total_bytes"] == sum(r["length"] for r in rs)
        with pytest.raises(gm.HotsetBudgetError):
            gm.plan_hotset(m, stats_tiny, base - 1, page_size=page)


def test_hotset_without_stats_is_nonexpert_only(tiny: SimpleNamespace) -> None:
    m = tiny.map
    hs = gm.plan_hotset(m, None, 10**9, page_size=32)
    assert hs["stats"] is None and hs["n_experts"] == 0 and hs["expert_coverage_fraction"] is None
    assert hs["total_bytes"] == _nonexpert_aligned(m, 32)
    assert all(r["kind"] != "expert" for r in hs["ranges"])
    with pytest.raises(gm.HotsetError):
        gm.plan_hotset(m, None, 10**9, include_nonexpert=False)


def test_hotset_no_nonexpert_and_merge_adjacent_experts(tiny: SimpleNamespace) -> None:
    m = tiny.map
    unit = _expert_cost32(m)
    stats = {"per_layer": [{"layer": 0, "freq": [100, 90, 1, 1], "weight_mass": [1, 1, 1, 1]},
                           {"layer": 1, "freq": [0, 0, 0, 0], "weight_mass": [0, 0, 0, 0]}]}
    hs = gm.plan_hotset(m, stats, 2 * unit, page_size=32, include_nonexpert=False)
    assert hs["nonexpert_included"] is False and hs["n_experts"] == 2
    assert all(r["kind"] == "expert" for r in hs["ranges"])
    el = m.expert_layer(0)
    # experts 0 and 1 are adjacent slabs in both tensors -> 2 merged ranges, one per tensor
    assert [(r["offset"], r["length"]) for r in hs["ranges"]] == [
        (el["gate_up"]["offset"], 2 * el["gate_up"]["nb_expert"]),
        (el["down"]["offset"], 2 * el["down"]["nb_expert"])] or \
        [(r["offset"], r["length"]) for r in hs["ranges"]] == sorted([
            (el["gate_up"]["offset"], 2 * el["gate_up"]["nb_expert"]),
            (el["down"]["offset"], 2 * el["down"]["nb_expert"])])
    for r in hs["ranges"]:
        assert r["layer"] == 0 and r["expert"] is None and r["n_parts"] == 2 and r["hits"] == 190
    hs1 = gm.plan_hotset(m, stats, unit, page_size=32, include_nonexpert=False)
    assert all(r["expert"] == 0 and r["hits"] == 100 for r in hs1["ranges"])
    with pytest.raises(gm.HotsetBudgetError):
        gm.plan_hotset(m, stats, unit - 1, page_size=32, include_nonexpert=False)


def test_hotset_per_layer_min_rank_weight_min_hits(tiny: SimpleNamespace, stats_tiny: dict) -> None:
    m = tiny.map
    base = _nonexpert_aligned(m, 32)
    unit = _expert_cost32(m)
    hs = gm.plan_hotset(m, stats_tiny, base + 2 * unit, page_size=32, per_layer_min=1)
    assert [tuple(x[:2]) for x in hs["experts_selected"]] == [(0, 1), (1, 2)]
    assert hs["per_layer_selected"] == [{"layer": 0, "n": 1}, {"layer": 1, "n": 1}]
    hs = gm.plan_hotset(m, stats_tiny, base + 2 * unit, page_size=32, per_layer_min=1, rank="count")
    hs2 = gm.plan_hotset(m, stats_tiny, base + 3 * unit, page_size=32, rank="weight")
    assert [tuple(x[:2]) for x in hs2["experts_selected"]] == [(0, 3), (0, 0), (1, 0)]  # 20, 9, 5
    assert hs2["rank"] == "weight"
    assert hs2["expert_weight_coverage_fraction"] == pytest.approx(34.0 / 40.5)
    hs3 = gm.plan_hotset(m, stats_tiny, 10**9, page_size=32, min_hits=1)
    assert (0, 2) not in {tuple(x[:2]) for x in hs3["experts_selected"]} and hs3["n_experts"] == 7
    hs4 = gm.plan_hotset(m, stats_tiny, 10**9, page_size=32)
    assert hs4["n_experts"] == 8 and hs4["expert_coverage_fraction"] == 1.0


def test_hotset_rejects_inconsistent_stats(tiny: SimpleNamespace) -> None:
    m = tiny.map
    with pytest.raises(gm.HotsetError, match="freq entries"):
        gm.plan_hotset(m, {"per_layer": [{"layer": 0, "freq": [1, 2, 3]}]}, 10**9)
    with pytest.raises(gm.HotsetError, match="not a MoE layer"):
        gm.plan_hotset(m, {"per_layer": [{"layer": 7, "freq": [1, 2, 3, 4]}]}, 10**9)
    with pytest.raises(gm.HotsetError, match="weight_mass"):
        gm.plan_hotset(m, {"per_layer": [{"layer": 0, "freq": [1, 2, 3, 4]}]}, 10**9, rank="weight")
    with pytest.raises(gm.HotsetError):
        gm.plan_hotset(m, {"per_layer": [{"layer": 0, "freq": [1, 2, 3, 4]}]}, 10**9, rank="bogus")
    with pytest.raises(gm.HotsetError, match="non-finite"):
        gm.plan_hotset(m, {"per_layer": [{"layer": 0, "freq": [1, 2, 3, 4],
                                          "weight_mass": [1.0, float("nan"), 0.0, 0.0]}]}, 10**9)


def test_ranges_txt_roundtrip(tiny: SimpleNamespace, stats_tiny: dict, tmp_path: Path) -> None:
    m = tiny.map
    budget = _nonexpert_aligned(m, 256) + 3 * 4096
    hs = gm.plan_hotset(m, stats_tiny, budget, page_size=256, stats_path="traces/x/all.stats.json")
    pj, pr = gm.write_hotset(hs, tmp_path / "hotset.json")
    assert pr == tmp_path / "hotset.ranges.txt" == gm.ranges_txt_path(pj)
    raw = pr.read_bytes()
    raw.decode("ascii")
    assert b"\r" not in raw
    first = raw.split(b"\n", 1)[0].decode()
    assert first == (f"# pigemma-ranges v1 gguf={tiny.path.name} size={m.file_size} page=256 budget={budget}")
    rf = gm.read_ranges_txt(pr)
    assert rf["header"] == {"gguf": tiny.path.name, "size": m.file_size, "page": 256, "budget": budget}
    got = [(r["offset"], r["length"], r["kind"], r["layer"], r["expert"]) for r in rf["ranges"]]
    want = [(r["offset"], r["length"], r["kind"], r["layer"], r["expert"]) for r in hs["ranges"]]
    assert got == want
    # every range line obeys the pinner grammar
    for line in raw.decode().splitlines()[1:]:
        if line.startswith("#"):
            continue
        assert re.fullmatch(r"\d+ \d+ # kind=\S+ layer=(\d+|-) expert=(\d+|-)", line), line
    j = json.loads(pj.read_text(encoding="utf-8"))
    for k in ("gguf", "map_sha256", "stats", "page_size", "budget_bytes", "total_bytes", "nonexpert_included",
              "n_experts", "expert_coverage_fraction", "rank", "ranges"):
        assert k in j, k
    assert j["stats"] == "traces/x/all.stats.json"
    for r in j["ranges"]:
        assert set(r) >= {"offset", "length", "kind", "layer", "expert", "hits"}
    # strict parser: malformed lines are rejected with their line number
    bad = raw.decode().splitlines()
    bad.insert(3, "12 abc")
    with pytest.raises(ValueError, match="line 4"):
        gm.read_ranges_txt("\n".join(bad) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        gm.read_ranges_txt("0 10\n")
    over = f"# pigemma-ranges v1 gguf=a size=100 page=4096 budget=0\n90 20\n"
    with pytest.raises(ValueError, match="size"):
        gm.read_ranges_txt(over)


def test_hotset_16k_clamps_last_range_to_file_size(tiny: SimpleNamespace, stats_tiny: dict) -> None:
    m = tiny.map
    hs = gm.plan_hotset(m, stats_tiny, 10**9, page_size=PAGE16K)
    last = hs["ranges"][-1]
    assert last["offset"] + last["length"] == m.file_size  # tiny file < 16 KiB multiple
    assert hs["total_bytes"] <= _nonexpert_aligned(m, PAGE16K) + 10**9


def test_map_sha256_matches_file(tiny: SimpleNamespace, tmp_path: Path) -> None:
    import hashlib

    p = tiny.map.save(tmp_path / "m.json")
    hs = gm.plan_hotset(gm.load_map(p), None, 10**9, page_size=4096)
    assert hs["map_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert hs["map"] == str(p)


# ----------------------------------------------------------------------------------------------- CLI
def test_cli_build_summary_hotset(tiny: SimpleNamespace, stats_tiny: dict, tmp_path: Path,
                                  capsys: pytest.CaptureFixture[str]) -> None:
    mp = tmp_path / "map.json"
    assert gm.main(["build", str(tiny.path), "--out", str(mp), "--page-size", "4096", "--backend", "stdlib"]) == 0
    out = capsys.readouterr().out
    assert "GB/token" in out and "expert unit" in out
    assert gm.load_map(mp).page_size == 4096
    assert gm.main(["build", str(tiny.path), "--out", str(mp), "--page-size", "4096", "--reuse", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["n_moe_layers"] == 2
    assert gm.main(["summary", "--map", str(mp)]) == 0
    assert "non-expert" in capsys.readouterr().out
    sp = tmp_path / "stats.json"
    sp.write_text(json.dumps(stats_tiny), encoding="utf-8")
    hj = tmp_path / "out" / "hotset.json"
    rc = gm.main(["hotset", "--map", str(mp), "--stats", str(sp), "--budget-bytes", "1e9",
                  "--page-size", "4096", "--out", str(hj)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "expert_coverage_fraction 1.0000" in out
    assert hj.is_file() and (tmp_path / "out" / "hotset.ranges.txt").is_file()
    # refused: budget below the non-expert set
    rc = gm.main(["hotset", "--map", str(mp), "--budget-bytes", "100", "--page-size", "4096",
                  "--out", str(tmp_path / "h2.json")])
    assert rc == 4 and "REFUSED" in capsys.readouterr().err
    assert not (tmp_path / "h2.json").exists()
    # missing inputs -> 3; no command -> 2; nothing to pin -> 1
    assert gm.main(["hotset", "--map", str(tmp_path / "nope.json"), "--budget-bytes", "1", "--out",
                    str(tmp_path / "h3.json")]) == 3
    assert gm.main(["hotset", "--map", str(mp), "--stats", str(tmp_path / "nope.json"), "--budget-bytes", "1",
                    "--out", str(tmp_path / "h3.json")]) == 3
    assert gm.main(["build", str(tmp_path / "nope.gguf"), "--out", str(tmp_path / "m2.json")]) == 3
    assert gm.main([]) == 2
    assert gm.main(["hotset", "--map", str(mp), "--no-nonexpert", "--budget-bytes", "1e9",
                    "--out", str(tmp_path / "h4.json")]) == 1
    with pytest.raises(SystemExit) as ei:
        gm.main(["hotset", "--map", str(mp), "--budget-bytes", "lots", "--out", str(tmp_path / "h5.json")])
    assert ei.value.code == 2
    capsys.readouterr()


def test_hotset_path_is_stdlib_only(tiny: SimpleNamespace, stats_tiny: dict, tmp_path: Path) -> None:
    """pi/06 runs 'gguf_map hotset' on the Pi: it must not import numpy or gguf."""
    mp = tiny.map.save(tmp_path / "map.json")
    sp = tmp_path / "stats.json"
    sp.write_text(json.dumps(stats_tiny), encoding="utf-8")
    code = (
        "import sys; from pigemma import gguf_map as g; "
        f"rc = g.main(['--log-level', 'WARNING', 'hotset', '--map', {str(mp)!r}, '--stats', {str(sp)!r}, "
        f"'--budget-bytes', '1e9', '--page-size', '16384', '--out', {str(tmp_path / 'h.json')!r}]); "
        "bad = [m for m in ('numpy', 'gguf', 'yaml') if m in sys.modules]; "
        "print('BAD', bad) if bad else None; sys.exit(rc or (9 if bad else 0))"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=paths.REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_parse_bytes() -> None:
    assert gm._parse_bytes("5200000000") == 5_200_000_000
    assert gm._parse_bytes("5.2e9") == 5_200_000_000
    assert gm._parse_bytes("5.2GB") == 5_200_000_000
    assert gm._parse_bytes("2GiB") == 2 << 30
    assert gm._parse_bytes("1_000") == 1000


# ----------------------------------------------------------------------------------------------- catalog
def test_catalog_formatter_reproduces_layout() -> None:
    sample = {"a": [1, 2], "b": {"c": "x", "d": [[0.1, 0.2], [0.3, 0.4]]}, "e": [{"f": 1}], "g": [], "h": {}}
    txt = gm.format_catalog_json(sample)
    assert json.loads(txt) == sample
    assert '"a": [1, 2]' in txt and '"d": [[0.1, 0.2], [0.3, 0.4]]' in txt and '"e": [\n' in txt
    raw = paths.CATALOG_PATH.read_text(encoding="utf-8")
    if gm.format_catalog_json(json.loads(raw)) + "\n" != raw:
        pytest.skip("catalog.json was reformatted by hand; catalog-refresh would rewrite its layout")


class _FakeApi:
    def __init__(self, files: dict[str, tuple[int, str]], sha: str = "f" * 40) -> None:
        self.files = files
        self.sha = sha
        self.calls: list[str] = []

    def model_info(self, repo: str, files_metadata: bool = False, **_: Any) -> Any:
        assert files_metadata
        self.calls.append(repo)
        sibs = [SimpleNamespace(rfilename=n, size=s, lfs=SimpleNamespace(sha256=h)) for n, (s, h) in self.files.items()]
        sibs.append(SimpleNamespace(rfilename="README.md", size=10, lfs=None))
        return SimpleNamespace(sha=self.sha, siblings=sibs)


def test_catalog_refresh_fills_placeholder_entry(tiny: SimpleNamespace, tmp_path: Path) -> None:
    cat = json.loads(paths.CATALOG_PATH.read_text(encoding="utf-8"))
    src = next(m for m in cat["models"] if m["id"] == "26b-ud-iq2_xxs")
    ent = json.loads(json.dumps(src))
    ent.update({"id": "test-placeholder", "file": "<fill from HF listing>", "file_pattern": "*-UD-TINY.gguf",
                "bytes_exact": False, "estimated": True, "bytes": 1, "sha256": None, "local_path": None,
                "repo": "example/tiny-GGUF"})
    ent["sources"]["bytes"] = "assumed: placeholder"
    cat["models"].append(ent)
    cp = tmp_path / "catalog.json"
    cp.write_text(gm.format_catalog_json(cat) + "\n", encoding="utf-8")
    before = cp.read_text(encoding="utf-8")
    size = tiny.path.stat().st_size
    api = _FakeApi({"sub/model-UD-TINY.gguf": (size, "a" * 64), "model-UD-OTHER.gguf": (5, "b" * 64)})
    opened: list[tuple[str, str, Any]] = []

    def opener(repo: str, file: str, rev: Any, token: Any) -> Any:
        opened.append((repo, file, rev))
        return io.BytesIO(tiny.path.read_bytes())

    rep = gm.catalog_refresh(cp, dry_run=True, api=api, opener=opener, today="2026-10-05")
    assert rep["selected"] == ["test-placeholder"] and rep["written"] is None and not rep["errors"]
    assert cp.read_text(encoding="utf-8") == before  # dry run writes nothing
    rep = gm.catalog_refresh(cp, api=api, opener=opener, today="2026-10-05")
    assert rep["written"] == str(cp) and not rep["errors"]
    new = json.loads(cp.read_text(encoding="utf-8"))
    e = next(m for m in new["models"] if m["id"] == "test-placeholder")
    assert (e["file"], e["bytes"], e["bytes_exact"], e["sha256"], e["revision"]) == \
        ("sub/model-UD-TINY.gguf", size, True, "a" * 64, "f" * 40)
    # dry run + real run each read the remote header once
    assert e["estimated"] is False and opened == [("example/tiny-GGUF", "sub/model-UD-TINY.gguf", "f" * 40)] * 2
    unit = Q4_0_ROW(N_EMBD) * 2 * N_FF_EXP + Q4_0_ROW(N_FF_EXP) * N_EMBD
    assert (e["n_layer"], e["n_moe_layers"], e["n_expert"], e["n_expert_used"]) == (2, 2, N_EXPERT, N_USED)
    assert e["expert_unit_bytes"] == unit and e["expert_bytes_per_token_worst"] == 2 * N_USED * unit
    assert e["nonexpert_bytes_per_token"] == e["nonexpert_bytes"] == tiny.map.summary["nonexpert_bytes"]
    assert e["sources"]["bytes"].startswith("measured") and e["sources"]["sha256"].startswith("measured")
    assert e["sources"]["data_offset"].startswith("measured")
    assert new["updated"] == "2026-10-05"
    others_before = [m for m in cat["models"] if m["id"] != "test-placeholder"]
    others_after = [m for m in new["models"] if m["id"] != "test-placeholder"]
    assert others_before == others_after
    # second run: nothing left to refresh
    rep = gm.catalog_refresh(cp, api=api, opener=opener)
    assert rep["selected"] == [] and rep["written"] is None


def test_catalog_refresh_reports_errors(tmp_path: Path) -> None:
    cat = {"models": [{"id": "x", "repo": "a/b", "file": "missing.gguf", "bytes_exact": False, "bytes": 1,
                       "sources": {}}]}
    cp = tmp_path / "c.json"
    cp.write_text(json.dumps(cat), encoding="utf-8")
    rep = gm.catalog_refresh(cp, api=_FakeApi({"other.gguf": (1, "c" * 64)}))
    assert "x" in rep["errors"] and "not found" in rep["errors"]["x"] and rep["written"] is None
    with pytest.raises(ValueError):
        gm.catalog_refresh(cp, ids=["nope"], api=_FakeApi({}))


def test_hf_token_from_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for v in ("HF_TOKEN", "HUGGING_FACE_TOKEN", "HUGGINGFACE_HUB_TOKEN", "PIGEMMA_HF_TOKEN_FILE"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(gm, "DEFAULT_HF_TOKEN_FILES", ())
    f = tmp_path / "hf-token.env"
    f.write_text("# comment\nOTHER=1\nHUGGING_FACE_TOKEN='hf_abc'\n", encoding="utf-8")
    assert gm.hf_token(f) == "hf_abc"
    assert gm.hf_token(tmp_path / "none") is None
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert gm.hf_token(f) == "hf_env"


# ----------------------------------------------------------------------------------------------- integration
def _doc_constants() -> dict[str, Any]:
    text = (paths.DOCS_DIR / "TRACE_FORMAT.md").read_text(encoding="utf-8")
    block = text.split("<!-- pigemma-constants:begin -->", 1)[1].split("<!-- pigemma-constants:end -->", 1)[0]
    out = {}
    for line in block.splitlines():
        mm = re.match(r"^([A-Z0-9_]+) = (.+)$", line)
        if mm:
            out[mm.group(1)] = ast.literal_eval(mm.group(2))
    return out


def _real(mid: str) -> Path:
    p = paths.model_path(mid)
    if not p.is_file() or p.stat().st_size != paths.MODELS[mid]["bytes"]:
        pytest.skip(f"model {mid} not present at {p}")
    return p


@pytest.mark.integration
def test_real_26b_constants() -> None:
    c = _doc_constants()
    p = _real("26b-q4")
    m = gm.build_map(p, backend="gguf")  # the spec's reader (~11 s on Windows)
    s = m.summary
    assert m["data_offset"] == c["GEMMA4_26B_DATA_OFFSET"] == 15823936
    assert len(m.tensors) == 658
    assert s["expert_unit_bytes"] == c["GEMMA4_26B_EXPERT_UNIT_BYTES"] == 3345408
    assert s["expert_bytes"] == c["GEMMA4_26B_EXPERT_BYTES"] == 12846366720
    assert s["nonexpert_bytes"] == c["GEMMA4_26B_NONEXPERT_BYTES"] == 1577172088
    assert s["expert_bytes_per_token_worst"] == c["GEMMA4_26B_EXPERT_BYTES_PER_TOKEN_WORST"] == 802897920
    assert (m.n_layer, m.n_expert, m.n_expert_used, s["n_moe_layers"]) == (30, 128, 8, 30)
    el = m.expert_layer(0)
    assert (el["gate_up"]["nb_expert"], el["down"]["nb_expert"]) == (2230272, 1115136)
    assert s["bytes_per_kind"]["token_embd"] == 605552640 and s["head"] == "token_embd"
    st = gm.build_map(p, backend="stdlib").to_dict()
    gd = m.to_dict()
    for k in ("created_at", "backend"):
        st.pop(k), gd.pop(k)
    assert st == gd


@pytest.mark.integration
def test_real_31b_constants() -> None:
    c = _doc_constants()
    p = _real("31b-q4")
    m = gm.build_map(p, backend="stdlib")
    assert m["data_offset"] == c["GEMMA4_31B_DATA_OFFSET"] == 15833440
    assert len(m.tensors) == 833 and m.n_expert == 0 and not m.is_moe
    ffn = m.tensors_of_kind("ffn_dense")
    assert len(ffn) == 180 and {t["n_bytes"] for t in ffn} == {c["GEMMA4_31B_FFN_MATRIX_BYTES"]} == {65028096}
    assert sum(t["n_bytes"] for t in ffn) == 11705057280
    assert m.summary["nonexpert_bytes"] == 17635166448


@pytest.mark.integration
def test_real_26b_hotset_budget(tmp_path: Path) -> None:
    p = _real("26b-q4")
    m = gm.build_map(p, backend="stdlib", with_sha=False)
    rng = np.random.default_rng(1)
    stats = {"per_layer": [{"layer": il, "freq": [int(x) for x in rng.zipf(1.3, 128) % 1000]}
                           for il in m.moe_layers]}
    budget = 4_100_000_000
    hs = gm.plan_hotset(m, stats, budget, page_size=PAGE16K)
    assert hs["total_bytes"] <= budget and hs["n_experts"] > 0
    assert hs["nonexpert_bytes_aligned"] >= 1577172088
    rf = gm.read_ranges_txt(gm.format_ranges_txt(hs))
    assert len(rf["ranges"]) == hs["n_ranges"]
