"""Tests for tests/synth.py, tests/conftest.py and the checked-in samples in tests/fixtures.

Manifest files[] "tests/conftest.py (+ tests/synth.py, tests/test_*.py per module)": the fixtures
must let every unit test run on Windows without models. This module checks that they do what
the consumers rely on:

* synthetic traces: shapes, Zipf popularity, exact cross-layer correlation, stickiness, the probs
  block, a TraceWriter round trip that pigemma.trace.validate accepts, and LRU / coverage
  semantics of the pattern traces (cyclic, constant);
* tiny GGUFs: KV values and tensor shapes from the manifest, the writer offsets equal gguf-py's
  reader offsets and pigemma.gguf_map's map, expert slabs, dequantisation, and the runnable llama
  model's tokenizer;
* the activation dump obeys ffn_out = W_down @ ffn_geglu;
* facts fixtures: exact data_formats keys, the probed values, and the feasibility scenario each
  one models;
* the real-binary sample files parse with the regexes the bench/quality specs name;
* conftest hooks: ``linux`` tests are skipped off Linux, ``isolated_lock`` isolates the bench lock.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

import synth  # noqa: E402
from pigemma import trace as tr  # noqa: E402

FIXTURES = synth.FIXTURES_DIR


def lru_hits(items: Any, capacity: int) -> int:
    cache: OrderedDict[int, None] = OrderedDict()
    hits = 0
    for x in np.asarray(items).tolist():
        if x in cache:
            hits += 1
            cache.move_to_end(x)
        else:
            cache[x] = None
            if len(cache) > capacity:
                cache.popitem(last=False)
    return hits


# =================================================================================================
# Synthetic traces
# =================================================================================================
def test_synth_trace_shapes_and_invariants() -> None:
    st = synth.synth_trace()
    assert (st.T, st.L, st.K, st.E, st.P) == (2000, 4, 8, 16, 0)
    assert st.ids.dtype == np.uint8 and st.weights.dtype == np.float32
    assert st.phase.dtype == np.uint8 and st.token_ids.dtype == np.int32
    assert st.ids.max() < 16
    srt = np.sort(st.ids, axis=2)
    assert not np.any(srt[..., 1:] == srt[..., :-1]), "K distinct experts per (token, layer)"
    np.testing.assert_allclose(st.weights.sum(axis=2), 1.0, atol=1e-5)
    assert np.all(np.diff(st.weights, axis=2) <= 0), "weights descending (ffn_moe_topk order)"
    # 4 equal chunks, last 20 % of each generated, pos restarts, BOS first
    assert st.n_chunks == 4 and st.n_gen == 400 and st.n_prefill == 1600
    for c in range(4):
        sl = st.chunk_slice(c)
        assert sl == slice(500 * c, 500 * (c + 1))
        assert st.pos[sl].tolist() == list(range(500))
        assert st.phase[sl].tolist() == [0] * 400 + [1] * 100
        assert st.token_ids[sl.start] == synth.GEMMA4_BOS_ID
    assert [c["source"] for c in st.chunks()] == ["en_synth", "he_synth", "en_synth", "he_synth"]
    assert st["ids"] is st.ids and set(st.keys()) >= {"ids", "weights", "phase", "chunk", "pos", "token_ids"}
    np.testing.assert_allclose(st.popularity.sum(axis=1), 1.0)
    assert st.freq().sum() == 2000 * 8 * 4


def test_synth_trace_deterministic_per_seed() -> None:
    a, b, c = synth.synth_trace(T=300, seed=5), synth.synth_trace(T=300, seed=5), synth.synth_trace(T=300, seed=6)
    for k in ("ids", "weights", "phase", "token_ids"):
        assert np.array_equal(a[k], b[k]), k
    assert not np.array_equal(a.ids, c.ids)


def test_zipf_popularity_shapes_frequencies() -> None:
    st = synth.synth_trace(T=4000, L=2, K=1, E=16, alpha=1.5, seed=1)
    f = st.freq()
    for j in range(2):
        hot, cold = int(np.argmax(st.popularity[j])), int(np.argmin(st.popularity[j]))
        assert f[j, hot] > 5 * max(f[j, cold], 1)
        # K=1: empirical frequency ~ popularity
        np.testing.assert_allclose(f[j] / f[j].sum(), st.popularity[j], atol=0.03)
    flat = synth.synth_trace(T=4000, L=1, K=8, E=16, alpha=0.0, seed=2).freq()[0]
    assert flat.max() / flat.min() < 1.25
    ent = [tr.entropy_bits(synth.synth_trace(T=3000, L=1, K=1, alpha=a, seed=3).freq()[0]) for a in (0.0, 1.0, 2.0)]
    assert ent[0] > ent[1] > ent[2]


def test_layer_corr_one_is_deterministic() -> None:
    st = synth.synth_trace(T=500, L=4, K=8, E=16, layer_corr=1.0, seed=4)
    assert st.corr_mask[:, 1:].all() and not st.corr_mask[:, 0].any()
    for j in range(1, 4):
        mapped = np.sort(st.layer_maps[j][st.ids[:, j - 1, :].astype(np.int64)], axis=1)
        assert np.array_equal(mapped, np.sort(st.ids[:, j, :].astype(np.int64), axis=1))


def test_layer_corr_partial_and_zero() -> None:
    def frac_mapped(st: synth.SynthTrace) -> float:
        hits = []
        for j in range(1, st.L):
            mapped = np.sort(st.layer_maps[j][st.ids[:, j - 1, :].astype(np.int64)], axis=1)
            hits.append(np.all(mapped == np.sort(st.ids[:, j, :], axis=1), axis=1))
        return float(np.mean(hits))

    half = synth.synth_trace(T=3000, layer_corr=0.5, seed=7)
    assert abs(frac_mapped(half) - 0.5) < 0.05
    assert abs(half.corr_mask[:, 1:].mean() - 0.5) < 0.05
    assert frac_mapped(synth.synth_trace(T=3000, layer_corr=0.0, seed=7)) < 0.02


def test_stickiness_copies_previous_token() -> None:
    st = synth.synth_trace(T=2000, stickiness=0.6, seed=8)
    same = np.all(st.ids[1:] == st.ids[:-1], axis=2)  # [T-1, L]
    assert same.mean() >= 0.55
    sm = st.sticky_mask
    assert np.all(same[sm[1:]]), "every sticky (t, layer) repeats token t-1"
    assert not sm[st.pos == 0].any(), "never sticky across a chunk boundary"


def test_probs_block_consistent_with_weights() -> None:
    st = synth.synth_trace(T=400, K=8, E=16, P=12, seed=9)
    assert st.P == 12 and st.probs is not None and st.probs_ids is not None
    assert np.array_equal(st.probs_ids[:, :, :8], st.ids)
    assert np.all(np.diff(st.probs, axis=2) <= 0)
    assert np.all(st.probs.sum(axis=2) <= 1.0 + 1e-6)
    renorm = st.probs[:, :, :8] / st.probs[:, :, :8].sum(axis=2, keepdims=True)
    np.testing.assert_allclose(renorm, st.weights, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kw", [dict(T=0), dict(K=17, E=16), dict(E=300, K=8), dict(P=17), dict(layer_corr=1.5),
                                dict(alpha=-1.0), dict(temperature=0.0)])
def test_synth_trace_rejects_bad_arguments(kw: dict) -> None:
    with pytest.raises(ValueError):
        synth.synth_trace(**kw)


def test_write_synth_trace_round_trip(tmp_path: Path) -> None:
    st = synth.write_synth_trace(tmp_path / "rt.moetrace", T=600, layer_corr=0.5, stickiness=0.2, P=10, seed=3,
                                 write_gen=True, write_meta=True)
    assert st.path == tmp_path / "rt.moetrace" and Path(st) == st.path and "T=600" in repr(st)
    assert tr.validate(st.path) == []
    with pytest.raises(TypeError):
        Path(synth.synth_trace(T=10))  # not written: no path
    with tr.TraceReader(st) as r:  # a written SynthTrace is path-like
        assert (r.n, r.L, r.K, r.E, r.P) == (600, 4, 8, 16, 10) and not r.truncated
        assert np.array_equal(r.ids, st.ids)
        assert np.array_equal(r.weights, st.weights_f16)
        assert np.array_equal(r.phase, st.phase)
        assert np.array_equal(r.pos.astype(np.int64), st.pos)
        assert np.array_equal(r.token_ids, st.token_ids) and np.array_equal(r.token_id, st.token_ids)
        assert r.header["chunks"] == st.chunks()
        assert r.header["synthetic"]["seed"] == 3 and r.header["created_at"] == synth.SYNTH_CREATED_AT
        pid, pp = r.probs
        assert np.array_equal(pid, st.probs_ids)
        assert np.array_equal(pp, st.probs.astype(np.float16).astype(np.float32))
    assert np.array_equal(tr.read_tokens(tr.tokens_path(st.path)), st.token_ids)
    gen = tr.read_gen(st.path)
    assert sorted(gen) == [0, 1, 2, 3] and "chunk 2" in gen[2]
    meta = tr.read_meta(st.path)
    assert meta is not None and meta["coherent"] is None and meta["chunks"] == st.chunks()


def test_explicit_trace_argument_and_conflict(tmp_path: Path) -> None:
    base = synth.synth_trace(T=50, seed=1)
    st = synth.write_synth_trace(tmp_path / "a.moetrace", base, header={"model_name": "x"})
    assert st is base and tr.read_header(st.path)["model_name"] == "x"
    with pytest.raises(TypeError):
        synth.write_synth_trace(tmp_path / "b.moetrace", base, T=10)


def test_session_trace_fixtures(synth_trace_file: synth.SynthTrace, synth_trace_corr_file: synth.SynthTrace,
                                make_synth_trace: Any) -> None:
    p = synth_trace_file.params
    assert (p["T"], p["L"], p["K"], p["E"], p["alpha"], p["layer_corr"], p["seed"]) == (2000, 4, 8, 16, 1.2, 0.0, 0)
    assert tr.validate(synth_trace_file.path) == [] and tr.validate(synth_trace_corr_file.path) == []
    assert tr.gen_path(synth_trace_file.path).is_file() and tr.meta_path(synth_trace_file.path).is_file()
    corr = tr.stats(synth_trace_corr_file.path)["cross_layer"]
    assert synth_trace_corr_file.E == 64
    assert corr["precision_at_16"] > corr["baseline_static_top16"] + 0.15
    flat = tr.stats(make_synth_trace("flat.moetrace", E=64, layer_corr=0.0).path)["cross_layer"]
    assert abs(flat["precision_at_16"] - flat["baseline_static_top16"]) < 0.05


def test_stats_lru_curve_matches_brute_force(synth_trace_file: synth.SynthTrace) -> None:
    st = tr.stats(synth_trace_file.path)
    curve = {int(c): h for c, h in st["global"]["lru_hit_curve"]}
    items = synth_trace_file.items()
    for cap in (8, 16, 40):
        assert curve[cap] == pytest.approx(lru_hits(items, cap) / items.size, abs=1e-9)


def test_make_synth_trace_factory(make_synth_trace: Any, tmp_path: Path) -> None:
    st = make_synth_trace("f.moetrace", T=100, E=32, K=4, L=2)
    assert st.path == tmp_path / "f.moetrace" and tr.validate(st.path) == []
    assert tr.read_header(st.path)["n_expert"] == 32


def test_cyclic_trace_defeats_lru() -> None:
    st = synth.cyclic_trace(200, period=5)
    items = st.items()
    assert items[:7].tolist() == [0, 1, 2, 3, 4, 0, 1]
    assert lru_hits(items, 4) == 0
    hist = tr.stack_distance_hist(items, st.L * st.E)
    assert tr.lru_hit_rate(hist, 4) == 0.0
    assert tr.lru_hit_rate(hist, 5) == pytest.approx((200 - 5) / 200)
    multi = synth.cyclic_trace(30, period=3, L=2, K=2)
    assert multi.E == 6 and multi.ids[4, 1].tolist() == [2, 3]


def test_constant_and_single_hot_traces(tmp_path: Path) -> None:
    st = synth.constant_trace(64, L=2, K=8, E=16)
    assert np.all(np.diff(st.weights, axis=2) < 0)
    synth.write_synth_trace(tmp_path / "c.moetrace", st)
    s = tr.stats(tmp_path / "c.moetrace")
    assert all(pl["coverage_at"]["8"] == pytest.approx(1.0) for pl in s["per_layer"])
    hot = synth.constant_trace(32, L=1, K=1, E=16, experts=[3])
    assert tr.entropy_bits(hot.freq()[0]) == 0.0
    assert tr.coverage_curve(hot.freq()[0], [1]) == [pytest.approx(1.0)]
    with pytest.raises(ValueError):
        synth.constant_trace(4, K=2, experts=[1, 1])


def test_trace_from_ids_defaults_and_errors() -> None:
    ids = np.array([[[0, 1]], [[1, 2]], [[2, 0]]], dtype=np.uint8)
    st = synth.trace_from_ids(ids, chunk=[0, 0, 1])
    assert st.E == 3 and st.pos.tolist() == [0, 1, 0]
    np.testing.assert_allclose(st.weights, 0.5)
    with pytest.raises(ValueError):
        synth.trace_from_ids(ids, E=2)
    with pytest.raises(ValueError):
        synth.trace_from_ids(ids[0])


# =================================================================================================
# Tiny GGUFs
# =================================================================================================
@pytest.fixture(scope="module")
def gguf_mod() -> Any:
    import pigemma  # noqa: F401

    return pytest.importorskip("gguf")


def _reader_fields(rd: Any) -> dict[str, Any]:
    out = {}
    for name, f in rd.fields.items():
        try:
            out[name] = f.contents()
        except Exception:  # noqa: BLE001 - older gguf-py: skip
            continue
    return out


def test_tiny_moe_gguf_kv_offsets_and_bytes(tiny_moe_gguf: synth.TinyGguf, gguf_mod: Any) -> None:
    g = tiny_moe_gguf
    rd = gguf_mod.GGUFReader(str(g.path))
    kv = _reader_fields(rd)
    a = "gemma4"
    expect = {f"{a}.block_count": 2, f"{a}.expert_count": 4, f"{a}.expert_used_count": 2,
              f"{a}.embedding_length": 64, f"{a}.feed_forward_length": 48, f"{a}.expert_feed_forward_length": 32,
              f"{a}.attention.head_count": 2, f"{a}.attention.head_count_kv": [2, 1],
              f"{a}.attention.key_length": 32, f"{a}.attention.key_length_swa": 16,
              f"{a}.attention.sliding_window": 8, f"{a}.attention.sliding_window_pattern": [True, False],
              "general.architecture": "gemma4"}
    for k, v in expect.items():
        assert kv[k] == v, k
        assert g.kv[k] == v, k
    seen = set()
    raw = g.path.read_bytes()
    for t in rd.tensors:
        tt = g.tensors[t.name]
        seen.add(t.name)
        assert int(t.data_offset) == tt.offset, t.name  # gguf-py data_offset is absolute
        assert int(t.n_bytes) == tt.n_bytes, t.name
        assert t.tensor_type.name == tt.type, t.name
        assert [int(x) for x in t.shape] == tt.shape, t.name
        assert raw[tt.offset:tt.offset + tt.n_bytes] == tt.raw, t.name
        assert tt.offset % g.alignment == 0
    assert seen == set(g.tensors)
    assert g.data_offset == min(g.offsets.values()) and g.file_size == len(raw)


def test_tiny_moe_gguf_manifest_shapes(tiny_moe_gguf: synth.TinyGguf) -> None:
    t = tiny_moe_gguf.tensors
    want = {
        "blk.0.ffn_gate_up_exps.weight": ("Q4_0", [64, 64, 4]),
        "blk.1.ffn_down_exps.weight": ("Q4_0", [32, 64, 4]),
        "blk.0.ffn_down_exps.scale": ("F32", [4]),
        "blk.1.ffn_gate_inp.weight": ("F32", [64, 4]),
        "blk.0.ffn_gate.weight": ("Q4_0", [64, 48]),
        "blk.0.ffn_up.weight": ("Q4_0", [64, 48]),
        "blk.0.ffn_down.weight": ("F16", [48, 64]),  # ne[0]=48 cannot be Q4_0 (documented fallback)
        "token_embd.weight": ("F16", [64, 16]),
        "output_norm.weight": ("F32", [64]),
        "blk.0.attn_q.weight": ("Q4_0", [64, 32]),   # sliding layer, head_dim 16
        "blk.1.attn_q.weight": ("Q4_0", [64, 64]),   # global layer, head_dim 32
    }
    for name, (typ, shape) in want.items():
        assert (t[name].type, t[name].shape) == (typ, shape), name
    assert "blk.0.attn_v.weight" in t and "blk.1.attn_v.weight" not in t  # K == V on global layers
    assert len(tiny_moe_gguf.params["type_fallbacks"]) == 2
    assert {k: tiny_moe_gguf.params[k] for k in synth.TINY_MOE_SPEC if k in tiny_moe_gguf.params} == {
        k: (list(v) if isinstance(v, tuple) else v) for k, v in synth.TINY_MOE_SPEC.items() if k in tiny_moe_gguf.params}
    kinds = set(tiny_moe_gguf.kinds.values())
    assert kinds == {"token_embd", "attn", "ffn_dense", "norm", "router", "expert_scale", "expert", "other"}


def test_tiny_gguf_pathlike_and_unpacking(tiny_moe_gguf: synth.TinyGguf) -> None:
    import os

    p, offsets = tiny_moe_gguf
    assert p == tiny_moe_gguf.path and Path(tiny_moe_gguf) == p and os.fspath(tiny_moe_gguf) == str(p)
    assert str(tiny_moe_gguf) == str(p) and offsets == tiny_moe_gguf.offsets
    assert len(repr(tiny_moe_gguf)) < 300 and "ffn_down_exps" in repr(tiny_moe_gguf.tensors["blk.0.ffn_down_exps.weight"])
    with open(tiny_moe_gguf, "rb") as f:
        assert f.read(4) == b"GGUF"


def test_tiny_moe_expert_slabs(tiny_moe_gguf: synth.TinyGguf) -> None:
    g = tiny_moe_gguf
    els = g.expert_layers()
    assert [e["layer"] for e in els] == [0, 1]
    for el in els:
        il = el["layer"]
        gu = g.tensors[f"blk.{il}.ffn_gate_up_exps.weight"]
        dn = g.tensors[f"blk.{il}.ffn_down_exps.weight"]
        assert el["gate_up"]["nb_expert"] * 4 == gu.n_bytes and el["down"]["nb_expert"] * 4 == dn.n_bytes
        assert el["unit_bytes"] == gu.n_bytes // 4 + dn.n_bytes // 4 == (2 * 32 * 64 + 32 * 64) // 32 * 18
        for e in range(4):
            (o1, l1), (o2, l2) = g.expert_ranges(il, e)
            assert o1 == gu.offset + e * l1 and o2 == dn.offset + e * l2
            assert g.raw_bytes(gu.name)[e * l1:(e + 1) * l1] == gu.stored[e].tobytes()
            assert g.raw_bytes(dn.name)[e * l2:(e + 1) * l2] == dn.stored[e].tobytes()
    s = g.summary()
    assert s["n_moe_layers"] == 2 and s["expert_unit_bytes"] == 3456
    assert s["expert_bytes_per_token_worst"] == 2 * 2 * 3456
    assert s["expert_bytes"] + s["nonexpert_bytes"] == s["tensor_bytes_total"] == sum(g.n_bytes.values())
    with pytest.raises(IndexError):
        g.expert_ranges(0, 4)


def test_tiny_gguf_agrees_with_gguf_map(tiny_moe_gguf: synth.TinyGguf, tiny_dense_gguf: synth.TinyGguf) -> None:
    gm = pytest.importorskip("pigemma.gguf_map")
    for g in (tiny_moe_gguf, tiny_dense_gguf):
        m = gm.build_map(g.path, 16384, backend="stdlib", with_sha=False)
        got = {t["name"]: t for t in m.tensors}
        assert set(got) == set(g.tensors)
        for name, tt in g.tensors.items():
            assert (got[name]["offset"], got[name]["n_bytes"], got[name]["kind"], got[name]["layer"]) == (
                tt.offset, tt.n_bytes, tt.kind, tt.layer), name
            assert gm.classify_tensor(name) == synth.expected_kind(name)
        s, mine = m.summary, g.summary()
        for k in ("expert_bytes", "nonexpert_bytes", "expert_unit_bytes", "n_moe_layers", "expert_bytes_per_token_worst"):
            assert s[k] == mine[k], k
        assert m["n_expert"] == g.n_expert
        assert [{k: el[k] for k in ("layer", "gate_up", "down")} for el in m.expert_layers] == [
            {k: el[k] for k in ("layer", "gate_up", "down")} for el in g.expert_layers()]


def test_tiny_dense_gguf_shapes(tiny_dense_gguf: synth.TinyGguf) -> None:
    g = tiny_dense_gguf
    assert not g.is_moe and g.n_expert == 0 and g.expert_layers() == []
    assert g.params["type_fallbacks"] == []
    for il in (0, 1):
        assert g.tensors[f"blk.{il}.ffn_gate.weight"].shape == [64, 256]
        assert g.tensors[f"blk.{il}.ffn_up.weight"].shape == [64, 256]
        assert g.tensors[f"blk.{il}.ffn_down.weight"].shape == [256, 64]
        assert {g.tensors[f"blk.{il}.ffn_{m}.weight"].type for m in ("gate", "up", "down")} == {"Q4_0"}
    assert not any("exps" in n or "gate_inp" in n for n in g.tensors)
    assert "gemma4.expert_count" not in g.kv


def test_tensor_values_dequantise(tiny_dense_gguf: synth.TinyGguf, gguf_mod: Any) -> None:
    t = tiny_dense_gguf.tensors["blk.0.ffn_down.weight"]
    W = t.matrix()
    assert W.shape == (64, 256) and W.dtype == np.float32  # [n_out, n_in], y = W @ x
    blocks = np.abs(t.source.reshape(64, 8, 32)).max(axis=2, keepdims=True)
    err = np.abs(W - t.source).reshape(64, 8, 32)
    assert np.all(err <= blocks / 7.0 + 1e-6), "within Q4_0 rounding of each 32-block"
    ref = gguf_mod.quants.dequantize(t.stored, gguf_mod.GGMLQuantizationType.Q4_0)
    assert np.array_equal(W, np.asarray(ref, np.float32).reshape(64, 256))
    norm = tiny_dense_gguf.tensors["blk.0.ffn_norm.weight"]
    assert np.array_equal(norm.values(), np.ones(64, np.float32))
    with pytest.raises(ValueError):
        norm.matrix()


def test_tiny_gguf_alignment_and_dropped_keys(tmp_path: Path, gguf_mod: Any) -> None:
    g = synth.write_tiny_gguf(tmp_path / "a64.gguf", alignment=64, n_ff=96, drop_keys=("expert_count",))
    rd = gguf_mod.GGUFReader(str(g.path))
    assert {t.name: int(t.data_offset) for t in rd.tensors} == g.offsets
    assert all(o % 64 == 0 for o in g.offsets.values()) and g.alignment == 64
    kv = _reader_fields(rd)
    assert "gemma4.expert_count" not in kv and kv["gemma4.expert_used_count"] == 2
    assert g.params["type_fallbacks"] == [] and g.params["dropped_keys"] == ["gemma4.expert_count"]


def test_tiny_llama_gguf_is_runnable_shape(tiny_llama_gguf: synth.TinyGguf, tiny_llama_gguf_f32: synth.TinyGguf,
                                           gguf_mod: Any) -> None:
    q, f = tiny_llama_gguf, tiny_llama_gguf_f32
    assert q.arch == f.arch == "llama"
    assert q.tensors["blk.0.attn_q.weight"].type == "Q4_0" and f.tensors["blk.0.attn_q.weight"].type == "F32"
    for name in q.tensors:
        assert np.array_equal(q.tensors[name].source, f.tensors[name].source), "same seed, same weights"
    kv = _reader_fields(gguf_mod.GGUFReader(str(q.path)))
    toks = kv["tokenizer.ggml.tokens"]
    assert len(toks) == q.params["n_vocab"] == 354 == q.tensors["output.weight"].shape[1]
    assert toks[:3] == ["<unk>", "<s>", "</s>"] and "<0x0A>" in toks and "\u2581" in toks
    assert kv["tokenizer.ggml.model"] == "llama" and kv["general.file_type"] == 2
    assert kv["llama.attention.head_count_kv"] == 2


# =================================================================================================
# Activation dump
# =================================================================================================
def test_synth_acts_dir_obeys_down_projection(synth_acts_dir: synth.SynthActs, tiny_dense_gguf: synth.TinyGguf) -> None:
    acts = tr.read_acts_dir(synth_acts_dir.path)
    assert acts.n_tokens == 32 and acts.layers == [0, 1]
    assert acts.kinds == {"ffn_norm": 64, "ffn_geglu": 256, "ffn_out": 64}
    assert acts.phase is not None and int(acts.phase.sum()) == 8
    assert acts.meta["gguf_path"] == str(tiny_dense_gguf.path)
    for il in (0, 1):
        geglu = np.asarray(acts.get(il, "ffn_geglu"), np.float32)
        out = np.asarray(acts.get(il, "ffn_out"), np.float32)
        Wd = tiny_dense_gguf.tensors[f"blk.{il}.ffn_down.weight"].matrix()
        assert np.array_equal(synth_acts_dir.W[(il, "down")], Wd)
        pred = geglu @ Wd.T
        assert np.max(np.abs(pred - out)) <= 2e-3 * max(1.0, float(np.abs(out).max()))
        np.testing.assert_array_equal(geglu, synth_acts_dir.geglu[il])
        x = np.asarray(acts.get(il, "ffn_norm"), np.float32)
        gate = x @ tiny_dense_gguf.tensors[f"blk.{il}.ffn_gate.weight"].matrix().T
        up = x @ tiny_dense_gguf.tensors[f"blk.{il}.ffn_up.weight"].matrix().T
        np.testing.assert_allclose(synth.gelu_tanh(gate) * up, geglu, rtol=2e-3, atol=2e-3)


def test_synth_acts_without_gguf(tmp_path: Path) -> None:
    a = synth.write_synth_acts_dir(tmp_path / "acts", layers=(3,), T=8, n_embd=32, n_ff=64)
    acts = tr.read_acts_dir(a.path)
    assert acts.layers == [3] and acts.kinds == {"ffn_norm": 32, "ffn_geglu": 64, "ffn_out": 32}
    assert a.W[(3, "down")].shape == (32, 64)


def test_gelu_tanh_reference_values() -> None:
    v = synth.gelu_tanh(np.array([-3.0, 0.0, 1.0, 3.0], np.float32))
    np.testing.assert_allclose(v, [-0.0036374, 0.0, 0.8411920, 2.9963627], atol=1e-5)


# =================================================================================================
# Facts fixtures
# =================================================================================================
STORAGE_TYPES = {"nvme", "usb-ssd", "microsd", "unknown"}


@pytest.mark.parametrize("name", synth.FACTS_NAMES)
def test_facts_fixture_shape(name: str, fake_facts: Any) -> None:
    f = fake_facts(name)
    keys = set(f)
    assert set(synth.FACTS_KEYS) <= keys and keys - set(synth.FACTS_KEYS) <= set(synth.FACTS_OPTIONAL_KEYS)
    assert set(f["storage"]) == set(synth.FACTS_STORAGE_KEYS)
    assert f["storage"]["type"] in STORAGE_TYPES and isinstance(f["storage"]["rotational"], bool)
    for k in ("page_size", "nproc", "ram_total_kb", "ram_available_kb", "swap_total_kb", "free_disk_bytes"):
        assert isinstance(f[k], int) and not isinstance(f[k], bool) and f[k] >= 0, k
    assert f["ram_available_kb"] <= f["ram_total_kb"]
    assert isinstance(f["loadavg1"], float)
    assert f["memlock_limit"] == "unlimited" or isinstance(f["memlock_limit"], int)
    assert f["override"] in (None, "PIGEMMA_I_KNOW")
    assert set(f["tools"]) == {"cmake", "gcc", "git", "python3"}
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", f["collected_at"])
    assert isinstance(f["running_services"], list) and isinstance(f["cpu_flags"], list)
    assert f["notes"] and "illustrative" in f["notes"][0]
    if name.startswith("pi5-"):
        assert f["page_size"] == 16384 and f["nproc"] == 4 and "asimddp" in f["cpu_flags"]
        assert "i8mm" not in f["cpu_flags"] and "sve" not in f["cpu_flags"]
    assert synth.facts_path(name).read_bytes().rstrip(b"\r\n").endswith(b"}")  # CRLF checkouts tolerated


def test_facts_4gb_is_the_probed_context_friend(fake_facts: Any) -> None:
    f = fake_facts("pi5-4gb-microsd")
    assert f["hostname"] == "context-friend-1" and f["ram_total_kb"] == 4_039_000
    assert f["storage"]["dev"].startswith("/dev/mmcblk0") and f["storage"]["type"] == "microsd"
    assert any(s.startswith("contextfriend-") for s in f["running_services"])


def test_facts_real_16gb_board_values(fake_facts: Any) -> None:
    f = fake_facts("pi5-16gb-microsd-real")
    assert f["ram_total_kb"] == 16_607_552 and f["free_disk_bytes"] == 15_501_496_320
    assert f["memlock_limit"] == 8192 and f["hat_product"] is None
    assert {"contextfriend-box.service", "hailort.service", "whisper-bench@base.service"} <= set(f["running_services"])


def test_fake_facts_overrides_and_isolation(fake_facts: Any, fake_facts_path: Any) -> None:
    a = fake_facts("pi5-8gb-usb3", storage={"read_ahead_kb": 2048}, loadavg1=1.5)
    assert a["storage"]["read_ahead_kb"] == 2048 and a["storage"]["type"] == "usb-ssd" and a["loadavg1"] == 1.5
    a["storage"]["type"] = "nvme"
    assert fake_facts("pi5-8gb-usb3")["storage"]["type"] == "usb-ssd"
    assert synth.load_facts("pi5-8gb-usb3") == fake_facts("pi5-8gb-usb3")
    assert fake_facts_path("pi5-8gb-usb3") == synth.facts_path("pi5-8gb-usb3")
    with pytest.raises(FileNotFoundError, match="pi5-4gb-microsd"):
        fake_facts("no-such-host")


# Exit codes of pigemma.feasibility (0 ok, 2 only slow streaming, 3 not viable / no disk, 4 refused).
FACTS_EXPECTED_EXIT = {
    "pi5-4gb-microsd": {4},           # protected services (also NOT_VIABLE)
    "pi5-8gb-usb3": {0, 2},           # streaming; OK vs SLOW depends on the assumed curve
    "pi5-8gb-nvme-gen3": {0, 2},
    "pi5-16gb-nvme-gen3": {0},
    "generic-linux-nodisk": {3},
    "pi5-16gb-microsd-real": {4},
    "pi5-8gb-nvme-busy": {4},
}


@pytest.mark.parametrize("name", synth.FACTS_NAMES)
def test_facts_fixture_feasibility_scenarios(name: str, fake_facts: Any) -> None:
    F = pytest.importorskip("pigemma.feasibility")
    cat = F.load_catalog()
    v = F.verdict(fake_facts(name), copy.deepcopy(cat), check_present=False)
    assert v.exit_code in FACTS_EXPECTED_EXIT[name], (name, v.exit_code, v.reasons[:3])
    rows = {r["model_id"]: r for r in v.rows}
    assert all(r["verdict"] == F.NEVER_GOOD_SPEED for r in v.rows if r["family"] == "31b")
    if name == "pi5-4gb-microsd":
        assert v.host_refused and all(r["verdict"] == F.NOT_VIABLE for r in v.rows if r["family"] == "26b")
    if name == "pi5-16gb-nvme-gen3":
        assert rows["26b-q4"]["verdict"] in (F.FITS_MAPPED, F.FITS_REPACKED)
    if name == "generic-linux-nodisk":
        assert rows["26b-q4"]["verdict"] == F.NO_DISK
    if name == "pi5-8gb-nvme-busy":
        assert v.busy and not v.host_refused
    if name in ("pi5-8gb-usb3", "pi5-8gb-nvme-gen3"):
        assert rows["26b-q4"]["verdict"] in (F.STREAM_OK, F.STREAM_SLOW)


def test_real_board_override_unlocks(fake_facts: Any) -> None:
    F = pytest.importorskip("pigemma.feasibility")
    v = F.verdict(fake_facts("pi5-16gb-microsd-real", override="PIGEMMA_I_KNOW"), F.load_catalog(),
                  check_present=False)
    assert not v.host_refused and v.exit_code == 0


# =================================================================================================
# Real-binary sample files
# =================================================================================================
OWNED_SAMPLES = (
    "llama_bench_sample.json", "completion_stderr_sample.txt", "completion_stderr_norepack_sample.txt",
    "completion_stderr_gemma26b_sample.txt", "server_timings_sample.json", "server_timings_nodraft_sample.json",
    "perplexity_kld_sample.txt", "perplexity_base_sample.txt", "thinking_sample.txt",
    "thinking_pi31b_he_sample.txt", "membw_sample.json", "fio_sample.json", "fio_sample.fio",
)


@pytest.mark.parametrize("name", OWNED_SAMPLES + tuple(f"facts_{n}.json" for n in synth.FACTS_NAMES))
def test_sample_files_present_utf8_documented(name: str) -> None:
    p = FIXTURES / name
    assert p.is_file(), name
    raw = p.read_bytes()
    raw.decode("utf-8")
    assert not raw.startswith(b"\xef\xbb\xbf")  # stored with LF; a CRLF checkout (autocrlf) is tolerated
    readme = (FIXTURES / "README.md").read_text(encoding="utf-8")
    assert f"`{name}`" in readme or (name.startswith("facts_") and f"`{name[6:-5]}`" in readme), name
    if name.endswith(".json"):
        json.loads(raw.decode("utf-8"))
    assert b"Users/person" not in raw and b"Users\\person" not in raw, "no user-specific paths in fixtures"


def test_llama_bench_sample_rows(fixture_json: Any) -> None:
    rows = fixture_json("llama_bench_sample.json")
    assert isinstance(rows, list) and len(rows) == 8
    need = {"build_commit", "build_number", "cpu_info", "model_size", "model_type", "n_threads", "n_prompt",
            "n_gen", "avg_ts", "stddev_ts", "type_k", "type_v", "flash_attn", "repack", "samples_ts"}
    for r in rows:
        assert need <= set(r)
        assert r["build_number"] == 11398 and r["build_commit"].startswith("a7b94df")
        assert (r["n_prompt"], r["n_gen"]) in ((32, 0), (0, 16))
        assert r["avg_ts"] > 0 and r["stddev_ts"] >= 0 and len(r["samples_ts"]) == 3
    assert sorted({(r["n_threads"], r["repack"]) for r in rows}) == [(1, False), (1, True), (2, False), (2, True)]


PERF_LOAD = re.compile(r"load time =\s+([\d.]+) ms")
PERF_PP = re.compile(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens \(\s*[\d.]+ ms per token,\s+([\d.]+) tokens per second\)")
PERF_TG = re.compile(r"(?<!prompt )eval time =\s+([\d.]+) ms /\s+(\d+) runs\s+\(\s*[\d.]+ ms per token,\s+([\d.]+) tokens per second\)")
BUFFER = re.compile(r"load_tensors:\s+(\S+) model buffer size =\s+([\d.]+) MiB")


def _perf(text: str) -> dict[str, Any]:
    return {"load_ms": float(PERF_LOAD.search(text).group(1)),
            "pp": float(PERF_PP.search(text).group(3)),
            "tg": float(PERF_TG.search(text).group(3)),
            "tg_runs": int(PERF_TG.search(text).group(2)),
            "buffers": {m.group(1): float(m.group(2)) for m in BUFFER.finditer(text)}}


def test_completion_stderr_samples_parse(fixture_text: Any) -> None:
    rep = _perf(fixture_text("completion_stderr_sample.txt"))
    assert rep["buffers"] == {"CPU_Mapped": 0.05, "CPU_REPACK": 0.04}
    assert rep["tg_runs"] == 31 and rep["pp"] > 0 and rep["load_ms"] > 0
    nr = _perf(fixture_text("completion_stderr_norepack_sample.txt"))
    assert nr["buffers"] == {"CPU_Mapped": 0.07}
    gem = _perf(fixture_text("completion_stderr_gemma26b_sample.txt"))
    assert gem == {"load_ms": 12747.86, "pp": 50.92, "tg": 8.71, "tg_runs": 74, "buffers": {}}


def test_server_timing_samples(fixture_json: Any) -> None:
    d = fixture_json("server_timings_sample.json")
    t = d["timings"]
    assert (t["draft_n"], t["draft_n_accepted"]) == (321, 91)
    assert t["draft_n_accepted"] / t["draft_n"] == pytest.approx(0.2835, abs=1e-4)
    assert t["predicted_per_second"] == pytest.approx(11.834, abs=1e-3)
    assert d["content"].startswith("<|channel>thought\n<channel|>")
    nd = fixture_json("server_timings_nodraft_sample.json")
    assert "draft_n" not in nd["timings"] and nd["generation_settings"]["speculative.types"] == "none"


def test_perplexity_samples_parse(fixture_text: Any) -> None:
    k = fixture_text("perplexity_kld_sample.txt")
    num = r"([-\d.]+)"
    got = {
        "mean_kld": re.search(rf"Mean\s+KLD:\s+{num}\s+±\s+{num}", k).group(1),
        "max_kld": re.search(rf"Maximum KLD:\s+{num}", k).group(1),
        "p999_kld": re.search(rf"99\.9%\s+KLD:\s+{num}", k).group(1),
        "same_top_p": re.search(rf"Same top p:\s+{num}\s+±\s+{num}\s*%", k).group(1),
        "ppl_q": re.search(rf"Mean PPL\(Q\)\s+:\s+{num}", k).group(1),
        "ppl_base": re.search(rf"Mean PPL\(base\)\s+:\s+{num}", k).group(1),
        "mean_dp": re.search(rf"Mean\s+Δp:\s+{num}\s+±\s+{num}\s*%", k).group(1),
    }
    assert {a: float(b) for a, b in got.items()} == {
        "mean_kld": 0.060170, "max_kld": 0.116063, "p999_kld": 0.115574, "same_top_p": 77.419,
        "ppl_q": 1100.227588, "ppl_base": 1165.064883, "mean_dp": 0.039}
    base = fixture_text("perplexity_base_sample.txt")
    m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", base)
    assert m and (float(m.group(1)), float(m.group(2))) == (1165.0598, 189.16265)


def test_thinking_samples(fixture_text: Any) -> None:
    s = fixture_text("thinking_sample.txt")
    assert "<|channel>thought\n<channel|>" in s and "שלום" in s and s.rstrip().endswith("[end of text]")
    assert s.count("<start_of_turn>") == 2 and "<end_of_turn>" in s
    he = fixture_text("thinking_pi31b_he_sample.txt")
    assert "model<|channel>thought\n<channel|>" in he and "Raspberry Pi" in he
    assert re.search(r"[\u0590-\u05FF]{3,}", he.split("<channel|>", 1)[1])


def test_membw_sample(fixture_json: Any) -> None:
    d = fixture_json("membw_sample.json")
    assert d["tool"] == "membw" and d["units"].startswith("GB/s = 1e9")
    assert [r["threads"] for r in d["results"]] == [1, 2, 4]
    for r in d["results"]:
        assert all(isinstance(r[k], float) and r[k] > 0 for k in ("read_gbps", "copy_gbps", "triad_gbps"))
    assert d["bytes_counted"]["read"] == d["size_mb"] * 2 ** 20


def test_fio_sample(fixture_json: Any, fixture_text: Any) -> None:
    d = fixture_json("fio_sample.json")
    names = [j["jobname"] for j in d["jobs"]]
    assert names == ["seq_read_qd1", "seq_read_qd8", "rand_read_3mb_qd1", "rand_read_3mb_qd4", "rand_read_16k_qd1"]
    for j in d["jobs"]:
        assert j["error"] == 0 and j["read"]["bw_bytes"] > 0 and j["read"]["iops"] > 0
        assert j["read"]["clat_ns"]["mean"] > 0
    assert d["jobs"][2]["job options"]["bs"] == "3345k" and d["jobs"][1]["job options"]["iodepth"] == "8"
    jobfile = fixture_text("fio_sample.fio")
    assert all(f"[{n}]" in jobfile for n in names)


# =================================================================================================
# conftest hooks and helper fixtures
# =================================================================================================
@pytest.mark.linux
def test_linux_marker_only_runs_on_linux() -> None:
    assert sys.platform.startswith("linux")  # the conftest hook skips this test elsewhere


def test_isolated_lock(isolated_lock: Path) -> None:
    import os

    from pigemma import paths

    assert os.environ["PIGEMMA_LOCK_FILE"] == str(isolated_lock)
    with paths.bench_lock(min_free_gb=0, cmd="test_synth") as rec:
        assert isolated_lock.is_file() and rec["cmd"] == "test_synth"
    assert not isolated_lock.exists() or paths.lock_status(isolated_lock)["state"] != "held"


def test_bash_fixture_runs_scripts(bash: str) -> None:
    assert "System32" not in bash, "WSL bash cannot run the repo scripts"
    r = subprocess.run([bash, "-c", "set -euo pipefail; echo ok"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.strip() == "ok"


def test_fixture_dirs(repo_root: Path, fixtures_dir: Path) -> None:
    assert (repo_root / "pigemma" / "__init__.py").is_file() and fixtures_dir == repo_root / "tests" / "fixtures"


@pytest.mark.integration
def test_tiny_llama_runs_in_real_llama_completion(tiny_llama_gguf: synth.TinyGguf, llama_bin: Any,
                                                  tmp_path: Path) -> None:
    """The runnable tiny model loads in the pinned llama-completion and its stderr parses like the
    checked-in sample (needs the native or Linux llama.cpp build; respects the bench lock)."""
    from pigemma import paths

    exe = llama_bin("llama-completion")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("The quick brown fox", encoding="utf-8")
    cmd = [str(exe), "-m", str(tiny_llama_gguf.path), "-f", str(prompt), "-n", "8", "--temp", "0", "-t", "2",
           "-c", "256", "-lv", "4", "--log-colors", "off"]
    try:
        with paths.bench_lock(min_free_gb=1, cmd="test_synth tiny llama-completion"):
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    except paths.BenchLockError as exc:
        pytest.skip(f"bench lock not available: {exc}")
    assert r.returncode == 0, r.stderr[-2000:]
    rep = _perf(r.stderr)
    assert set(rep["buffers"]) >= {"CPU_Mapped"} and rep["tg_runs"] == 7 and rep["pp"] > 0

