"""Unit tests for pigemma.trace (no models, no Docker; Windows and Linux).

Manifest test field (docs/design/manifest.json, files[] "pigemma/trace.py"):
* write 1000 synthetic records (L=4, K=8, E=16) and read them back bit-exact;
* a sentinel n_records is inferred with truncated=True;
* validate rejects a bad id, duplicate ids, a weight sum of 1.5 and a truncated record;
* the coverage curve is monotone and 1.0 at K=E;
* a single-hot-expert trace gives coverage@1 == 1.0 and entropy 0;
* the stack-distance LRU curve equals an LRU cache simulation within 1e-9 at 3 capacities
  (brute-force OrderedDict reference here, plus pigemma.cache_sim when that module exists);
* cross-layer predictor precision@16 >= the static baseline on a trace with injected layer
  correlation;
* merge renumbers token_index.

Extra checks: the constants match the docs/TRACE_FORMAT.md block, the header-rewrite path at
close, every validator error code that is cheap to provoke, the CLI exit codes, the sidecars
(meta, gen.txt, probe summary), and the activation-dump reader including an unpatched .npy. The
golden 26B trace is validated when tests/fixtures/golden_26b_72tok.moetrace exists.

The synthetic generator below is local on purpose: tests/synth.py (owned by the tests agent) may
not exist yet when this file runs.
"""

from __future__ import annotations

import ast
import json
import re
import struct
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from pigemma import trace as tr

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "fixtures" / "golden_26b_72tok.moetrace"


# --------------------------------------------------------------------------------------------
# Synthetic traces
# --------------------------------------------------------------------------------------------
def synth_routing(T=2000, L=4, K=8, E=16, alpha=1.2, layer_corr=0.0, seed=0, gen_frac=0.2, n_chunks=4):
    """Zipf expert popularity per layer, optional cross-layer correlation, chunked phases.

    Returns dict(ids uint8 [T,L,K], weights float32 [T,L,K] (sorted desc, sum 1),
    phase uint8 [T], chunk int [T], pos int [T], token_ids int32 [T]).
    """
    rng = np.random.default_rng(seed)
    base = 1.0 / np.arange(1, E + 1) ** alpha
    pops = [rng.permutation(base) for _ in range(L)]
    perms = [rng.permutation(E) for _ in range(L)]
    ids = np.empty((T, L, K), dtype=np.uint8)
    for j in range(L):
        logits = np.log(pops[j])[None, :] + rng.gumbel(size=(T, E))
        if j > 0 and layer_corr > 0:
            mapped = perms[j - 1][ids[:, j - 1, :].astype(np.int64)]
            boost = np.zeros((T, E))
            np.put_along_axis(boost, mapped, 50.0, axis=1)
            boost[rng.random(T) >= layer_corr] = 0.0
            logits = logits + boost
        ids[:, j, :] = np.argsort(-logits, axis=1)[:, :K]
    raw = rng.random((T, L, K)) + 0.05
    raw = -np.sort(-raw, axis=2)
    w = (raw / raw.sum(axis=2, keepdims=True)).astype(np.float32)
    bounds = np.linspace(0, T, n_chunks + 1).astype(int)
    chunk = np.zeros(T, dtype=np.int64)
    phase = np.zeros(T, dtype=np.uint8)
    pos = np.zeros(T, dtype=np.int64)
    for c in range(n_chunks):
        a, b = bounds[c], bounds[c + 1]
        chunk[a:b] = c
        pos[a:b] = np.arange(b - a)
        n_gen = int((b - a) * gen_frac)
        phase[b - n_gen:b] = 1
    token_ids = rng.integers(0, 262144, size=T).astype(np.int32)
    return {"ids": ids, "weights": w, "phase": phase, "chunk": chunk, "pos": pos, "token_ids": token_ids}


def _header(L=4, K=8, E=16, P=0, **kw):
    return tr.make_header(n_expert=E, n_expert_used=K, moe_layers=list(range(L)), probs_top=P,
                          gguf_path="/work/models/x.gguf", gguf_size_bytes=123456, **kw)


def write_synth(path, *, T=2000, L=4, K=8, E=16, P=0, sources=None, **kw):
    d = synth_routing(T=T, L=L, K=K, E=E, **kw)
    n_chunks = int(d["chunk"].max()) + 1
    with tr.TraceWriter(path, _header(L, K, E, P)) as w:
        for c in range(n_chunks):
            rows = np.flatnonzero(d["chunk"] == c)
            w.begin_chunk(sources[c % len(sources)] if sources else f"src{c % 2}")
            probs = None
            if P:
                pid = np.empty((rows.size, L, P), np.uint8)
                pid[:, :, :K] = d["ids"][rows]
                for t_i, t in enumerate(rows):
                    for j in range(L):
                        rest = [e for e in range(E) if e not in set(d["ids"][t, j].tolist())]
                        pid[t_i, j, K:] = rest[: P - K]
                pp = np.sort(np.random.default_rng(1).random((rows.size, L, P)), axis=2)[:, :, ::-1] * 0.5
                probs = (pid, pp)
            w.write_many(d["ids"][rows], d["weights"][rows], phase=d["phase"][rows], pos=d["pos"][rows],
                         token_ids=d["token_ids"][rows], chunk_ids=np.full(rows.size, c), probs=probs)
    return d


def lru_brute(items, capacity):
    """Reference: overall hit rate of an LRU cache (OrderedDict), cold misses counted, no warm-up."""
    cache: OrderedDict[int, None] = OrderedDict()
    hits = 0
    for x in items:
        if x in cache:
            hits += 1
            cache.move_to_end(x)
        else:
            cache[x] = None
            if len(cache) > capacity:
                cache.popitem(last=False)
    return hits / len(items)


def stack_distances_brute(items):
    stack: list[int] = []
    out = []
    for x in items:
        if x in stack:
            i = stack.index(x)
            out.append(i + 1)
            stack.pop(i)
        else:
            out.append(0)
        stack.insert(0, x)
    return np.array(out)


@pytest.fixture()
def synth_trace(tmp_path):
    p = tmp_path / "s.moetrace"
    d = write_synth(p, T=1000, L=4, K=8, E=16)
    return p, d


# --------------------------------------------------------------------------------------------
# Constants and layout
# --------------------------------------------------------------------------------------------
def _doc_constants() -> dict:
    text = (REPO / "docs" / "TRACE_FORMAT.md").read_text(encoding="utf-8")
    block = text.split("<!-- pigemma-constants:begin -->")[1].split("<!-- pigemma-constants:end -->")[0]
    out = {}
    for line in block.splitlines():
        m = re.match(r"^([A-Z0-9_]+) = (.+)$", line.strip())
        if m:
            out[m.group(1)] = ast.literal_eval(m.group(2))
    return out


def test_constants_match_trace_format_doc():
    consts = _doc_constants()
    assert len(consts) > 40
    required = [k for k in consts if k.startswith(("MOETRACE_", "SIDECAR_", "NPY_", "ACTS_"))] + [
        "GEN_CHUNK_SEPARATOR", "PROBE_SUMMARY_PREFIX", "SHA256_PREFIX_BYTES"]
    for k in required:
        assert hasattr(tr, k), f"pigemma.trace lacks {k}"
        assert getattr(tr, k) == consts[k], k


def test_record_dtype_sizes():
    assert tr.record_dtype(30, 8, 0).itemsize == 736 == tr.MOETRACE_RECORD_SIZE_26B_P0
    assert tr.record_dtype(30, 8, 16).itemsize == 2176 == tr.MOETRACE_RECORD_SIZE_26B_P16
    assert tr.record_dtype(4, 8, 0).itemsize == 112
    assert tr.record_dtype(2, 2, 0).itemsize == 28
    assert tr.record_size(30, 8) * 100_000 == tr.MOETRACE_BYTES_26B_100K_TOKENS
    assert len(tr.REQUIRED_HEADER_KEYS) == 22


def test_fixed_header_and_layout(synth_trace):
    p, d = synth_trace
    raw = p.read_bytes()
    magic, h, r, n, res = struct.unpack("<8sIIQQ", raw[:32])
    assert magic == b"MOETRC01" and h % 8 == 0 and r == 112 and n == 1000 and res == 0
    assert len(raw) == 32 + h + 1000 * 112
    hdr = json.loads(raw[32:32 + h].decode("utf-8"))
    assert set(tr.REQUIRED_HEADER_KEYS) <= set(hdr)
    # record 5, layer 2: ids at 16 + 24*2, weights right after (TRACE_FORMAT 3.6)
    rec = raw[32 + h + 5 * 112: 32 + h + 6 * 112]
    assert struct.unpack("<I", rec[0:4])[0] == 5
    assert list(rec[16 + 24 * 2: 16 + 24 * 2 + 8]) == d["ids"][5, 2].tolist()
    w = np.frombuffer(rec[16 + 24 * 2 + 8: 16 + 24 * 3], dtype="<f2")
    assert np.array_equal(w, d["weights"][5, 2].astype(np.float16))
    assert tr.tokens_path(p).stat().st_size == 4000


# --------------------------------------------------------------------------------------------
# Round trip, sentinel, rewrite
# --------------------------------------------------------------------------------------------
def test_roundtrip_1000_records_bit_exact(tmp_path):
    L, K, E = 4, 8, 16
    d = synth_routing(T=1000, L=L, K=K, E=E, seed=3)
    p = tmp_path / "rt.moetrace"
    with tr.TraceWriter(p, _header(L, K, E)) as w:
        for t in range(1000):
            w.write(t, int(d["pos"][t]), int(d["token_ids"][t]), int(d["phase"][t]), int(d["chunk"][t]),
                    d["ids"][t], d["weights"][t])
    with tr.TraceReader(p) as r:
        assert r.n == 1000 and not r.truncated
        assert (r.L, r.K, r.E, r.P) == (L, K, E, 0)
        assert r.ids.dtype == np.uint8 and r.ids.shape == (1000, L, K)
        assert np.array_equal(r.ids, d["ids"])
        assert r.weights.dtype == np.float32
        assert np.array_equal(r.weights, d["weights"].astype(np.float16).astype(np.float32))
        assert np.array_equal(r.phase, d["phase"])
        assert np.array_equal(r.chunk, d["chunk"].astype(np.uint8))
        assert np.array_equal(r.pos, d["pos"])
        assert np.array_equal(r.token_index, np.arange(1000))
        assert np.array_equal(r.token_ids, d["token_ids"]) and r.token_ids_source == "tokens.bin"
        assert r.probs is None
        segs = r.segments()
        assert [s["end"] - s["start"] for s in segs] == [250] * 4 and not segs[0]["derived"]
        assert [c["n_gen"] for c in r.header["chunks"]] == [50] * 4
    assert tr.validate(p) == []


def test_roundtrip_with_probs(tmp_path):
    p = tmp_path / "pr.moetrace"
    write_synth(p, T=120, L=3, K=4, E=12, P=6)
    with tr.TraceReader(p) as r:
        pid, pp = r.probs
        assert pid.shape == (120, 3, 6) and pp.dtype == np.float32
        assert np.array_equal(np.sort(pid[:, :, :4], axis=2), np.sort(r.ids, axis=2))
    rep = tr.validate_report(p)
    assert rep.errors == [] and not any(w.startswith("W4") for w in rep.warnings)
    st = tr.stats(p)
    assert st["probs"]["margin_p8_p9"] is not None and 0 < st["probs"]["eighth_expert_mass_mean"] < 1


def test_sentinel_n_records_inferred(synth_trace):
    p, _d = synth_trace
    with open(p, "r+b") as f:
        f.seek(16)
        f.write(struct.pack("<Q", tr.MOETRACE_N_RECORDS_OPEN))
    with pytest.warns(tr.TraceWarning):
        r = tr.TraceReader(p)
    assert r.truncated and r.n == 1000
    r.close()
    rep = tr.validate_report(p)
    assert rep.errors == [] and rep.truncated
    assert any(w.startswith("W1") for w in rep.warnings)


def test_unclosed_writer_is_truncated(tmp_path):
    p = tmp_path / "open.moetrace"
    d = synth_routing(T=10, L=2, K=2, E=4)
    w = tr.TraceWriter(p, _header(2, 2, 4))
    w.write_many(d["ids"], d["weights"])
    w.flush()
    w._f.close()  # simulate a crash: no close() -> sentinel stays
    w._tok.close()
    with pytest.warns(tr.TraceWarning):
        r = tr.TraceReader(p)
    assert r.truncated and r.n == 10
    r.close()
    rep = tr.validate_report(p)
    assert rep.errors == []
    assert {w[:2] for w in rep.warnings} >= {"W1", "W2"}


def test_header_rewrite_when_final_header_grows(tmp_path):
    p = tmp_path / "grow.moetrace"
    d = synth_routing(T=300, L=2, K=2, E=4, n_chunks=150)
    with tr.TraceWriter(p, _header(2, 2, 4), reserve=0) as w:
        h0 = w.H
        for c in range(150):
            rows = np.flatnonzero(d["chunk"] == c)
            w.begin_chunk(f"source_with_a_long_name_{c}", prompt_index=c)
            w.write_many(d["ids"][rows], d["weights"][rows], phase=d["phase"][rows],
                         chunk_ids=np.full(rows.size, c))
    assert w.H > h0  # rewritten
    assert not p.with_name(p.name + ".tmp").exists()
    assert tr.validate(p) == []
    with tr.TraceReader(p) as r:
        assert len(r.header["chunks"]) == 150 and r.header["chunks"][149]["prompt_index"] == 149
        assert np.array_equal(r.ids, d["ids"])


def test_writer_strict_rejects(tmp_path):
    w = tr.TraceWriter(tmp_path / "x.moetrace", _header(2, 2, 4))
    ids = np.array([[0, 1], [2, 3]])
    wt = np.full((2, 2), 0.5)
    with pytest.raises(ValueError):
        w.write(1, 0, 0, 0, 0, ids, wt)  # token_index must be 0
    with pytest.raises(ValueError):
        w.write(0, 0, 0, 0, 0, np.array([[0, 0], [1, 2]]), wt)  # duplicate
    with pytest.raises(ValueError):
        w.write(0, 0, 0, 0, 0, np.array([[0, 4], [1, 2]]), wt)  # id >= E
    with pytest.raises(ValueError):
        w.write(0, 0, 0, 2, 0, ids, wt)  # phase
    w.write(0, 0, 0, 0, 0, ids, wt)
    with pytest.raises(ValueError):
        w.write(1, 1, 0, 0, 3, ids, wt)  # chunk jumps from 0 to 3
    w.close()
    with pytest.raises(ValueError):
        tr.TraceWriter(tmp_path / "y.moetrace", {"n_expert": 300, "n_expert_used": 2, "moe_layers": [0]})


# --------------------------------------------------------------------------------------------
# Validator
# --------------------------------------------------------------------------------------------
def _bad_trace(tmp_path, name, mutate):
    d = synth_routing(T=50, L=4, K=8, E=16, seed=7)
    mutate(d)
    p = tmp_path / f"{name}.moetrace"
    with tr.TraceWriter(p, _header(), strict=False) as w:
        w.write_many(d["ids"], d["weights"], phase=d["phase"], pos=d["pos"], token_ids=d["token_ids"])
    return p


def _codes(errs):
    return {e.split(":")[0] for e in errs}


def test_validate_rejects_bad_id(tmp_path):
    p = _bad_trace(tmp_path, "badid", lambda d: d["ids"].__setitem__((3, 1, 2), 16))
    errs = tr.validate(p)
    assert "E10" in _codes(errs)
    assert any(e.startswith("E10: layer 1 token 3: id 16 >= n_expert 16") for e in errs)


def test_validate_rejects_duplicate_ids(tmp_path):
    def mut(d):
        d["ids"][7, 0, 1] = d["ids"][7, 0, 0]
    assert "E11" in _codes(tr.validate(_bad_trace(tmp_path, "dup", mut)))


def test_validate_rejects_weight_sum(tmp_path):
    def mut(d):
        d["weights"][9, 2] *= 1.5
    errs = tr.validate(_bad_trace(tmp_path, "wsum", mut))
    assert "E12" in _codes(errs) and "1.5" in next(e for e in errs if e.startswith("E12"))


def test_validate_rejects_nonfinite_weight(tmp_path):
    def mut(d):
        d["weights"][1, 0, 0] = np.inf
    assert "E12" in _codes(tr.validate(_bad_trace(tmp_path, "inf", mut)))


def test_validate_rejects_truncated_record(synth_trace):
    p, _d = synth_trace
    with open(p, "ab") as f:
        f.write(b"\x00" * 5)
    errs = tr.validate(p)
    assert "E6" in _codes(errs)


def test_validate_other_errors(tmp_path, synth_trace):
    p, _d = synth_trace
    raw = bytearray(p.read_bytes())
    h = struct.unpack("<I", raw[8:12])[0]
    # E1 magic
    q = tmp_path / "e1.moetrace"
    q.write_bytes(b"XXXXXXXX" + bytes(raw[8:]))
    assert _codes(tr.validate(q)) == {"E1"}
    # E2 header length not a multiple of 8
    q = tmp_path / "e2.moetrace"
    bad = bytearray(raw)
    bad[8:12] = struct.pack("<I", h + 3)
    q.write_bytes(bytes(bad))
    assert _codes(tr.validate(q)) == {"E2"}
    # E3 missing key / E4 bad moe_layers / E5 record size: rebuild the JSON header in place
    hdr = json.loads(bytes(raw[32:32 + h]).decode("utf-8"))

    def with_header(name, hd, rsize=None):
        js = json.dumps(hd).encode()
        assert len(js) <= h
        b = bytearray(raw)
        b[32:32 + h] = js + b" " * (h - len(js))
        if rsize is not None:
            b[12:16] = struct.pack("<I", rsize)
        f = tmp_path / f"{name}.moetrace"
        f.write_bytes(bytes(b))
        return f

    hd = dict(hdr)
    del hd["n_ubatch"]
    assert "E3" in _codes(tr.validate(with_header("e3", hd)))
    hd = dict(hdr, moe_layers=[0, 2, 1, 3])
    assert "E4" in _codes(tr.validate(with_header("e4", hd)))
    assert "E5" in _codes(tr.validate(with_header("e5", dict(hdr), rsize=113)))
    # E7 n_records disagrees with the file size
    q = tmp_path / "e7.moetrace"
    bad = bytearray(raw)
    bad[16:24] = struct.pack("<Q", 999)
    q.write_bytes(bytes(bad))
    assert "E7" in _codes(tr.validate(q))
    # E14 tokens.bin length
    tr.tokens_path(p).write_bytes(b"\x00" * 8)
    assert "E14" in _codes(tr.validate(p))


def test_validate_token_index_and_phase(tmp_path):
    d = synth_routing(T=20, L=2, K=2, E=4)
    p = tmp_path / "ti.moetrace"
    with tr.TraceWriter(p, _header(2, 2, 4), strict=False) as w:
        for t in range(20):
            w.write(t if t != 11 else 99, t, 0, 0 if t != 5 else 7, 0, d["ids"][t], d["weights"][t])
    codes = _codes(tr.validate(p))
    assert {"E8", "E9"} <= codes


def test_validate_cli_exit_codes(synth_trace, tmp_path, capsys):
    p, _d = synth_trace
    assert tr.main(["validate", str(p)]) == 0
    assert "OK" in capsys.readouterr().out
    bad = _bad_trace(tmp_path, "cli", lambda d: d["ids"].__setitem__((0, 0, 0), 200))
    assert tr.main(["validate", str(bad)]) == 1
    assert "E10" in capsys.readouterr().out
    assert tr.main(["validate", str(tmp_path / "missing.moetrace")]) == 1
    capsys.readouterr()
    assert tr.main(["validate", "--json", str(p)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep[0]["errors"] == [] and rep[0]["n_records"] == 1000
    assert tr.main([]) == 2
    assert tr.main(["validate"]) == 2


@pytest.mark.skipif(not GOLDEN.is_file(), reason="golden trace not captured yet (pc/trace_collect.sh --smoke)")
def test_golden_26b_trace_validates():
    assert tr.validate(GOLDEN) == []
    with tr.TraceReader(GOLDEN) as r:
        assert r.n == 72 and r.L == 30 and r.K == 8 and r.E == 128
        assert r.moe_layers == list(range(30))
        assert np.allclose(r.weights.sum(axis=2), 1.0, atol=0.02)


# --------------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------------
def test_stack_distances_match_brute_force():
    rng = np.random.default_rng(11)
    for n_items, n, batch in ((5, 300, 7), (40, 2000, 256), (64, 1500, 33), (3, 50, 1), (100, 1000, 1024)):
        items = rng.integers(0, n_items, size=n)
        got = tr.stack_distances(items, n_items, batch=batch)
        assert np.array_equal(got, stack_distances_brute(items.tolist())), (n_items, n, batch)


def test_lru_curve_equals_cache_simulation(synth_trace):
    p, d = synth_trace
    st = tr.stats(p)
    items, n_items = tr.access_items(d["ids"], 16)
    assert n_items == 64
    hist = st["global"]["stack_distance_hist"]
    assert len(hist) == 64 + 1 and sum(hist) == 1000 * 4 * 8
    curve = dict((int(c), h) for c, h in st["global"]["lru_hit_curve"])
    assert sorted(curve) == list(range(1, 65))
    for cap in (8, 24, 50):
        ref = lru_brute(items.tolist(), cap)
        assert abs(curve[cap] - ref) < 1e-9, (cap, curve[cap], ref)
    assert abs(curve[64] - (1 - hist[0] / sum(hist))) < 1e-12


def test_lru_curve_equals_cache_sim_module(synth_trace):
    cache_sim = pytest.importorskip("pigemma.cache_sim")
    p, _d = synth_trace
    st = tr.stats(p)
    curve = dict((int(c), h) for c, h in st["global"]["lru_hit_curve"])
    unit = 1000
    fake_map = {"n_expert": 16, "n_expert_used": 8, "summary": {"expert_unit_bytes": unit, "n_moe_layers": 4,
                                                                 "nonexpert_bytes": 0, "expert_bytes": 64 * unit}}
    with tr.TraceReader(p) as r:
        for cap in (8, 24, 50):
            try:
                res = cache_sim.simulate(r, fake_map, "lru", cap * unit, warmup_tokens=0, read_granularity=1)
            except (TypeError, KeyError, AttributeError) as e:
                pytest.skip(f"pigemma.cache_sim API differs from the manifest guess ({e}); "
                            "equality is asserted by tests/test_cache_sim.py")
            overall = res["overall"] if isinstance(res, dict) else getattr(res, "overall", None)
            hit = overall["hit_rate"] if isinstance(overall, dict) else getattr(overall, "hit_rate")
            assert abs(hit - curve[cap]) < 1e-9


def test_coverage_monotone_and_full_at_k_equals_e(tmp_path):
    p = tmp_path / "zipf.moetrace"
    write_synth(p, T=400, L=3, K=4, E=32, alpha=1.3)
    st = tr.stats(p)
    for e in st["per_layer"]:
        curve = tr.coverage_curve(e["freq"], range(1, 33))
        assert all(b >= a for a, b in zip(curve, curve[1:]))
        assert curve[-1] == 1.0
        vals = [e["coverage_at"][k] for k in ("8", "16", "32", "48", "64", "96")]
        assert all(b >= a for a, b in zip(vals, vals[1:])) and vals[2:] == [1.0] * 4
        assert sum(e["freq"]) == 400 * 4
    q = tmp_path / "full.moetrace"
    d = synth_routing(T=200, L=2, K=8, E=8)
    with tr.TraceWriter(q, _header(2, 8, 8)) as w:
        w.write_many(d["ids"], d["weights"])
    st = tr.stats(q)
    for e in st["per_layer"]:
        assert all(v == 1.0 for v in e["coverage_at"].values())
        assert e["entropy_bits"] == pytest.approx(3.0) and e["gini"] == pytest.approx(0.0)
        assert e["adjacent_reuse"] == 1.0


def test_single_hot_expert(tmp_path):
    p = tmp_path / "hot.moetrace"
    T, L, E = 64, 2, 4
    with tr.TraceWriter(p, _header(L, 1, E)) as w:
        w.write_many(np.full((T, L, 1), 2), np.ones((T, L, 1)))
    st = tr.stats(p)
    for e in st["per_layer"]:
        assert tr.coverage_curve(e["freq"], [1]) == [1.0]
        assert e["entropy_bits"] == 0.0 and e["effective_experts"] == 1.0
        assert e["gini"] == pytest.approx((E - 1) / E)
        assert e["adjacent_reuse"] == 1.0
        assert e["working_set"]["16"] == {"median": 1.0, "p95": 1.0}
    g = st["global"]
    assert g["items_never_used"] == L * (E - 1)
    assert g["stack_distance_hist"][0] == L and g["lru_hit_curve"][L - 1][1] == pytest.approx(1 - L / (T * L))


def test_predictor_beats_static_with_layer_correlation(tmp_path):
    p = tmp_path / "corr.moetrace"
    write_synth(p, T=3000, L=4, K=8, E=64, alpha=1.0, layer_corr=0.9, seed=5)
    cl = tr.stats(p, train_frac=0.7)["cross_layer"]
    assert cl["precision_at_16"] >= cl["baseline_static_top16"]  # the manifest assertion
    assert cl["precision_at_16"] > cl["baseline_static_top16"] + 0.2
    assert cl["precision_at_16"] >= cl["precision_at_8"]
    assert cl["n_train"] == 2100 and cl["n_test"] == 900 and len(cl["per_pair"]) == 3
    q = tmp_path / "indep.moetrace"
    write_synth(q, T=3000, L=4, K=8, E=64, alpha=1.0, layer_corr=0.0, seed=5)
    cl0 = tr.stats(q)["cross_layer"]
    # no correlation: the predictor cannot beat the frequency baseline by much
    assert abs(cl0["precision_at_16"] - cl0["baseline_static_top16"]) < 0.05
    assert cl["precision_at_16"] > cl0["precision_at_16"] + 0.15
    assert cl["mutual_info_bits_mean"] > cl0["mutual_info_bits_mean"]
    # plug-in MI is biased upward for independent layers; the shuffle null measures that bias
    assert cl0["mutual_info_bits_mean"] > 0.3
    assert abs(cl0["mutual_info_bits_mean"] - cl0["mutual_info_bits_null_mean"]) < 0.1
    assert cl["mutual_info_bits_mean"] - cl["mutual_info_bits_null_mean"] > 0.1


def test_mutual_info_known_answers():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, 20000)
    assert tr.mutual_info_bits(a, a, 4) == pytest.approx(2.0, abs=0.01)
    assert tr.mutual_info_bits(a, rng.integers(0, 4, 20000), 4, miller_madow=True) == pytest.approx(0.0, abs=0.01)
    assert tr.js_divergence_bits([1, 0], [0, 1]) == pytest.approx(1.0)
    assert tr.js_divergence_bits([1, 1], [2, 2]) == pytest.approx(0.0)
    assert tr.js_divergence_bits([0, 0], [1, 1]) is None
    assert tr.entropy_bits([1, 1, 1, 1]) == pytest.approx(2.0) and tr.gini([5, 5, 5]) == 0.0


def test_stats_schema_and_json(tmp_path, synth_trace):
    p, _d = synth_trace
    st = tr.stats(p)
    for k in ("trace", "n_tokens", "n_prefill", "n_gen", "train_frac", "per_layer", "global", "cross_layer"):
        assert k in st
    assert st["n_tokens"] == 1000 and st["n_prefill"] + st["n_gen"] == 1000 and st["n_gen"] == 200
    e = st["per_layer"][0]
    for k in ("layer", "freq", "weight_mass", "entropy_bits", "effective_experts", "gini", "coverage_at",
              "coverage_weight_at", "adjacent_reuse", "jaccard_next_mean", "working_set", "prefill_vs_gen_js_bits"):
        assert k in e
    assert set(e["working_set"]) == {"16", "64", "256", "1024"} and e["working_set"]["1024"] is None
    assert 0.0 <= e["prefill_vs_gen_js_bits"] <= 1.0
    assert sum(e["weight_mass"]) == pytest.approx(1000, rel=1e-3)
    g = st["global"]
    for k in ("unique_experts_per_token_mean", "items_never_used", "lru_hit_curve", "stack_distance_hist",
              "adjacent_reuse_mean", "corpus_overlap"):
        assert k in g
    assert set(g["corpus_overlap"]) == {"src0|src1"} and 0 <= g["corpus_overlap"]["src0|src1"] <= 1
    assert 8 <= g["unique_experts_per_token_mean"] <= 16
    for k in ("precision_at_8", "precision_at_16", "baseline_static_top16", "baseline_same_ids", "mutual_info_bits_mean"):
        assert isinstance(st["cross_layer"][k], float)
    assert "probs" not in st
    json.dumps(st, allow_nan=False)
    one = tr.stats(p, train_frac=1.0)["cross_layer"]
    assert one["precision_at_8"] is None
    assert len(tr.stats(p, item="expert")["global"]["stack_distance_hist"]) == 17


def test_stats_cli_and_csv(tmp_path, synth_trace, capsys):
    p, _d = synth_trace
    out = tmp_path / "st" / "s.stats.json"
    assert tr.main(["stats", str(p), "--train-frac", "0.5", "--out", str(out), "--csv-dir", str(tmp_path / "csv")]) == 0
    st = json.loads(out.read_text(encoding="utf-8"))
    assert st["train_frac"] == 0.5
    assert {f.name for f in (tmp_path / "csv").iterdir()} == {
        "s.per_layer.csv", "s.freq.csv", "s.lru_curve.csv", "s.cross_layer.csv"}
    assert "LRU overall hit" in capsys.readouterr().out
    assert tr.main(["stats", str(tmp_path / "nope.moetrace")]) == 3
    assert tr.main(["info", str(p)]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["n_records"] == 1000 and info["n_gen"] == 200 and info["sidecars"]["tokens"]


# --------------------------------------------------------------------------------------------
# Merge and sidecars
# --------------------------------------------------------------------------------------------
def test_merge_renumbers_token_index_and_chunks(tmp_path):
    a, b = tmp_path / "a.moetrace", tmp_path / "b.moetrace"
    da = write_synth(a, T=300, n_chunks=3, sources=["en"], seed=1)
    db = write_synth(b, T=200, n_chunks=2, sources=["he"], seed=2)
    tr.gen_path(a).write_bytes(b"\n--- chunk 0 ---\nhello\n--- chunk 1 ---\n\n--- chunk 2 ---\nx")
    tr.gen_path(b).write_bytes("\n--- chunk 0 ---\nשלום\n--- chunk 1 ---\nbye".encode())
    tr.write_meta(a, {"gguf_sha256_prefix": "ab" * 32, "coherent": True, "corpus": [{"path": "a.txt"}]})
    out = tmp_path / "all.moetrace"
    res = tr.merge([a, b], out)
    assert res["n_records"] == 500 and res["n_chunks"] == 5
    assert tr.validate(out) == []
    with tr.TraceReader(out) as r:
        assert np.array_equal(r.token_index, np.arange(500))
        assert np.array_equal(r.ids, np.concatenate([da["ids"], db["ids"]]))
        assert np.array_equal(r.token_ids, np.concatenate([da["token_ids"], db["token_ids"]]))
        assert [c["chunk_id"] for c in r.header["chunks"]] == [0, 1, 2, 3, 4]
        assert [c["from_source"] for c in r.header["chunks"]] == [0, 0, 0, 1, 1]
        assert [c["source"] for c in r.header["chunks"]] == ["en", "en", "en", "he", "he"]
        assert sorted(set(r.chunk.tolist())) == [0, 1, 2, 3, 4]
        assert r.header["corpus_id"] == "merged" and r.header["sources"] == [str(a), str(b)]
    gen = tr.read_gen(out)
    assert gen == {0: "hello", 1: "", 2: "x", 3: "שלום", 4: "bye"}
    meta = tr.read_meta(out)
    assert meta["coherent"] is None and len(meta["sources"]) == 2 and len(meta["chunks"]) == 5
    # the glob in pc/trace_collect.sh includes the output: it must be skipped
    res2 = tr.merge([str(tmp_path / "*.moetrace")], out)
    assert res2["n_records"] == 500 and res2["sources"] == [str(a), str(b)]
    assert tr.main(["merge", str(tmp_path / "m2.moetrace"), str(a), str(b), "--no-meta"]) == 0


def test_merge_refuses_incompatible(tmp_path):
    a, b = tmp_path / "a.moetrace", tmp_path / "b.moetrace"
    write_synth(a, T=50, E=16)
    write_synth(b, T=50, E=32)
    with pytest.raises(tr.MergeError, match="n_expert"):
        tr.merge([a, b], tmp_path / "o.moetrace")
    assert tr.main(["merge", str(tmp_path / "o.moetrace"), str(a), str(b)]) == 1
    assert tr.main(["merge", str(tmp_path / "o.moetrace"), str(tmp_path / "none.moetrace")]) == 3


def test_meta_and_probe_summary(tmp_path, synth_trace):
    p, _d = synth_trace
    corpus = tmp_path / "c.txt"
    corpus.write_text("text", encoding="utf-8")
    log = tmp_path / "probe.err"
    log.write_text("noise\nPIGEMMA_SUMMARY {\"mode\":\"moe-trace\",\"tok_s_prompt\":55.5,\"tok_s_gen\":12.0,"
                   "\"callback_ms\":3.25,\"exit_code\":0}\n", encoding="utf-8")
    assert tr.parse_probe_summary(log.read_text())["tok_s_gen"] == 12.0
    assert tr.parse_probe_summary("nothing here") is None
    rc = tr.main(["meta", str(p), "--corpus", str(corpus), "--summary-file", str(log), "--wall-ms", "1234",
                  "--coherent", "true", "--no-sha", "--note", "unit test"])
    assert rc == 0
    meta = tr.read_meta(p)
    assert meta["tok_s"] == {"prompt": 55.5, "gen": 12.0} and meta["callback_overhead_ms"] == 3.25
    assert meta["coherent"] is True and meta["wall_ms"] == 1234.0
    assert meta["corpus"][0]["bytes"] == 4 and len(meta["corpus"][0]["sha256"]) == 64
    assert meta["chunks"] == json.loads(json.dumps(tr.read_header(p)["chunks"]))
    assert "unit test" in meta["notes"] and meta["host"]["hostname"]
    assert tr.meta_path(p) == Path(str(p) + ".meta.json")


# --------------------------------------------------------------------------------------------
# Activation dumps
# --------------------------------------------------------------------------------------------
def test_npy_header_matches_numpy(tmp_path):
    for shape in ((512, 21504), (8, 5376), (0, 64), (2**40, 21504)):
        hdr = tr.npy_header_f16(*shape)
        assert len(hdr) == 128 and hdr.endswith(b"\n") and hdr[8:10] == b"v\x00"
        if shape[0] < 10_000:
            f = tmp_path / "x.npy"
            np.save(f, np.zeros(shape, dtype="<f2"))
            assert f.read_bytes()[:128] == hdr


def test_read_acts_dir_roundtrip_and_unpatched(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((10, 64)).astype(np.float16)
    g = rng.standard_normal((10, 256)).astype(np.float16)
    d = tr.write_acts_dir(tmp_path / "acts", {(0, "ffn_norm"): a, (0, "ffn_geglu"): g, (5, "ffn_norm"): a},
                          phase=[0] * 7 + [1] * 3, meta={"model": "synthetic"})
    acts = tr.read_acts_dir(d)
    assert acts.n_tokens == 10 and acts.layers == [0, 5] and acts.kinds == {"ffn_geglu": 256, "ffn_norm": 64}
    assert np.array_equal(acts.get(0, "ffn_geglu"), g) and acts.get(5, "ffn_norm").dtype == np.float16
    assert int(acts.phase.sum()) == 3 and np.array_equal(acts.token_ids, np.arange(10))
    assert np.array_equal(np.load(d / "L0.ffn_norm.npy"), a)  # numpy reads our files
    # unpatched: header says T_MAX=100 rows, file holds 6 complete rows + a partial one
    bad = d / "L7.ffn_out.npy"
    bad.write_bytes(tr.npy_header_f16(100, 64) + a[:6].tobytes() + b"\x00" * 10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        arr = acts.get(7, "ffn_out")
    msgs = [str(w.message) for w in caught if issubclass(w.category, tr.TraceWarning)]
    assert any("unpatched" in m for m in msgs) and any("< meta n_tokens 10" in m for m in msgs)
    assert arr.shape == (6, 64) and np.array_equal(arr, a[:6])
    with pytest.raises(FileNotFoundError):
        acts.get(3, "ffn_norm")
    del arr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tr.main(["acts", str(d)]) == 0
