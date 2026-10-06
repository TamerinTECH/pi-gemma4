"""Unit tests for pigemma.cache_sim (no models, no Docker; Windows and Linux).

Manifest test field (docs/design/manifest.json, files[] "pigemma/cache_sim.py"):
* infinite budget -> steady-state bytes 0 and cold_bytes == unique items * unit;
* zero budget -> bytes/token == L*K*unit;
* belady misses <= lru misses <= lru_perlayer misses on 30 random Zipf traces;
* cyclic pattern of budget+1 items -> LRU hit 0, Belady > 0;
* hit rate monotone in budget;
* lru_prefetch with a perfect predictor (table from the same trace, deterministic routing) ->
  stall bytes 0 and wasted 0;
* LRU hit rate equals the trace.stats stack-distance curve within 1e-9;
* projection known-answer (hand-computed numbers);
* profile-derived budget formula;
* verdict thresholds.

Extra checks: Belady equals an exhaustive-search optimum on tiny sequences, LFU and byte-budgeted
LRU equal brute-force references, the read-granularity cost model, the CLI (outputs, sim.csv
header, envelopes, exit codes 0/1/2/3), sim.json feeding pigemma.feasibility --sim, and the
io-point / validation.json comparison (pc/pagecache_limit.sh step 3) on a fixture io.csv.

The synthetic generator is local on purpose: tests/synth.py (owned by the tests agent) may not exist
when this file runs. Tests that need the real artefacts (results/map26.json, the golden 72-token
trace) skip when those files are absent.
"""

from __future__ import annotations

import copy
import functools
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from pigemma import cache_sim as cs
from pigemma import feasibility as feas
from pigemma import paths
from pigemma import results as rs
from pigemma import trace as tr

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "fixtures" / "golden_26b_72tok.moetrace"
MAP26 = REPO / "results" / "map26.json"
CATALOG = REPO / "pigemma" / "data" / "catalog.json"
UNIT = 1000


# --------------------------------------------------------------------------------------------
# Synthetic traces and maps
# --------------------------------------------------------------------------------------------
def zipf_ids(T=400, L=4, K=4, E=16, alpha=1.2, seed=0, alphas=None):
    """Gumbel-top-K sampling of Zipf expert popularity per layer -> uint8 [T, L, K]."""
    rng = np.random.default_rng(seed)
    ids = np.empty((T, L, K), dtype=np.uint8)
    for j in range(L):
        a = alphas[j] if alphas is not None else alpha
        pop = rng.permutation(1.0 / np.arange(1, E + 1) ** a)
        logits = np.log(pop)[None, :] + rng.gumbel(size=(T, E))
        ids[:, j, :] = np.argsort(-logits, axis=1)[:, :K]
    return ids


def write_trace(path, ids, E, weights=None, phase=None):
    T, L, K = ids.shape
    if weights is None:
        weights = np.full((T, L, K), 1.0 / K, dtype=np.float32)
    hdr = tr.make_header(n_expert=E, n_expert_used=K, moe_layers=list(range(L)),
                         gguf_path="/work/models/x.gguf", gguf_size_bytes=123456)
    with tr.TraceWriter(path, hdr) as w:
        w.write_many(ids, weights, phase=phase)
    return path


def minimal_map(L, E, K, unit=UNIT, nonexpert=0, **summary):
    s = {"expert_unit_bytes": unit, "n_moe_layers": L, "nonexpert_bytes": nonexpert,
         "expert_bytes": L * E * unit}
    s.update(summary)
    return {"n_expert": E, "n_expert_used": K, "summary": s}


def full_map_doc(L, E, K, gate_up=600, down=400, nonexpert=10_000, base=4096, units=None):
    """A map.json that gguf_map.load_map accepts (expert_layers with offsets, tensors list)."""
    els = []
    off = base
    for j in range(L):
        gu, dn = (units[j] if units is not None else (gate_up, down))
        els.append({"layer": j, "gate_up": {"offset": off, "n_bytes": gu * E, "nb_expert": gu},
                    "down": {"offset": off + gu * E, "n_bytes": dn * E, "nb_expert": dn},
                    "n_expert": E, "unit_bytes": gu + dn})
        off += (gu + dn) * E
    unit_list = [el["unit_bytes"] for el in els]
    return {
        "format_version": 1, "gguf": "models/synthetic/x.gguf", "gguf_basename": "x.gguf",
        "file_size": off + 64, "data_offset": base, "page_size": 16384, "n_layer": L, "n_expert": E,
        "n_expert_used": K, "tensors": [], "expert_layers": els, "moe_layers": list(range(L)),
        "summary": {"nonexpert_bytes": nonexpert, "nonexpert_bytes_per_token": nonexpert,
                    "expert_bytes": sum(unit_list) * E, "expert_unit_bytes": int(np.mean(unit_list)),
                    "n_moe_layers": L, "expert_bytes_per_token_worst": K * sum(unit_list),
                    "kv_bytes_f16_at_4096": 1_000_000},
    }


def misses(res) -> int:
    o = res.overall
    return int(round((1.0 - o["hit_rate"]) * o["n_accesses"]))


@pytest.fixture(scope="module")
def zipf_trace(tmp_path_factory):
    p = tmp_path_factory.mktemp("cs") / "zipf.moetrace"
    ids = zipf_ids(T=600, L=4, K=4, E=16, alpha=1.1, seed=3)
    write_trace(p, ids, 16)
    return p, ids


# --------------------------------------------------------------------------------------------
# Helpers and map adapter
# --------------------------------------------------------------------------------------------
def test_parse_helpers():
    assert cs.parse_token_range("10:20", 100) == (10, 20)
    assert cs.parse_token_range("-10:", 100) == (90, 100)
    assert cs.parse_token_range(":", 100) == (0, 100)
    assert cs.parse_token_range(None, 7) == (0, 7)
    assert cs.parse_token_range([5, None], 7) == (5, 7)
    assert cs.parse_token_range("50:10", 100) == (50, 50)
    with pytest.raises(cs.CacheSimError):
        cs.parse_token_range("a:b", 10)
    assert cs.parse_budgets_gb("0.5, 1,12.85") == [0.5, 1.0, 12.85]
    with pytest.raises(cs.CacheSimError):
        cs.parse_budgets_gb("1,-2")


def test_map_info_minimal_and_full():
    mi = cs.map_info(minimal_map(4, 16, 8, unit=1000, nonexpert=5))
    assert mi.n_moe_layers == 4 and mi.uniform and mi.unit_bytes == [1000] * 4
    assert mi.expert_bytes == 64_000 and mi.expert_bytes_per_token_worst == 8 * 4000
    assert mi.nonexpert_bytes_per_token == 5 and mi.part_offsets is None
    cost, rm = mi.read_costs(4096)
    assert rm == "round" and set(cost.tolist()) == {4096}
    cost1, _ = mi.read_costs(1)
    assert set(cost1.tolist()) == {1000}
    # aligned falls back to round without offsets
    _, rm2 = mi.read_costs(4096, "aligned")
    assert rm2 == "round"
    full = cs.map_info(full_map_doc(3, 8, 2, gate_up=600, down=400))
    assert full.unit_bytes == [1000] * 3 and full.part_offsets is not None
    c_round, _ = full.read_costs(512)
    assert set(c_round.tolist()) == {1024 + 512}  # 600 -> 1024, 400 -> 512
    c_al, rm3 = full.read_costs(512, "aligned")
    assert rm3 == "aligned" and np.all(c_al >= 1000) and np.all(c_al % 512 == 0)
    # brute force: granules touched by expert e's two slabs
    el = full_map_doc(3, 8, 2)["expert_layers"][1]
    for e in range(8):
        exp = 0
        for part in ("gate_up", "down"):
            s = el[part]["offset"] + e * el[part]["nb_expert"]
            exp += (-(-(s + el[part]["nb_expert"]) // 512) - s // 512) * 512
        assert c_al[1 * 8 + e] == exp
    kv, src = full.kv_bytes(4096)
    assert kv == 1_000_000 * 34 // 64 and "kv_bytes_f16_at_4096" in src
    with pytest.raises(cs.CacheSimError):
        cs.map_info({"n_expert": 0, "n_expert_used": 0, "summary": {}})
    with pytest.raises(cs.CacheSimError):
        full.check_trace(3, 2, 16, None)


@pytest.mark.skipif(not MAP26.is_file(), reason="results/map26.json not present")
def test_real_map26_numbers():
    mi = cs.map_info(MAP26)
    assert mi.uniform and mi.unit_bytes[0] == 3_345_408 and mi.n_moe_layers == 30
    assert mi.expert_bytes == 12_846_366_720 and mi.expert_bytes_per_token_worst == 802_897_920
    cost, _ = mi.read_costs(131072)
    assert int(cost[0]) == 18 * 131072 + 9 * 131072  # gate_up 2,230,272 -> 18 granules, down 1,115,136 -> 9
    kv, _ = mi.kv_bytes(4096, "q8_0")
    assert kv == 155_975_680  # == map summary.kv_bytes_q8_at_4096
    assert mi.model_id() == "26b-q4"


# --------------------------------------------------------------------------------------------
# Manifest test field
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("policy", ["lru", "lfu", "belady", "lru_perlayer", "lru_prefetch", "static"])
def test_infinite_budget(tmp_path, policy):
    ids = zipf_ids(T=300, L=3, K=4, E=12, alpha=0.8, seed=5)
    p = write_trace(tmp_path / "t.moetrace", ids, 12)
    items = (np.arange(3)[None, :, None] * 12 + ids.astype(np.int64)).ravel()
    uniq, first = np.unique(items, return_index=True)
    last_first_token = int(first.max() // (3 * 4))
    warm = last_first_token + 1
    res = cs.simulate(p, minimal_map(3, 12, 4), policy, 10**12, warmup_tokens=warm, read_granularity=1,
                      train_split=0.0 if policy != "lru_prefetch" else 0.5)
    ss = res.steady
    assert ss["n_tokens"] > 0
    assert ss["miss_bytes_per_token"]["mean"] == 0 and ss["miss_bytes_per_token"]["max"] == 0
    assert ss["hit_rate"] == 1.0
    assert res.unique_items_touched == uniq.size
    if policy in ("lru", "lfu", "belady", "lru_perlayer"):
        assert res.cold_bytes == uniq.size * UNIT
        assert misses(res) == uniq.size
    elif policy == "static":
        assert res.cold_bytes == 0  # every item pre-filled (never-seen ones fill the budget too)
    else:  # prefetch may read some items ahead: demand cold misses can only shrink
        assert res.cold_bytes <= uniq.size * UNIT


@pytest.mark.parametrize("policy", list(cs.POLICIES))
@pytest.mark.parametrize("granularity", [1, 4096])
def test_zero_budget(zipf_trace, policy, granularity):
    p, ids = zipf_trace
    T, L, K = ids.shape
    res = cs.simulate(p, minimal_map(L, 16, K), policy, 0, warmup_tokens=10, read_granularity=granularity)
    cost = -(-UNIT // granularity) * granularity
    for blk in (res.steady, res.overall, res.test):
        assert blk["hit_rate"] == 0.0
        assert blk["miss_bytes_per_token"]["mean"] == L * K * cost
        assert blk["miss_bytes_per_token"]["max"] == L * K * cost
        assert blk["n_miss_per_token"] == L * K
        assert blk["prefetch_bytes_per_token"] == 0
    assert res.capacity_items == 0


def test_policy_ordering_on_30_zipf_traces(tmp_path):
    """belady <= lru <= lru_perlayer on 30 Zipf traces.

    belady <= anything is a theorem (uniform sizes). lru <= lru_perlayer is NOT: it is the manifest's
    expectation for the control and holds when the layers differ in skew (global LRU moves capacity
    to the flat layers) and the cache holds >= 2 tokens of accesses (cap >= 2*L*K). Measured while
    writing this test: 200/200 traces in that regime (minimum margin 19 misses); with equal skew per
    layer and cap near a multiple of L, per-layer LRU wins instead (see the next test).
    """
    rng = np.random.default_rng(1234)
    L, K, E = 4, 4, 16
    for s in range(30):
        alphas = np.array([2.5, 0.4, 2.5, 0.4])[rng.permutation(L)] + rng.uniform(-0.1, 0.1, size=L)
        ids = zipf_ids(T=250, L=L, K=K, E=E, alphas=alphas, seed=s)
        p = write_trace(tmp_path / f"z{s}.moetrace", ids, E)
        prep = cs.prepare(p)
        mp = minimal_map(L, E, K)
        cap = int(rng.integers(2 * L * K, L * E - 11))
        m = {pol: misses(cs.simulate(prep, mp, pol, cap * UNIT, warmup_tokens=0, read_granularity=1))
             for pol in ("belady", "lru", "lru_perlayer")}
        assert m["belady"] <= m["lru"] <= m["lru_perlayer"], (s, cap, m)
        nb = misses(cs.simulate(prep, mp, "belady", cap * UNIT, warmup_tokens=0, belady_bypass=False))
        assert m["belady"] <= nb <= m["lru"], (s, cap, nb, m)
        assert m["belady"] <= misses(cs.simulate(prep, mp, "lfu", cap * UNIT, warmup_tokens=0))


def test_perlayer_can_beat_global_lru_near_the_working_set(tmp_path):
    """Documents the limit of the ordering above: equal skew, cap = 1.5 tokens of accesses."""
    L, K, E = 4, 4, 16
    ids = zipf_ids(T=250, L=L, K=K, E=E, alpha=1.0, seed=0)
    p = write_trace(tmp_path / "eq.moetrace", ids, E)
    prep = cs.prepare(p)
    mp = minimal_map(L, E, K)
    lru = misses(cs.simulate(prep, mp, "lru", 24 * UNIT, warmup_tokens=0))
    per = misses(cs.simulate(prep, mp, "lru_perlayer", 24 * UNIT, warmup_tokens=0))
    bel = misses(cs.simulate(prep, mp, "belady", 24 * UNIT, warmup_tokens=0))
    assert bel <= per < lru


def test_cyclic_pattern_lru_zero_belady_positive(tmp_path):
    C = 6  # budget in items; the cycle has C + 1 items
    seq = np.tile(np.arange(C + 1), 20)
    ids = seq.astype(np.uint8).reshape(-1, 1, 1)
    p = write_trace(tmp_path / "cyc.moetrace", ids, C + 1)
    mp = minimal_map(1, C + 1, 1)
    lru = cs.simulate(p, mp, "lru", C * UNIT, warmup_tokens=0, read_granularity=1)
    bel = cs.simulate(p, mp, "belady", C * UNIT, warmup_tokens=0, read_granularity=1)
    assert lru.capacity_items == C
    assert lru.overall["hit_rate"] == 0.0 and lru.steady["hit_rate"] == 0.0
    assert bel.overall["hit_rate"] > 0.5
    lfu = cs.simulate(p, mp, "lfu", C * UNIT, warmup_tokens=0)
    assert lfu.overall["hit_rate"] >= 0.0


def test_hit_rate_monotone_in_budget(zipf_trace):
    p, ids = zipf_trace
    T, L, K = ids.shape
    prep = cs.prepare(p)
    mp = minimal_map(L, 16, K)
    for pol in ("lru", "belady", "static", "lru_perlayer"):
        prev = -1.0
        prev_bytes = math.inf
        for cap in range(0, L * 16 + 1, 4):
            r = cs.simulate(prep, mp, pol, cap * UNIT, warmup_tokens=20, read_granularity=1)
            h = r.steady["hit_rate"]
            b = r.steady["miss_bytes_per_token"]["mean"]
            assert h >= prev - 1e-12, (pol, cap, h, prev)
            assert b <= prev_bytes + 1e-9, (pol, cap)
            prev, prev_bytes = h, b
        assert prev == 1.0 or pol != "static"


def _deterministic_routing(T, L, E, K, rng, n_pairs=4):
    """Slot 0 draws one of n_pairs fixed pairs; slot j+1 = perm_j(slot j): fully determined."""
    assert K == 2
    pairs = [(2 * i, 2 * i + 1) for i in range(n_pairs)]
    perms = [rng.permutation(E) for _ in range(L - 1)]
    ids = np.empty((T, L, K), dtype=np.uint8)
    choice = rng.integers(0, n_pairs, size=T)
    for t in range(T):
        ids[t, 0] = pairs[choice[t]]
        for j in range(1, L):
            ids[t, j] = perms[j - 1][ids[t, j - 1].astype(np.int64)]
    return ids


def test_lru_prefetch_perfect_predictor(tmp_path):
    rng = np.random.default_rng(7)
    T, L, E, K = 400, 4, 16, 2
    ids = _deterministic_routing(T, L, E, K, rng)
    p = write_trace(tmp_path / "det.moetrace", ids, E)
    mp = minimal_map(L, E, K)
    table = tr.cross_layer_table(ids, T, E)  # the same trace: a perfect predictor
    budget = 6 * UNIT  # far below the 32 distinct items: plain LRU thrashes
    pf = cs.simulate(p, mp, "lru_prefetch", budget, warmup_tokens=20, read_granularity=1, prefetch_table=table)
    lru = cs.simulate(p, mp, "lru", budget, warmup_tokens=20, read_granularity=1)
    ss = pf.steady
    assert pf.in_sample  # injected table: in-sample by construction
    # every slot after the first is prefetched exactly: no stall there, nothing wasted
    assert pf.per_layer_hit_rate[1:] == [1.0] * (L - 1)
    assert ss["prefetch_bytes_per_token"] > 0
    assert ss["wasted_prefetch_fraction"] == 0.0 and ss["wasted_prefetch_bytes_per_token"] == 0.0
    # the only stall left is slot 0 (nothing predicts it); the stall equals its misses exactly
    slot0_miss = (1.0 - pf.per_layer_hit_rate[0]) * K * UNIT
    assert ss["stall_bytes_per_token"] == pytest.approx(slot0_miss)
    assert max(lru.per_layer_hit_rate[1:]) < 1.0  # plain LRU does stall there
    # identical routing on every token and a cache that holds its 8-item working set: steady-state
    # stall 0 and no wasted prefetch (with 6 items only slot 0 would still stall: nothing predicts it)
    same = np.repeat(ids[:1], T, axis=0)
    q = write_trace(tmp_path / "same.moetrace", same, E)
    tab = tr.cross_layer_table(same, T, E)
    pf2 = cs.simulate(q, mp, "lru_prefetch", L * K * UNIT, warmup_tokens=5, read_granularity=1, prefetch_table=tab)
    assert pf2.steady["stall_bytes_per_token"] == 0.0
    pf3 = cs.simulate(q, mp, "lru_prefetch", budget, warmup_tokens=5, read_granularity=1, prefetch_table=tab)
    assert pf3.per_layer_hit_rate[1:] == [1.0] * (L - 1) and pf3.steady["wasted_prefetch_fraction"] == 0.0
    assert pf3.steady["stall_bytes_per_token"] == pytest.approx((1.0 - pf3.per_layer_hit_rate[0]) * K * UNIT)
    assert pf2.steady["wasted_prefetch_fraction"] in (None, 0.0)
    assert pf2.overall["wasted_prefetch_bytes_per_token"] == 0.0


def test_lru_prefetch_learned_table_is_out_of_sample(zipf_trace):
    p, ids = zipf_trace
    T, L, K = ids.shape
    r = cs.simulate(p, minimal_map(L, 16, K), "lru_prefetch", 20 * UNIT, warmup_tokens=10, train_split=0.3)
    n_train = int(math.floor(0.3 * T))
    assert r.steady_window == [n_train, T] and r.test_window == [n_train, T] and not r.in_sample
    wf = r.steady["wasted_prefetch_fraction"]
    assert wf is None or 0.0 <= wf <= 1.0
    assert r.steady["io_bytes_per_token"] == pytest.approx(
        r.steady["miss_bytes_per_token"]["mean"] + r.steady["prefetch_bytes_per_token"])
    assert len(r.steady["prefetch_bytes_per_token_by_layer"]) == L
    assert r.steady["prefetch_bytes_per_token_by_layer"][0] == 0.0  # slot 0 is never predicted


def test_lru_equals_stack_distance_curve(zipf_trace):
    p, ids = zipf_trace
    T, L, K = ids.shape
    st = tr.stats(p)
    curve = {int(c): h for c, h in st["global"]["lru_hit_curve"]}
    prep = cs.prepare(p)
    mp = minimal_map(L, 16, K)
    for cap in (1, 5, 17, 23, 40, 63, 64):
        r = cs.simulate(prep, mp, "lru", cap * UNIT, warmup_tokens=0, read_granularity=1)
        assert r.capacity_items == cap
        assert abs(r.overall["hit_rate"] - curve[cap]) < 1e-9, (cap, r.overall["hit_rate"], curve[cap])
    # a byte budget between two capacities floors to the lower one
    r = cs.simulate(prep, mp, "lru", 17 * UNIT + 999, warmup_tokens=0)
    assert r.capacity_items == 17 and abs(r.overall["hit_rate"] - curve[17]) < 1e-9


# --------------------------------------------------------------------------------------------
# Brute-force references
# --------------------------------------------------------------------------------------------
def _opt_misses(seq, cap):
    """Exhaustive optimum (eviction choice or bypass on every miss)."""
    n = len(seq)

    @functools.lru_cache(maxsize=None)
    def f(i, cache):
        if i == n:
            return 0
        x = seq[i]
        if x in cache:
            return f(i + 1, cache)
        best = 1 + f(i + 1, cache)
        if cap > 0:
            if len(cache) < cap:
                best = min(best, 1 + f(i + 1, cache | {x}))
            else:
                for y in cache:
                    best = min(best, 1 + f(i + 1, (cache - {y}) | {x}))
        return best

    return f(0, frozenset())


def test_belady_equals_exhaustive_optimum(tmp_path):
    rng = np.random.default_rng(99)
    for s in range(25):
        n_items = int(rng.integers(3, 6))
        cap = int(rng.integers(1, n_items))
        seq = rng.integers(0, n_items, size=int(rng.integers(8, 15))).tolist()
        ids = np.asarray(seq, dtype=np.uint8).reshape(-1, 1, 1)
        p = write_trace(tmp_path / f"b{s}.moetrace", ids, n_items)
        r = cs.simulate(p, minimal_map(1, n_items, 1), "belady", cap * UNIT, warmup_tokens=0)
        assert misses(r) == _opt_misses(tuple(seq), cap), (seq, cap)


def _lfu_ref(ids, cap, decay):
    T, L, K = ids.shape
    E = int(ids.max()) + 1
    cnt, tl, last = {}, {}, {}
    cache = set()
    hits = 0
    i = 0
    for t in range(T):
        for j in range(L):
            for k in range(K):
                x = j * 64 + int(ids[t, j, k])
                c = cnt.get(x, 0.0)
                if c > 0 and t != tl[x]:
                    c *= decay ** (t - tl[x])
                cnt[x] = c + 1.0
                tl[x] = t
                if x in cache:
                    hits += 1
                elif cap > 0:
                    if len(cache) >= cap:
                        y = min(cache, key=lambda y: (cnt[y] * decay ** (t - tl[y]), last[y]))
                        cache.remove(y)
                    cache.add(x)
                if x in cache:
                    last[x] = i
                i += 1
    del E
    return hits / (T * L * K)


@pytest.mark.parametrize("decay", [1.0, 0.5])
def test_lfu_matches_reference(tmp_path, decay):
    for s in range(6):
        ids = zipf_ids(T=120, L=2, K=3, E=10, alpha=1.0, seed=50 + s)
        p = write_trace(tmp_path / f"l{s}.moetrace", ids, 10)
        for cap in (0, 3, 7, 12):
            r = cs.simulate(p, minimal_map(2, 10, 3), "lfu", cap * UNIT, warmup_tokens=0, lfu_decay=decay)
            assert r.overall["hit_rate"] == pytest.approx(_lfu_ref(ids, cap, decay), abs=1e-12), (s, cap)


def _lru_bytes_ref(items, sizes, budget):
    od = OrderedDict()
    used = hits = 0
    for x in items:
        if x in od:
            od.move_to_end(x)
            hits += 1
            continue
        if sizes[x] > budget:
            continue
        od[x] = sizes[x]
        used += sizes[x]
        while used > budget:
            used -= od.popitem(last=False)[1]
        assert used <= budget
    return hits


def test_variable_unit_sizes(tmp_path):
    L, E, K = 3, 8, 2
    doc = full_map_doc(L, E, K, units=[(600, 400), (1200, 800), (300, 200)])
    mi = cs.map_info(doc)
    assert not mi.uniform and mi.unit_bytes == [1000, 2000, 500]
    ids = zipf_ids(T=200, L=L, K=K, E=E, alpha=0.9, seed=11)
    p = write_trace(tmp_path / "v.moetrace", ids, E)
    prep = cs.prepare(p)
    sizes = mi.item_sizes().tolist()
    for budget in (0, 1500, 4000, 9000, 20000):
        r = cs.simulate(prep, doc, "lru", budget, warmup_tokens=0, read_granularity=1)
        assert not r.capacity_items_exact
        assert misses(r) == prep.items.size - _lru_bytes_ref(prep.items.tolist(), sizes, budget)
        for pol in ("lfu", "belady", "static", "lru_prefetch", "lru_perlayer"):
            cs.simulate(prep, doc, pol, budget, warmup_tokens=0)
    # miss bytes are per-layer unit sized at granularity 1
    r0 = cs.simulate(prep, doc, "lru", 0, warmup_tokens=0, read_granularity=1)
    assert r0.overall["miss_bytes_per_token"]["mean"] == K * (1000 + 2000 + 500)


def test_static_policy(zipf_trace):
    p, ids = zipf_trace
    T, L, K = ids.shape
    prep = cs.prepare(p)
    mp = minimal_map(L, 16, K)
    cap = 20
    r = cs.simulate(prep, mp, "static", cap * UNIT, warmup_tokens=10, train_split=0.3)
    n_train = int(math.floor(0.3 * T))
    sel = cs.static_selection(prep.items[: n_train * L * K], mp_sizes := np.full(L * 16, UNIT), cap * UNIT)
    assert sel.sum() == cap and r.info["static_items"] == cap and mp_sizes.size == L * 16
    test_items = prep.items[n_train * L * K:]
    assert r.steady["hit_rate"] == pytest.approx(sel[test_items].mean())
    assert r.steady_window == [n_train, T] and not r.in_sample
    freq = np.bincount(prep.items[: n_train * L * K], minlength=L * 16)
    assert freq[sel].min() >= freq[~sel].max()  # the top items by train frequency
    inj = cs.simulate(prep, mp, "static", cap * UNIT, warmup_tokens=10, static_items=[0, 1, 2])
    assert inj.in_sample and inj.steady_window[0] == 10 and inj.info["selection"] == "injected"
    assert inj.overall["hit_rate"] == pytest.approx(np.isin(prep.items, [0, 1, 2]).mean())


def test_weighted_hit_rate(tmp_path):
    ids = np.array([[[0, 1]], [[0, 2]], [[0, 1]]], dtype=np.uint8)  # T=3, L=1, K=2
    w = np.array([[[0.9, 0.1]], [[0.75, 0.25]], [[0.5, 0.5]]], dtype=np.float32)
    p = write_trace(tmp_path / "w.moetrace", ids, 3, weights=w)
    r = cs.simulate(p, minimal_map(1, 3, 2), "lru", 1 * UNIT, warmup_tokens=0)
    # cap 1: t0 miss/miss; t1: 0 evicted by 1 -> miss, 2 miss; t2: 0 miss, 1 miss -> all misses
    assert r.overall["hit_rate"] == 0.0 and r.overall["weighted_hit_rate"] == 0.0
    r2 = cs.simulate(p, minimal_map(1, 3, 2), "lru", 2 * UNIT, warmup_tokens=0)
    # cap 2: t0 miss miss; t1 0 hit (w .75), 2 miss; t2 0 hit (.5), 1 miss
    assert r2.overall["hit_rate"] == pytest.approx(2 / 6)
    assert r2.overall["weighted_hit_rate"] == pytest.approx((0.75 + 0.5) / 3.0, rel=1e-3)


# --------------------------------------------------------------------------------------------
# Projection, budgets, verdicts
# --------------------------------------------------------------------------------------------
def _proj_map():
    # L=2, E=4, K=2, unit 100 MB: expert_bytes 800 MB, worst 400 MB/token; non-expert 1 GB
    return minimal_map(2, 4, 2, unit=100_000_000, nonexpert=1_000_000_000,
                       nonexpert_bytes_per_token=1_000_000_000, expert_bytes_per_token_worst=400_000_000)


def _proj_profile(ram=2.0e9, rr=100.0, acc=1.0):
    return {"name": "test-dev", "platform": "pi5", "ram_bytes": int(ram), "os_reserve_bytes": 500_000_000,
            "compute_buffers_bytes": 200_000_000, "n_ctx": 4096,
            "mem_bw_gbps": {"value": 10.0, "source": "assumed: test"},
            "storage": {"type": "nvme", "class": "x", "seq_read_mbps": {"value": 800, "source": "assumed: t"},
                        "rand_read_3mb_mbps": {"value": rr, "source": "assumed: t"},
                        "access_ms": {"value": acc, "source": "assumed: t"}},
            "sources": {}}


def _proj_run(budget=300_000_000, **steady):
    blk = {"n_tokens": 10, "hit_rate": 0.5, "miss_bytes_per_token": {"mean": 2e8, "p50": 2e8, "p95": 2e8, "max": 2e8},
           "n_miss_per_token": 2.0, "hit_bytes_per_token": 2e8, "prefetch_bytes_per_token": 0.0,
           "prefetch_items_per_token": 0.0, "window": [0, 10]}
    blk.update(steady)
    return {"policy": "lru", "budget_bytes": budget, "steady": blk, "overall": blk}


def test_projection_known_answer():
    mp = _proj_map()
    # budget: 2e9 - 0.5e9 - 1e9 - kv 0 (no attention header) - 0.2e9 = 0.3e9 < 0.8e9 expert bytes
    pr = cs.project(_proj_run(), _proj_profile(), mp, 0.5)
    assert pr["kind"] == "projected" and pr["expert_cache_bytes"] == 300_000_000
    assert pr["resident_bytes_per_token"] == pytest.approx(1.2e9)
    assert pr["t_ram_s"] == pytest.approx(1.2e9 / 5e9)                # 0.24 s
    assert pr["t_io_s"] == pytest.approx(2e8 / 1e8 + 2 * 1.0 / 1e3)    # 2.002 s
    assert pr["tg_serial"] == pytest.approx(1 / 2.242)
    assert pr["tg_overlap"] == pytest.approx(1 / 2.002)
    assert pr["floor_all_in_ram_tok_s"] == pytest.approx(5e9 / 1.4e9)
    assert pr["verdict"] == cs.NOT_VIABLE
    for k in ("mem_bw_gbps", "llama_efficiency", "rand_read_3mb_mbps", "access_ms", "os_reserve_bytes", "kv_bytes"):
        assert set(pr["assumptions"][k]) >= {"value", "source"}
    assert pr["assumptions"]["llama_efficiency"]["value"] == 0.5
    # fast storage: t_io = 0.2 + 0.0002 -> tg_serial 1/0.4402, tg_overlap 1/0.24 -> STREAM_GOOD
    pr2 = cs.project(_proj_run(), _proj_profile(rr=1000.0, acc=0.1), mp, (0.5, "measured: test"))
    assert pr2["tg_serial"] == pytest.approx(1 / 0.4402) and pr2["tg_overlap"] == pytest.approx(1 / 0.24)
    assert pr2["verdict"] == cs.STREAM_GOOD
    assert pr2["assumptions"]["llama_efficiency"]["source"] == "measured: test"
    # prefetch: 1e8 B / 1 item per token, all at slot 1; t_layer = 0.12 s >= 0.1001 s -> fully hidden
    run = _proj_run(prefetch_bytes_per_token=1e8, prefetch_items_per_token=1.0,
                    prefetch_bytes_per_token_by_layer=[0.0, 1e8], prefetch_items_per_token_by_layer=[0.0, 1.0])
    pr3 = cs.project(run, _proj_profile(rr=1000.0, acc=0.1), mp, {"value": 0.5, "source": "assumed: t"})
    assert pr3["t_io_prefetch_s"] == pytest.approx(0.1001) and pr3["overlapped_s"] == pytest.approx(0.1001)
    assert pr3["t_io_s"] == pytest.approx(0.3003)
    assert pr3["tg_serial"] == pytest.approx(1 / 0.4402) and pr3["tg_overlap"] == pytest.approx(1 / 0.3003)
    # default calibration is the 0.65 assumption
    pr4 = cs.project(_proj_run(), _proj_profile(), mp)
    assert pr4["assumptions"]["llama_efficiency"]["value"] == 0.65
    assert pr4["assumptions"]["llama_efficiency"]["source"].startswith("assumed")
    # empty steady window -> overall block with a note
    run5 = _proj_run()
    run5["steady"] = {"n_tokens": 0}
    pr5 = cs.project(run5, _proj_profile(), mp, 0.5)
    assert pr5["window"] == "overall" and pr5["tg_serial"] == pytest.approx(pr["tg_serial"])


def test_profile_budget_formula():
    mp = full_map_doc(2, 4, 2, gate_up=60_000_000, down=40_000_000, nonexpert=1_000_000_000)
    prof = _proj_profile(ram=3.0e9)
    b = cs.profile_budget(prof, mp)
    kv = 1_000_000 * 34 // 64  # q8_0 from summary.kv_bytes_f16_at_4096
    assert b["kv_bytes"] == kv
    assert b["expert_cache_bytes"] == int(3.0e9) - 500_000_000 - 1_000_000_000 - kv - 200_000_000
    assert b["budget_bytes"] == b["expert_cache_bytes"] and not b["infeasible"]
    assert b["fits_all_experts"] == (b["expert_cache_bytes"] >= 800_000_000)
    tiny = cs.profile_budget(_proj_profile(ram=1.5e9), mp)
    assert tiny["infeasible"] and tiny["budget_bytes"] == 0 and tiny["expert_cache_bytes"] < 0
    pr = cs.project(_proj_run(budget=0), _proj_profile(ram=1.5e9), mp, 0.5)
    assert pr["verdict"] == cs.INFEASIBLE
    big = _proj_profile(ram=4.0e9)
    assert cs.project(_proj_run(budget=10**9), big, mp, 0.5)["verdict"] == cs.FITS_RAM


@pytest.mark.skipif(not MAP26.is_file(), reason="results/map26.json not present")
def test_profile_budget_real_catalog():
    cat = feas.load_catalog(CATALOG)
    devs = {d["name"]: d for d in cat["devices"]}
    b8 = cs.profile_budget(devs["pi5-8gb-nvme-gen3"], MAP26)
    assert b8["expert_cache_bytes"] == 8_000_000_000 - 800_000_000 - 1_577_172_088 - 155_975_680 - 300_000_000
    assert b8["expert_cache_bytes"] == 5_166_852_232
    b16 = cs.profile_budget(devs["pi5-16gb-nvme-gen3"], MAP26)
    assert b16["fits_all_experts"] and not cs.profile_budget(devs["pi5-4gb-microsd"], MAP26)["infeasible"]


def test_verdict_thresholds():
    c = cs.classify
    eb = 1000
    assert c(1.0, 2.0, 10, eb) == cs.STREAM_GOOD  # argument order: (tg_serial, tg_overlap)
    assert c(2.0, 1.99, 10, eb) == cs.STREAM_WEAK  # tg_overlap below 'good'
    assert c(1.999, 1.5, 10, eb) == cs.STREAM_WEAK
    assert c(0.999, 3.0, 10, eb) == cs.STREAM_WEAK
    assert c(0.5, 3.0, 10, eb) == cs.STREAM_WEAK
    assert c(0.4999, 3.0, 10, eb) == cs.NOT_VIABLE
    assert c(None, None, 10, eb) == cs.STREAM_GOOD
    assert c(100.0, 100.0, 0, eb) == cs.INFEASIBLE
    assert c(100.0, 100.0, -5, eb) == cs.INFEASIBLE
    assert c(0.01, 0.01, eb, eb) == cs.FITS_RAM
    assert c(1.2, 2.5, 10, eb, good=3.0) == cs.STREAM_WEAK
    assert set(cs.VERDICTS) == {"FITS_RAM", "STREAM_GOOD", "STREAM_WEAK", "NOT_VIABLE", "INFEASIBLE"}


def test_efficiency_resolution(tmp_path):
    pi = {"name": "p", "platform": "pi5"}
    desk = {"name": "d", "platform": "desktop"}
    assert cs.efficiency_for_profile(pi)["value"] == 0.65
    dc = {"value": 0.9, "source": "measured: desktop", "files": []}
    t = cs.efficiency_for_profile(pi, desktop_cal=dc)
    assert t["value"] == 0.9 and t["source"].startswith("assumed") and "transferred" in t["source"]
    assert cs.efficiency_for_profile(desk, desktop_cal=dc)["source"] == "measured: desktop"
    pc = {"value": 0.7, "source": "measured: pi", "files": []}
    assert cs.efficiency_for_profile(pi, pi_cal=pc, desktop_cal=dc)["value"] == 0.7
    assert cs.efficiency_for_profile(desk, pi_cal=pc)["value"] == 0.65
    # desktop E0 derivation: envelope tg x bytes/token / membw read
    e0 = tmp_path / "e0"
    e0.mkdir()
    (e0 / "membw.json").write_text(json.dumps({"tool": "membw", "results": [
        {"threads": 1, "read_gbps": 30.0}, {"threads": 4, "read_gbps": 40.0}]}), encoding="utf-8")
    rs.write_result(e0 / "E0_a.json", experiment="E0", kind="measured", llama_build=11398, llama_commit="a7b94df",
                    model={"id": "26b-q4", "file": "x", "quant": "Q4_0", "bytes": 1, "sha_prefix": None},
                    variant="default", flags=[], threads=4, n_ctx=None, tg={"tok_s": 15.0, "sd": None, "n": 3})
    cat = feas.load_catalog(CATALOG)
    m = {x["id"]: x for x in cat["models"]}["26b-q4"]
    bpt = m["nonexpert_bytes_per_token"] + m["expert_bytes_per_token_worst"]
    got = cs.efficiency_from_e0(e0, cat, prefer_model="26b-q4")
    assert got is not None and got["value"] == pytest.approx(round(15.0 * bpt / 40e9, 4))
    assert got["source"].startswith("measured")
    assert cs.efficiency_from_e0(tmp_path / "nope", cat) is None


# --------------------------------------------------------------------------------------------
# Full run, CLI, outputs
# --------------------------------------------------------------------------------------------
@pytest.fixture()
def cli_inputs(tmp_path):
    L, E, K = 3, 8, 2
    ids = zipf_ids(T=300, L=L, K=K, E=E, alpha=1.0, seed=21)
    tp = write_trace(tmp_path / "t.moetrace", ids, E)
    mp = tmp_path / "map.json"
    mp.write_text(json.dumps(full_map_doc(L, E, K, gate_up=6_000_000, down=4_000_000, nonexpert=7_000_000_000)),
                  encoding="utf-8")
    return tp, mp, (L, E, K)


def test_cli_end_to_end(cli_inputs, tmp_path):
    tp, mp, (L, E, K) = cli_inputs
    out = tmp_path / "h1" / "sim.json"
    csvp = tmp_path / "h1" / "sim.csv"
    ser = tmp_path / "h1" / "series.npz"
    env = tmp_path / "env"
    argv = [str(tp), "--map", str(mp), "--catalog", str(CATALOG), "--out", str(out), "--csv", str(csvp),
            "--series", str(ser), "--results-dir", str(env), "--budgets-gb", "0,0.02,1",
            "--profiles", "pi5-8gb-nvme-gen3,pi5-4gb-microsd", "--policies", "lru,belady,static,lru_prefetch,lfu",
            "--warmup-tokens", "20", "--train-split", "0.3", "--io-mem", "none", "--calibration", "none",
            "--log-level", "WARNING"]
    assert cs.main(argv) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    for k in ("trace", "map", "catalog", "token_range", "warmup_tokens", "read_granularity", "train_split", "runs",
              "dense_31b_line"):
        assert k in doc
    assert doc["token_range"] == [0, 300] and len(doc["runs"]) == 5 * (3 + 2)
    for r in doc["runs"]:
        for k in ("policy", "budget_bytes", "capacity_items", "profile", "steady", "overall", "cold_bytes",
                  "per_layer_hit_rate", "projection"):
            assert k in r
        assert len(r["per_layer_hit_rate"]) == L
        for blk in ("steady", "overall"):
            b = r[blk]
            for k in ("hit_rate", "weighted_hit_rate", "miss_bytes_per_token", "n_miss_per_token",
                      "prefetch_bytes_per_token", "wasted_prefetch_fraction"):
                assert k in b
            assert set(b["miss_bytes_per_token"]) >= {"mean", "p50", "p95", "max"}
        if r["profile"] is None:
            assert r["projection"] is None
        else:
            pr = r["projection"]
            assert pr["kind"] == "projected" and pr["verdict"] in cs.VERDICTS
            for k in ("resident_bytes_per_token", "t_ram_s", "t_io_s", "tg_serial", "tg_overlap",
                      "floor_all_in_ram_tok_s", "assumptions"):
                assert k in pr
            for k in ("mem_bw_gbps", "llama_efficiency", "rand_read_3mb_mbps", "access_ms", "os_reserve_bytes",
                      "kv_bytes"):
                assert set(pr["assumptions"][k]) >= {"value", "source"}
    # 7 GB of non-expert weights leave nothing on the 4 GB profile
    micro = [r for r in doc["runs"] if r["profile"] == "pi5-4gb-microsd"]
    assert micro and all(r["projection"]["verdict"] == cs.INFEASIBLE and r["budget_bytes"] == 0 for r in micro)
    assert doc["checks"]["lru_vs_stack_distance"]["ok"] is True
    assert set(doc["dense_31b_line"]) >= {"bytes_per_token", "tok_s_at_mem_bw", "s_per_token_at_storage_bw"}
    # sim.csv
    lines = csvp.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "policy,budget_gb,profile,hit_rate_ss,weighted_hit_ss,miss_mb_tok_mean,miss_mb_tok_p95," \
                       "tg_serial,tg_overlap,verdict"
    assert len(lines) == 1 + len(doc["runs"])
    raw = [ln for ln in lines[1:] if ln.split(",")[2] == ""]
    assert raw and all(ln.endswith(",,,") for ln in raw)  # raw budgets: no projection cells
    # series
    with np.load(ser) as z:
        assert z["labels"].shape[0] == len(doc["runs"]) and z["token_index"].shape[0] == 300
        assert z["r000_miss_bytes"].shape[0] == 300
    # envelopes: one per (profile, policy) + one dense line per profile, all valid E4 projected
    files = sorted(env.glob("*.json"))
    assert len(files) == 5 * 2 + 2
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        errs, _w = rs.check_result(d)
        assert not errs, (f.name, errs)
        assert d["experiment"] == "E4" and d["kind"] == "projected" and d["assumptions"]
    # re-running overwrites the same envelope files instead of accumulating
    assert cs.main(argv) == 0
    assert len(sorted(env.glob("*.json"))) == len(files)


def test_cli_exit_codes(cli_inputs, tmp_path):
    tp, mp, (L, E, K) = cli_inputs
    out = str(tmp_path / "o.json")
    base = ["--map", str(mp), "--out", out, "--io-mem", "none", "--calibration", "none", "--log-level", "ERROR",
            "--budgets-gb", "0.01", "--no-envelopes"]
    assert cs.main([str(tmp_path / "missing.moetrace")] + base) == paths.EXIT_MISSING
    assert cs.main([str(tp)] + base[:1] + [str(tmp_path / "nomap.json")] + base[2:]) == paths.EXIT_MISSING
    assert cs.main([str(tp)] + base + ["--policies", "lru,fifo"]) == paths.EXIT_USAGE
    assert cs.main([str(tp)] + base + ["--profiles", "no-such-device"]) == paths.EXIT_USAGE
    assert cs.main([str(tp)] + base + ["--train-split", "1.5"]) == paths.EXIT_USAGE
    assert cs.main([str(tp), "--out", out]) == paths.EXIT_USAGE  # --map missing
    assert cs.main(["--help"]) == 0
    bad = tmp_path / "bad_map.json"
    bad.write_text(json.dumps(full_map_doc(L, E * 2, K)), encoding="utf-8")  # n_expert mismatch
    assert cs.main([str(tp)] + ["--map", str(bad)] + base[2:]) == paths.EXIT_ERROR
    assert cs.main([str(tp)] + base) == 0 and Path(out).is_file()


@pytest.mark.skipif(not (GOLDEN.is_file() and MAP26.is_file()), reason="golden trace / map26.json not present")
def test_golden_trace_feeds_feasibility():
    doc, runs, prep = cs.run_simulation(GOLDEN, MAP26, catalog=CATALOG, policies=["lru", "belady"],
                                        budgets_gb=[1, 3, 5, 8], profiles=None, warmup_tokens=8,
                                        io_mem="none", calibration="none")
    assert prep.n == 72 and prep.L == 30
    assert doc["checks"]["lru_vs_stack_distance"]["ok"] is True
    assert doc["model"]["id"] == "26b-q4"
    for r in runs:
        assert len(r.per_layer_hit_rate) == 30
    bel = {r.budget_bytes: r for r in runs if r.policy == "belady"}
    for r in runs:
        if r.policy == "lru":
            assert misses(bel[r.budget_bytes]) <= misses(r)
    facts = {"hostname": "pi-lab", "kernel": "6.18.0-v8-16k+", "page_size": 16384, "cpu_model": "Cortex-A76",
             "cpu_flags": ["asimddp", "fphp"], "nproc": 4, "ram_total_kb": 7_812_500, "ram_available_kb": 7_000_000,
             "swap_total_kb": 0, "loadavg1": 0.2, "model_dir": "/nonexistent", "free_disk_bytes": 400_000_000_000,
             "storage": {"dev": "/dev/sda1", "type": "usb-ssd", "rotational": False, "pcie_link_speed": None,
                         "pcie_link_width": None, "read_ahead_kb": 128},
             "running_services": [], "hat_product": None, "pciex1_gen_cfg": None, "governor": "performance",
             "memlock_limit": "unlimited", "tools": {}, "override": None, "collected_at": "2026-10-05T12:00:00Z"}
    v = feas.verdict(facts, feas.load_catalog(CATALOG), sim=copy.deepcopy(doc), check_present=False)
    assert v.hit_source == "simulated"


# --------------------------------------------------------------------------------------------
# E5/E6 io.csv comparison
# --------------------------------------------------------------------------------------------
def _write_io_csv(path, rows):
    lines = [cs.IO_CSV_HEADER] + [",".join(str(x) for x in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _io_rows():
    rows = []
    for t in range(10):
        rb = 10_000_000 if t < 5 else 4_000_000
        rows.append([t, 0, "100.000", rb, 1000 if t < 5 else 500, 600_000, 1_000_000 + t])
    return rows


def _sim_doc():
    blk = lambda m, n: {"miss_bytes_per_token": {"mean": m, "p50": m, "p95": m, "max": m},  # noqa: E731
                        "n_miss_per_token": n, "hit_rate": 0.5, "n_tokens": 5}
    return {"warmup_tokens": 5, "runs": [
        {"policy": "lru", "budget_bytes": 3_000_000_000, "profile": None, "steady": blk(3e6, 1.0)},
        {"policy": "belady", "budget_bytes": 3_000_000_000, "profile": None, "steady": blk(2e6, 0.6)}]}


def test_io_measure_and_point(tmp_path):
    io = _write_io_csv(tmp_path / "io.csv", _io_rows())
    m = cs.measure_io(io, warmup_tokens=5)
    assert m["measured_tokens"] == 5 and m["measured_rows"] == 5
    assert m["measured_bytes_per_token"] == 4e6 and m["measured_majflt_per_token"] == 500
    assert m["measured_tok_s"] == pytest.approx(10.0)
    mb = cs.measure_io(io, warmup_tokens=5, page_size=4096, bind_mount=True)
    assert mb["measured_bytes_per_token"] == 500 * 4096
    pt = cs.validation_point(io, _sim_doc(), mem_cap_bytes=4_000_000_000, tier_rand_read_3mb_mbps=100.0,
                             ram_tok_s=20.0, warmup_tokens=5)
    for k in ("mem_cap_bytes", "pinned", "pin_budget_bytes", "pin_locked_bytes", "anon_rss_bytes",
              "file_cache_bytes", "tier_rand_read_3mb_mbps", "measured_bytes_per_token",
              "measured_majflt_per_token", "measured_tok_s", "sim_budget_bytes", "sim_lru_bytes_per_token",
              "sim_belady_bytes_per_token", "ratio_sim_lru_over_measured", "predicted_tok_s", "oom", "notes"):
        assert k in pt
    assert pt["ratio_sim_lru_over_measured"] == pytest.approx(0.75)
    assert pt["predicted_tok_s"] == pytest.approx(1 / (0.05 + 0.03))  # 12.5 tok/s
    assert pt["projected"]["kind"] == "projected" and pt["projected"]["predicted_tok_s"] == pt["predicted_tok_s"]
    assert pt["anon_rss_bytes"] == 600_000 * 1024 and pt["within_belady_lru_band"] is False  # 4e6 > 1.3 x 3e6
    vp = tmp_path / "validation.json"
    cs.write_validation(vp, pt, gguf="/models/x.gguf", token_range=[0, 1500], warmup_tokens=5)
    cs.write_validation(vp, dict(pt, measured_tok_s=9.0), gguf="/models/x.gguf", warmup_tokens=5)
    cs.write_validation(vp, dict(pt, mem_cap_bytes=5_000_000_000), warmup_tokens=5)
    v = json.loads(vp.read_text(encoding="utf-8"))
    assert v["kind"] == "measured" and v["token_range"] == [0, 1500] and v["volume"] == "pigemma-models"
    assert [p["mem_cap_bytes"] for p in v["points"]] == [4_000_000_000, 5_000_000_000]
    assert v["points"][0]["measured_tok_s"] == 9.0  # replaced, not duplicated
    bm = cs.validation_point(io, _sim_doc(), mem_cap_bytes=1, volume="bind-mount", warmup_tokens=5)
    assert cs.BIND_MOUNT_NOTE in bm["notes"] and bm["predicted_tok_s"] is None


def test_io_point_cli(tmp_path, capsys):
    io = _write_io_csv(tmp_path / "io.csv", _io_rows())
    simp = tmp_path / "sim.json"
    simp.write_text(json.dumps(_sim_doc()), encoding="utf-8")
    vp = tmp_path / "v.json"
    code = cs.main(["io-point", "--io-csv", str(io), "--sim", str(simp), "--mem-cap-bytes", "4000000000",
                    "--tier-mbps", "100", "--ram-tok-s", "20", "--warmup-tokens", "5", "--validation", str(vp),
                    "--token-range", "0:1500", "--log-level", "ERROR"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["predicted_tok_s"] == pytest.approx(12.5)
    assert json.loads(vp.read_text(encoding="utf-8"))["points"][0]["mem_cap_bytes"] == 4_000_000_000
    bad = tmp_path / "bad.csv"
    bad.write_text("token_index,wall_ms\n0,1\n", encoding="utf-8")
    assert cs.main(["io-point", "--io-csv", str(bad), "--sim", str(simp), "--mem-cap-bytes", "1",
                    "--log-level", "ERROR"]) == paths.EXIT_ERROR
    assert cs.main(["io-point", "--io-csv", str(tmp_path / "nope.csv"), "--sim", str(simp), "--mem-cap-bytes", "1",
                    "--log-level", "ERROR"]) == paths.EXIT_MISSING


def test_io_csv_header_matches_trace_format():
    doc = (REPO / "docs" / "TRACE_FORMAT.md").read_text(encoding="utf-8")
    assert f"`{cs.IO_CSV_HEADER}`" in doc
    assert cs.SIM_CSV_HEADER in doc
