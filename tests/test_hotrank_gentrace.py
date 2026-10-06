"""Stage T1: pigemma.gentrace (server.log batch parsing, probe items, pass comparison) and
pigemma.hotrank (access events, score, ordering, static hit rates). No model, no Docker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pigemma import gentrace as gt
from pigemma import hotrank as hr
from pigemma import trace as tr

# llama-server a7b94df log lines (-lv 4): tasks 0 and 331 are copied from a real run; task 400 is a constructed
# follow-up in the same line formats (real example: "restored context checkpoint (pos_min = 0, pos_max = 23,
# n_tokens = 24, n_past = 23, size = 4.688 MiB)")
SERVER_LOG = """\
0.03.109.267 I slot   operator(): id  0 | task 0 | new prompt, n_ctx_slot = 4096, n_keep = 0, task.n_tokens = 23
0.03.109.283 I slot   operator(): id  0 | task 0 | cached n_tokens = 0, memory_seq_rm [0, end)
0.03.874.622 I slot   operator(): id  0 | task 0 | cached n_tokens = 19, memory_seq_rm [19, end)
0.05.423.774 I slot      release: id  0 | task 0 | stop processing: n_tokens = 36, truncated = 0
0.44.497.473 I slot   operator(): id  0 | task 331 | new prompt, n_ctx_slot = 4096, n_keep = 0, task.n_tokens = 687
0.44.497.493 I slot   operator(): id  0 | task 331 | cached n_tokens = 0, memory_seq_rm [0, end)
0.49.029.054 I slot   operator(): id  0 | task 331 | cached n_tokens = 171, memory_seq_rm [171, end)
1.01.555.845 I slot   operator(): id  0 | task 331 | cached n_tokens = 683, memory_seq_rm [683, end)
1.11.723.763 I slot      release: id  0 | task 331 | stop processing: n_tokens = 766, truncated = 0
1.12.000.000 I slot   operator(): id  0 | task 400 | new prompt, n_ctx_slot = 4096, n_keep = 0, task.n_tokens = 85
1.12.000.100 I slot   operator(): id  0 | task 400 | restored context checkpoint (pos_min = 0, pos_max = 22, n_tokens = 23, n_past = 23, size = 1.5 MiB)
1.12.000.200 I slot   operator(): id  0 | task 400 | cached n_tokens = 23, memory_seq_rm [23, end)
1.12.500.000 I slot   operator(): id  0 | task 400 | cached n_tokens = 81, memory_seq_rm [81, end)
1.14.000.000 I slot      release: id  0 | task 400 | stop processing: n_tokens = 132, truncated = 0
"""


def test_parse_server_batches_boundaries_and_reuse():
    tasks = gt.parse_server_batches(SERVER_LOG)
    assert [t["task"] for t in tasks] == [0, 331, 400]
    assert tasks[0]["n_past"] == 0 and tasks[0]["batches"] == [19, 4] and tasks[0]["stop_n_tokens"] == 36
    assert tasks[1]["batches"] == [171, 512, 4] and sum(tasks[1]["batches"]) == 687
    # a follow-up: the first cut is the reused prefix, the batches cover only the new tokens
    assert tasks[2]["n_past"] == 23 and tasks[2]["batches"] == [58, 4]
    assert tasks[2]["checkpoint_restored"]["n_past"] == 23
    assert not tasks[2]["full_reprocess"]


def _server_pass():
    def rec(rid, kind, ptoks, cache_n, batches, sampled):
        n = len(ptoks)
        return {"kind": kind, "rid": rid, "prompt_tokens": ptoks, "n_prompt_tokens": n, "n_predict": 8,
                "cache_n": cache_n, "prompt_n": n - cache_n, "token_ids": sampled, "text": "x", "stop_type": "eos",
                "ttft_s": 1.0, "prompt_ms_server": 900.0, "decode_s_client": 2.0, "decode_tok_s_client": 1.0,
                "decode_tok_s_server": 1.0, "lang": "en", "category": "chat",
                "server_task": {"batches": batches, "checks": {"a": True, "b": True}}}
    return {"tag": "t-server", "requests": [
        rec("warmup", "warmup", [2, 5, 6], 0, [3], [7, 8]),
        rec("c-1#0", "workload", [2, 9, 9, 9, 4], 0, [1, 4], [11, 12, 13]),
        rec("c-1#1", "workload", [2, 9, 9, 9, 4, 11, 12, 5, 5], 6, [3], [21, 22, 23]),
    ]}


def test_build_items_drops_warmup_and_carries_keep_and_batches():
    sp = _server_pass()
    items = gt.build_items(sp, with_warmup=False)
    assert [i["id"] for i in items] == ["c-1#0", "c-1#1"]
    assert items[1]["keep"] == 6 and items[1]["batches"] == [3] and items[1]["tokens"] == sp["requests"][2]["prompt_tokens"]
    assert [i["id"] for i in gt.build_items(sp, with_warmup=True, only_warmup=True)] == ["warmup"]
    sp["requests"][1]["server_task"]["checks"]["b"] = False
    with pytest.raises(RuntimeError):
        gt.build_items(sp, with_warmup=False)


def test_compare_pass_reports_identity_and_first_divergence():
    sp = _server_pass()

    def pr(rid, keep, npre, gen, t0=1000.0):
        return {"id": rid, "n_keep": keep, "n_prefill": npre, "gen_tokens": gen, "n_gen_evaluated": len(gen) - 1,
                "stop": "eos", "text": "x", "t_prefill_ms": 950.0, "sampled_ms": [t0 + 1000.0 * i for i in range(len(gen))]}
    same = gt.compare_pass(sp, [pr("c-1#0", 0, 5, [11, 12, 13]), pr("c-1#1", 6, 3, [21, 22, 23])])
    s = same["summary"]
    assert s["workload_tokens_identical"] == 2 and s["workload_evaluated_prefill_equal"] == 2 and s["divergent"] == []
    assert s["decode"]["time_ratio_probe_over_server_pooled"] == pytest.approx(1.0)
    diff = gt.compare_pass(sp, [pr("c-1#0", 0, 5, [11, 99, 13]), pr("c-1#1", 6, 3, [21, 22])])
    assert diff["summary"]["workload_tokens_identical"] == 0
    assert {d["rid"]: d["first_divergence"] for d in diff["summary"]["divergent"]} == {"c-1#0": 1, "c-1#1": 2}


def test_ratio_summary_uses_only_identical_requests():
    a = {"r1": {"tokens": [1, 2, 3], "n_new": 4, "prefill_s": 2.0, "decode_s": 2.2},
         "r2": {"tokens": [1, 2], "n_new": 4, "prefill_s": 1.0, "decode_s": 1.0}}
    b = {"r1": {"tokens": [1, 2, 3], "n_new": 4, "prefill_s": 2.0, "decode_s": 2.0},
         "r2": {"tokens": [1, 9], "n_new": 4, "prefill_s": 1.0, "decode_s": 0.1}}
    r = gt.ratio_summary(a, b)
    assert r["n_common_requests"] == 2 and r["n_identical"] == 1
    assert r["decode"]["time_ratio_pooled"] == pytest.approx(1.1) and r["decode"]["evaluations"] == 2


# ------------------------------------------------------------------------------------------------
# hotrank
# ------------------------------------------------------------------------------------------------
L, K, E, UB = 3, 2, 8, 4


def _write_trace(tmp_path: Path, corpus_id: str = "calibration") -> Path:
    """Two requests. Request 0: prefill 6 tokens in server calls [5, 1] -> ubatches 4, 1, 1; 2 generated.
    Request 1 (follow-up, keep 3): prefill 2 tokens in one call; 3 generated."""
    p = tmp_path / "cal.moetrace"
    w = tr.TraceWriter(p, {"n_expert": E, "n_expert_used": K, "moe_layers": list(range(L)), "corpus_id": corpus_id,
                           "n_ubatch": UB, "n_batch": 8})
    wts = np.full((L, K), 0.5, np.float32)

    def ids(a, b):
        return np.array([[a, b]] * L, dtype=np.uint8)
    plan = [
        # (chunk, phase, pos, ids)
        (0, 0, 0, ids(0, 1)), (0, 0, 1, ids(0, 1)), (0, 0, 2, ids(0, 2)), (0, 0, 3, ids(0, 1)),   # ubatch 1 (4 tokens)
        (0, 0, 4, ids(3, 4)),                                                                      # ubatch 2 (1 token)
        (0, 0, 5, ids(0, 5)),                                                                      # ubatch 3 (call 2)
        (0, 1, 6, ids(0, 6)), (0, 1, 7, ids(0, 6)),
        (1, 0, 3, ids(1, 2)), (1, 0, 4, ids(1, 2)),                                                # one ubatch
        (1, 1, 5, ids(0, 6)), (1, 1, 6, ids(0, 7)), (1, 1, 7, ids(6, 7)),
    ]
    cur = -1
    for c, ph, pos, i in plan:
        if c != cur:
            w.begin_chunk("items", prompt_id=f"cal-0{c + 1}#0", **({"pos0": 3} if c == 1 else {}))
            cur = c
        w.write(None, pos, 100 + pos, ph, c, i, wts)
    w.close()
    side = [
        {"chunk_id": 0, "id": "cal-01#0", "n_prompt_tokens": 6, "n_keep": 0, "n_prefill": 6, "prefill_calls": [5, 1],
         "n_sampled": 3, "n_gen_evaluated": 2, "stop": "eos"},
        {"chunk_id": 1, "id": "cal-01#1", "n_prompt_tokens": 5, "n_keep": 3, "n_prefill": 2, "prefill_calls": [2],
         "n_sampled": 4, "n_gen_evaluated": 3, "stop": "limit"},
    ]
    Path(str(p) + ".requests.jsonl").write_text("\n".join(json.dumps(x) for x in side) + "\n", encoding="utf-8")
    return p


def test_trace_with_reused_prefix_validates(tmp_path):
    rep = tr.validate_report(_write_trace(tmp_path))
    assert rep.ok, rep.errors


def test_load_events_cuts_prefill_into_ubatches_and_counts(tmp_path):
    ev = hr.load_events(_write_trace(tmp_path))
    assert [r["ubatches"] for r in ev["requests"]] == [[4, 1, 1], [2]]
    c = hr.counts_of(ev)
    assert (c["n_requests"], c["n_decode_events"], c["n_prefill_ubatches"], c["n_prefill_tokens"]) == (2, 5, 4, 8)
    for layer in range(L):
        # decode events per expert: e0 in 4 generated tokens, e6 in 4, e7 in 2
        assert c["decode"][layer].tolist() == [4, 0, 0, 0, 0, 0, 4, 2]
        # ubatch presence: e0 in ubatches 1 and 3; e1 in ubatch 1 and request 1's; e2 the same; e3, e4 once; e5 once
        assert c["prefill_ubatch"][layer].tolist() == [2, 2, 2, 1, 1, 1, 0, 0]
        # token level: e0 x5, e1 x3 + 2, e2 x1 + 2
        assert c["prefill_token"][layer].tolist() == [5, 5, 3, 1, 1, 1, 0, 0]
    one = hr.counts_of(ev, [1])
    assert one["n_decode_events"] == 3 and one["decode"][0].tolist() == [2, 0, 0, 0, 0, 0, 2, 2]


def test_load_events_needs_the_requests_sidecar(tmp_path):
    p = _write_trace(tmp_path)
    Path(str(p) + ".requests.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        hr.load_events(p)


def _geom(pages, runs, page=4096):
    pages = np.array(pages, np.int64)
    return {"pages": pages, "bytes": pages * page, "runs": np.array(runs, np.int64)}


def test_score_is_frequency_times_fetch_cost_per_retained_byte():
    counts = {"decode": np.array([[10, 4, 0]]), "prefill_ubatch": np.array([[1, 2, 6]]), "n_requests": 2}
    g = _geom([[100, 100, 200]], [[2, 2, 2]])
    r = hr.build_ranking(counts, g, w_dec=1.0, w_pre=1.0, rate_bps=1e6, t_op_s=1e-3, readaround=1000)
    c_run = 1e6 * 1e-3 + 1000
    f = np.array([[11, 6, 6]]) / 2
    cost = (g["bytes"] + 2 * c_run) / g["bytes"]
    assert r["c_run_bytes"] == c_run
    assert np.allclose(r["score"], f * cost)
    # equal frequency: the smaller expert has the larger per-fetch share, so it ranks first
    assert r["order"] == [(0, 0), (0, 1), (0, 2)]
    # decode-only weighting changes the order of the tail (expert 2 has no decode events)
    d = hr.build_ranking(counts, g, w_dec=1.0, w_pre=0.0, rate_bps=1e6, t_op_s=1e-3, readaround=1000)
    assert d["order"][-1] == (0, 2) and d["score"][0, 2] == 0.0
    # without a per-fetch term the cost factor is exactly 1 (retained bytes == fetch bytes)
    n = hr.build_ranking(counts, g, rate_bps=1e6, t_op_s=0.0, readaround=0)
    assert np.allclose(n["cost_factor"], 1.0) and np.allclose(n["score"], f)


def test_order_breaks_score_ties_by_decode_count_then_index():
    score = np.array([[1.0, 1.0, 2.0], [1.0, 0.0, 1.0]])
    tie = np.array([[0, 5, 0], [0, 0, 5]])
    assert hr._order(score, tie) == [(0, 2), (0, 1), (1, 2), (0, 0), (1, 0), (1, 1)]


def test_hit_rates_are_static_shares_of_accesses():
    counts = {"decode": np.array([[6, 2, 0, 0]]), "prefill_ubatch": np.array([[1, 1, 1, 1]]), "n_decode_events": 4}
    g = _geom([[10, 10, 10, 10]], [[2, 2, 2, 2]])
    pinned = np.array([[True, False, False, False]])
    h = hr.hit_rates(counts, pinned, g)
    assert h["decode_access_hit_rate"] == pytest.approx(0.75)
    assert h["prefill_event_hit_rate"] == pytest.approx(0.25)
    assert h["decode_unpinned_accesses_per_token"] == pytest.approx(0.5)
    assert h["decode_unpinned_bytes_per_token_worst_case"] == pytest.approx(2 * 10 * 4096 / 4)
    assert hr.set_from_ranking([(0, 1), (0, 3)], 1, 1, 4).tolist() == [[False, True, False, False]]


def test_spearman_and_topn_helpers():
    a = np.array([5, 4, 3, 2, 1, 0])
    assert hr._spearman(a, a) == pytest.approx(1.0)
    assert hr._spearman(a, a[::-1].copy()) == pytest.approx(-1.0)
    assert hr._topn(np.array([1, 9, 9, 3]), 2) == {1, 2}      # ties go to the lower index
    assert hr._prefix_agreement([(0, 0), (0, 1), (0, 2)], [(0, 1), (0, 0), (0, 3)], [2, 3]) == {"2": 1.0, "3": 0.6667}
