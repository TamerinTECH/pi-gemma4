"""Unit tests for pigemma.controlled_summary (P5 controlled-memory harness analysis).

Synthetic cells only (no Docker, no model): each test writes config.json / requests.jsonl /
outputs/*.json exactly in the layout pigemma.controlled_cell produces.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from pigemma import controlled_summary as cs

GIB = 1024**3


def make_cell(root: Path, cell_id: str, method: str, *, budget_gib: float = 8, repeat: int = 0,
              wset: str = "heldout", requests: list[dict] | None = None, status: str = "ok",
              storage: str = "fuse-throttled", rate: float = 91.6e6, phases: dict | None = None,
              tokens: dict[str, list[int]] | None = None, prompts: dict[str, str] | None = None) -> Path:
    d = root / cell_id
    (d / "outputs").mkdir(parents=True)
    cfg = {"schema": "pigemma-controlled-cell v1", "cell_id": cell_id, "method": method,
           "budget_bytes": int(budget_gib * GIB), "repeat": repeat, "workload": {"set": wset},
           "storage": {"mode": storage, "rate_bps": rate}, "status": status,
           "effective": {"phases": phases or {"setup_s": 0.0, "load_s": 100.0, "warmup_s": 5.0,
                                              "setup_backing_bytes": 0, "load_backing_bytes": 12_000_000_000},
                         "max_usage_by_phase_sampled": {"load": 8 * GIB - 1000},
                         "setup": {"pinner_summary": {"locked_bytes": 0}}},
           "host": {"host_cpu_during_cell": {"host_cpu_pct_mean": 20.0}}}
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    lines = [{"kind": "warmup", "rid": "warmup", "decode_tok_s_client": 99.0}]
    for r in requests or []:
        rec = {"kind": "workload", "output_tokens": 10, "total_s": 10.0, **r}
        lines.append(rec)
        toks = (tokens or {}).get(r["rid"], list(range(10)))
        out = {"rid": r["rid"], "token_ids": toks,
               "prompt_sha256": (prompts or {}).get(r["rid"], "p-" + r["rid"])}
        safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in r["rid"])
        (d / "outputs" / f"{safe}.json").write_text(json.dumps(out), encoding="utf-8")
    (d / "requests.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return d


def req(rid: str, dec: float, *, pre: float = 10.0, ttft: float = 2.0, p99: float = 1.0, cat: str = "chat",
        bpt: int | None = None) -> dict:
    return {"rid": rid, "conv_id": rid.split("#")[0], "category": cat, "decode_tok_s_client": dec,
            "prefill_tok_s_client": pre, "ttft_s": ttft, "total_s": 10.0, "itl_p99_s": p99, "itl_median_s": 0.5,
            "decode_backing_bytes_per_token": bpt, "io_decode": {"fuse_backing_bytes": (bpt or 0) * 9},
            "io_prefill": {"fuse_backing_bytes": 1000}}


RIDS = ["ho-01#0", "ho-02#0", "ho-03#0"]


def build_pair(root: Path, factor: dict[str, float] | float, repeats: int = 2, method: str = "C",
               pre_factor: float = 1.0, p99_factor: float = 1.0, base_dec: float = 1.0) -> None:
    for r in range(repeats):
        make_cell(root, f"A-r{r}", "A", repeat=r, requests=[req(x, base_dec, bpt=60_000_000) for x in RIDS])
        reqs = []
        for x in RIDS:
            f = factor[x] if isinstance(factor, dict) else factor
            reqs.append(req(x, base_dec * f, pre=10.0 * pre_factor, p99=1.0 * p99_factor, bpt=int(60_000_000 / f)))
        make_cell(root, f"{method}-r{r}", method, repeat=r, requests=reqs)


def test_pairs_and_uniform_speedup(tmp_path: Path) -> None:
    build_pair(tmp_path, 2.0)
    cells = cs.load_cells(tmp_path)
    pairs = cs.pair_requests(cells, "A")
    assert len(pairs) == 1
    (cond, meth), ps = next(iter(pairs.items()))
    assert meth == "C" and len(ps) == 6
    s = cs.summarize_pairs(ps, n_boot=200)
    d = s["metrics"]["decode_tok_s"]
    assert d["median_ratio"] == pytest.approx(2.0)
    assert d["ci95"] == [pytest.approx(2.0), pytest.approx(2.0)]
    assert s["n_requests"] == 3 and s["repeats"] == [0, 1]
    assert s["flags"]["promising"] is True
    assert s["flags"]["strong"] is True  # 2x and treatment 2.0 tok/s
    assert s["decode_ratio_by_rid"]["ho-01#0"]["median"] == pytest.approx(2.0)


def test_heterogeneous_ratios_median_and_ci(tmp_path: Path) -> None:
    build_pair(tmp_path, {"ho-01#0": 1.0, "ho-02#0": 1.5, "ho-03#0": 3.0}, repeats=3)
    s = cs.summarize_pairs(next(iter(cs.pair_requests(cs.load_cells(tmp_path)).values())), n_boot=500)
    d = s["metrics"]["decode_tok_s"]
    assert d["median_ratio"] == pytest.approx(1.5)
    lo, hi = d["ci95"]
    assert lo <= 1.5 <= hi
    assert lo >= 1.0 and hi <= 3.0
    assert d["share_better"] == pytest.approx(6 / 9)


def test_flags_prefill_and_tail_regression_block_promising(tmp_path: Path) -> None:
    build_pair(tmp_path, 1.8, pre_factor=0.5)
    s = cs.summarize_pairs(next(iter(cs.pair_requests(cs.load_cells(tmp_path)).values())), n_boot=100)
    assert s["metrics"]["prefill_tok_s"]["median_ratio"] == pytest.approx(0.5)
    assert s["flags"]["no_prefill_regression"] is False
    assert s["flags"]["promising"] is False
    (tmp_path / "t").mkdir()
    build_pair(tmp_path / "t", 1.8, p99_factor=1.5)
    s2 = cs.summarize_pairs(next(iter(cs.pair_requests(cs.load_cells(tmp_path / "t")).values())), n_boot=100)
    assert s2["flags"]["no_tail_regression"] is False and s2["flags"]["promising"] is False


def test_strong_needs_absolute_rate_and_incremental_label(tmp_path: Path) -> None:
    build_pair(tmp_path, 2.5, base_dec=0.5)  # 2.5x but only 1.25 tok/s
    s = cs.summarize_pairs(next(iter(cs.pair_requests(cs.load_cells(tmp_path)).values())), n_boot=100)
    assert s["flags"]["promising"] is True and s["flags"]["strong"] is False
    (tmp_path / "i").mkdir()
    build_pair(tmp_path / "i", 1.2)
    s2 = cs.summarize_pairs(next(iter(cs.pair_requests(cs.load_cells(tmp_path / "i")).values())), n_boot=100)
    assert s2["flags"]["incremental"] is True and s2["flags"]["promising"] is False
    (tmp_path / "r").mkdir()
    build_pair(tmp_path / "r", 0.8)
    s3 = cs.summarize_pairs(next(iter(cs.pair_requests(cs.load_cells(tmp_path / "r")).values())), n_boot=100)
    assert s3["flags"]["regression"] is True


def test_conditions_not_mixed_and_unpaired_dropped(tmp_path: Path) -> None:
    make_cell(tmp_path, "A8", "A", budget_gib=8, requests=[req("ho-01#0", 1.0), req("ho-02#0", 1.0)])
    make_cell(tmp_path, "C8", "C", budget_gib=8, requests=[req("ho-01#0", 2.0)])
    make_cell(tmp_path, "C6", "C", budget_gib=6, requests=[req("ho-01#0", 5.0)])  # no A at 6 GiB
    make_cell(tmp_path, "Cunthr", "C", budget_gib=8, storage="fuse-unthrottled", rate=0.0,
              requests=[req("ho-01#0", 9.0)])
    pairs = cs.pair_requests(cs.load_cells(tmp_path))
    assert len(pairs) == 1
    (cond, meth), ps = next(iter(pairs.items()))
    assert cond[0] == 8 * GIB and cond[2] == "fuse-throttled"
    assert [p["rid"] for p in ps] == ["ho-01#0"]


def test_smoke_cells_excluded_and_subset_not_paired_with_full(tmp_path: Path) -> None:
    make_cell(tmp_path, "A", "A", requests=[req("ho-01#0", 1.0)])
    d = make_cell(tmp_path, "Csmoke", "C", requests=[req("ho-01#0", 9.0)])
    cfg = json.loads((d / "config.json").read_text("utf-8"))
    cfg["purpose"] = "smoke"
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert [c["config"]["cell_id"] for c in cs.load_cells(tmp_path)] == ["A"]
    assert len(cs.load_cells(tmp_path, include_smoke=True)) == 2
    e = make_cell(tmp_path, "Ccap", "C", requests=[req("ho-01#0", 9.0)])
    cfg = json.loads((e / "config.json").read_text("utf-8"))
    cfg["workload"]["n_predict_cap"] = 16
    (e / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert cs.pair_requests(cs.load_cells(tmp_path)) == {}
    assert "/cap16" in cs.cond_label(cs.condition(cfg))


def test_failed_cells_excluded_and_duplicate_repeat_rejected(tmp_path: Path) -> None:
    make_cell(tmp_path, "A", "A", requests=[req("ho-01#0", 1.0)])
    make_cell(tmp_path, "Cfail", "C", status="fail", requests=[req("ho-01#0", 9.0)])
    assert [c["config"]["cell_id"] for c in cs.load_cells(tmp_path)] == ["A"]
    assert len(cs.load_cells(tmp_path, include_failed=True)) == 2
    make_cell(tmp_path, "A-dup", "A", requests=[req("ho-01#0", 1.0)])
    with pytest.raises(ValueError, match="duplicate"):
        cs.pair_requests(cs.load_cells(tmp_path))


def test_first_divergence() -> None:
    assert cs.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert cs.first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert cs.first_divergence([1, 2, 3], [1, 2]) == 2
    assert cs.first_divergence([], []) is None
    assert cs.first_divergence([], [5]) == 0


def test_exactness_reports_divergence_and_prompt_mismatch(tmp_path: Path) -> None:
    rq = [req(x, 1.0) for x in RIDS]
    make_cell(tmp_path, "A0", "A", requests=rq)
    make_cell(tmp_path, "C0", "C", requests=rq, tokens={"ho-02#0": [0, 1, 2, 7, 4, 5, 6, 7, 8, 9]})
    make_cell(tmp_path, "D0", "D", requests=rq, prompts={"ho-03#0": "other"})
    rep = cs.exactness(cs.load_cells(tmp_path))
    by = {c["cell"]: c for c in rep["cells"]}
    assert by["C0"]["ref_cell"] == "A0" and by["C0"]["n_identical"] == 2 and by["C0"]["all_identical"] is False
    assert by["C0"]["divergent"][0]["rid"] == "ho-02#0" and by["C0"]["divergent"][0]["first_divergence"] == 3
    assert by["D0"]["all_identical"] is True
    assert by["D0"]["divergent"][0]["prompt_identical"] is False
    assert "A0" not in by  # a single A cell has no other reference


def test_bootstrap_helpers_deterministic() -> None:
    xs = [1.0, 2.0, 3.0, 4.0, 10.0]
    assert cs.bootstrap_ci(xs, seed=3) == cs.bootstrap_ci(xs, seed=3)
    lo, hi = cs.bootstrap_ci(xs, n_boot=1000)
    assert lo <= statistics.median(xs) <= hi
    assert cs.bootstrap_ci([1.0]) == (None, None)
    lo2, hi2 = cs.cluster_bootstrap_ci({"a": [2.0, 2.0, 2.0]})  # one cluster -> plain bootstrap
    assert lo2 == hi2 == 2.0


def test_spearman() -> None:
    assert cs.spearman([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)
    assert cs.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert cs.spearman([1, 2], [3, 4]) is None
    assert cs.spearman([1, 1, 2, 2], [1, 2, 3, 4]) == pytest.approx(0.894427, rel=1e-4)


def test_budget_readbytes_setup_tables(tmp_path: Path) -> None:
    build_pair(tmp_path, 2.0, repeats=1)
    cells = cs.load_cells(tmp_path)
    bt = {r["method"]: r for r in cs.budget_table(cells)}
    assert bt["A"]["decode_tok_s_median"] == pytest.approx(1.0) and bt["C"]["decode_tok_s_median"] == pytest.approx(2.0)
    assert bt["A"]["budget_gib"] == 8 and bt["A"]["n_requests"] == 3
    rb = cs.read_bytes_table(cells)
    assert rb["spearman_bytes_per_token_vs_decode_tok_s"] == pytest.approx(-1.0)
    agg = {r["method"]: r for r in rb["by_method_budget"]}
    assert agg["A"]["decode_MB_per_token_median"] == pytest.approx(60.0)
    st = {r["method"]: r for r in cs.setup_table(cells)}
    # 30 tokens / (0 + 100 + 5 + 30 s)
    assert st["A"]["amortized_tok_s"] == pytest.approx(30 / 135, rel=1e-3)
    assert st["A"]["host_cpu_pct_mean"] == 20.0


def test_summarize_markdown_and_cli(tmp_path: Path) -> None:
    root = tmp_path / "controlled"
    root.mkdir()
    build_pair(root, 1.6, repeats=3)
    s = cs.summarize(cs.load_cells(root), n_boot=100)
    md = cs.to_markdown(s)
    assert "8GiB/heldout/fuse-throttled@91.6MB/s" in md and "| C vs A |" in md and "promising" in md
    out, mdp = tmp_path / "s.json", tmp_path / "s.md"
    rc = cs.main(["summarize", "--root", str(root), "--out", str(out), "--md", str(mdp), "--n-boot", "100"])
    assert rc == 0 and out.exists() and mdp.exists()
    doc = json.loads(out.read_text("utf-8"))
    assert doc["paired"][0]["metrics"]["decode_tok_s"]["median_ratio"] == pytest.approx(1.6)
    assert cs.main(["exactness", "--root", str(root), "--out", str(tmp_path / "e.json")]) == 0


# ------------------------------------------------------------------ added 2026-10-05 22:55 (resume): engines
# (variants, own-parent comparison), background CPU next to requests, text exactness, per-category markdown
def _set_engine(d: Path, engine_id: str, parent: str | None = None, **extra) -> None:
    cfg = json.loads((d / "config.json").read_text("utf-8"))
    cfg["engine"] = {"id": engine_id, "parent": parent}
    cfg.update(extra)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")


def _engine_cells(root: Path, repeats: int = 2) -> None:
    for r in range(repeats):
        make_cell(root, f"A-r{r}", "A", repeat=r, requests=[req(x, 1.0) for x in RIDS])
        _set_engine(make_cell(root, f"hp-parent-r{r}", "A", repeat=r, requests=[req(x, 0.8) for x in RIDS]),
                    "hotpin-parent")
        _set_engine(make_cell(root, f"hp-r{r}", "A", repeat=r, requests=[req(x, 1.6) for x in RIDS]),
                    "hotpin", parent="hotpin-parent")
        _set_engine(make_cell(root, f"up-r{r}", "A", repeat=r, requests=[req(x, 1.1) for x in RIDS]),
                    "upstream-8345f33", parent="a7b94df")


def test_variant_labels_and_parent() -> None:
    assert cs.variant({"method": "C"}) == "C"
    assert cs.variant({"method": "C", "engine": {"id": "a7b94df"}}) == "C"
    assert cs.variant({"method": "A", "engine": {"id": "ik-plain"}}) == "ik-plain:A"
    assert cs.variant({"method": "A", "variant": "x:A", "engine": {"id": "ik-plain"}}) == "x:A"
    assert cs.parent_variant({"method": "A", "engine": {"id": "hotpin", "parent": "hotpin-parent"}}) == "hotpin-parent:A"
    assert cs.parent_variant({"method": "A", "engine": {"id": "up", "parent": "a7b94df"}}) == "A"
    assert cs.parent_variant({"method": "A", "engine": {"id": "up", "parent": "llama.cpp-a7b94df"}}) == "A"
    assert cs.variant({"method": "D", "engine": {"id": "llama.cpp-a7b94df"}}) == "D"
    assert cs.parent_variant({"method": "A", "engine": {"id": "ik-plain"}}) is None
    assert cs.parent_variant({"method": "A"}) is None


def test_engine_variants_vs_reference_and_vs_own_parent(tmp_path: Path) -> None:
    _engine_cells(tmp_path)
    cells = cs.load_cells(tmp_path)
    vs_ref = cs.pair_requests(cells, "A")
    assert sorted(k[1] for k in vs_ref) == ["hotpin-parent:A", "hotpin:A", "upstream-8345f33:A"]
    assert all(len(v) == 6 for v in vs_ref.values())
    vs_par = cs.pair_requests_vs_parent(cells, "A")
    assert [(k[1], k[2]) for k in vs_par] == [("hotpin:A", "hotpin-parent:A")]  # upstream's parent IS the baseline
    s = cs.summarize(cells, n_boot=100)
    by = {(c["treatment"], c["baseline"]): c for c in s["paired"]}
    assert by[("hotpin:A", "A")]["metrics"]["decode_tok_s"]["median_ratio"] == pytest.approx(1.6)
    assert by[("hotpin:A", "A")]["comparison"] == "vs reference baseline"
    own = by[("hotpin:A", "hotpin-parent:A")]
    assert own["metrics"]["decode_tok_s"]["median_ratio"] == pytest.approx(2.0)   # 1.6 / 0.8: the patch's own effect
    assert own["comparison"] == "vs own unmodified parent engine" and own["flags"]["promising"] is True
    assert by[("hotpin-parent:A", "A")]["flags"]["regression"] is True
    assert by[("upstream-8345f33:A", "A")]["flags"]["incremental"] is True
    bt = {r["method"]: r for r in cs.budget_table(cells)}
    assert set(bt) == {"A", "hotpin-parent:A", "hotpin:A", "upstream-8345f33:A"}
    assert bt["hotpin:A"]["decode_tok_s_median"] == pytest.approx(1.6) and bt["hotpin:A"]["n_cells"] == 2
    assert bt["A"]["decode_tok_s_per_cell_median"] == [pytest.approx(1.0), pytest.approx(1.0)]
    md = cs.to_markdown(s)
    assert "| hotpin:A vs hotpin-parent:A |" in md and "per workload category" in md and "| chat |" in md


def test_exactness_reference_chain_same_engine_parent_cross_engine(tmp_path: Path) -> None:
    rq = [req(x, 1.0) for x in RIDS]
    diff = {"ho-01#0": [0, 1, 2, 3, 4, 99, 6, 7, 8, 9]}
    make_cell(tmp_path, "A0", "A", requests=rq)
    _set_engine(make_cell(tmp_path, "P0", "A", requests=rq, tokens=diff), "hotpin-parent")
    _set_engine(make_cell(tmp_path, "H0", "A", requests=rq, tokens=diff), "hotpin", parent="hotpin-parent")
    _set_engine(make_cell(tmp_path, "HC0", "C", requests=rq), "hotpin", parent="hotpin-parent")
    rep = cs.exactness(cs.load_cells(tmp_path))
    by = {c["cell"]: c for c in rep["cells"]}
    # the patched engine equals its own parent token for token ...
    assert by["H0"]["ref_cell"] == "P0" and by["H0"]["ref_kind"] == "unmodified parent engine"
    assert by["H0"]["all_identical"] is True
    # ... a cache method of that engine is checked against the SAME engine's method A ...
    assert by["HC0"]["ref_cell"] == "H0" and by["HC0"]["ref_kind"] == "same engine"
    assert by["HC0"]["all_identical"] is False and by["HC0"]["divergent"][0]["first_divergence"] == 5
    # ... and the parent differs from the reference engine (cross-engine difference, reported as such)
    assert by["P0"]["ref_cell"] == "A0" and by["P0"]["ref_kind"] == "reference engine (cross-engine)"
    assert by["P0"]["n_identical"] == 2


def test_exactness_text_difference_and_request_subset(tmp_path: Path) -> None:
    rq = [req(x, 1.0) for x in RIDS]
    a = make_cell(tmp_path, "A0", "A", requests=rq)
    c = make_cell(tmp_path, "C0", "C", requests=rq)
    for d, text in ((a, "hello"), (c, "hellO")):
        p = d / "outputs" / "ho-02_0.json"
        o = json.loads(p.read_text("utf-8"))
        o.update({"text": text, "token_ids_source": "re-tokenized from the output text (/tokenize), not the sampled ids"})
        p.write_text(json.dumps(o), encoding="utf-8")
    capped = make_cell(tmp_path, "Dcap", "D", requests=rq)
    cfg = json.loads((capped / "config.json").read_text("utf-8"))
    cfg["workload"]["n_predict_cap"] = 16
    (capped / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    by = {x["cell"]: x for x in cs.exactness(cs.load_cells(tmp_path))["cells"]}
    assert by["C0"]["n_identical"] == 3                       # token ids equal ...
    assert [d["rid"] for d in by["C0"]["divergent"]] == ["ho-02#0"]  # ... but the text differs: reported
    assert by["C0"]["divergent"][0]["text_identical"] is False
    assert "Dcap" not in by                                   # ran other requests (cap 16): no reference


def test_host_cpu_joined_to_requests(tmp_path: Path) -> None:
    t0 = 1_800_000_000.0
    for name, meth, dec, start, load in (("A0", "A", 1.0, 100.0, 5.0), ("C0", "C", 2.0, 100.0, 80.0)):
        rq = [dict(req("ho-01#0", dec), t_start_epoch=t0 + start, total_s=10.0, bg_vm_other_cores_decode=0.25)]
        d = make_cell(tmp_path, name, meth, requests=rq)
        cfg = json.loads((d / "config.json").read_text("utf-8"))
        cfg["host"]["host_cpu_t0_epoch"] = t0
        (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        rows = ["# t0_epoch=1800000000.000 (Windows host clock; t_s is relative to it)", "t_s,host_cpu_pct,vmmem_cpu_pct_of_host"]
        rows += [f"{t:.1f},{load if 100 <= t <= 111 else 1.0},3.0" for t in range(0, 200)]
        (d / "host_cpu.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    cells = cs.load_cells(tmp_path)
    ps = next(iter(cs.pair_requests(cells).values()))
    assert ps[0]["treat"]["_host_cpu_pct"] == pytest.approx(80.0) and ps[0]["base"]["_host_cpu_pct"] == pytest.approx(5.0)
    s = cs.summarize_pairs(ps, n_boot=50)
    assert s["background_treatment"]["host_cpu_pct_mean_over_requests"] == pytest.approx(80.0)
    assert s["background_baseline"]["host_cpu_pct_max_request"] == pytest.approx(5.0)
    assert s["background_treatment"]["vm_other_cores_decode_mean"] == pytest.approx(0.25)
    # cells written before the host clock was recorded: no number, no crash
    build_pair(tmp_path / "old", 2.0, repeats=1)
    old = cs.summarize_pairs(next(iter(cs.pair_requests(cs.load_cells(tmp_path / "old")).values())), n_boot=50)
    assert old["background_treatment"]["host_cpu_pct_mean_over_requests"] is None


def test_setup_table_lock_accounting_and_read_bytes_extras(tmp_path: Path) -> None:
    d = make_cell(tmp_path, "C0", "C", requests=[req(x, 2.0, bpt=30_000_000) for x in RIDS])
    cfg = json.loads((d / "config.json").read_text("utf-8"))
    cfg["effective"]["lock_accounting"] = {"requested_bytes_page_aligned_union": 4096000, "pinner_locked_bytes": 4096000,
                                           "effective_lock_verified": True}
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    make_cell(tmp_path, "A0", "A", requests=[req(x, 1.0, bpt=60_000_000) for x in RIDS])
    cells = cs.load_cells(tmp_path)
    st = {r["cell"]: r for r in cs.setup_table(cells)}
    assert st["C0"]["requested_lock_bytes"] == 4096000 and st["C0"]["effective_lock_verified"] is True
    assert st["A0"]["requested_lock_bytes"] is None
    rb = cs.read_bytes_table(cells)
    assert set(rb["spearman_within_method_budget"]) == {"A@8GiB", "C@8GiB"}
    agg = {r["method"]: r for r in rb["by_method_budget"]}
    assert agg["C"]["decode_MB_per_token_median"] == pytest.approx(30.0)
    assert agg["A"]["prefill_GB_per_request_median"] == pytest.approx(1e-6)
