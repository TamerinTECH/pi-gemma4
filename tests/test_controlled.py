"""Unit tests for the P5 controlled-memory harness: pigemma.controlled (host side) and the
pure helpers of pigemma.controlled_cell (in-cgroup runner). No Docker, no model loads; the
pin-planner integration test uses the real UD-Q3_K_M map JSON when it is present."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pigemma import controlled as ct
from pigemma import controlled_cell as cc

REPO = Path(__file__).resolve().parent.parent
REAL_MAP = REPO / "results" / "maps" / "map26_ud_q3_k_m.json"
GIB = 1024**3


# ------------------------------------------------------------------ controlled_cell helpers
def test_itl_stats_stalls_and_percentiles() -> None:
    times = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 5.5]  # gaps .5 .5 .5 .5 2.0 .5
    s = cc.itl_stats(times)
    assert s["n_gaps"] == 6
    assert s["itl_median_s"] == pytest.approx(0.5)
    assert s["itl_max_s"] == pytest.approx(2.0)
    assert s["stalls_gt_3x_median"] == 1 and s["stall_threshold_s"] == pytest.approx(1.5)
    assert s["itl_p99_s"] == pytest.approx(2.0)
    assert cc.itl_stats([1.0])["n_gaps"] == 0
    assert cc.itl_stats([1.0, 1.0, 2.0])["zero_gaps"] == 1


def test_pct_nearest_rank() -> None:
    assert cc.pct([], 50) is None
    assert cc.pct([3, 1, 2], 50) == 2
    assert cc.pct(list(range(101)), 90) == 90


def test_server_argv_fixed_flags_and_no_speculation() -> None:
    scfg = {"bin": "/x/llama-server", "threads": 4, "ctx": 4096, "batch": 2048, "ubatch": 512,
            "cache_ram_mib": 0, "ctx_checkpoints": 32, "warmup": True}
    a = cc.server_argv(scfg, "/mnt/m.gguf", 18000)
    assert a[0] == "/x/llama-server" and "-nr" in a
    for flag, val in (("-t", "4"), ("-tb", "4"), ("-c", "4096"), ("-b", "2048"), ("-ub", "512"),
                      ("-ctk", "f16"), ("-ctv", "f16"), ("-np", "1"), ("--cache-ram", "0"), ("--port", "18000")):
        assert a[a.index(flag) + 1] == val
    assert "--warmup" in a and "--no-warmup" not in a
    assert a[a.index("-lv") + 1] == "4"
    assert "--no-warmup" in cc.server_argv(dict(scfg, warmup=False), "/m", 1)
    with pytest.raises(ValueError, match="speculative"):
        cc.server_argv(dict(scfg, extra=["-md", "draft.gguf"]), "/m", 1)
    with pytest.raises(ValueError):
        cc.server_argv(dict(scfg, extra=["--spec-type", "draft-mtp"]), "/m", 1)
    assert cc.server_argv(dict(scfg, nice=10), "/m", 1)[:3] == ["nice", "-n", "10"]


def test_parse_server_log() -> None:
    log = ("build: 11398 (a7b94df) with GNU 13.3.0\n"
           "load_tensors:   CPU_Mapped model buffer size = 12124.05 MiB\n"
           "llama_kv_cache:        CPU KV buffer size =   380.00 MiB\n"
           "sched_reserve:        CPU compute buffer size =   538.01 MiB\n"
           "llama_context: flash_attn = enabled\n"
           "llama_context:        CPU  output buffer size =     1.00 MiB\n"
           "srv  load_model: context checkpoints enabled, max = 32, min spacing = 256\n"
           "slot x: forcing full prompt re-processing due to lack of cache data\n")
    p = cc.parse_server_log(log)
    assert p["buffers_mib"]["CPU_Mapped model"] == pytest.approx(12124.05)
    assert p["buffers_mib"]["CPU KV"] == pytest.approx(380.0)
    assert p["repack_present"] is False and p["build"] == {"number": 11398, "commit": "a7b94df"}
    assert p["n_forcing_full_reprocess"] == 1 and p["flash_attn_effective"] == "enabled"
    assert p["buffers_mib"]["CPU output"] == pytest.approx(1.0) and p["ctx_checkpoints_max"] == 32
    assert cc.parse_server_log("CPU_REPACK model buffer size = 10 MiB")["repack_present"] is True


def _snap(t: float, **kw) -> dict:
    base = {"t": t, "cg": {"usage": 0, "pgmajfault": 0, "ev_max": 0, "ev_oom_kill": 0},
            "cg_cpu_ns": 0, "cg_io": None, "proc": {}, "fuse": None, "disk": {"sdc": 0},
            "vm": {"pgmajfault": 0, "pgscan_kswapd": 0, "pgscan_direct": 0}, "cpu": (0, 100)}
    for k, v in kw.items():
        base[k] = v
    return base


def test_io_row_deltas() -> None:
    a = _snap(10.0, proc={"server": {"read_bytes": 0, "majflt": 5, "rchar": 0, "minflt": 0}},
              fuse={"backing_bytes": 1000, "requests": 1, "queue_wait_ns": 0, "sleep_ns": 0})
    b = _snap(12.5, proc={"server": {"read_bytes": 0, "majflt": 25, "rchar": 7, "minflt": 3}},
              fuse={"backing_bytes": 1000 + 131072 * 10, "requests": 11, "queue_wait_ns": 2_000_000_000, "sleep_ns": 10**9},
              disk={"sdc": 4096}, cg={"usage": 77, "pgmajfault": 20, "ev_max": 2, "ev_oom_kill": 0,
                                      "file": 5, "anon": 6, "unevictable": 7},
              vm={"pgmajfault": 30, "pgscan_kswapd": 4, "pgscan_direct": 6}, cpu=(300, 1300), cg_cpu_ns=2 * 10**9)
    r = cc.io_row("decode", "x#0", a, b)
    assert r["wall_s"] == pytest.approx(2.5)
    assert r["fuse_backing_bytes"] == 1310720 and r["fuse_requests"] == 10
    assert r["fuse_queue_wait_s"] == pytest.approx(2.0) and r["fuse_sleep_s"] == pytest.approx(1.0)
    assert r["server_majflt"] == 20 and r["server_read_bytes"] == 0 and r["server_rchar"] == 7
    assert r["disk_read_bytes"] == 4096 and r["disk_devices"] == "sdc"
    assert r["cg_pgmajfault"] == 20 and r["cg_failcnt_delta"] == 2 and r["vm_pgscan"] == 10
    assert r["cg_usage_end"] == 77 and r["cg_unevictable_end"] == 7
    assert r["cg_cpu_s"] == pytest.approx(2.0)
    assert set(r) == set(cc.IO_FIELDS)


def test_categories_from_synthetic_map(tmp_path: Path) -> None:
    page = 4096
    size = 10 * page
    doc = {"expert_map": {
        "files": [{"index": 0, "size": size, "data_offset": page}],
        "nonexpert_ranges": [{"file": 0, "offset": page, "length": 2 * page}],
        "experts": [{"layer": 0, "expert": 0, "ranges": [{"offset": 3 * page, "n_bytes": 2 * page}]},
                    {"layer": 1, "expert": 0, "ranges": [{"offset": 5 * page, "n_bytes": page + 10}]}]}}
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(doc))
    cat = cc.Categories(str(mp), page, size)
    assert cat.ok
    vec = bytes([1, 1, 0, 1, 1, 1, 1, 0, 0, 1])
    s = cat.summarize(vec)
    assert s["header_pages"] == 1 and s["nonexpert_pages"] == 1
    assert s["expert_pages_by_layer"] == {"expert_L0": 2, "expert_L1": 2}
    assert s["unowned_pages"] == 1 and s["resident_pages_total"] == 7
    assert not cc.Categories(str(mp), page, size + 1).ok  # map belongs to another file


# ------------------------------------------------------------------ controlled (host side)
def test_default_config_and_docker_argv() -> None:
    cfg = ct.default_config(method="A", budget_bytes=8 * GIB, workload_set="tuning", repeat=1, cell_id="t-A-8g-r1")
    assert cfg["server"]["cache_ram_mib"] == 0 and cfg["storage"]["rate_bps"] == pytest.approx(91.6e6)
    a = ct.docker_argv(cfg)
    assert f"--memory={8 * GIB}b" in a and f"--memory-swap={8 * GIB}b" in a and "--memory-swappiness=0" in a
    assert "--cpuset-cpus=2,4,6,8" in a and "--cpus=4.0" in a
    for x in ("IPC_LOCK", "memlock=-1:-1", "SYS_ADMIN", "/dev/fuse", "apparmor:unconfined", "none",
              "pigemma-models:/models:ro", "pi-gemma4-build:fuse"):
        assert x in a
    assert a[-1] == "/work/results/controlled/t-A-8g-r1/config.json"
    with pytest.raises(ValueError):
        ct.default_config(method="C", budget_bytes=GIB, workload_set="tuning", repeat=0, cell_id="x")
    with pytest.raises(ValueError):
        ct.default_config(method="E", budget_bytes=GIB, workload_set="tuning", repeat=0, cell_id="x")
    u = ct.default_config(method="B", budget_bytes=GIB, workload_set="heldout", repeat=0, cell_id="x",
                          storage_mode="fuse-unthrottled")
    assert u["storage"]["rate_bps"] == 0.0
    assert cfg["purpose"] == "measurement" and cfg["server"]["verbosity"] == 4
    with pytest.raises(ValueError):
        ct.default_config(method="A", budget_bytes=GIB, workload_set="tuning", repeat=0, cell_id="x", purpose="x")


def test_pin_budget_arithmetic() -> None:
    r = ct.pin_budget(8 * GIB, anon_bytes=1_300_000_000, kernel_bytes=100_000_000, nonexpert_bytes=2_400_129_024,
                      miss_reserve_bytes=644_549_568, safety_bytes=int(0.05 * 8 * GIB))
    left = 8 * GIB - 1_300_000_000 - 100_000_000 - 2_400_129_024 - 644_549_568 - int(0.05 * 8 * GIB)
    assert r["expert_pin_bytes"] == left and r["feasible"]
    r2 = ct.pin_budget(3 * GIB, anon_bytes=1_300_000_000, kernel_bytes=0, nonexpert_bytes=2_400_000_000,
                       miss_reserve_bytes=0, safety_bytes=0)
    assert r2["expert_pin_bytes"] == 0 and r2["feasible"] is False


def test_load_ranking_formats_and_heldout_guard(tmp_path: Path) -> None:
    p1 = tmp_path / "r1.json"
    p1.write_text(json.dumps({"set": "calibration", "ranking": [[0, 1], [2, 3], [0, 1], {"layer": 1, "expert": 0}]}))
    items, meta = ct.load_ranking(p1, 3, 4)
    assert items == [(0, 1), (2, 3), (1, 0)] and meta["set"] == "calibration" and len(meta["sha256"]) == 64
    p2 = tmp_path / "r2.json"
    p2.write_text(json.dumps({"counts": [[0, 5, 1, 0], [9, 0, 0, 0], [0, 0, 0, 5]]}))
    items2, _ = ct.load_ranking(p2, 3, 4)
    assert items2[:3] == [(1, 0), (0, 1), (2, 3)]
    p3 = tmp_path / "r3.json"
    p3.write_text(json.dumps({"set": "heldout", "ranking": [[0, 0]]}))
    with pytest.raises(ValueError, match="calibration"):
        ct.load_ranking(p3, 3, 4)
    p4 = tmp_path / "r4.json"
    p4.write_text(json.dumps([[5, 0]]))
    with pytest.raises(ValueError, match="out of range"):
        ct.load_ranking(p4, 3, 4)


def test_page_planner_merges_overlapping_runs() -> None:
    assert ct.PagePlanner.merged([(5, 7), (0, 2), (1, 3), (7, 8)]) == [(0, 3), (5, 8)]


@pytest.mark.skipif(not REAL_MAP.exists(), reason="real UD-Q3_K_M map not present")
def test_plan_pins_real_map_c_and_d(tmp_path: Path) -> None:
    from pigemma.expert_map import load_expert_map
    em = load_expert_map(REAL_MAP)
    ranking = [(l, e) for e in range(128) for l in range(30)]  # arbitrary but fixed order
    c = ct.plan_pins(em, 4096, "C", ranking=ranking, expert_pin_bytes=1_000_000_000)
    assert c["check_ok"] and c["nonexpert_pages"] == em.pages_for([], 4096, with_nonexpert=True)
    assert (c["expert_pages"]) * 4096 <= 1_000_000_000
    # the next ranked expert would not have fitted (strict prefix)
    nxt = ranking[c["n_experts"]]
    assert em.pages_for([tuple(x) for x in c["experts"]] + [nxt], 4096, True) * 4096 - c["nonexpert_pages"] * 4096 > 1_000_000_000
    assert c["experts"] == [list(k) for k in ranking[:c["n_experts"]]]
    d = ct.plan_pins(em, 4096, "D", target_total_pages=c["total_unique_pages"], seed=3)
    assert d["check_ok"] and d["total_unique_pages"] <= c["total_unique_pages"]
    assert c["total_unique_pages"] - d["total_unique_pages"] < 700  # within one expert (~644-1030 pages)
    assert d["experts"] != c["experts"]
    d2 = ct.plan_pins(em, 4096, "D", target_total_pages=c["total_unique_pages"], seed=3)
    assert d2["experts"] == d["experts"]  # seeded
    chk = ct.write_ranges(em, c, tmp_path / "c.ranges.txt")
    assert chk["union_matches_plan"]
    head = (tmp_path / "c.ranges.txt").read_text("ascii").splitlines()[0]
    assert head.startswith("# pigemma-ranges v1 gguf=gemma-4-26B-A4B-it-UD-Q3_K_M.gguf size=12728497888 page=4096")


# ------------------------------------------------------------------ added 2026-10-05 22:50 (resume): engines,
# CPU lock, series order, reclaim counters, streaming fallback, lock accounting
import http.server  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

FROZEN_SMOKE_ARGV = ("/work/third_party/llama.cpp/build-linux-x86/bin/llama-server -m /mnt/pigemma-thr/"
                     "gemma-4-26B-A4B-it-UD-Q3_K_M.gguf -nr -t 4 -tb 4 -c 4096 -b 2048 -ub 512 -ctk f16 -ctv f16 "
                     "-fa auto -np 1 --cache-ram 0 --ctx-checkpoints 32 --seed 0 -lv 4 --host 127.0.0.1 --port 57563 "
                     "--no-webui --warmup")


def test_server_argv_identical_to_the_smoke_cells() -> None:
    """The flag-map refactoring must not change the reference engine's command line (the argv of
    results/controlled/20261005-smoke2-tuning-A-8g-r0, port included)."""
    cfg = ct.default_config(method="A", budget_bytes=8 * GIB, workload_set="tuning", repeat=0, cell_id="x")
    a = cc.server_argv(cfg["server"], "/mnt/pigemma-thr/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf", 57563)
    assert " ".join(a) == FROZEN_SMOKE_ARGV


def test_server_argv_flag_overrides() -> None:
    scfg = {"bin": "/e/llama-server", "threads": 4, "warmup": True,
            "flag_overrides": {"no_repack": [], "cache_ram": None, "verbosity": ["-v"],
                               "threads": ["--threads", "{threads}"], "listen": ["--port", "{port}"]},
            "extra": ["--prefetch-experts"]}
    a = cc.server_argv(scfg, "/m.gguf", 9)
    assert "-nr" not in a and "--cache-ram" not in a and "-lv" not in a and "-tb" not in a
    assert a[a.index("--threads") + 1] == "4" and a[a.index("--port") + 1] == "9" and "--host" not in a
    assert "-v" in a and a[-1] == "--prefetch-experts" and "--warmup" in a
    with pytest.raises(ValueError, match="unknown flag_overrides"):
        cc.server_argv(dict(scfg, flag_overrides={"nope": []}), "/m", 1)
    with pytest.raises(ValueError, match="model"):
        cc.server_argv(dict(scfg, flag_overrides={"model": []}), "/m", 1)
    with pytest.raises(ValueError, match="speculative"):
        cc.server_argv(dict(scfg, extra=["--spec-draft-n-max", "4"]), "/m", 1)
    # dropping the warm-up flags of an engine that has neither spelling
    b = cc.server_argv(dict(scfg, flag_overrides={"warmup_on": None, "warmup_off": None}), "/m", 1)
    assert "--warmup" not in b and "--no-warmup" not in b


def test_psi_totals_and_new_io_fields() -> None:
    txt = "some avg10=0.00 avg60=1.50 avg300=0.10 total=123456\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=789\n"
    assert cc.psi_totals(txt) == {"some": 123456, "full": 789}
    assert cc.psi_totals(None) == {"some": None, "full": None}
    a = _snap(0.0, cg={"usage": 0, "pgmajfault": 0, "ev_max": 0, "ev_oom_kill": 0, "pgpgin": 100, "pgpgout": 40,
                       "psi_some_us": 1_000_000, "psi_full_us": 0, "pgscan": 5, "pgsteal": 4, "ev_high": 1,
                       "workingset_refault_file": 10})
    b = _snap(2.0, cg={"usage": 0, "pgmajfault": 0, "ev_max": 0, "ev_oom_kill": 0, "pgpgin": 350, "pgpgout": 240,
                       "psi_some_us": 3_500_000, "psi_full_us": 250_000, "pgscan": 25, "pgsteal": 14, "ev_high": 4,
                       "workingset_refault_file": 110, "swap": 0})
    r = cc.io_row("decode", "x", a, b)
    assert r["cg_pgpgin"] == 250 and r["cg_pgpgout"] == 200
    assert r["cg_psi_some_s"] == pytest.approx(2.5) and r["cg_psi_full_s"] == pytest.approx(0.25)
    assert r["cg_pgscan"] == 20 and r["cg_pgsteal"] == 10 and r["cg_refault_file"] == 100
    assert r["cg_swap_end"] == 0 and r["cg_ev_high_delta"] == 3
    v1 = cc.io_row("decode", "x", _snap(0.0), _snap(1.0))  # cgroup v1 snapshot: no PSI / pgscan keys
    assert v1["cg_psi_some_s"] is None and v1["cg_pgscan"] is None and v1["cg_pgpgin"] is None
    assert set(r) == set(cc.IO_FIELDS) and len(cc.IO_FIELDS) == len(set(cc.IO_FIELDS))
    assert cc.SAMPLE_FIELDS[-1] == "temp_mC" and len(cc.SAMPLE_FIELDS) == len(set(cc.SAMPLE_FIELDS))


class _SSE(http.server.BaseHTTPRequestHandler):
    events: list[dict] = []
    gap_s = 0.02
    seen: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length", "0"))
        type(self).seen.append({"path": self.path, "body": json.loads(self.rfile.read(n) or b"{}")})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for e in type(self).events:
            time.sleep(type(self).gap_s)
            self.wfile.write(("data: " + json.dumps(e) + "\n\n").encode())
            self.wfile.flush()

    def log_message(self, *a) -> None:  # silence
        pass


def _serve(events: list[dict]) -> tuple[http.server.HTTPServer, int]:
    _SSE.events, _SSE.seen = events, []
    srv = http.server.HTTPServer(("127.0.0.1", 0), _SSE)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_stream_completion_token_ids_and_event_fallback() -> None:
    final = {"content": "", "tokens": [], "stop": True, "stop_type": "limit",
             "timings": {"prompt_n": 7, "cache_n": 0, "predicted_n": 3}}
    srv, port = _serve([{"content": "a", "tokens": [11]}, {"content": "b", "tokens": [12]},
                        {"content": "c", "tokens": [13]}, final])
    try:
        first: list[float] = []
        r = cc.stream_completion(port, {"prompt": "p", "n_predict": 3}, lambda: first.append(time.monotonic()))
    finally:
        srv.shutdown()
    assert r["token_ids"] == [11, 12, 13] and r["text"] == "abc" and r["timestamps_per"] == "token"
    assert len(r["token_times"]) == 3 and r["token_times"] == sorted(r["token_times"]) and len(first) == 1
    assert r["final"]["timings"]["predicted_n"] == 3 and r["t_end"] >= r["token_times"][-1]
    assert _SSE.seen[0]["path"] == "/completion"
    assert _SSE.seen[0]["body"]["stream"] is True and _SSE.seen[0]["body"]["return_tokens"] is True
    # an engine that returns no token ids: one timestamp per content-bearing SSE event
    srv, port = _serve([{"content": "a"}, {"content": ""}, {"content": "bc"}, dict(final, tokens=None)])
    try:
        first2: list[int] = []
        r2 = cc.stream_completion(port, {"prompt": "p"}, lambda: first2.append(1), path="/v1/x")
    finally:
        srv.shutdown()
    assert r2["token_ids"] == [] and r2["timestamps_per"] == "sse_event" and len(r2["token_times"]) == 2
    assert r2["text"] == "abc" and len(first2) == 1 and _SSE.seen[0]["path"] == "/v1/x"


def test_first_text_diff() -> None:
    assert cc.first_text_diff("abc", "abc") is None
    assert cc.first_text_diff("abc", "abd") == 2
    assert cc.first_text_diff("abc", "ab") == 2
    assert cc.first_text_diff("", "") is None


def test_lock_accounting_checks(tmp_path: Path) -> None:
    c = cc.Cell.__new__(cc.Cell)  # no files: only the pure accounting
    page = 4096
    req_b = 1000 * page
    plan = tmp_path / "c.plan.json"
    plan.write_text(json.dumps({"method": "C", "page": page, "n_experts": 3, "total_unique_pages": 1000,
                                "total_bytes": req_b, "nominal_expert_bytes": 3_000_000,
                                "nominal_nonexpert_bytes": 1_000_000, "alignment_overhead_bytes": req_b - 4_000_000,
                                "ranking": {"set": "calibration"}}), encoding="utf-8")
    c.cfg = {"method": "C", "pinner": {"ranges": str(tmp_path / "c.ranges.txt")}}
    c.eff = {"page_size": page, "pinner_verify_end": {"resident_bytes": req_b}}
    c.pinner_summary = {"requested_bytes": req_b, "requested_raw_bytes": req_b - 5000, "locked_bytes": req_b,
                        "vm_lck_kb": req_b // 1024, "failed": 0, "errors": {"ENOMEM": 0}, "lock_s": 12.5}
    c.cg_phases = [{"tag": "after_setup", "stat": {"unevictable": req_b},
                    "procs": {"pinner": {"status": {"VmLck": req_b // 1024}}}},
                   {"tag": "end", "stat": {"unevictable": req_b + page},
                    "procs": {"pinner": {"status": {"VmLck": req_b // 1024}}}}]
    a = c.lock_accounting()
    assert a["requested_unique_pages"] == 1000 and "alignment_overhead_bytes" not in a
    assert a["plan"]["plan_total_eq_pinner_requested"] is True and a["plan"]["n_experts"] == 3
    assert a["plan"]["alignment_overhead_bytes"] == req_b - 4_000_000 and len(a["plan"]["sha256"]) == 64
    assert a["plan"]["ranking"] == {"set": "calibration"}
    assert a["pinner_vm_lck_bytes_end"] == req_b and a["cg_unevictable_bytes_end"] == req_b + page
    assert a["effective_lock_verified"] is True and all(a["checks"].values())
    assert a["server_vm_lck_bytes_end"] is None           # no server entry in these marks
    c.cg_phases[1]["procs"]["server"] = {"status": {"VmLck": 2048}}
    assert c.lock_accounting()["server_vm_lck_bytes_end"] == 2048 * 1024   # engine-internal pinning is visible
    del c.cg_phases[1]["procs"]["server"]
    # the cgroup's unevictable counter may trail by a few pages (per-CPU LRU batches): tolerated up to 1 MiB
    c.cg_phases[1]["stat"]["unevictable"] = req_b - 7 * page
    assert c.lock_accounting()["effective_lock_verified"] is True
    c.cg_phases[1]["stat"]["unevictable"] = req_b - 2 * 1024**2
    assert c.lock_accounting()["effective_lock_verified"] is False
    c.cg_phases[1]["stat"]["unevictable"] = req_b
    c.pinner_summary["locked_bytes"] = req_b - page          # one page failed to lock
    assert c.lock_accounting()["effective_lock_verified"] is False
    c.pinner_summary["locked_bytes"] = req_b
    c.eff["pinner_verify_end"] = {"resident_bytes": req_b - page}  # mincore disagrees
    bad = c.lock_accounting()
    assert bad["checks"]["mincore_end_eq_requested"] is False and bad["effective_lock_verified"] is False
    c.cfg = {"method": "A"}
    c.pinner_summary = None
    assert c.lock_accounting()["effective_lock_verified"] is None


def test_engines_file_schema_and_resolution(tmp_path: Path) -> None:
    assert ct.variant_label(None, "A") == "A" and ct.variant_label("a7b94df", "C") == "C"
    assert ct.variant_label("hotpin", "A") == "hotpin:A"
    f = tmp_path / "engines.json"
    f.write_text(json.dumps([
        {"engine_id": "upstream-8345f33", "binary_path_in_container": "/work/third_party/up/bin/llama-server",
         "revision": "8345f33", "build_flags": "-DGGML_NATIVE=ON", "notes": "n", "status": "locally tested"},
        {"id": "hotpin", "bin": "third_party/hotpin-llm/build/bin/llama-server", "parent": "hotpin-parent",
         "extra_args": "--hotpin-profile /work/x.json --hotpin-budget-mb 4096", "status": "runnable",
         "server_flag_overrides": {"cache_ram": None, "verbosity": "-v"}, "env": {"OMP_NUM_THREADS": 4},
         "docker_volumes": "pigemma-hotpin:/opt/hotpin:ro"},
        {"name": "ik-plain", "server_bin": "/opt/ik/bin/llama-server", "status": "blocked: does not load the GGUF"},
        {"engine_id": "a7b94df", "bin": "/somewhere/else"},
    ]), encoding="utf-8")
    engs = ct.load_engines(f)
    assert set(engs) == {"a7b94df", "upstream-8345f33", "hotpin", "ik-plain"}
    assert engs["a7b94df"]["bin"] == ct.LLAMA_SERVER  # reserved id: the built-in definition wins
    h = engs["hotpin"]
    assert h["bin"] == "/work/third_party/hotpin-llm/build/bin/llama-server" and h["parent"] == "hotpin-parent"
    assert h["extra"] == ["--hotpin-profile", "/work/x.json", "--hotpin-budget-mb", "4096"]
    assert h["flag_overrides"] == {"cache_ram": None, "verbosity": ["-v"]}
    assert h["volumes"] == ["pigemma-hotpin:/opt/hotpin:ro"] and len(h["profile_sha256"]) == 64
    assert ct.engine_runnable(engs["ik-plain"])[0] is False
    assert ct.engine_runnable(engs["hotpin"])[0] is False  # /work binary does not exist in the repo
    with pytest.raises(ct.EngineError, match="not runnable"):
        ct.resolve_engine("ik-plain", f)
    with pytest.raises(ct.EngineError, match="unknown engine"):
        ct.resolve_engine("nope", f)
    e = ct.resolve_engine("hotpin", f, allow_not_runnable=True)
    cfg = ct.default_config(method="A", budget_bytes=8 * GIB, workload_set="heldout", repeat=2, cell_id="c",
                            engine=e, server_extra=["--foo"])
    assert cfg["variant"] == "hotpin:A" and cfg["engine"]["parent"] == "hotpin-parent"
    assert cfg["server"]["bin"] == h["bin"] and cfg["server"]["extra"][-1] == "--foo"
    assert cfg["server"]["flag_overrides"] == {"cache_ram": None, "verbosity": ["-v"]}
    argv = ct.docker_argv(cfg)
    assert argv[argv.index("pigemma-hotpin:/opt/hotpin:ro") - 1] == "-v"
    sargv = cc.server_argv(cfg["server"], "/m.gguf", 5)
    assert "--cache-ram" not in sargv and "-v" in sargv and "--hotpin-profile" in sargv and "-nr" in sargv
    # other accepted shapes + errors
    g = tmp_path / "e2.json"
    g.write_text(json.dumps({"engines": {"x": {"bin": "/opt/x/llama-server", "status": "ok"}}}), encoding="utf-8")
    assert ct.resolve_engine("x", g)["bin"] == "/opt/x/llama-server"
    g.write_text(json.dumps([{"engine_id": "d", "bin": "/a"}, {"engine_id": "d", "bin": "/b"}]), encoding="utf-8")
    with pytest.raises(ct.EngineError, match="duplicate"):
        ct.load_engines(g)
    g.write_text(json.dumps([{"engine_id": "nobin"}]), encoding="utf-8")
    with pytest.raises(ct.EngineError, match="binary"):
        ct.load_engines(g)
    assert set(ct.load_engines(tmp_path / "missing.json")) == {"a7b94df"}
    assert ct.new_cell_id("A", 8 * GIB, "heldout", 1, engine_id="hotpin").split("-")[3] == "hotpin"
    assert "hotpin" not in ct.new_cell_id("A", 8 * GIB, "heldout", 1, engine_id="a7b94df")
    ref = ct.default_config(method="A", budget_bytes=GIB, workload_set="tuning", repeat=0, cell_id="r")
    assert ref["variant"] == "A" and ref["engine"]["id"] == "a7b94df" and ref["docker"]["extra_volumes"] == []


def test_cpu_lock_acquire_release_busy_stale(tmp_path: Path) -> None:
    p = tmp_path / ".cpu.lock"
    assert ct.cpu_lock_status(p)["held"] is False
    got = ct.cpu_lock_acquire("M1", "pilot", path=p)
    assert got["acquired"] and json.loads(p.read_text("utf-8"))["owner"] == "M1"
    st = ct.cpu_lock_status(p)
    assert st["held"] and st["owner"] == "M1" and st["stale"] is False
    with pytest.raises(ct.CpuLockBusy, match="M1"):
        ct.cpu_lock_acquire("other", "x", path=p)
    t0 = time.time()
    with pytest.raises(ct.CpuLockBusy):
        ct.cpu_lock_acquire("other", "x", wait_s=1.0, poll_s=0.2, path=p)
    assert 0.9 <= time.time() - t0 < 5
    assert ct.cpu_lock_release("other", path=p) is False and p.exists()  # never remove a foreign lock
    assert ct.cpu_lock_release("M1", path=p) is True and not p.exists()
    assert ct.cpu_lock_release("M1", path=p) is False
    # the format written by pc/x1a/cpu_lock.sh and by D1 (start vs start_utc) is understood
    p.write_text('{"owner": "D1", "purpose": "dense", "start_utc": "2026-10-05T20:38:07Z"}\n', encoding="utf-8")
    assert ct.cpu_lock_status(p)["start"] == "2026-10-05T20:38:07Z"
    old = time.time() - ct.CPU_LOCK_STALE_S - 60
    os.utime(p, (old, old))
    assert ct.cpu_lock_status(p)["stale"] is True
    with pytest.raises(ct.CpuLockBusy, match="STALE"):
        ct.cpu_lock_acquire("M1", "pilot", wait_s=30, path=p)  # reported, never removed automatically
    assert p.exists()


def test_series_order_randomized_complete_blocks() -> None:
    vs = ["A", "B", "C", "D", "ik-plain:A"]
    o = ct.series_order(vs, 4, seed=11)
    assert len(o) == 20 and [x["index"] for x in o] == list(range(20))
    for r in range(4):
        block = [x["variant"] for x in o if x["repeat"] == r]
        assert sorted(block) == sorted(vs)                     # every variant once per block
    assert o == ct.series_order(vs, 4, seed=11)                # reproducible
    assert [x["variant"] for x in o] != [x["variant"] for x in ct.series_order(vs, 4, seed=12)]
    firsts = {x["variant"] for x in ct.series_order(vs, 40, seed=1) if x["position_in_block"] == 0}
    assert len(firsts) == len(vs)                              # no variant owns the first slot
    with pytest.raises(ValueError):
        ct.series_order(["A", "A"], 2, seed=0)
    assert ct.parse_variant("C") == ("a7b94df", "C") and ct.parse_variant("ik-plain:A") == ("ik-plain", "A")
    with pytest.raises(ValueError):
        ct.parse_variant("ik-plain:X")


# ------------------------------------------------------------------ added 2026-10-05 23:05: runtime checks of the
# prompt template (/apply-template) and of KV-prefix reuse (/tokenize), against a fake llama-server
class _API(http.server.BaseHTTPRequestHandler):
    template_prompt = ""
    fail_template = False

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/apply-template" and not type(self).fail_template:
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            out = {"prompt": type(self).template_prompt}
        elif self.path == "/tokenize":
            # one "token" per whitespace-separated word, plus BOS when add_special
            out = {"tokens": ([2] if body.get("add_special") else []) + list(range(len(body["content"].split())))}
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a) -> None:
        pass


def _api_server() -> tuple[http.server.HTTPServer, int]:
    srv = http.server.HTTPServer(("127.0.0.1", 0), _API)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _bare_cell() -> "cc.Cell":
    c = cc.Cell.__new__(cc.Cell)
    c.cfg, c.eff = {"request": {}}, {}
    c.template_api, c.template_checks, c.reuse_checks = None, [], []
    return c


def test_template_check_match_mismatch_and_unavailable() -> None:
    ours = "<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    msgs = [{"role": "user", "content": "Hi"}]
    srv, port = _api_server()
    try:
        c = _bare_cell()
        _API.fail_template, _API.template_prompt = False, "<bos>" + ours     # the server prepends BOS
        r = c.template_check(port, "a#0", msgs, ours)
        assert r["available"] and r["match"] and r["server_had_bos"] and r["first_diff"] is None
        _API.template_prompt = ours.replace("<|channel>thought\n<channel|>", "")  # e.g. thinking not disabled
        r2 = c.template_check(port, "a#1", msgs, ours)
        assert r2["match"] is False and r2["first_diff"] == len(ours) - len("<|channel>thought\n<channel|>")
        assert "ours_at_diff" in r2 and len(c.template_checks) == 2 and c.template_api is True
        assert c.template_check(port, "x", None, ours) is None            # nothing to compare without messages
        _API.fail_template = True                                          # an engine without the endpoint
        c2 = _bare_cell()
        assert c2.template_check(port, "b#0", msgs, ours) == {"rid": "b#0", "available": False}
        assert c2.template_api is False and "template_check_unavailable" in c2.eff
        assert c2.template_check(port, "b#1", msgs, ours) is None         # not retried for every request
        c3 = _bare_cell()
        c3.cfg = {"request": {"template_check": False}}
        assert c3.template_check(port, "c#0", msgs, ours) is None
    finally:
        _API.fail_template = False
        srv.shutdown()


def test_expected_reuse_counts_the_common_prefix_tokens() -> None:
    prev = "<|turn>user\nfirst question here<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    new = "<|turn>user\nfirst question here<turn|>\n<|turn>model\nthe reply<turn|>\n<|turn>user\nnext<turn|>\n<|turn>model\n"
    srv, port = _api_server()
    try:
        r = _bare_cell().expected_reuse(port, prev, new)
    finally:
        srv.shutdown()
    cp = os.path.commonprefix([prev, new])
    assert cp.endswith("<|turn>model\n") and r["common_prefix_chars"] == len(cp)
    assert r["expected_reusable_prefix_tokens"] == 1 + len(cp.split())   # BOS + the words of the common prefix
    assert r["prev_prompt_chars"] == len(prev)
    dead = _bare_cell().expected_reuse(1, prev, new)                      # no server: recorded, not raised
    assert dead["expected_reusable_prefix_tokens"] is None and "error" in dead


# ------------------------------------------------------------------ added 2026-10-05 23:20: bugs found by the
# direct-storage smoke cell (process started inside a phase; a comma in a CSV value)
import csv as _csv  # noqa: E402


def test_io_row_process_started_inside_the_phase_and_read_source() -> None:
    # llama-server starts during `load`: no start snapshot, so its counters count from 0
    r = cc.io_row("load", "-", _snap(0.0),
                  _snap(5.0, proc={"server": {"read_bytes": 40960, "majflt": 7, "rchar": 1, "minflt": 2}}, disk={"sdc": 50000}))
    assert r["server_read_bytes"] == 40960 and r["server_majflt"] == 7 and r["pinner_read_bytes"] is None
    assert r["device_read_bytes"] == 40960 and r["device_read_source"].startswith("/proc/<pid>/io")
    # a process present at both ends: plain delta
    r1 = cc.io_row("x", "-", _snap(0.0, proc={"server": {"read_bytes": 100, "majflt": 1}}),
                   _snap(1.0, proc={"server": {"read_bytes": 350, "majflt": 4}, "pinner": {"read_bytes": 50}}))
    assert r1["server_read_bytes"] == 250 and r1["server_majflt"] == 3 and r1["device_read_bytes"] == 300
    # FUSE mode: the emulated device's own counter wins over every other source
    r2 = cc.io_row("x", "-", _snap(0.0, fuse={"backing_bytes": 0, "requests": 0, "queue_wait_ns": 0, "sleep_ns": 0}),
                   _snap(1.0, fuse={"backing_bytes": 777, "requests": 1, "queue_wait_ns": 0, "sleep_ns": 0},
                         proc={"server": {"read_bytes": 5}}, disk={"sdc": 999}))
    assert r2["device_read_bytes"] == 777 and r2["device_read_source"].startswith("fuse_backing_bytes")
    # cgroup v2 io.stat is preferred over process counters; the bare block device is the last resort
    r3 = cc.io_row("x", "-", _snap(0.0, cg_io={"179:0": {"rbytes": 100}}),
                   _snap(1.0, cg_io={"179:0": {"rbytes": 600}}, proc={"server": {"read_bytes": 999}}))
    assert r3["device_read_bytes"] == 500 and r3["device_read_source"] == "cgroup io.stat rbytes"
    r4 = cc.io_row("x", "-", _snap(0.0), _snap(1.0, disk={"sdc": 4096}))
    assert r4["device_read_bytes"] == 4096 and "diskstats" in r4["device_read_source"]
    assert all("," not in x["device_read_source"] for x in (r, r1, r2, r3, r4))


def test_io_csv_is_valid_csv_even_with_commas(tmp_path: Path) -> None:
    c = cc.Cell.__new__(cc.Cell)
    with open(tmp_path / "io.csv", "w", encoding="utf-8", newline="") as fh:
        c.io_fh = fh
        c.io_csv = _csv.writer(fh, lineterminator="\n")
        c.io_csv.writerow(cc.IO_FIELDS)
        c.io("workload_decode", "conv,with,commas#0", _snap(0.0), _snap(1.0, disk={"sdc": 4096}))
    rows = list(_csv.DictReader(open(tmp_path / "io.csv", encoding="utf-8")))
    assert len(rows) == 1 and None not in rows[0] and set(rows[0]) == set(cc.IO_FIELDS)
    assert rows[0]["rid"] == "conv,with,commas#0" and rows[0]["device_read_bytes"] == "4096"
    assert rows[0]["cg_io_rbytes"] == "" and rows[0]["device_read_source"].startswith("/proc/diskstats")


# ------------------------------------------------------------------ added 2026-10-06 00:20: the engines.json that
# the comparison stages really wrote (nested harness.extra_server_args, parent_engine_id, an alias of the reference)
REAL_ENGINES = REPO / "results" / "comparisons" / "engines.json"


def test_engines_reader_accepts_the_comparison_stages_conventions(tmp_path: Path) -> None:
    f = tmp_path / "engines.json"
    f.write_text(json.dumps({"schema": "pigemma-engines v1", "pending_engines": ["x"], "engines": [
        {"engine_id": "llama.cpp-a7b94df", "binary_path_in_container": ct.LLAMA_SERVER, "status": "runnable",
         "harness": {"extra_server_args": [], "methods": ["A", "B", "C", "D"]}},
        {"engine_id": "ik-prefetch-experts", "binary_path_in_container": "/opt/ik/llama-server", "status": "runnable",
         "parent_engine_id": "ik-plain",
         "harness": {"extra_server_args": ["--prefetch-experts"], "server_flag_overrides": {"no_repack": None},
                     "methods": ["A"]}},
        {"engine_id": "hotpin", "binary_path_in_container": "/opt/hp/llama-server", "status": "runnable",
         "parent": "hotpin-parent", "parent_engine_id": "hotpin-parent", "extra_args": [],
         "env": {"LLAMA_HOT_EXPERTS": "/work/hot.txt"}, "harness": {"extra_server_args": []}},
    ]}), encoding="utf-8")
    engs = ct.load_engines(f)
    assert engs["llama.cpp-a7b94df"]["alias_of"] == "a7b94df"
    ref = ct.resolve_engine("llama.cpp-a7b94df", f)
    assert ref["id"] == "a7b94df" and ref["bin"] == ct.LLAMA_SERVER and "alias" in ref["notes"]
    cfg = ct.default_config(method="C", budget_bytes=8 * GIB, workload_set="heldout", repeat=0, cell_id="c",
                            ranges="/work/x.ranges.txt", engine=ref)
    assert cfg["variant"] == "C" and cfg["engine"]["id"] == "a7b94df"      # a plain variant, not 'llama.cpp-a7b94df:C'
    ik = ct.resolve_engine("ik-prefetch-experts", f)
    assert ik["parent"] == "ik-plain" and ik["extra"] == ["--prefetch-experts"]
    assert ik["flag_overrides"] == {"no_repack": None} and ik["methods"] == ["A"]
    argv = cc.server_argv(ct.default_config(method="A", budget_bytes=GIB, workload_set="tuning", repeat=0,
                                            cell_id="i", engine=ik)["server"], "/m.gguf", 1)
    assert "-nr" not in argv and argv[-1] == "--prefetch-experts"
    assert ct.engine_allows(ik, "A") == (True, "") and ct.engine_allows(ik, "C")[0] is False
    assert ct.engine_allows(ref, "D")[0] is True            # the built-in reference has no restriction
    assert ct.main(["cell", "--method", "C", "--ranges", "x.ranges.txt", "--workload", "tuning", "--engine",
                    "ik-prefetch-experts", "--engines-file", str(f), "--dry-run"]) == ct.EXIT_REFUSED
    hp = ct.resolve_engine("hotpin", f)
    assert hp["parent"] == "hotpin-parent" and hp["env"] == {"LLAMA_HOT_EXPERTS": "/work/hot.txt"} and hp["extra"] == []
    hcfg = ct.default_config(method="A", budget_bytes=GIB, workload_set="tuning", repeat=0, cell_id="h", engine=hp)
    assert hcfg["variant"] == "hotpin:A" and hcfg["server"]["env"] == {"LLAMA_HOT_EXPERTS": "/work/hot.txt"}


@pytest.mark.skipif(not REAL_ENGINES.exists(), reason="results/comparisons/engines.json not written yet")
def test_real_engines_file_is_readable() -> None:
    engs = ct.load_engines(REAL_ENGINES)
    assert "a7b94df" in engs and len(engs) >= 2
    for eid, e in engs.items():
        assert e["bin"].startswith("/") and isinstance(e["extra"], list) and isinstance(e["flag_overrides"], dict)
        if e.get("parent"):
            assert e["parent"] in engs, f"{eid}: parent {e['parent']} is not a registered engine"
        ok, _why = ct.engine_runnable(e)
        if ok:  # every runnable profile must yield a valid command line
            cfg = ct.default_config(method="A", budget_bytes=GIB, workload_set="tuning", repeat=0, cell_id="t",
                                    engine=ct.resolve_engine(eid, REAL_ENGINES))
            assert cc.server_argv(cfg["server"], "/m.gguf", 1)[0] == cfg["server"]["bin"]
