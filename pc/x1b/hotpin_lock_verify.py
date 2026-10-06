#!/usr/bin/env python3
"""X1b: which pages does HotPin lock, and are they the pages the kernels read? (runs INSIDE the container)

For every case in --cases (JSON list) it starts llama-server with the controlled harness's own argv
(pigemma.controlled_cell.server_argv; -nr unless the case says "nr": false) and the case's env
(LLAMA_HOT_EXPERTS=<hot file>), then records:

  * the tool's own log line ("expert-cache v1: pinned N hot expert-slices (X MiB)") = REQUESTED bytes;
  * failed lock requests = "failed to mlock" warnings in the server log;
  * ACTUAL locks from /proc/<pid>/smaps: every VMA with the "lo" VmFlag (mlock splits VMAs, so each
    locked range is its own VMA) -> unique locked pages, split into pages of the GGUF mapping
    (file-offset page runs) and anonymous memory; VmLck from /proc/<pid>/status;
  * PREDICTED locked page runs for the hot file (pc/x1b/hotpin_pinset.py lock model on the P4 expert
    map, the process page size) and the exact set difference actual vs predicted;
  * buffers from the load log (CPU_Mapped / CPU_REPACK) = repack evidence;
  * a short greedy generation on TUNING prompts (functional, not a benchmark) with
    /proc/<pid>/clear_refs before it: smaps "Referenced" afterwards shows which locked GGUF pages,
    unlocked GGUF pages and anonymous pages the compute kernels touched; token ids + top-5 logprobs
    are stored so engines can be compared.

It starts and stops only its own child process. Nothing of HotPin is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, "/work")
sys.path.insert(0, "/work/pc/x1b")
from pigemma import controlled as ctl  # noqa: E402
from pigemma import controlled_cell as cc  # noqa: E402
from pigemma import workloads as wl  # noqa: E402
import hotpin_pinset as hp  # noqa: E402

PAGE = os.sysconf("SC_PAGESIZE")


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_text(p: str) -> str:
    try:
        return open(p, encoding="utf-8", errors="replace").read()
    except OSError as e:
        return f"ERR {e!r}"


def status_fields(pid: int) -> dict[str, int]:
    out = {}
    for ln in read_text(f"/proc/{pid}/status").splitlines():
        m = re.match(r"(Vm\w+|Rss\w+):\s+(\d+) kB", ln)
        if m:
            out[m.group(1) + "_kB"] = int(m.group(2))
    return out


def stat_faults(pid: int) -> dict[str, int]:
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        return {"minflt": int(f[7]), "majflt": int(f[9])}
    except Exception:  # noqa: BLE001
        return {}


def parse_smaps(pid: int) -> list[dict]:
    vmas: list[dict] = []
    cur: dict | None = None
    head = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) (\S+) ([0-9a-f]+) (\S+) (\d+)\s*(.*)$")
    with open(f"/proc/{pid}/smaps", encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = head.match(ln)
            if m:
                cur = {"start": int(m.group(1), 16), "end": int(m.group(2), 16), "perms": m.group(3),
                       "offset": int(m.group(4), 16), "inode": int(m.group(6)), "path": m.group(7).strip(),
                       "flags": []}
                vmas.append(cur)
                continue
            if cur is None:
                continue
            k, _, v = ln.partition(":")
            if k == "VmFlags":
                cur["flags"] = v.split()
            elif k in ("Size", "Rss", "Pss", "Referenced", "Anonymous", "Locked", "Shared_Clean", "Private_Clean",
                       "Private_Dirty", "Swap"):
                cur[k] = int(v.split()[0])
    return vmas


def merge_runs(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(runs):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def runs_pages(runs: list[tuple[int, int]]) -> int:
    return sum(b - a for a, b in runs)


def runs_diff(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """pages in a but not in b (both merged, sorted)."""
    out = []
    j = 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            if b[k][0] > cur:
                out.append((cur, b[k][0]))
            cur = max(cur, b[k][1])
            k += 1
        if cur < e:
            out.append((cur, e))
    return out


def smaps_summary(vmas: list[dict], model_real: str) -> dict:
    def is_model(v: dict) -> bool:
        return v["path"] == model_real or v["path"].endswith(os.path.basename(model_real))
    model = [v for v in vmas if is_model(v)]
    locked = [v for v in vmas if "lo" in v["flags"]]
    lm = [v for v in locked if is_model(v)]
    la = [v for v in locked if not is_model(v)]
    um = [v for v in model if "lo" not in v["flags"]]
    anon = [v for v in vmas if not v["path"] or v["path"].startswith("[")]
    kb = lambda vs, k: sum(v.get(k, 0) for v in vs)  # noqa: E731
    file_runs = merge_runs([(v["offset"] // PAGE, v["offset"] // PAGE + (v["end"] - v["start"]) // PAGE) for v in lm])
    return {
        "n_vmas": len(vmas), "n_model_vmas": len(model),
        "model_mapping": {"size_kB": kb(model, "Size"), "rss_kB": kb(model, "Rss"), "referenced_kB": kb(model, "Referenced")},
        "locked_vmas": len(locked),
        "locked_model": {"n_vmas": len(lm), "size_kB": kb(lm, "Size"), "rss_kB": kb(lm, "Rss"),
                         "locked_field_kB": kb(lm, "Locked"), "referenced_kB": kb(lm, "Referenced"),
                         "unique_locked_pages": runs_pages(file_runs), "n_merged_file_runs": len(file_runs),
                         "vmas_with_any_reference": sum(1 for v in lm if v.get("Referenced", 0) > 0)},
        "locked_non_model": {"n_vmas": len(la), "size_kB": kb(la, "Size"), "rss_kB": kb(la, "Rss"),
                             "locked_field_kB": kb(la, "Locked"), "referenced_kB": kb(la, "Referenced"),
                             "anonymous_kB": kb(la, "Anonymous"),
                             "examples": [{"path": v["path"] or "(anonymous)", "size_kB": v.get("Size"),
                                           "anonymous_kB": v.get("Anonymous")} for v in la[:6]]},
        "unlocked_model": {"n_vmas": len(um), "size_kB": kb(um, "Size"), "rss_kB": kb(um, "Rss"),
                           "referenced_kB": kb(um, "Referenced")},
        "anonymous_all": {"n_vmas": len(anon), "size_kB": kb(anon, "Size"), "rss_kB": kb(anon, "Rss"),
                          "referenced_kB": kb(anon, "Referenced"),
                          "largest_kB": sorted((v.get("Size", 0) for v in anon), reverse=True)[:5]},
        "_file_runs": file_runs,
    }


def probs_of(resp: dict) -> list[dict]:
    out = []
    for e in resp.get("completion_probabilities") or []:
        top = e.get("top_logprobs") or e.get("top_probs") or []
        out.append({"id": e.get("id"), "logprob": e.get("logprob"),
                    "top": [{"id": t.get("id"), "logprob": t.get("logprob")} for t in top]})
    return out


def run_case(case: dict, scfg_base: dict, model: str, map_path: str, prompts: list[dict], n_predict: int,
             out: Path, skip_generation: bool = False) -> dict:
    cid = case["case_id"]
    cdir = out / cid
    cdir.mkdir(parents=True, exist_ok=True)
    rec: dict = {"case_id": cid, "bin": case["bin"], "nr": case.get("nr", True), "env": case.get("env") or {},
                 "note": case.get("note"), "ok": False, "page_size": PAGE}
    rec["bin_sha256"] = sha256_file(case["bin"])
    rec["limits_memlock"] = next((ln for ln in read_text("/proc/self/limits").splitlines() if "locked memory" in ln), None)
    rec["cap_eff"] = next((ln.split()[1] for ln in read_text("/proc/self/status").splitlines() if ln.startswith("CapEff")), None)
    scfg = dict(scfg_base, bin=case["bin"])
    if "warmup" in case:
        scfg["warmup"] = bool(case["warmup"])
    if not case.get("nr", True):
        scfg["flag_overrides"] = dict(scfg.get("flag_overrides") or {}, no_repack=[])
    port = cc.free_port()
    argv = cc.server_argv(scfg, model, port)
    rec["server_argv"] = argv
    hot_file = (case.get("env") or {}).get("LLAMA_HOT_EXPERTS")
    pred_runs: list[tuple[int, int]] | None = None
    if hot_file and os.path.exists(hot_file):
        hot = hp.read_hot(Path(hot_file))
        rec["hot_file"] = {"path": hot_file, "sha256": sha256_file(hot_file), "lines": len(hot)}
        rec["predicted"] = hp.predict(hot, Path(map_path), pages=(PAGE,))["by_page"][str(PAGE)]
        pred_runs = hp.locked_runs(hot, Path(map_path), PAGE)
        rec["predicted"]["n_merged_file_runs"] = len(pred_runs)
    logf = open(cdir / "server.log", "wb")
    t0 = time.monotonic()
    srv = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT,
                           env=dict(os.environ, **{str(k): str(v) for k, v in (case.get("env") or {}).items()}))
    try:
        while True:
            if srv.poll() is not None:
                rec["error"] = f"llama-server exited during load (rc={srv.returncode}); see server.log"
                return rec
            try:
                if cc.http_json(port, "/health", timeout=5).get("status") == "ok":
                    break
            except (urllib.error.URLError, OSError, ValueError):
                pass
            if time.monotonic() - t0 > 2400:
                rec["error"] = "load timeout (2400 s)"
                return rec
            time.sleep(0.25)
        rec["server_ready_s"] = round(time.monotonic() - t0, 2)
        pid = srv.pid
        model_real = os.path.realpath(model)
        rec["status_after_load"] = status_fields(pid)
        s0 = smaps_summary(parse_smaps(pid), model_real)
        act_runs = s0.pop("_file_runs")
        rec["smaps_after_load"] = s0
        if pred_runs is not None:
            missing = runs_diff(pred_runs, act_runs)     # predicted but not locked
            extra = runs_diff(act_runs, pred_runs)       # locked but not predicted
            rec["locked_vs_predicted"] = {
                "predicted_pages": runs_pages(pred_runs), "actual_locked_model_pages": runs_pages(act_runs),
                "predicted_not_locked_pages": runs_pages(missing), "locked_not_predicted_pages": runs_pages(extra),
                "exact_match": not missing and not extra,
                "first_missing_runs": missing[:5], "first_extra_runs": extra[:5]}
            cc.write_json(cdir / "locked_file_page_runs.json",
                          {"page": PAGE, "unit": "file-offset pages [first, end)", "actual": act_runs,
                           "predicted": pred_runs})
        if skip_generation:
            rec["generation"] = "skipped (--skip-generation)"
            rec["ok"] = True
            return rec
        # --- generation with reference tracking -------------------------------------------------
        try:
            open(f"/proc/{pid}/clear_refs", "w").write("1")
            rec["clear_refs"] = "ok"
        except OSError as e:
            rec["clear_refs"] = f"failed: {e!r}"
        io0, f0 = cc.proc_io(pid), stat_faults(pid)
        rec["requests"] = {}
        for p in prompts:
            body = {"prompt": p["prompt"], "n_predict": n_predict, "temperature": 0.0, "seed": 0,
                    "cache_prompt": False, "id_slot": 0, "n_probs": 5, "return_tokens": True}
            t1 = time.monotonic()
            r = cc.http_json(port, "/completion", body, timeout=3600)
            tm = r.get("timings") or {}
            rec["requests"][p["id"]] = {
                "token_ids": r.get("tokens"), "text": r.get("content"), "tokens_predicted": r.get("tokens_predicted"),
                "tokens_evaluated": r.get("tokens_evaluated"), "probs": probs_of(r),
                "wall_s": round(time.monotonic() - t1, 3),
                "timings_functional_only": {k: tm.get(k) for k in ("prompt_n", "prompt_per_second", "predicted_n",
                                                                   "predicted_per_second")}}
        io1, f1 = cc.proc_io(pid), stat_faults(pid)
        rec["generation_deltas"] = {"read_bytes": (io1.get("read_bytes", 0) - io0.get("read_bytes", 0)),
                                    "majflt": f1.get("majflt", 0) - f0.get("majflt", 0),
                                    "minflt": f1.get("minflt", 0) - f0.get("minflt", 0)}
        s1 = smaps_summary(parse_smaps(pid), model_real)
        s1.pop("_file_runs")
        rec["smaps_after_generation"] = s1
        rec["status_after_generation"] = status_fields(pid)
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)
    finally:
        if srv.poll() is None:
            srv.send_signal(signal.SIGTERM)  # our own child only
            try:
                srv.wait(timeout=120)
            except subprocess.TimeoutExpired:
                srv.kill()
                srv.wait(timeout=30)
        logf.close()
        rec["server_rc"] = srv.returncode
        text = (cdir / "server.log").read_text("utf-8", "replace")
        parsed = cc.parse_server_log(text)
        m = re.search(r"expert-cache v1: pinned (\d+) hot expert-slices \(([\d.]+) MiB\)", text)
        fails = re.findall(r"failed to mlock (\d+)-byte buffer \(after previously locking (\d+) bytes\): ([^\n]*)", text)
        rec["server_log"] = {
            "buffers_mib": parsed.get("buffers_mib"), "repack_present": parsed.get("repack_present"),
            "tool_pinned_line": m.group(0) if m else None,
            "tool_pinned_slices": int(m.group(1)) if m else None,
            "tool_pinned_mib_requested": float(m.group(2)) if m else None,
            "failed_mlock_warnings": len(fails),
            "failed_mlock_bytes": sum(int(a) for a, _, _ in fails),
            "failed_mlock_errors": sorted({c.strip() for _, _, c in fails}),
            "cannot_open_hot_file": "cannot open LLAMA_HOT_EXPERTS" in text,
            "system_info": (re.search(r"system_info:[^\n]*", text) or [None])[0],
            "arg_errors": [ln[:300] for ln in text.splitlines()
                           if re.search(r"(?i)\b(invalid argument|unknown argument|error:)", ln)][:10]}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--map", default="/work/results/maps/map26_ud_q3_k_m.json")
    ap.add_argument("--conversations", default="tun-01,tun-02")
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-generation", action="store_true",
                    help="load + lock accounting only (emulated aarch64: generation would take very long)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = json.load(open(a.cases, encoding="utf-8"))
    cfg = ctl.default_config(method="A", budget_bytes=8 * ctl.GIB, workload_set="tuning", repeat=0,
                             cell_id="x1b-lock-verify", purpose="smoke")
    scfg = cfg["server"]
    convs = {c["id"]: c for c in wl.load_set("tuning")}
    prompts = []
    for cid in a.conversations.split(","):
        c = convs[cid]
        text = wl.render_gemma4(wl.request_messages(c, 0, []), bos=False)
        prompts.append({"id": cid, "lang": c.get("lang"), "prompt": text,
                        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    res: dict = {"schema": "pigemma-x1b-hotpin-lock-verify v1", "started": cc.now_iso(),
                 "kind": "FUNCTIONAL / accounting check - not a benchmark (no memory budget, no storage throttle, "
                         "desktop not verified idle)",
                 "model": a.model, "model_bytes": os.path.getsize(a.model), "map": a.map, "page_size": PAGE,
                 "n_predict": a.n_predict, "prompts": prompts, "server_cfg": scfg, "nproc": os.cpu_count(),
                 "machine": os.uname().machine, "skip_generation": a.skip_generation,
                 "cases": []}
    for case in cases:
        print(f"== case {case['case_id']}: {case['bin']} nr={case.get('nr', True)} env={case.get('env')}", flush=True)
        r = run_case(case, scfg, a.model, a.map, prompts, a.n_predict, out, skip_generation=a.skip_generation)
        lv = r.get("locked_vs_predicted") or {}
        print(f"   ok={r.get('ok')} ready_s={r.get('server_ready_s')} err={r.get('error')} "
              f"log={(r.get('server_log') or {}).get('tool_pinned_line')} "
              f"fails={(r.get('server_log') or {}).get('failed_mlock_warnings')} "
              f"VmLck_kB={(r.get('status_after_load') or {}).get('VmLck_kB')} exact={lv.get('exact_match')}", flush=True)
        res["cases"].append(r)
        cc.write_json(out / "lock_verify.json", res)
    res["finished"] = cc.now_iso()
    cc.write_json(out / "lock_verify.json", res)
    return 0 if all(c.get("ok") for c in res["cases"]) else 2


if __name__ == "__main__":
    sys.exit(main())
