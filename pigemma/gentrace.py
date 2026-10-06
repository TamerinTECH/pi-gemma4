"""pigemma.gentrace - real-generation expert-routing traces that replay the controlled harness's
requests (stage T1 of the decision workflow; brief 2026-10-05 sections 4 and 5).

Why this exists: the routing traces must describe exactly the requests the controlled comparison
measures (same GGUF chat template, same max_tokens, same context reuse), and tracing overhead must
be measured against a tracing-off run of the same requests. A ``session`` therefore runs, in ONE
Linux container without a memory limit, passes of two kinds over a workload set:

``server`` pass (tracing OFF)
    The harness's own ``llama-server`` binary with the argv of
    ``pigemma.controlled_cell.server_argv`` (identical flags to a controlled cell), the cell's fixed
    warm-up request, and the workload sent exactly like ``Cell.run_workload``: prompts rendered by
    ``pigemma.workloads.render_gemma4``, ``/completion`` with stream/return_tokens, temperature 0,
    seed 0, ``n_predict`` = the conversation's max_tokens, ``cache_prompt`` only for follow-ups.
    Per request it records the prompt token ids (``/tokenize``), the reused prefix
    (``timings.cache_n``), the prompt-batch boundaries the server used (``cached n_tokens = X``
    lines of server.log), the sampled token ids and client + server timings.

``probe`` pass (tracing ON)
    ``pigemma-probe`` 1.1.0 with ``--server-stop`` and one item per request holding the server
    pass's prompt ``tokens``, ``keep`` (= cache_n) and ``batches`` (= the server's batch sizes).
    The probe evaluates exactly the token set the server evaluated, in the same llama_process
    calls, generates greedily on its own and records the routing of every evaluated token
    (phase 0 = prefill, phase 1 = generated). Follow-up prompts contain the SERVER's replies;
    ``compare`` checks that the probe sampled the same token ids, so they are also its own.

``compare`` per request: prompt/keep/prefill counts, sampled token ids equal or first
divergence, and the timing ratios probe / server (prefill and decode) = tracing overhead.

Host side (Windows, repo root)::

    python -m pigemma.gentrace host-session --tag <tag> --sets calibration,tuning,heldout \
        --timing-set heldout --timing-repeats 3 --cpu-lock-owner <owner holding results/.cpu.lock>

writes ``traces/q3km/_sessions/<tag>/`` (session.json, per-pass directories, host_cpu.csv) and
copies the canonical traces to ``traces/q3km/<set>.moetrace`` (+ sidecars). In-container entry:
``python3 -m pigemma.gentrace run --config /work/traces/q3km/_sessions/<tag>/session.json``.

Stdlib only (the in-container half runs on the cell image's Python 3.12).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["parse_server_batches", "build_items", "compare_pass", "main"]

PROBE_NATIVE = "/work/tools/build-linux-x86-native/bin/pigemma-probe"
SESSIONS_REL = "traces/q3km/_sessions"
TRACES_REL = "traces/q3km"
SCHEMA = "pigemma-gentrace-session v1"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _write_json(p: Path, obj: Any) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def _sha256(p: str | os.PathLike[str]) -> str | None:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()
    except OSError:
        return None


# ----------------------------------------------------------------------------------------------
# server.log -> prompt batch boundaries
# ----------------------------------------------------------------------------------------------
_RE_NEW = re.compile(r"\| task (\d+) \| new prompt, n_ctx_slot = \d+, n_keep = \d+, task\.n_tokens = (\d+)")
_RE_CUT = re.compile(r"\| task (\d+) \| cached n_tokens = (\d+), memory_seq_rm \[(\d+), end\)")
_RE_STOP = re.compile(r"\| task (\d+) \| stop processing: n_tokens = (\d+)")
_RE_RESTORE = re.compile(r"\| task (\d+) \| restored context checkpoint \(pos_min = (-?\d+), pos_max = (-?\d+), "
                         r"n_tokens = (\d+), n_past = (\d+)")
_RE_FULL = re.compile(r"\| task (\d+) \| forcing full prompt re-processing")


def parse_server_batches(log_text: str) -> list[dict[str, Any]]:
    """Per completion task, in log order: the prompt size, the token index at which each prompt
    batch started ("cached n_tokens = X, memory_seq_rm [X, end)" is logged before every prompt
    batch), hence the reused prefix (first cut) and the batch sizes, plus the context size when
    the slot was released. Source: llama-server a7b94df, tools/server/server-context.cpp."""
    tasks: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for line in log_text.splitlines():
        m = _RE_NEW.search(line)
        if m:
            tid = int(m.group(1))
            tasks[tid] = {"task": tid, "n_tokens": int(m.group(2)), "cuts": [], "stop_n_tokens": None,
                          "checkpoint_restored": None, "full_reprocess": False}
            order.append(tid)
            continue
        m = _RE_CUT.search(line)
        if m and int(m.group(1)) in tasks:
            tasks[int(m.group(1))]["cuts"].append(int(m.group(2)))
            continue
        m = _RE_RESTORE.search(line)
        if m and int(m.group(1)) in tasks:
            tasks[int(m.group(1))]["checkpoint_restored"] = {
                "pos_min": int(m.group(2)), "pos_max": int(m.group(3)), "n_tokens": int(m.group(4)),
                "n_past": int(m.group(5))}
            continue
        m = _RE_FULL.search(line)
        if m and int(m.group(1)) in tasks:
            tasks[int(m.group(1))]["full_reprocess"] = True
            continue
        m = _RE_STOP.search(line)
        if m and int(m.group(1)) in tasks:
            tasks[int(m.group(1))]["stop_n_tokens"] = int(m.group(2))
    out = []
    for tid in order:
        t = tasks[tid]
        cuts = t["cuts"]
        if cuts:
            edges = cuts + [t["n_tokens"]]
            t["n_past"] = cuts[0]
            t["batches"] = [b - a for a, b in zip(edges, edges[1:])]
        else:
            t["n_past"] = None
            t["batches"] = []
        out.append(t)
    return out


# ----------------------------------------------------------------------------------------------
# in-container: CPU accounting
# ----------------------------------------------------------------------------------------------
def _cfs_stat() -> dict[str, int] | None:
    """CFS bandwidth counters of this container (--cpus quota): nr_periods, nr_throttled and the
    throttled time. cgroup v1 (cpu.stat of the cpu controller) or v2 (cpu.stat of the cgroup)."""
    for p in ("/sys/fs/cgroup/cpu/cpu.stat", "/sys/fs/cgroup/cpu,cpuacct/cpu.stat", "/sys/fs/cgroup/cpu.stat"):
        try:
            kv = {}
            with open(p, encoding="ascii") as fh:
                for line in fh:
                    f = line.split()
                    if len(f) == 2 and f[1].lstrip("-").isdigit():
                        kv[f[0]] = int(f[1])
            if "nr_periods" in kv:
                thr_ns = kv.get("throttled_time", kv.get("throttled_usec", 0) * 1000)
                return {"nr_periods": kv["nr_periods"], "nr_throttled": kv.get("nr_throttled", 0), "throttled_ns": thr_ns}
        except OSError:
            continue
    return None


def _cpu_snap(cg: Any) -> dict[str, Any]:
    from pigemma import controlled_cell as cc
    return {"mono": time.monotonic(), "epoch": time.time(), "cg_cpu_ns": cg.cpu_usage_ns(), "vm": cc.cpu_totals(),
            "cfs": _cfs_stat()}


def _cpu_delta(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    wall = b["mono"] - a["mono"]
    own = (b["cg_cpu_ns"] - a["cg_cpu_ns"]) / 1e9 if a.get("cg_cpu_ns") is not None and b.get("cg_cpu_ns") is not None else None
    busy = None
    if a.get("vm") and b.get("vm"):
        busy = (b["vm"][0] - a["vm"][0]) / float(os.sysconf("SC_CLK_TCK"))
    other = (busy - own) if busy is not None and own is not None else None
    cfs = None
    if a.get("cfs") and b.get("cfs"):
        cfs = {"periods": b["cfs"]["nr_periods"] - a["cfs"]["nr_periods"],
               "throttled_periods": b["cfs"]["nr_throttled"] - a["cfs"]["nr_throttled"],
               "throttled_s": round((b["cfs"]["throttled_ns"] - a["cfs"]["throttled_ns"]) / 1e9, 3)}
    return {"wall_s": round(wall, 3), "t_start_epoch": round(a["epoch"], 3), "cfs_quota": cfs,
            "own_cpu_s": round(own, 3) if own is not None else None,
            "vm_busy_cpu_s": round(busy, 3) if busy is not None else None,
            # CPU cores busy in the Docker VM OUTSIDE this container while the pass ran (background load)
            "vm_other_cores": round(max(0.0, other) / wall, 3) if other is not None and wall > 0 else None,
            "own_cores": round(own / wall, 3) if own is not None and wall > 0 else None}


# ----------------------------------------------------------------------------------------------
# in-container: server pass (tracing off)
# ----------------------------------------------------------------------------------------------
def _server_pass(sc: Mapping[str, Any], p: Mapping[str, Any], out: Path) -> dict[str, Any]:
    from pigemma import controlled_cell as cc
    from pigemma import workloads as wl
    out.mkdir(parents=True, exist_ok=True)
    cg = cc.Cgroup()
    port = cc.free_port()
    argv = cc.server_argv(sc["server"], sc["model"], port)
    res: dict[str, Any] = {"schema": SCHEMA, "kind": "server", "tag": p["tag"], "set": p["set"], "started": _now(),
                           "server_argv": argv, "requests": []}
    log_fh = open(out / "server.log", "wb")
    c_start = _cpu_snap(cg)
    proc = subprocess.Popen(argv, stdout=log_fh, stderr=subprocess.STDOUT)
    try:
        t0 = time.monotonic()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited during load ({proc.returncode}); see {out / 'server.log'}")
            try:
                if cc.http_json(port, "/health", timeout=5).get("status") == "ok":
                    break
            except (urllib.error.URLError, OSError, ValueError):
                pass
            if time.monotonic() - t0 > 3600:
                raise RuntimeError("llama-server load timeout")
            time.sleep(0.25)
        res["load_s"] = round(time.monotonic() - t0, 3)
        rq = sc.get("request") or {}

        def request(kind: str, rid: str, prompt: str, n_predict: int, cache_prompt: bool,
                    meta: Mapping[str, Any], messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            tk = cc.http_json(port, "/tokenize", {"content": prompt, "add_special": True, "parse_special": True},
                              timeout=60)
            ptoks = [int(x["id"] if isinstance(x, Mapping) else x) for x in tk["tokens"]]
            tmpl: dict[str, Any] = {"available": False}
            try:
                r = cc.http_json(port, "/apply-template", {"messages": list(messages),
                                                            "chat_template_kwargs": {"enable_thinking": False}}, timeout=30)
                srv = r.get("prompt") if isinstance(r, Mapping) else None
                if isinstance(srv, str):
                    cand = srv[5:] if srv.startswith("<bos>") and not prompt.startswith("<bos>") else srv
                    tmpl = {"available": True, "match": cand == prompt}
            except Exception as e:  # noqa: BLE001
                tmpl = {"available": False, "error": repr(e)[:200]}
            body = {"prompt": prompt, "n_predict": int(n_predict), "temperature": float(rq.get("temperature", 0.0)),
                    "seed": int(rq.get("seed", 0)), "cache_prompt": bool(cache_prompt), "id_slot": 0}
            c0 = _cpu_snap(cg)
            sres = cc.stream_completion(port, body, lambda: None)
            c1 = _cpu_snap(cg)
            fin = sres["final"]
            tim = fin.get("timings", {}) if fin else {}
            tt = sres["token_times"]
            dec_s = (tt[-1] - tt[0]) if len(tt) >= 2 else None
            rec = {"kind": kind, "rid": rid, **dict(meta), "prompt": prompt,
                   "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                   "prompt_tokens": ptoks, "n_prompt_tokens": len(ptoks), "n_predict": int(n_predict),
                   "cache_prompt": bool(cache_prompt), "template_check": tmpl,
                   "token_ids": sres["token_ids"], "text": sres["text"],
                   "token_times_s": [round(x, 6) for x in tt], "t_end_s": round(sres["t_end"], 6),
                   "stop_type": fin.get("stop_type") if fin else None,
                   "cache_n": tim.get("cache_n"), "prompt_n": tim.get("prompt_n"), "predicted_n": tim.get("predicted_n"),
                   "prompt_ms_server": tim.get("prompt_ms"), "predicted_ms_server": tim.get("predicted_ms"),
                   "prefill_tok_s_server": tim.get("prompt_per_second"),
                   "decode_tok_s_server": tim.get("predicted_per_second"),
                   "ttft_s": round(tt[0], 6) if tt else None,
                   # the harness's definitions (pigemma.controlled_cell.Cell.request)
                   "prefill_tok_s_client": round(tim["prompt_n"] / tt[0], 4) if tim.get("prompt_n") and tt else None,
                   "decode_tok_s_client": round((len(tt) - 1) / dec_s, 4) if dec_s and len(tt) >= 2 else None,
                   "decode_s_client": round(dec_s, 6) if dec_s is not None else None,
                   "cpu": _cpu_delta(c0, c1)}
            res["requests"].append(rec)
            print(f"[server {p['tag']}] {rid}: prompt {len(ptoks)} (cache_n {rec['cache_n']}, new {rec['prompt_n']}) "
                  f"out {len(sres['token_ids'])} {rec['stop_type']} ttft {rec['ttft_s']} s "
                  f"dec {rec['decode_tok_s_client']} tok/s", flush=True)
            return rec

        if p.get("with_warmup", True):
            wu = sc.get("warmup_request") or {}
            wmsgs = [{"role": "user", "content": wu.get("text", "Count from one to five.")}]
            request("warmup", "warmup", wl.render_gemma4(wmsgs, bos=False), int(wu.get("n_predict", 16)), False, {}, wmsgs)
        convs = wl.load_set(p["set"])
        if p.get("conversations"):
            by_id = {c["id"]: c for c in convs}
            convs = [by_id[x] for x in p["conversations"]]
        c_w0 = _cpu_snap(cg)
        for conv in convs:  # identical to pigemma.controlled_cell.Cell.run_workload
            replies: list[str] = []
            n_user = len(wl.user_turns(conv))
            for k in range(n_user):
                msgs = wl.request_messages(conv, k, replies)
                prompt = wl.render_gemma4(msgs, bos=False)
                follow = conv.get("context_reuse") == "follow_up_same_conversation" and k > 0
                rec = request("workload", f"{conv['id']}#{k}", prompt, int(conv["max_tokens"]), follow,
                              {"conv_id": conv["id"], "turn": k, "set": conv.get("set"), "lang": conv.get("lang"),
                               "category": conv.get("category"), "context_reuse": conv.get("context_reuse"),
                               "max_tokens": conv["max_tokens"]}, msgs)
                replies.append(rec["text"])
        res["workload_cpu"] = _cpu_delta(c_w0, _cpu_snap(cg))
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
        log_fh.close()
    res["pass_cpu"] = _cpu_delta(c_start, _cpu_snap(cg))
    # batch boundaries from the server's own log
    tasks = parse_server_batches((out / "server.log").read_text("utf-8", "replace"))
    problems: list[str] = []
    if len(tasks) != len(res["requests"]):
        problems.append(f"server.log has {len(tasks)} completion tasks, {len(res['requests'])} requests were sent")
    for rec, t in zip(res["requests"], tasks):
        rec["server_task"] = {k: t[k] for k in ("task", "n_tokens", "n_past", "batches", "stop_n_tokens",
                                                 "checkpoint_restored", "full_reprocess")}
        n = rec["n_prompt_tokens"]
        chk = {"prompt_tokens_eq_task_n_tokens": t["n_tokens"] == n,
               "log_n_past_eq_cache_n": t["n_past"] == rec["cache_n"],
               "batches_sum_eq_prompt_n": sum(t["batches"]) == rec["prompt_n"],
               "cache_n_plus_prompt_n_eq_tokens": (rec["cache_n"] or 0) + (rec["prompt_n"] or 0) == n,
               # the last sampled token (EOG or the n_predict-th) is not evaluated by the server
               "stop_n_tokens_eq_prompt_plus_sampled_minus_1":
                   t["stop_n_tokens"] == n + max(0, len(rec["token_ids"]) - 1)}
        rec["server_task"]["checks"] = chk
        bad = [k for k, v in chk.items() if not v]
        if bad:
            problems.append(f"{rec['rid']}: {bad}")
    res["problems"] = problems
    res["finished"] = _now()
    _write_json(out / "server_pass.json", res)
    return res


# ----------------------------------------------------------------------------------------------
# probe items from a server pass
# ----------------------------------------------------------------------------------------------
def build_items(server_pass: Mapping[str, Any], *, with_warmup: bool, only_warmup: bool = False) -> list[dict[str, Any]]:
    """One probe item per request: the server's prompt tokens, reused prefix and batch sizes."""
    items = []
    for rec in server_pass["requests"]:
        if rec["kind"] == "warmup" and not with_warmup:
            continue
        if only_warmup and rec["kind"] != "warmup":
            continue
        st = rec["server_task"]
        if not all(st["checks"].values()):
            raise RuntimeError(f"{rec['rid']}: the server pass is not self-consistent ({st['checks']}); no plan built")
        items.append({"id": rec["rid"], "tokens": rec["prompt_tokens"], "max_tokens": rec["n_predict"],
                      "keep": int(rec["cache_n"] or 0), "batches": list(st["batches"])})
    return items


# ----------------------------------------------------------------------------------------------
# in-container: probe pass (tracing on)
# ----------------------------------------------------------------------------------------------
def _probe_argv(sc: Mapping[str, Any], items_path: str, trace: str, requests: str, corpus_id: str,
                mode: str = "moe-trace") -> list[str]:
    s = sc["server"]
    rec = ["--trace-out", trace, "--corpus-id", corpus_id] if mode == "moe-trace" else []
    return [sc["probe_bin"], "--mode", mode, "-m", sc["model"], "--gen-prompts", items_path,
            "--server-stop", "--requests-out", requests, *rec,
            "-nr", "-t", str(s.get("threads", 4)), "-tb", str(s.get("threads", 4)), "-c", str(s.get("ctx", 4096)),
            "-b", str(s.get("batch", 2048)), "-ub", str(s.get("ubatch", 512)),
            "-ctk", str(s.get("ctk", "f16")), "-ctv", str(s.get("ctv", "f16")), "-fa", str(s.get("flash_attn", "auto"))]


def _probe_pass(sc: Mapping[str, Any], p: Mapping[str, Any], out: Path, server_pass: Mapping[str, Any]) -> dict[str, Any]:
    from pigemma import controlled_cell as cc
    out.mkdir(parents=True, exist_ok=True)
    cg = cc.Cgroup()
    items = build_items(server_pass, with_warmup=bool(p.get("with_warmup", False)),
                        only_warmup=bool(p.get("only_warmup", False)))
    if p.get("drop_batches"):
        # sensitivity control: the whole new prompt in ONE llama_process call (the probe's default
        # for plain --gen-prompts items) instead of the server's batch boundaries
        for it in items:
            it.pop("batches", None)
    if p.get("drop_keep"):
        # sensitivity control: no prefix reuse - every request re-evaluates its whole prompt
        for it in items:
            it["keep"] = 0
            it.pop("batches", None)
    items_path = out / "items.json"
    items_path.write_text(json.dumps(items), encoding="utf-8")
    name = p.get("trace_name") or p["set"]
    tmp = Path("/tmp") / f"pigemma-gentrace-{p['tag']}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    trace = tmp / f"{name}.moetrace"       # container-local: no bind-mount writes inside the timed loop
    reqs = tmp / f"{name}.moetrace.requests.jsonl"
    mode = str(p.get("mode") or "moe-trace")   # "run" = the same loop without the eval callback (tracing off)
    argv = _probe_argv(sc, str(items_path), str(trace), str(reqs), str(p.get("corpus_id") or name), mode)
    c0 = _cpu_snap(cg)
    t0 = time.monotonic()
    with open(out / "probe.log", "wb") as log_fh:
        rc = subprocess.run(argv, stdout=log_fh, stderr=subprocess.STDOUT, check=False).returncode
    wall = time.monotonic() - t0
    c1 = _cpu_snap(cg)
    log_text = (out / "probe.log").read_text("utf-8", "replace")
    summary = None
    for line in log_text.splitlines():
        if line.startswith("PIGEMMA_SUMMARY "):
            try:
                summary = json.loads(line[len("PIGEMMA_SUMMARY "):])
            except ValueError:
                pass
    copied = []
    for f in sorted(tmp.iterdir()):
        shutil.copy2(f, out / f.name)
        copied.append(f.name)
    shutil.rmtree(tmp, ignore_errors=True)
    recs = []
    rp = out / reqs.name
    if rp.is_file():
        recs = [json.loads(ln) for ln in rp.read_text("utf-8").splitlines() if ln.strip()]
    res = {"schema": SCHEMA, "kind": "probe", "mode": mode, "tag": p["tag"], "set": p["set"], "probe_argv": argv,
           "exit_code": rc,
           "wall_s": round(wall, 3), "summary": summary, "files": copied, "trace": str(out / trace.name),
           "n_items": len(items), "n_request_records": len(recs), "pass_cpu": _cpu_delta(c0, c1),
           "with_warmup": bool(p.get("with_warmup", False)), "server_pass_tag": server_pass["tag"], "finished": _now()}
    _write_json(out / "probe_pass.json", res)
    print(f"[probe {p['tag']}] exit {rc}, {len(recs)}/{len(items)} requests, wall {wall:.1f} s, "
          f"records {summary.get('n_records') if summary else None}", flush=True)
    if rc != 0:
        raise RuntimeError(f"pigemma-probe exited {rc}; see {out / 'probe.log'}")
    return res


# ----------------------------------------------------------------------------------------------
# compare a probe pass with a server pass
# ----------------------------------------------------------------------------------------------
def _median(xs: Sequence[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 6) if xs else None


def compare_pass(server_pass: Mapping[str, Any], probe_requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Request by request: same evaluated tokens? same sampled tokens? timing ratios probe/server.

    Timing definitions (both sides, per request):
      prefill_s  server: client time to the first token (TTFT, includes HTTP + tokenisation) and the
                 server's own prompt_ms; probe: wall time of the prefill llama_process calls
                 (incl. the callback and the trace writes).
      decode_s   time from the first to the last sampled token, over (n_sampled - 1) evaluations:
                 server: client timestamps (the harness's decode tok/s) and predicted_ms; probe:
                 its own sampling timestamps.
    """
    by_id = {r["id"]: r for r in probe_requests}
    rows = []
    n_same = n_cmp = 0
    for s in server_pass["requests"]:
        pr = by_id.get(s["rid"])
        if pr is None:
            continue
        n_cmp += 1
        st, pt = list(s["token_ids"]), list(pr["gen_tokens"])
        div = next((i for i, (a, b) in enumerate(zip(st, pt)) if a != b), None)
        if div is None and len(st) != len(pt):
            div = min(len(st), len(pt))
        same = div is None
        n_same += int(same)
        n_s = len(st)
        p_dec_s = (pr["sampled_ms"][-1] - pr["sampled_ms"][0]) / 1000.0 if len(pr["sampled_ms"]) >= 2 else None
        s_dec_s = s.get("decode_s_client")
        row = {
            "rid": s["rid"], "kind": s["kind"], "lang": s.get("lang"), "category": s.get("category"),
            "n_prompt_tokens": s["n_prompt_tokens"], "cache_n": s["cache_n"], "prompt_n_new": s["prompt_n"],
            "probe_n_keep": pr["n_keep"], "probe_n_prefill": pr["n_prefill"],
            "evaluated_prefill_equal": pr["n_keep"] == (s["cache_n"] or 0) and pr["n_prefill"] == s["prompt_n"],
            "server_n_sampled": n_s, "probe_n_sampled": len(pt), "server_stop": s["stop_type"], "probe_stop": pr["stop"],
            "tokens_identical": same, "first_divergence": div,
            "text_identical": s["text"] == pr["text"],
            "evaluated_gen_equal": pr["n_gen_evaluated"] == max(0, n_s - 1),
            "server_prefill_s_client": s.get("ttft_s"),
            "server_prefill_s_server": round(s["prompt_ms_server"] / 1000.0, 6) if s.get("prompt_ms_server") else None,
            "probe_prefill_s": round(pr["t_prefill_ms"] / 1000.0, 6),
            "server_decode_s_client": s_dec_s,
            "server_decode_tok_s_client": s.get("decode_tok_s_client"),
            "server_decode_tok_s_server": s.get("decode_tok_s_server"),
            "probe_decode_s": round(p_dec_s, 6) if p_dec_s is not None else None,
            "probe_decode_tok_s": round((len(pt) - 1) / p_dec_s, 4) if p_dec_s else None,
        }
        if same and s_dec_s and p_dec_s:
            row["decode_time_ratio_probe_over_server"] = round(p_dec_s / s_dec_s, 5)
        if row["evaluated_prefill_equal"] and s.get("ttft_s") and pr["t_prefill_ms"]:
            row["prefill_time_ratio_probe_over_server_client"] = round(pr["t_prefill_ms"] / 1000.0 / s["ttft_s"], 5)
            if s.get("prompt_ms_server"):
                row["prefill_time_ratio_probe_over_server_internal"] = round(pr["t_prefill_ms"] / s["prompt_ms_server"], 5)
        rows.append(row)
    wl_rows = [r for r in rows if r["kind"] == "workload"]
    ok_rows = [r for r in wl_rows if r["tokens_identical"] and r.get("decode_time_ratio_probe_over_server")]

    def tot(key: str, rs: Sequence[Mapping[str, Any]]) -> float:
        return sum(float(r[key]) for r in rs if r.get(key))
    dec_tok = sum(r["server_n_sampled"] - 1 for r in ok_rows)
    summary = {
        "n_requests_compared": n_cmp, "n_tokens_identical": n_same,
        "n_workload_requests": len(wl_rows),
        "workload_tokens_identical": sum(int(r["tokens_identical"]) for r in wl_rows),
        "workload_text_identical": sum(int(r["text_identical"]) for r in wl_rows),
        "workload_evaluated_prefill_equal": sum(int(r["evaluated_prefill_equal"]) for r in wl_rows),
        "workload_evaluated_gen_equal": sum(int(r["evaluated_gen_equal"]) for r in wl_rows),
        "divergent": [{"rid": r["rid"], "first_divergence": r["first_divergence"]} for r in rows if not r["tokens_identical"]],
        "decode": {
            "n_requests": len(ok_rows), "evaluations": dec_tok,
            "server_total_s_client": round(tot("server_decode_s_client", ok_rows), 4),
            "probe_total_s": round(tot("probe_decode_s", ok_rows), 4),
            "server_tok_s_pooled": round(dec_tok / tot("server_decode_s_client", ok_rows), 4) if ok_rows else None,
            "probe_tok_s_pooled": round(dec_tok / tot("probe_decode_s", ok_rows), 4) if ok_rows else None,
            "time_ratio_probe_over_server_pooled": (round(tot("probe_decode_s", ok_rows) / tot("server_decode_s_client", ok_rows), 5)
                                                     if ok_rows else None),
            "time_ratio_probe_over_server_median": _median([r["decode_time_ratio_probe_over_server"] for r in ok_rows]),
        },
        "prefill": {
            "n_requests": len([r for r in wl_rows if r.get("prefill_time_ratio_probe_over_server_client")]),
            "new_tokens": sum(r["prompt_n_new"] or 0 for r in wl_rows),
            "server_total_s_client_ttft": round(tot("server_prefill_s_client", wl_rows), 4),
            "server_total_s_internal": round(tot("server_prefill_s_server", wl_rows), 4),
            "probe_total_s": round(tot("probe_prefill_s", wl_rows), 4),
            "time_ratio_probe_over_server_client_pooled": (round(tot("probe_prefill_s", wl_rows) / tot("server_prefill_s_client", wl_rows), 5)
                                                           if wl_rows and tot("server_prefill_s_client", wl_rows) else None),
            "time_ratio_probe_over_server_internal_pooled": (round(tot("probe_prefill_s", wl_rows) / tot("server_prefill_s_server", wl_rows), 5)
                                                             if wl_rows and tot("server_prefill_s_server", wl_rows) else None),
            "time_ratio_probe_over_server_client_median": _median([r.get("prefill_time_ratio_probe_over_server_client") for r in wl_rows]),
            "time_ratio_probe_over_server_internal_median": _median([r.get("prefill_time_ratio_probe_over_server_internal") for r in wl_rows]),
        },
    }
    return {"schema": SCHEMA, "kind": "compare", "server_tag": server_pass["tag"], "summary": summary, "requests": rows}


def timing_rows_server(sp: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per workload request of a server pass: tokens, prefill and decode times (see compare_pass)."""
    out = {}
    for s in sp["requests"]:
        if s["kind"] != "workload":
            continue
        out[s["rid"]] = {"tokens": list(s["token_ids"]), "n_new": s["prompt_n"],
                         "prefill_s": (s["prompt_ms_server"] / 1000.0) if s.get("prompt_ms_server") else None,
                         "prefill_s_client": s.get("ttft_s"), "decode_s": s.get("decode_s_client"),
                         "decode_s_server_internal": (s["predicted_ms_server"] / 1000.0) if s.get("predicted_ms_server") else None,
                         "lang": s.get("lang"), "category": s.get("category")}
    return out


def timing_rows_probe(recs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for r in recs:
        if r["id"] == "warmup":
            continue
        ms = r["sampled_ms"]
        out[r["id"]] = {"tokens": list(r["gen_tokens"]), "n_new": r["n_prefill"], "prefill_s": r["t_prefill_ms"] / 1000.0,
                        "decode_s": (ms[-1] - ms[0]) / 1000.0 if len(ms) >= 2 else None}
    return out


def ratio_summary(a: Mapping[str, Mapping[str, Any]], b: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Time ratios a / b over the requests both ran with identical sampled tokens and equal new
    prefill counts. decode: first-to-last sampled token, (n - 1) evaluations; prefill: the prompt
    evaluation. pooled = sum of times; median = median of the per-request ratios."""
    ids = [k for k in a if k in b and a[k]["tokens"] == b[k]["tokens"] and a[k]["n_new"] == b[k]["n_new"]]
    dec = [(k, a[k]["decode_s"], b[k]["decode_s"], len(a[k]["tokens"]) - 1) for k in ids if a[k]["decode_s"] and b[k]["decode_s"]]
    pre = [(k, a[k]["prefill_s"], b[k]["prefill_s"], a[k]["n_new"]) for k in ids if a[k]["prefill_s"] and b[k]["prefill_s"]]

    def block(rows: Sequence[tuple[str, float, float, int]]) -> dict[str, Any]:
        if not rows:
            return {"n_requests": 0}
        ta, tb, n = sum(r[1] for r in rows), sum(r[2] for r in rows), sum(r[3] for r in rows)
        ratios = sorted(r[1] / r[2] for r in rows)
        return {"n_requests": len(rows), "evaluations": n, "a_total_s": round(ta, 4), "b_total_s": round(tb, 4),
                "a_tok_s_pooled": round(n / ta, 4), "b_tok_s_pooled": round(n / tb, 4),
                "time_ratio_pooled": round(ta / tb, 5), "time_ratio_median": round(statistics.median(ratios), 5),
                "time_ratio_min": round(ratios[0], 5), "time_ratio_max": round(ratios[-1], 5),
                "per_request_tok_s_median_a": round(statistics.median(r[3] / r[1] for r in rows), 4),
                "per_request_tok_s_median_b": round(statistics.median(r[3] / r[2] for r in rows), 4)}
    return {"n_common_requests": len([k for k in a if k in b]), "n_identical": len(ids),
            "decode": block(dec), "prefill": block(pre),
            "per_request": {k: {"decode_ratio": round(x / y, 5)} for k, x, y, _ in dec}}


# ----------------------------------------------------------------------------------------------
# in-container: the session
# ----------------------------------------------------------------------------------------------
def _run(config_path: str) -> int:
    from pigemma import controlled_cell as cc
    sc = json.loads(Path(config_path).read_text("utf-8"))
    root = Path(sc["out_dir"])
    cg = cc.Cgroup()
    mi = cc.meminfo()
    env = {"started": _now(), "meminfo_kB": {k: mi.get(k) for k in ("MemTotal", "MemAvailable", "Cached")},
           "nproc_online": os.cpu_count(), "cpus_allowed": sorted(os.sched_getaffinity(0)),
           "page_size": os.sysconf("SC_PAGE_SIZE"),
           "cgroup_version": cg.version, "cgroup_memory_limit": (cg.memory() or {}).get("max"),
           "server_bin_sha256": _sha256(sc["server"]["bin"]), "probe_bin_sha256": _sha256(sc["probe_bin"]),
           "probe_version": subprocess.run([sc["probe_bin"], "--version"], capture_output=True, text=True).stdout.strip()[-200:],
           "kernel": os.uname().release}
    state: dict[str, Any] = {"schema": SCHEMA, "config": config_path, "env": env, "passes": [], "status": "running"}
    _write_json(root / "run_state.json", state)
    server_passes: dict[str, dict[str, Any]] = {}
    probe_recs: dict[str, list[dict[str, Any]]] = {}
    rc = 0
    try:
        for p in sc["passes"]:
            out = root / p["tag"]
            entry: dict[str, Any] = {"tag": p["tag"], "kind": p["kind"], "set": p["set"], "started": _now()}
            if p["kind"] == "server":
                sp = _server_pass(sc, p, out)
                server_passes[p["tag"]] = sp
                entry.update({"load_s": sp.get("load_s"), "n_requests": len(sp["requests"]), "problems": sp["problems"],
                              "pass_cpu": sp["pass_cpu"], "workload_cpu": sp.get("workload_cpu")})
                if sp["problems"]:
                    raise RuntimeError(f"server pass {p['tag']} is not self-consistent: {sp['problems'][:5]}")
            elif p["kind"] == "probe":
                if p.get("server_pass_json"):
                    src = json.loads(Path(p["server_pass_json"]).read_text("utf-8"))
                else:
                    src = server_passes.get(p["server_pass"]) or json.loads((root / p["server_pass"] / "server_pass.json").read_text("utf-8"))
                pp = _probe_pass(sc, p, out, src)
                recs = [json.loads(ln) for ln in Path(pp["trace"] + ".requests.jsonl").read_text("utf-8").splitlines() if ln.strip()]
                probe_recs[p["tag"]] = recs
                # exactness: against the server pass the items were built from (same session, other process)
                cmp = compare_pass(src, recs)
                cmp["probe_tag"] = p["tag"]
                _write_json(out / "compare_plan_source.json", cmp)
                entry.update({"wall_s": pp["wall_s"], "summary": pp["summary"], "pass_cpu": pp["pass_cpu"],
                              "plan_source": src["tag"], "compare_plan_source": cmp["summary"]})
                print(f"[compare {p['tag']} vs {src['tag']}] sampled tokens identical "
                      f"{cmp['summary']['workload_tokens_identical']}/{cmp['summary']['n_workload_requests']} "
                      f"(all requests {cmp['summary']['n_tokens_identical']}/{cmp['summary']['n_requests_compared']})",
                      flush=True)
            else:
                raise ValueError(f"unknown pass kind {p['kind']!r}")
            entry["finished"] = _now()
            state["passes"].append(entry)
            _write_json(root / "run_state.json", state)
        # timing blocks: within each repeat, trace-on probe vs trace-off probe (same binary) and vs the server
        timing = []
        order = [q["tag"] for q in sc["passes"]]
        reps = sorted({int(p["timing_repeat"]) for p in sc["passes"] if p.get("timing_repeat")})
        for r in reps:
            blk = {p["timing_role"]: p for p in sc["passes"] if p.get("timing_repeat") == r}
            rows: dict[str, dict[str, dict[str, Any]]] = {}
            if "server" in blk and blk["server"]["tag"] in server_passes:
                rows["server"] = timing_rows_server(server_passes[blk["server"]["tag"]])
            for role in ("trace", "notrace"):
                if role in blk and blk[role]["tag"] in probe_recs:
                    rows[role] = timing_rows_probe(probe_recs[blk[role]["tag"]])
            ent: dict[str, Any] = {"repeat": r, "order": [p["timing_role"] for p in sc["passes"] if p.get("timing_repeat") == r],
                                   "tags": {k: v["tag"] for k, v in blk.items()}, "ratios": {}}
            for x, y in (("trace", "notrace"), ("trace", "server"), ("notrace", "server")):
                if x in rows and y in rows:
                    ent["ratios"][f"{x}_over_{y}"] = ratio_summary(rows[x], rows[y])
            timing.append(ent)
            for k, v in ent["ratios"].items():
                print(f"[timing r{r} {k}] decode time ratio pooled {v['decode'].get('time_ratio_pooled')} median "
                      f"{v['decode'].get('time_ratio_median')} | prefill pooled {v['prefill'].get('time_ratio_pooled')} "
                      f"| identical {v['n_identical']}/{v['n_common_requests']}", flush=True)
        state["timing"] = timing
        _write_json(root / "timing.json", {"schema": SCHEMA, "pass_order": order, "blocks": timing})
        state["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        state["status"] = "failed"
        state["failure"] = repr(e)[:2000]
        print(f"SESSION FAILED: {e!r}", file=sys.stderr, flush=True)
        rc = 2
    mi = cc.meminfo()
    state["env"]["meminfo_kB_end"] = {k: mi.get(k) for k in ("MemTotal", "MemAvailable", "Cached")}
    state["finished"] = _now()
    _write_json(root / "run_state.json", state)
    return rc


# ----------------------------------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------------------------------
def _plan_passes(sets: Sequence[str], timing_set: str | None, timing_repeats: int,
                 plan_source: str | None = None) -> list[dict[str, Any]]:
    """Canonical passes (one server + one probe pass per set, the probe WITHOUT the warm-up item so
    the trace holds the set's requests only), one warm-up-only trace, then the timing blocks on
    ``timing_set``: per repeat three fresh processes that each run the fixed warm-up request first -
    the harness's llama-server (tracing off), the probe with the recorder (moe-trace) and the probe
    without any eval callback (run) - in a rotating order (S T N | T N S | N S T)."""
    passes: list[dict[str, Any]] = []
    for s in sets:
        passes.append({"kind": "server", "set": s, "tag": f"{s}-server", "with_warmup": True})
        passes.append({"kind": "probe", "set": s, "tag": f"{s}-probe", "server_pass": f"{s}-server",
                       "with_warmup": False, "trace_name": s, "corpus_id": s, "canonical": True})
    if sets:
        s0 = sets[0]
        passes.append({"kind": "probe", "set": s0, "tag": "warmup-probe", "server_pass": f"{s0}-server",
                       "with_warmup": True, "only_warmup": True, "trace_name": "warmup", "corpus_id": "warmup"})
    if timing_set and timing_repeats > 0:
        for r in range(1, timing_repeats + 1):
            common = {"set": timing_set, "with_warmup": True, "timing_repeat": r}
            trio = [
                {"kind": "server", "tag": f"timing-r{r}-server", "timing_role": "server", **common},
                {"kind": "probe", "mode": "moe-trace", "tag": f"timing-r{r}-trace", "timing_role": "trace",
                 "server_pass": plan_source or f"{timing_set}-server", "trace_name": f"{timing_set}-timing-r{r}",
                 "corpus_id": timing_set, **common},
                {"kind": "probe", "mode": "run", "tag": f"timing-r{r}-notrace", "timing_role": "notrace",
                 "server_pass": plan_source or f"{timing_set}-server", "trace_name": f"{timing_set}-timing-r{r}-notrace",
                 **common},
            ]
            k = (r - 1) % 3          # rotate: S T N | T N S | N S T - every variant once in every position
            passes += trio[k:] + trio[:k]
    return passes


def _host_session(a: argparse.Namespace) -> int:
    from pigemma import controlled as ct
    repo = ct.REPO
    root = repo / SESSIONS_REL / a.tag
    if root.exists() and any(root.iterdir()) and not a.force:
        print(f"REFUSED: {root} exists (use another --tag or --force)", file=sys.stderr)
        return 4
    root.mkdir(parents=True, exist_ok=True)
    st = ct.cpu_lock_status()
    if not a.no_cpu_lock and (not st.get("held") or st.get("owner") != a.cpu_lock_owner):
        print(f"REFUSED: results/.cpu.lock is not held by {a.cpu_lock_owner!r} (status {st})", file=sys.stderr)
        return 4
    jobs = ct.other_model_jobs(own_name=f"pigemma-gentrace-{a.tag}".lower())
    if jobs and not a.allow_other_jobs:
        print(f"REFUSED: other model jobs are running: {jobs}", file=sys.stderr)
        return 4
    free = ct.disk_free()
    cfg = ct.default_config(method="A", budget_bytes=0, workload_set="heldout", repeat=0, cell_id=f"gentrace-{a.tag}")
    sets = [s for s in a.sets.split(",") if s]
    passes = (json.loads(Path(a.passes_json).read_text("utf-8")) if a.passes_json
              else _plan_passes(sets, a.timing_set or None, a.timing_repeats))
    if a.conversations:
        for p in passes:
            p["conversations"] = a.conversations.split(",")
    sc = {
        "schema": SCHEMA, "tag": a.tag, "created": _now(),
        "purpose": "T1: real-generation routing traces that replay the controlled harness's requests; tracing overhead",
        "memory_limit": None, "storage": "direct (the GGUF on the pigemma-models volume, no FUSE throttle)",
        "model": ct.MODEL_IN_VOLUME, "model_sha256": ct.MODEL_SHA256, "model_size": ct.MODEL_SIZE,
        "server": cfg["server"], "warmup_request": cfg["warmup_request"], "request": cfg["request"],
        "probe_bin": a.probe_bin, "out_dir": f"/work/{SESSIONS_REL}/{a.tag}", "passes": passes,
        "docker": {"image": cfg["docker"]["image"], "cpus": cfg["docker"]["cpus"],
                   "cpuset_cpus": cfg["docker"]["cpuset_cpus"], "volume": cfg["docker"]["volume"]},
        "host": {"disk_free_GB_before": free, "cpu_lock": st, "idle_probe_before": ct.idle_probe(10.0),
                 "other_model_jobs_at_start": jobs},
    }
    _write_json(root / "session.json", sc)
    name = f"pigemma-gentrace-{a.tag}".lower()
    d = sc["docker"]
    argv = ["docker", "run", "--rm", "--name", name, "--label", "pigemma.stage=T1-gentrace",
            f"--cpus={d['cpus']}", f"--cpuset-cpus={d['cpuset_cpus']}", "--network", "none",
            "-v", f"{d['volume']}:/models:ro", "-v", f"{str(repo).replace(chr(92), '/')}:/work",
            "-e", "PYTHONPATH=/work", "-e", "PYTHONUNBUFFERED=1", "-e", "PYTHONUTF8=1",
            "-e", "PYTHONPYCACHEPREFIX=/tmp/pycache", "-w", "/work",
            d["image"], "python3", "-m", "pigemma.gentrace", "run", "--config", f"{sc['out_dir']}/session.json"]
    sc["host"]["docker_argv"] = argv
    _write_json(root / "session.json", sc)
    if a.dry_run:
        print(json.dumps({"passes": [p["tag"] for p in passes], "docker": " ".join(argv)}, indent=1))
        return 0
    cpu = ct.HostCpu(root / "host_cpu.csv")
    cpu.start()
    t0 = time.time()
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    with open(root / "docker.log", "wb") as log_fh:
        rc = subprocess.run(argv, stdout=log_fh, stderr=subprocess.STDOUT, env=env, check=False).returncode
    cpu.stop_ev.set()
    cpu.join(timeout=5)
    sc["host"].update({"docker_exit_code": rc, "wall_s": round(time.time() - t0, 1),
                       "host_cpu_during_session": cpu.summary(), "host_cpu_t0_epoch": cpu.t0_epoch,
                       "disk_free_GB_after": ct.disk_free(), "idle_probe_after": ct.idle_probe(5.0)})
    _write_json(root / "session.json", sc)
    print(json.dumps({"tag": a.tag, "docker_exit_code": rc, "out": str(root),
                      "host_cpu": sc["host"]["host_cpu_during_session"]}))
    return 0 if rc == 0 else 2


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pigemma.gentrace", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="in-container: run the passes of a session.json")
    r.add_argument("--config", required=True)
    h = sub.add_parser("host-session", help="host: plan a session, run it in one container (no memory limit)")
    h.add_argument("--tag", required=True)
    h.add_argument("--sets", default="calibration,tuning,heldout")
    h.add_argument("--timing-set", default="heldout")
    h.add_argument("--timing-repeats", type=int, default=3)
    h.add_argument("--conversations", help="comma list: restrict every pass to these conversation ids (smoke tests)")
    h.add_argument("--probe-bin", default=PROBE_NATIVE)
    h.add_argument("--passes-json", help="a JSON list of passes to run instead of the standard plan")
    h.add_argument("--cpu-lock-owner", help="owner that already holds results/.cpu.lock (the stage)")
    h.add_argument("--no-cpu-lock", action="store_true", help="functional smoke only: do not require the CPU lock")
    h.add_argument("--allow-other-jobs", action="store_true")
    h.add_argument("--force", action="store_true")
    h.add_argument("--dry-run", action="store_true")
    c = sub.add_parser("compare", help="compare a probe pass (requests.jsonl) with a server pass (server_pass.json)")
    c.add_argument("--server", required=True)
    c.add_argument("--probe-requests", required=True)
    c.add_argument("--out")
    b = sub.add_parser("batches", help="print the prompt batch boundaries parsed from a llama-server log")
    b.add_argument("log")
    a = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    if a.cmd == "run":
        return _run(a.config)
    if a.cmd == "host-session":
        return _host_session(a)
    if a.cmd == "compare":
        sp = json.loads(Path(a.server).read_text("utf-8"))
        recs = [json.loads(ln) for ln in Path(a.probe_requests).read_text("utf-8").splitlines() if ln.strip()]
        res = compare_pass(sp, recs)
        if a.out:
            _write_json(Path(a.out), res)
        print(json.dumps(res["summary"], indent=1))
        return 0
    if a.cmd == "batches":
        print(json.dumps(parse_server_batches(Path(a.log).read_text("utf-8", "replace")), indent=1))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
