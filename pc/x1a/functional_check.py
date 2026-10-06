#!/usr/bin/env python3
"""X1a functional check (runs INSIDE the harness image, stdlib only).

For every engine in --engines (JSON list of {engine_id, bin}) it starts llama-server with EXACTLY the
argv the controlled harness uses (pigemma.controlled_cell.server_argv on the harness's own server
config, -nr included), then for each selected TUNING conversation renders the first request with
pigemma.workloads.render_gemma4 and generates --n-predict greedy tokens:

  * run "a" and "b": POST /completion (non-streaming, n_probs=5, return_tokens) -> token ids, text,
    chosen-token logprobs, top-5 candidates (within-engine determinism = a vs b);
  * run "stream": the harness's own client (controlled_cell.stream_completion) -> token ids.

It is a FUNCTIONAL check: no memory budget, no FUSE throttle, no idle-desktop guarantee. The timings it
stores are labelled accordingly and must not be used as benchmark results.
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
from pigemma import controlled_cell as cc  # noqa: E402
from pigemma import workloads as wl  # noqa: E402


def meminfo() -> dict[str, int]:
    out = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        out[k] = int(v.split()[0]) * 1024
    return out


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def probs_of(resp: dict) -> list[dict]:
    out = []
    for e in resp.get("completion_probabilities") or []:
        top = e.get("top_logprobs") or e.get("top_probs") or []
        out.append({"id": e.get("id"), "token": e.get("token"),
                    "logprob": e.get("logprob"), "prob": e.get("prob"),
                    "top": [{"id": t.get("id"), "logprob": t.get("logprob"), "prob": t.get("prob")} for t in top]})
    return out


def run_engine(eng: dict, scfg_base: dict, model: str, prompts: list[dict], n_predict: int, out: Path) -> dict:
    eid = eng["engine_id"]
    edir = out / eid
    edir.mkdir(parents=True, exist_ok=True)
    rec: dict = {"engine_id": eid, "bin": eng["bin"], "ok": False}
    try:
        v = subprocess.run([eng["bin"], "--version"], capture_output=True, text=True, timeout=60)
        rec["version_output"] = (v.stdout + v.stderr).strip()
        rec["bin_sha256"] = sha256_file(eng["bin"])
        rec["bin_bytes"] = os.path.getsize(eng["bin"])
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"--version failed: {e!r}"
        return rec
    scfg = dict(scfg_base, bin=eng["bin"])
    port = cc.free_port()
    argv = cc.server_argv(scfg, model, port)
    rec["server_argv"] = argv
    logf = open(edir / "server.log", "wb")
    t0 = time.monotonic()
    srv = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT)
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
            if time.monotonic() - t0 > 1800:
                rec["error"] = "load timeout (1800 s)"
                return rec
            time.sleep(0.25)
        rec["server_ready_s"] = round(time.monotonic() - t0, 2)
        try:
            props = cc.http_json(port, "/props", timeout=10)
            gs = props.get("default_generation_settings", {})
            tmpl = props.get("chat_template") or ""
            rec["props"] = {"build_info": props.get("build_info"), "n_ctx": gs.get("n_ctx"),
                            "total_slots": props.get("total_slots"), "model_path": props.get("model_path"),
                            "chat_template_sha256": hashlib.sha256(tmpl.encode("utf-8")).hexdigest(),
                            "chat_template_len": len(tmpl)}
        except Exception as e:  # noqa: BLE001
            rec["props"] = {"error": repr(e)}
        rec["requests"] = {}
        for p in prompts:
            pid = p["id"]
            pr: dict = {}
            try:
                tk = cc.http_json(port, "/tokenize",
                                  {"content": p["prompt"], "add_special": True, "parse_special": True}, timeout=60)
                pr["prompt_token_ids"] = tk.get("tokens")
            except Exception as e:  # noqa: BLE001
                pr["tokenize_error"] = repr(e)
            body = {"prompt": p["prompt"], "n_predict": n_predict, "temperature": 0.0, "seed": 0,
                    "cache_prompt": False, "id_slot": 0}
            for tag in ("a", "b"):
                t1 = time.monotonic()
                r = cc.http_json(port, "/completion", dict(body, n_probs=5, return_tokens=True), timeout=1800)
                tm = r.get("timings") or {}
                pr[tag] = {"token_ids": r.get("tokens"), "text": r.get("content"),
                           "tokens_predicted": r.get("tokens_predicted"),
                           "tokens_evaluated": r.get("tokens_evaluated"),
                           "stop_type": r.get("stop_type"), "truncated": r.get("truncated"),
                           "probs": probs_of(r), "wall_s": round(time.monotonic() - t1, 3),
                           "timings_functional_only": {k: tm.get(k) for k in (
                               "prompt_n", "prompt_ms", "prompt_per_second",
                               "predicted_n", "predicted_ms", "predicted_per_second", "cache_n")}}
            s = cc.stream_completion(port, body, lambda: None, timeout=1800)
            fin = s.get("final") or {}
            pr["stream"] = {"token_ids": s["token_ids"], "text": s["text"], "n_events": len(s["events"]),
                            "final_keys": sorted(fin.keys()),
                            "timings_functional_only": {k: (fin.get("timings") or {}).get(k) for k in (
                                "prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second")}}
            rec["requests"][pid] = pr
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)
    finally:
        if srv.poll() is None:
            srv.send_signal(signal.SIGTERM)  # our own child only
            try:
                srv.wait(timeout=60)
            except subprocess.TimeoutExpired:
                srv.kill()
                srv.wait(timeout=30)
        logf.close()
        rec["server_rc"] = srv.returncode
        try:
            text = (edir / "server.log").read_text("utf-8", "replace")
            m = re.search(r"system_info:[^\n]*", text)
            rec["system_info"] = m.group(0).strip() if m else None
            parsed = cc.parse_server_log(text)
            bad = re.compile(r"(?i)\b(invalid argument|unknown argument|error:|deprecated)")
            rec["server_log"] = {"buffers_mib": parsed.get("buffers_mib"),
                                 "repack_present": parsed.get("repack_present"),
                                 "flash_attn": parsed.get("flash_attn"),
                                 "arg_or_error_lines": [ln[:300] for ln in text.splitlines() if bad.search(ln)][:20]}
        except Exception as e:  # noqa: BLE001
            rec["server_log"] = {"error": repr(e)}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", required=True)
    ap.add_argument("--server-cfg", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-sha256", default=None)
    ap.add_argument("--conversations", default="tun-01,tun-02")
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-avail-gib", type=float, default=16.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    engines = json.load(open(a.engines, encoding="utf-8"))
    scfg = json.load(open(a.server_cfg, encoding="utf-8"))
    mi = meminfo()
    res: dict = {"schema": "pigemma-x1a-functional-check v1", "started": cc.now_iso(),
                 "kind": "FUNCTIONAL CHECK - not a benchmark (no memory budget, no storage throttle, "
                         "desktop not verified idle)",
                 "model": a.model, "model_bytes": os.path.getsize(a.model), "n_predict": a.n_predict,
                 "vm_mem_available_bytes_start": mi.get("MemAvailable"), "nproc": os.cpu_count(),
                 "server_cfg": scfg, "engines": []}
    if mi.get("MemAvailable", 0) < a.min_avail_gib * 2**30:
        res["refused"] = f"MemAvailable {mi.get('MemAvailable')} < {a.min_avail_gib} GiB"
        cc.write_json(out / "functional_check.json", res)
        print("REFUSED:", res["refused"])
        return 4
    t = time.monotonic()
    res["model_sha256_measured"] = sha256_file(a.model)  # also warms the VM page cache for every engine alike
    res["model_sha256_seconds"] = round(time.monotonic() - t, 1)
    res["model_sha256_expected"] = a.model_sha256
    res["model_sha256_ok"] = (a.model_sha256 is None) or (res["model_sha256_measured"] == a.model_sha256)
    print(f"model sha256 {res['model_sha256_measured']} ok={res['model_sha256_ok']} "
          f"({res['model_sha256_seconds']} s)", flush=True)
    convs = {c["id"]: c for c in wl.load_set("tuning")}
    prompts = []
    for cid in a.conversations.split(","):
        c = convs[cid]
        text = wl.render_gemma4(wl.request_messages(c, 0, []), bos=False)
        prompts.append({"id": cid, "lang": c.get("lang"), "category": c.get("category"), "set": c.get("set"),
                        "prompt": text, "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    res["prompts"] = prompts
    for e in engines:
        print(f"== engine {e['engine_id']}: {e['bin']}", flush=True)
        r = run_engine(e, scfg, a.model, prompts, a.n_predict, out)
        print(f"   ok={r.get('ok')} ready_s={r.get('server_ready_s')} err={r.get('error')}", flush=True)
        res["engines"].append(r)
        cc.write_json(out / "functional_check.json", res)
    res["finished"] = cc.now_iso()
    cc.write_json(out / "functional_check.json", res)
    return 0 if all(e.get("ok") for e in res["engines"]) else 2


if __name__ == "__main__":
    sys.exit(main())
