#!/usr/bin/env python3
"""X1c engine check (runs INSIDE the harness image or pi-gemma4-build:x1c-tools; stdlib only).

For every engine entry in --engines it starts llama-server with the argv the controlled harness would
build (pigemma.controlled_cell.server_argv on the harness's own server config + the entry's
flag_overrides / extra args / env, i.e. exactly what an engines.json profile produces), then for each
selected TUNING conversation renders the first request with pigemma.workloads.render_gemma4 and
generates --n-predict greedy tokens:

  mode "a"/"b"  POST /completion non-streaming, n_probs=5 -> token ids, text, logprobs, top-5
  mode "stream" the harness's own streaming client (controlled_cell.stream_completion)

Around the requests it records what the stage has to prove: where the weights live
(/proc/<pid>/smaps of the model mapping: Rss / Anonymous / Private_Dirty / Locked; anonymous Rss),
the server's threads (count, names, CPU ticks), the page-cache residency of the model file
(vmtouch) and the container's cgroup-v1 memory counters.

Entry keys: engine_id, bin, flag_overrides {}, extra [], env {}, wrapper [argv prefix, e.g. strace],
evict_before (bool: `vmtouch -e` the model file before the server starts; file-local, no drop_caches).

FUNCTIONAL / BEHAVIOURAL check only: no storage throttle, desktop not verified idle. Timings stored
here are labelled and must not be used as benchmark results.
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

CG = "/sys/fs/cgroup/memory"


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


def cgroup_mem() -> dict:
    out: dict = {}
    for k in ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes", "memory.failcnt",
              "memory.memsw.limit_in_bytes", "memory.memsw.usage_in_bytes", "memory.swappiness"):
        v = cc.read_text(f"{CG}/{k}")
        out[k] = int(v.strip()) if v and v.strip().lstrip("-").isdigit() else None
    st = cc.read_text(f"{CG}/memory.stat") or ""
    keep = ("cache", "rss", "mapped_file", "unevictable", "pgfault", "pgmajfault", "inactive_file", "active_file",
            "inactive_anon", "active_anon", "pgpgin", "pgpgout", "swap")
    out["stat"] = {k: int(v) for k, v in (ln.split() for ln in st.splitlines() if len(ln.split()) == 2) if k in keep}
    oc = cc.read_text(f"{CG}/memory.oom_control") or ""
    out["oom_control"] = dict(ln.split() for ln in oc.splitlines() if len(ln.split()) == 2)
    return out


def vmtouch(path: str) -> dict:
    try:
        r = subprocess.run(["vmtouch", "-f", path], capture_output=True, text=True, timeout=600)
        m = re.search(r"Resident Pages:\s*(\d+)/(\d+)", r.stdout)
        if m:
            return {"resident_pages": int(m.group(1)), "total_pages": int(m.group(2)),
                    "resident_fraction": round(int(m.group(1)) / max(1, int(m.group(2))), 5)}
        return {"error": (r.stdout + r.stderr)[-300:]}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def proc_snapshot(pid: int, model_path: str) -> dict:
    snap: dict = {"t": time.time()}
    st = cc.read_text(f"/proc/{pid}/status") or ""
    keys = ("VmPeak", "VmSize", "VmLck", "VmPin", "VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem", "VmData",
            "VmSwap", "Threads")
    snap["status"] = {}
    for ln in st.splitlines():
        k, _, v = ln.partition(":")
        if k in keys:
            snap["status"][k] = int(v.split()[0]) * (1024 if "kB" in v else 1)
    # smaps: aggregate the model-file mapping(s) and the anonymous memory
    sm = cc.read_text(f"/proc/{pid}/smaps") or ""
    model_real = os.path.realpath(model_path)
    cur = None
    maps = []
    for ln in sm.splitlines():
        m = re.match(r"^([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+([0-9a-f]+)\s+\S+\s+\d+\s*(.*)$", ln)
        if m:
            cur = {"start": int(m.group(1), 16), "end": int(m.group(2), 16), "perms": m.group(3),
                   "offset": int(m.group(4), 16), "path": m.group(5).strip(), "f": {}}
            maps.append(cur)
            continue
        if cur is None:
            continue
        k, _, v = ln.partition(":")
        if k == "VmFlags":
            cur["vmflags"] = v.strip()
        elif v.strip().endswith("kB"):
            cur["f"][k] = int(v.split()[0]) * 1024
    fields = ("Size", "Rss", "Pss", "Shared_Clean", "Shared_Dirty", "Private_Clean", "Private_Dirty", "Anonymous",
              "Swap", "Locked", "Referenced")
    mm = [x for x in maps if x["path"] and os.path.realpath(x["path"]) == model_real]
    snap["model_mappings"] = [{"start": hex(x["start"]), "end": hex(x["end"]), "perms": x["perms"],
                               "file_offset": x["offset"], "vmflags": x.get("vmflags"),
                               **{k: x["f"].get(k, 0) for k in fields}} for x in mm]
    snap["model_mapping_total"] = {k: sum(x["f"].get(k, 0) for x in mm) for k in fields}
    anon = [x for x in maps if not x["path"] or x["path"].startswith("[")]
    snap["anon_total"] = {k: sum(x["f"].get(k, 0) for x in anon) for k in ("Size", "Rss", "Anonymous", "Locked")}
    big = sorted(anon, key=lambda x: -x["f"].get("Rss", 0))[:8]
    snap["anon_largest_by_rss"] = [{"path": x["path"] or "(anon)", "perms": x["perms"], "Size": x["f"].get("Size", 0),
                                    "Rss": x["f"].get("Rss", 0)} for x in big]
    other = [x for x in maps if x["path"] and not x["path"].startswith("[") and x not in mm]
    snap["other_file_rss"] = sum(x["f"].get("Rss", 0) for x in other)
    # threads
    th = []
    try:
        for tid in sorted(os.listdir(f"/proc/{pid}/task"), key=int):
            comm = (cc.read_text(f"/proc/{pid}/task/{tid}/comm") or "").strip()
            stat = cc.read_text(f"/proc/{pid}/task/{tid}/stat") or ""
            rest = stat.rsplit(")", 1)[-1].split()
            th.append({"tid": int(tid), "comm": comm, "state": rest[0] if rest else None,
                       "utime_ticks": int(rest[11]) if len(rest) > 12 else None,
                       "stime_ticks": int(rest[12]) if len(rest) > 12 else None})
    except OSError as e:
        snap["threads_error"] = repr(e)
    snap["threads"] = th
    snap["n_threads"] = len(th)
    snap["model_file_residency"] = vmtouch(model_path)
    snap["cgroup"] = cgroup_mem()
    return snap


def probs_of(resp: dict) -> list[dict]:
    """Per generated token: id, piece, logprob/prob and the top-n candidates.

    Two server formats exist (v2 of this script, 2026-10-06 03:45):
      llama.cpp      {"id", "token", "logprob", "top_logprobs": [{"id", "token", "logprob"}]}
      ik_llama.cpp   {"content": piece, "probs": [{"tok_str", "prob"}]}   (no ids, probabilities)
    v1 only knew the first one and stored id = None / top = [] for ik_llama.cpp."""
    out = []
    for e in resp.get("completion_probabilities") or []:
        if "probs" in e and "id" not in e:
            out.append({"id": None, "token": e.get("content"), "logprob": None, "prob": None, "format": "ik-legacy",
                        "top": [{"id": None, "token": t.get("tok_str"), "logprob": None, "prob": t.get("prob")}
                                for t in e.get("probs") or []]})
            continue
        top = e.get("top_logprobs") or e.get("top_probs") or []
        out.append({"id": e.get("id"), "token": e.get("token"), "logprob": e.get("logprob"), "prob": e.get("prob"),
                    "top": [{"id": t.get("id"), "token": t.get("token"), "logprob": t.get("logprob"), "prob": t.get("prob")}
                            for t in top]})
    return out


def retokenize(port: int, text: str | None) -> list | None:
    """ids of the generated text via the server's own /tokenize (same procedure for every engine)."""
    if not text:
        return None
    try:
        tk = cc.http_json(port, "/tokenize", {"content": text, "add_special": False, "parse_special": True}, timeout=60)
        return tk.get("tokens")
    except Exception:  # noqa: BLE001
        return None


def find_server_pid(parent: int, name: str = "llama-server") -> int | None:
    """The server itself, or (when wrapped by strace) the wrapper's child."""
    comm = (cc.read_text(f"/proc/{parent}/comm") or "").strip()
    if comm.startswith(name[:15]):
        return parent
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        st = cc.read_text(f"/proc/{p}/stat") or ""
        rest = st.rsplit(")", 1)[-1].split()
        if len(rest) > 2 and int(rest[1]) == parent and (cc.read_text(f"/proc/{p}/comm") or "").strip().startswith(name[:15]):
            return int(p)
    return None


def run_engine(eng: dict, scfg_base: dict, model: str, prompts: list[dict], n_predict: int, out: Path,
               modes: list[str]) -> dict:
    eid = eng["engine_id"]
    edir = out / eid
    edir.mkdir(parents=True, exist_ok=True)
    rec: dict = {"engine_id": eid, "bin": eng["bin"], "ok": False, "notes": eng.get("notes")}
    try:
        v = subprocess.run([eng["bin"], "--version"], capture_output=True, text=True, timeout=60)
        rec["version_output"] = (v.stdout + v.stderr).strip()
        rec["bin_sha256"] = sha256_file(eng["bin"])
        rec["bin_bytes"] = os.path.getsize(eng["bin"])
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"--version failed: {e!r}"
        return rec
    scfg = dict(scfg_base, bin=eng["bin"], flag_overrides=dict(eng.get("flag_overrides") or {}),
                extra=[str(x) for x in (eng.get("extra") or [])])
    port = cc.free_port()
    argv = cc.server_argv(scfg, model, port)
    wrapper = [str(w).replace("{edir}", str(edir)) for w in (eng.get("wrapper") or [])]
    env = dict(os.environ)
    for k, val in (eng.get("env") or {}).items():
        env[k] = str(val).replace("{edir}", str(edir))
    rec["server_argv"] = argv
    rec["wrapper"] = wrapper
    rec["env_added"] = {k: env[k] for k in (eng.get("env") or {})}
    if eng.get("evict_before"):
        r = subprocess.run(["vmtouch", "-e", model], capture_output=True, text=True, timeout=600)
        rec["evict_before"] = {"cmd": "vmtouch -e <model> (posix_fadvise DONTNEED on this one file)",
                               "rc": r.returncode, "residency_after_evict": vmtouch(model)}
    rec["cgroup_before_start"] = cgroup_mem()
    rec["model_file_residency_before_start"] = vmtouch(model)
    logf = open(edir / "server.log", "wb")
    t0 = time.monotonic()
    rec["t_start_epoch"] = time.time()
    srv = subprocess.Popen(wrapper + argv, stdout=logf, stderr=subprocess.STDOUT, env=env)
    pid = None
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
        rec["t_ready_epoch"] = time.time()
        pid = find_server_pid(srv.pid)
        rec["server_pid"] = pid
        rec["snapshots"] = {}
        if pid:
            rec["snapshots"]["after_load"] = proc_snapshot(pid, model)
            rec["proc_maps_model_lines"] = [ln for ln in (cc.read_text(f"/proc/{pid}/maps") or "").splitlines()
                                            if os.path.basename(model) in ln]
        try:
            props = cc.http_json(port, "/props", timeout=10)
            gs = props.get("default_generation_settings", {})
            tmpl = props.get("chat_template") or ""
            rec["props"] = {"build_info": props.get("build_info"), "n_ctx": gs.get("n_ctx"),
                            "total_slots": props.get("total_slots"), "model_path": props.get("model_path"),
                            "chat_template_sha256": hashlib.sha256(tmpl.encode("utf-8")).hexdigest(),
                            "chat_template_len": len(tmpl), "keys": sorted(props.keys())}
        except Exception as e:  # noqa: BLE001
            rec["props"] = {"error": repr(e)}
        rec["requests"] = {}
        for p in prompts:
            pid_ = p["id"]
            pr: dict = {}
            try:
                tk = cc.http_json(port, "/tokenize",
                                  {"content": p["prompt"], "add_special": True, "parse_special": True}, timeout=60)
                pr["prompt_token_ids"] = tk.get("tokens")
            except Exception as e:  # noqa: BLE001
                pr["tokenize_error"] = repr(e)
            body = {"prompt": p["prompt"], "n_predict": n_predict, "temperature": 0.0, "seed": 0,
                    "cache_prompt": False, "id_slot": 0}
            for tag in [m for m in modes if m in ("a", "b")]:
                t1 = time.monotonic()
                te = time.time()
                r = cc.http_json(port, "/completion", dict(body, n_probs=5, return_tokens=True), timeout=3600)
                tm = r.get("timings") or {}
                pb = probs_of(r)
                srv_ids = r.get("tokens") or [x.get("id") for x in pb]
                if not srv_ids or any(i is None for i in srv_ids):
                    srv_ids = None  # ik_llama.cpp: /completion returns pieces, not ids
                pr[tag] = {"token_ids": srv_ids,
                           "token_ids_source": ("tokens" if r.get("tokens") else "completion_probabilities[].id") if srv_ids
                                               else "none (server returns no token ids)",
                           "retokenized_ids": retokenize(port, r.get("content")),
                           "probs_raw_sample": (r.get("completion_probabilities") or [])[:1],
                           "text": r.get("content"),
                           "tokens_predicted": r.get("tokens_predicted"),
                           "tokens_evaluated": r.get("tokens_evaluated"),
                           "stop_type": r.get("stop_type"), "truncated": r.get("truncated"),
                           "probs": pb, "wall_s": round(time.monotonic() - t1, 3),
                           "t_request_start_epoch": te, "t_request_end_epoch": time.time(),
                           "timings_functional_only": {k: tm.get(k) for k in (
                               "prompt_n", "prompt_ms", "prompt_per_second",
                               "predicted_n", "predicted_ms", "predicted_per_second", "cache_n")}}
            if "stream" in modes:
                te = time.time()
                t1 = time.monotonic()
                s = cc.stream_completion(port, body, lambda: None, timeout=3600)
                fin = s.get("final") or {}
                pr["stream"] = {"token_ids": s.get("token_ids"), "text": s.get("text"),
                                "n_events": len(s.get("events") or []),
                                "timestamps_per": s.get("timestamps_per"),
                                "t_request_start_epoch": te, "t_request_end_epoch": time.time(),
                                "t_monotonic_wall_s": round(time.monotonic() - t1, 3),
                                "token_times_rel_s": s.get("token_times"),
                                "final_keys": sorted(fin.keys()),
                                "timings_functional_only": {k: (fin.get("timings") or {}).get(k) for k in (
                                    "prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms",
                                    "predicted_per_second")}}
            if pid:
                rec["snapshots"][f"after_{pid_}"] = proc_snapshot(pid, model)
            rec["requests"][pid_] = pr
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)
    finally:
        target = pid if (pid and pid != srv.pid) else None
        try:
            if target:
                os.kill(target, signal.SIGTERM)  # our own grandchild (the server under the wrapper)
            elif srv.poll() is None:
                srv.send_signal(signal.SIGTERM)  # our own child only
        except OSError:
            pass
        try:
            srv.wait(timeout=120)
        except subprocess.TimeoutExpired:
            srv.kill()
            srv.wait(timeout=30)
        logf.close()
        rec["server_rc"] = srv.returncode
        rec["t_stop_epoch"] = time.time()
        rec["cgroup_after_stop"] = cgroup_mem()
        try:
            text = (edir / "server.log").read_text("utf-8", "replace")
            m = re.search(r"system_info:[^\n]*", text) or re.search(r'system_info="[^"\n]*"', text)
            rec["system_info"] = m.group(0).strip() if m else None
            m = re.search(r"^stop: jobs=[^\n]*", text, re.M)
            rec["prefetch_exit_counters"] = m.group(0).strip() if m else None
            parsed = cc.parse_server_log(text)
            bad = re.compile(r"(?i)\b(invalid argument|unknown argument|error:|deprecated|not supported)")
            want = re.compile(r"(?i)(buffer size|mmap|mlock|repack|prefetch|ser\s+=|swa|flash_attn|fused|"
                              r"KV self size|graph_reuse|merge|n_ctx\s+=|warm|hadamard|thresh)")
            rec["server_log"] = {"buffers_mib": parsed.get("buffers_mib"),
                                 "repack_present": parsed.get("repack_present"),
                                 "flash_attn": parsed.get("flash_attn_effective"),
                                 "arg_or_error_lines": [ln[:300] for ln in text.splitlines() if bad.search(ln)][:20],
                                 "selected_lines": [ln[:260] for ln in text.splitlines() if want.search(ln)][:120]}
        except Exception as e:  # noqa: BLE001
            rec["server_log"] = {"error": repr(e)}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", required=True)
    ap.add_argument("--server-cfg", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-sha256", default=None)
    ap.add_argument("--skip-sha256", action="store_true")
    ap.add_argument("--conversations", default="tun-01,tun-02")
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--modes", default="a,b,stream")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-avail-gib", type=float, default=16.0)
    ap.add_argument("--kind", default="FUNCTIONAL CHECK - not a benchmark (no memory budget, no storage throttle, "
                                      "desktop not verified idle)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    engines = json.load(open(a.engines, encoding="utf-8"))
    scfg = json.load(open(a.server_cfg, encoding="utf-8"))
    mi = meminfo()
    res: dict = {"schema": "pigemma-x1c-ik-check v1", "started": cc.now_iso(), "kind": a.kind,
                 "model": a.model, "model_bytes": os.path.getsize(a.model), "n_predict": a.n_predict,
                 "modes": a.modes.split(","), "page_size": os.sysconf("SC_PAGESIZE"),
                 "kernel": os.uname().release, "nproc": os.cpu_count(),
                 "sched_affinity": sorted(os.sched_getaffinity(0)),
                 "vm_mem_available_bytes_start": mi.get("MemAvailable"), "vm_mem_total_bytes": mi.get("MemTotal"),
                 "cgroup_at_start": cgroup_mem(), "server_cfg": scfg, "engines": []}
    if mi.get("MemAvailable", 0) < a.min_avail_gib * 2**30:
        res["refused"] = f"MemAvailable {mi.get('MemAvailable')} < {a.min_avail_gib} GiB"
        cc.write_json(out / "check.json", res)
        print("REFUSED:", res["refused"])
        return 4
    if not a.skip_sha256:
        t = time.monotonic()
        res["model_sha256_measured"] = sha256_file(a.model)  # also warms the page cache for every engine alike
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
        r = run_engine(e, scfg, a.model, prompts, a.n_predict, out, a.modes.split(","))
        print(f"   ok={r.get('ok')} ready_s={r.get('server_ready_s')} err={r.get('error')}", flush=True)
        res["engines"].append(r)
        cc.write_json(out / "check.json", res)
    res["finished"] = cc.now_iso()
    cc.write_json(out / "check.json", res)
    return 0 if all(e.get("ok") for e in res["engines"]) else 2


if __name__ == "__main__":
    sys.exit(main())
