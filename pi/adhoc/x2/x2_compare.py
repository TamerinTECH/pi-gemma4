#!/usr/bin/env python3
"""X2 (comparisons workflow, 2026-10-05): same-model runtime comparison driver for the real Pi (stdlib only).

Runs a PLAN = ordered list of BLOCKS. One block = one llama-server process of one engine build
(our llama.cpp a7b94df, or ik_llama.cpp) with one fixed argument list:
  guard checks -> start server (nice -n 10, 127.0.0.1) -> wait /health -> /props -> /tokenize every prompt
  -> one discarded warm-up request -> the block's requests (POST /completion, stream, greedy, cache off)
  -> stop the server (SIGTERM to the child this script started; nothing else is ever signalled).
The SAME pre-rendered prompt bytes (GGUF Gemma 4 chat template, thinking off) are sent to every engine.

Per request it records: client-side time to first event / first visible content, total latency, the
timestamp of every streamed event (inter-token gaps: median / p90 / p99 / max), the server's own timings
(prompt_n, prompt tok/s, predicted_n, decode tok/s), process read_bytes / major+minor faults / CPU time,
block-device read bytes, RssAnon / RssFile / VmHWM, temperature max, CPU frequency min, throttle flags.

Safety (the real Pi runs ContextFriend services): refuses to start a block if another llama-*/powerinfer*
process exists, if MemAvailable < 3 GiB, or if a protected service is not active; aborts the plan (stopping
only its own server) when MemAvailable drops below 3 GiB or a service becomes inactive during a block.
Run under the Pi lock:  flock -n ~/pigemma/.bench.lock python3 x2_compare.py --plan PLAN.json --out DIR
"""
import argparse
import ctypes
import json
import mmap
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request

SERVICES = ["contextfriend-box", "contextfriend-gpu-tunnel", "hailort", "whisper-bench@base"]
MIN_AVAIL_KB = 3 * 1024 * 1024


def meminfo():
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            d[k] = int(v.split()[0])
    return d


def proc_status(pid):
    d = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap", "VmPeak", "VmSize", "Threads"):
                    d[k] = int(v.split()[0])
    except OSError:
        pass
    return d


def proc_io_stat(pid):
    d = {}
    try:
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                k, v = line.split(":")
                d[k.strip()] = int(v)
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        d["minflt"], d["majflt"] = int(fields[7]), int(fields[9])
        d["utime_ticks"], d["stime_ticks"] = int(fields[11]), int(fields[12])
    except (OSError, ValueError, IndexError):
        pass
    return d


def smaps_rollup(pid):
    d = {}
    keep = ("Rss", "Pss", "Pss_Anon", "Pss_File", "Pss_Shmem", "Swap", "Private_Clean", "Private_Dirty",
            "Shared_Clean", "Anonymous")
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    if k in keep:
                        d[k] = int(v.split()[0])
    except OSError:
        pass
    return d


def temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000.0
    except (OSError, ValueError):
        return None


def cpu_freq_khz():
    out = []
    for c in range(os.cpu_count() or 4):
        try:
            with open(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq") as f:
                out.append(int(f.read()))
        except (OSError, ValueError):
            pass
    return out


def throttled():
    try:
        return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def dev_stat_path(path):
    try:
        st = os.stat(path)
        p = f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}/stat"
        return p if os.path.exists(p) else None
    except OSError:
        return None


def dev_read_bytes(p):
    try:
        with open(p) as f:
            return int(f.read().split()[2]) * 512
    except (OSError, ValueError, IndexError, TypeError):
        return None


def other_llama_processes(own_pids=()):
    found = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) in own_pids:
            continue
        try:
            with open(f"/proc/{pid}/comm") as f:
                comm = f.read().strip()
        except OSError:
            continue
        if comm.startswith("llama-") or comm.startswith("powerinfer"):
            found.append((int(pid), comm))
    return found


def services_state():
    st = {}
    for s in SERVICES:
        try:
            st[s] = subprocess.run(["systemctl", "is-active", s], capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            st[s] = "unknown"
    return st


def page_cache_residency(path):
    """Fraction of the file's pages that are in the page cache (mincore on a shared read-only mapping)."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    size = os.path.getsize(path)
    ps = os.sysconf("SC_PAGE_SIZE")
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr is None or addr == ctypes.c_void_p(-1).value:
            return None
        n = (size + ps - 1) // ps
        vec = ctypes.create_string_buffer(n)
        rc = libc.mincore(addr, size, vec)
        libc.munmap(addr, size)
        if rc != 0:
            return None
        res = sum(b & 1 for b in vec.raw)
        return {"pages": n, "resident_pages": res, "fraction": round(res / n, 6), "page_size": ps}
    finally:
        os.close(fd)


def read_whole_file(path):
    """Sequential read (fills the page cache); returns seconds."""
    t0 = time.time()
    with open(path, "rb", buffering=0) as f:
        while f.read(16 << 20):
            pass
    return round(time.time() - t0, 1)


class Sampler(threading.Thread):
    COLS = ["t", "VmHWM_kB", "VmRSS_kB", "RssAnon_kB", "RssFile_kB", "RssShmem_kB", "VmSwap_kB", "read_bytes", "majflt",
            "minflt", "cpu_ticks", "MemAvailable_kB", "Cached_kB", "SwapFree_kB", "temp_c", "freq_min_khz", "freq_max_khz",
            "throttled"]

    def __init__(self, pid, path):
        super().__init__(daemon=True)
        self.pid, self.path, self.stop_ev, self.rows, self.abort_reason = pid, path, threading.Event(), [], None

    def run(self):
        n = 0
        thr = None
        with open(self.path, "w") as f:
            f.write(",".join(self.COLS) + "\n")
            while not self.stop_ev.is_set():
                st, io, mi, fr = proc_status(self.pid), proc_io_stat(self.pid), meminfo(), cpu_freq_khz()
                if not st:
                    break
                if n % 5 == 0:
                    thr = throttled()
                row = [round(time.time(), 2), st.get("VmHWM"), st.get("VmRSS"), st.get("RssAnon"), st.get("RssFile"),
                       st.get("RssShmem"), st.get("VmSwap"), io.get("read_bytes"), io.get("majflt"), io.get("minflt"),
                       (io.get("utime_ticks", 0) + io.get("stime_ticks", 0)), mi.get("MemAvailable"), mi.get("Cached"),
                       mi.get("SwapFree"), temp_c(), min(fr) if fr else None, max(fr) if fr else None, thr]
                self.rows.append(row)
                f.write(",".join("" if x is None else str(x) for x in row) + "\n")
                f.flush()
                if mi.get("MemAvailable", 1 << 40) < MIN_AVAIL_KB:
                    self.abort_reason = f"MemAvailable {mi.get('MemAvailable')} kB < {MIN_AVAIL_KB} kB"
                if n % 20 == 19:
                    bad = [s for s, v in services_state().items() if v != "active"]
                    if bad:
                        self.abort_reason = f"service not active: {bad}"
                n += 1
                self.stop_ev.wait(1.0)

    def colvals(self, k, rows=None):
        i = self.COLS.index(k)
        return [r[i] for r in (self.rows if rows is None else rows) if r[i] is not None]

    def window(self, t0, t1):
        rows = [r for r in self.rows if t0 - 1.0 <= r[0] <= t1 + 1.0]
        return {"n_samples": len(rows),
                "temp_max_c": max(self.colvals("temp_c", rows), default=None),
                "temp_min_c": min(self.colvals("temp_c", rows), default=None),
                "freq_min_khz": min(self.colvals("freq_min_khz", rows), default=None),
                "RssAnon_max_kB": max(self.colvals("RssAnon_kB", rows), default=None),
                "RssFile_max_kB": max(self.colvals("RssFile_kB", rows), default=None),
                "VmRSS_max_kB": max(self.colvals("VmRSS_kB", rows), default=None),
                "MemAvailable_min_kB": min(self.colvals("MemAvailable_kB", rows), default=None),
                "throttled_values": sorted(set(self.colvals("throttled", rows)))}


def http_json(port, path, body=None, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))]


def completion_stream(port, body, timeout=3600):
    """POST /completion (stream). Event times are relative to the moment the request is sent."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t_wall0 = time.time()
    t0 = time.monotonic()
    events, final, text, tok_ids = [], {}, [], []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except ValueError:
                continue
            now = time.monotonic() - t0
            toks = d.get("tokens") if isinstance(d.get("tokens"), list) else None
            content = d.get("content") or ""
            if content or toks:
                events.append((round(now, 4), len(content), len(toks) if toks is not None else None))
                text.append(content)
                if toks:
                    tok_ids.extend(toks)
            if d.get("stop"):
                final = d
                break
    t_end = time.monotonic() - t0
    return {"t_wall0": t_wall0, "events": events, "final": final, "text": "".join(text), "token_ids": tok_ids,
            "total_s": round(t_end, 4)}


def run_block(blk, plan, prompts, out_root, own, index=0):
    name = blk["label"]
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    model = os.path.expanduser(plan["model"])
    # one port per block: neither server could re-bind the previous block's port inside the 60 s TIME_WAIT
    port = int(plan.get("port", 18090)) + index
    rec = {"label": name, "engine": blk["engine"], "config": blk.get("config"), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    # ---- guards
    others = other_llama_processes(own)
    mi = meminfo()
    sv = services_state()
    vfs = os.statvfs("/")
    res_pre = page_cache_residency(model)
    prewarm_s = None
    if plan.get("prewarm", True) and res_pre is not None and res_pre["fraction"] < 0.9999 and not others             and mi["MemAvailable"] >= MIN_AVAIL_KB:
        prewarm_s = read_whole_file(model)
        mi = meminfo()
    rec["pre"] = {"residency_before_prewarm": res_pre, "prewarm_read_s": prewarm_s, "port": port,
                  "other_llama_processes": others, "MemAvailable_kB": mi["MemAvailable"], "Cached_kB": mi["Cached"],
                  "SwapFree_kB": mi["SwapFree"], "services": sv, "temp_c": temp_c(), "throttled": throttled(),
                  "root_free_bytes": vfs.f_bavail * vfs.f_frsize,
                  "loadavg": open("/proc/loadavg").read().split()[:3],
                  "model_residency": page_cache_residency(model)}
    if others:
        rec["aborted"] = f"other llama/powerinfer process running: {others}"
    elif mi["MemAvailable"] < MIN_AVAIL_KB:
        rec["aborted"] = "MemAvailable < 3 GiB before start"
    elif any(v != "active" for v in sv.values()):
        rec["aborted"] = f"service not active: {sv}"
    if rec.get("aborted"):
        with open(os.path.join(out, "block.json"), "w") as f:
            json.dump(rec, f, indent=1)
        return rec, True
    eng = plan["engines"][blk["engine"]]
    binary = os.path.expanduser(eng["bin"])
    cmd = ["nice", "-n", "10", binary, "-m", model] + list(eng.get("args", [])) + list(blk.get("args", [])) + \
          ["--host", "127.0.0.1", "--port", str(port)]
    rec["command"] = " ".join(shlex.quote(c) for c in cmd)
    dstat = dev_stat_path(model)
    dev0 = dev_read_bytes(dstat)
    log = open(os.path.join(out, "server.log"), "w")
    t_start = time.time()
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    own.add(srv.pid)
    smp = Sampler(srv.pid, os.path.join(out, "samples.csv"))
    smp.start()
    fatal = False
    try:
        while True:
            if srv.poll() is not None:
                raise RuntimeError(f"server exited during load (rc={srv.returncode}), see server.log")
            if smp.abort_reason:
                raise RuntimeError("GUARD: " + smp.abort_reason)
            if time.time() - t_start > float(plan.get("load_timeout_s", 900)):
                raise RuntimeError("load timeout")
            try:
                h = http_json(port, "/health", timeout=5)
                if h.get("status") == "ok":
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
        rec["load_s"] = round(time.time() - t_start, 1)
        io0 = proc_io_stat(srv.pid)
        d1 = dev_read_bytes(dstat)
        rec["after_load"] = {"read_bytes": io0.get("read_bytes"), "majflt": io0.get("majflt"), "minflt": io0.get("minflt"),
                             "status_kB": proc_status(srv.pid), "smaps_rollup_kB": smaps_rollup(srv.pid),
                             "dev_read_bytes": (d1 - dev0) if dev0 is not None and d1 is not None else None,
                             "MemAvailable_kB": meminfo()["MemAvailable"]}
        try:
            props = http_json(port, "/props", timeout=30)
            with open(os.path.join(out, "props.json"), "w", encoding="utf-8") as f:
                json.dump(props, f, ensure_ascii=False, indent=1)
            rec["props_keys"] = sorted(props.keys())
        except Exception as e:  # noqa: BLE001
            rec["props_error"] = repr(e)
        # ---- tokenisation of every prompt in this block (parity check across engines is done offline)
        ids = [plan.get("warmup_prompt_id", "warmup")] + list(blk["prompts"])
        toks = {}
        for pid_ in ids:
            try:
                toks[pid_] = http_json(port, "/tokenize", {"content": prompts[pid_]["prompt"], "add_special": True,
                                                           "parse_special": True}, timeout=60).get("tokens")
            except Exception as e:  # noqa: BLE001
                toks[pid_] = {"error": repr(e)}
        with open(os.path.join(out, "prompt_tokens.json"), "w") as f:
            json.dump(toks, f)
        # ---- diagnostic only: what would this engine's chat template produce for the same messages (thinking off)?
        tmpl = {}
        for pid_ in ids:
            if not prompts[pid_].get("messages"):
                continue
            try:
                r_t = http_json(port, "/apply-template", {"messages": prompts[pid_]["messages"],
                                                          "chat_template_kwargs": {"enable_thinking": False}}, timeout=60)
                tmpl[pid_] = {"prompt": r_t.get("prompt"), "equals_sent_prompt": r_t.get("prompt") == prompts[pid_]["prompt"],
                              "equals_with_bos_prefix": r_t.get("prompt") == "<bos>" + prompts[pid_]["prompt"]}
            except Exception as e:  # noqa: BLE001
                tmpl[pid_] = {"error": repr(e)}
        with open(os.path.join(out, "apply_template.json"), "w", encoding="utf-8") as f:
            json.dump(tmpl, f, ensure_ascii=False, indent=1)
        rec["requests"] = []
        clk = os.sysconf("SC_CLK_TCK")
        for k, pid_ in enumerate(ids):
            if smp.abort_reason:
                raise RuntimeError("GUARD: " + smp.abort_reason)
            p = prompts[pid_]
            body = {"prompt": p["prompt"], "n_predict": int(p["max_tokens"]), "temperature": 0, "top_k": 1, "seed": 1,
                    "cache_prompt": False, "stream": True, "return_tokens": True}
            a_io, a_dev, a_thr = proc_io_stat(srv.pid), dev_read_bytes(dstat), throttled()
            r = completion_stream(port, body)
            b_io, b_dev = proc_io_stat(srv.pid), dev_read_bytes(dstat)
            ev_all = r["events"]
            # identical event criterion for every engine: only events that carry visible text
            # (ik_llama.cpp streams no token ids; mainline may emit id-only events)
            ev = [e for e in ev_all if e[1] > 0]
            fin = r["final"] or {}
            t = fin.get("timings", {}) or {}
            gaps = [round(ev[i][0] - ev[i - 1][0], 4) for i in range(1, len(ev))]
            med = pct(gaps, 50)
            n_ev = len(ev)
            dec_client = round((n_ev - 1) / (ev[-1][0] - ev[0][0]), 4) if n_ev > 1 and ev[-1][0] > ev[0][0] else None
            win = smp.window(r["t_wall0"], r["t_wall0"] + r["total_s"])
            try:
                out_ids = http_json(port, "/tokenize", {"content": r["text"], "add_special": False, "parse_special": True},
                                    timeout=60).get("tokens")
            except Exception:  # noqa: BLE001
                out_ids = None
            ticks = (b_io.get("utime_ticks", 0) + b_io.get("stime_ticks", 0)
                     - a_io.get("utime_ticks", 0) - a_io.get("stime_ticks", 0))
            q = {"id": pid_, "order": k, "discarded_warmup": k == 0, "set": p.get("set"), "lang": p.get("lang"),
                 "category": p.get("category"), "n_predict": int(p["max_tokens"]),
                 "ttft_s": ev[0][0] if ev else None, "total_s": r["total_s"], "n_events": n_ev, "n_events_all": len(ev_all),
                 "decode_tok_s_client": dec_client,
                 "itl_median_s": med, "itl_p90_s": pct(gaps, 90), "itl_p99_s": pct(gaps, 99),
                 "itl_max_s": max(gaps) if gaps else None, "itl_mean_s": round(sum(gaps) / len(gaps), 4) if gaps else None,
                 "stalls_gt_2x_median": sum(1 for g in gaps if med and g > 2 * med),
                 "stalls_gt_1s": sum(1 for g in gaps if g > 1.0),
                 "server_timings": t,
                 "prompt_n": t.get("prompt_n"), "prompt_tok_s": t.get("prompt_per_second"), "prompt_ms": t.get("prompt_ms"),
                 "gen_n": t.get("predicted_n"), "gen_tok_s": t.get("predicted_per_second"), "gen_ms": t.get("predicted_ms"),
                 "tokens_evaluated": fin.get("tokens_evaluated"), "tokens_cached": fin.get("tokens_cached"),
                 "tokens_predicted": fin.get("tokens_predicted"),
                 "stop_fields": {k2: fin.get(k2) for k2 in ("stop_type", "stopped_eos", "stopped_limit", "stopped_word",
                                                             "stopping_word", "truncated")},
                 "read_bytes": b_io.get("read_bytes", 0) - a_io.get("read_bytes", 0),
                 "majflt": b_io.get("majflt", 0) - a_io.get("majflt", 0),
                 "minflt": b_io.get("minflt", 0) - a_io.get("minflt", 0),
                 "cpu_s": round(ticks / clk, 2),
                 "dev_read_bytes": (b_dev - a_dev) if a_dev is not None and b_dev is not None else None,
                 "throttled_before": a_thr, "throttled_after": throttled(), "window": win,
                 "status_kB_after": proc_status(srv.pid),
                 "stream_token_ids": r["token_ids"] or None, "output_token_ids_retokenized": out_ids,
                 "output": r["text"]}
            rec["requests"].append(q)
            with open(os.path.join(out, f"gen-{pid_}.txt"), "w", encoding="utf-8") as g:
                g.write(r["text"])
            with open(os.path.join(out, f"events-{pid_}.json"), "w") as g:
                json.dump({"id": pid_, "events_t_s__content_chars__n_tokens": ev_all, "n_events_with_text": n_ev,
                           "gaps_s": gaps}, g)
            print(json.dumps({"block": name, "id": pid_, "prompt_n": q["prompt_n"], "pp": q["prompt_tok_s"], "gen_n": q["gen_n"],
                              "tg": q["gen_tok_s"], "ttft": q["ttft_s"], "total": q["total_s"], "itl_max": q["itl_max_s"],
                              "rd": q["read_bytes"], "temp": win["temp_max_c"]}), flush=True)
        rec["after_requests"] = {"status_kB": proc_status(srv.pid), "smaps_rollup_kB": smaps_rollup(srv.pid),
                                 "MemAvailable_kB": meminfo()["MemAvailable"], "temp_c": temp_c(), "throttled": throttled()}
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)
        fatal = "GUARD" in str(e)
        print("BLOCK_ERROR", name, repr(e), flush=True)
    finally:
        smp.stop_ev.set()
        if srv.poll() is None:
            srv.send_signal(signal.SIGTERM)
            try:
                srv.wait(timeout=60)
            except subprocess.TimeoutExpired:
                srv.kill()
                srv.wait(timeout=30)
        rec["server_rc"] = srv.returncode
        own.discard(srv.pid)
        log.close()
        smp.join(timeout=5)
        rec["block_peaks"] = {
            "VmHWM_kB": max(smp.colvals("VmHWM_kB"), default=None),
            "RssAnon_max_kB": max(smp.colvals("RssAnon_kB"), default=None),
            "RssFile_max_kB": max(smp.colvals("RssFile_kB"), default=None),
            "VmSwap_max_kB": max(smp.colvals("VmSwap_kB"), default=None),
            "MemAvailable_min_kB": min(smp.colvals("MemAvailable_kB"), default=None),
            "temp_max_c": max(smp.colvals("temp_c"), default=None),
            "freq_min_khz": min(smp.colvals("freq_min_khz"), default=None),
            "throttled_values": sorted(set(smp.colvals("throttled"))),
            "n_samples": len(smp.rows)}
        rec["post"] = {"services": services_state(), "MemAvailable_kB": meminfo()["MemAvailable"], "throttled": throttled(),
                       "temp_c": temp_c()}
        rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with open(os.path.join(out, "block.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        if any(v != "active" for v in rec["post"]["services"].values()):
            fatal = True
    return rec, fatal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.plan, encoding="utf-8") as f:
        plan = json.load(f)
    with open(os.path.expanduser(plan["prompts_file"]), encoding="utf-8") as f:
        pj = json.load(f)
    prompts = {p["id"]: p for p in pj["prompts"]}
    os.makedirs(a.out, exist_ok=True)
    model = os.path.expanduser(plan["model"])
    summary = {"plan": plan, "host": os.uname().nodename, "kernel": os.uname().release,
               "page_size": os.sysconf("SC_PAGE_SIZE"), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "model_bytes": os.path.getsize(model), "blocks": []}
    res0 = page_cache_residency(model)
    summary["model_residency_at_start"] = res0
    if plan.get("prewarm", True) and (res0 is None or res0["fraction"] < 0.9999):
        if meminfo()["MemAvailable"] < MIN_AVAIL_KB:
            print("ABORT: MemAvailable < 3 GiB before prewarm", flush=True)
            sys.exit(5)
        summary["prewarm_read_s"] = read_whole_file(model)
        summary["model_residency_after_prewarm"] = page_cache_residency(model)
    print("START", json.dumps({k: summary[k] for k in summary if k.startswith("model_res") or k.startswith("prewarm")}),
          flush=True)
    own = set()
    gap = float(plan.get("gap_s", 45))
    for i, blk in enumerate(plan["blocks"]):
        if i:
            time.sleep(gap)
        rec, fatal = run_block(blk, plan, prompts, a.out, own, index=i)
        summary["blocks"].append({"label": rec["label"], "engine": rec["engine"], "error": rec.get("error"),
                                  "aborted": rec.get("aborted"), "load_s": rec.get("load_s")})
        with open(os.path.join(a.out, "plan_run.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        if fatal or rec.get("aborted"):
            print("PLAN_ABORTED", rec.get("aborted") or rec.get("error"), flush=True)
            break
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(os.path.join(a.out, "plan_run.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("PLAN_DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
