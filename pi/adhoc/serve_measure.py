#!/usr/bin/env python3
"""One reproducible, fully logged measurement run on the Pi (stdlib only).

Starts its own llama-server (nice 10, 127.0.0.1 only), samples the server process every second
(read_bytes, majflt, smaps_rollup, swap, MemAvailable, temperature), then sends a fixed prompt set and
records, PER REQUEST, the server's own timings (prompt tok/s, generation tok/s, draft stats) together
with the storage bytes and major faults that happened during that request. Optionally first pins an
mlocked memory balloon (sudo) so the page cache has less room - the Pi kernel has no memory cgroup.

Usage:
  python3 serve_measure.py --model M.gguf --label NAME [--threads 4] [--ctx 4096] [--n-predict 64]
         [--prompts prompts.json] [--extra "-md X.gguf --spec-type draft-mtp"] [--balloon-gib 6]
         [--cold] [--stream] [--out ~/pigemma/results]
Writes <out>/<label>/{run.json, samples.csv, server.log, gen-<id>.txt} (+ itl-<id>.json with --stream).

Added 2026-10-05 (P1 baseline audit, opt-in / additive; the default request path is unchanged):
  * every request also records system-wide /proc/vmstat deltas (pgmajfault, pgpgin, pswpin/out,
    pgscan/pgsteal, workingset refaults) and the read bytes of the block device holding the model
    (/sys/dev/block/<maj:min>/stat; device-level, includes unrelated traffic on that device).
  * --stream: requests use stream=true and timestamp every server-sent event, giving time to first
    token and the inter-token latency distribution (median / p90 / p99 / max, stalls), and split the
    request's process / device I/O into prefill (request start -> first token) and decode.
  * a prompt entry may carry "format": "canonical" to use the GGUF's Gemma 4 turn markers
    (<|turn>user ... <turn|> <|turn>model + empty thought channel, i.e. thinking off) instead of the
    legacy Gemma-3 text markers used by all runs before 2026-10-05 19:00 ("format": "legacy", default).
Never touches other processes: it only signals the server and balloon it started itself.
"""
import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.expanduser("~/pigemma/llama.cpp/build/bin/llama-server")


def proc_io(pid):
    d = {}
    try:
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                k, v = line.split(":")
                d[k.strip()] = int(v)
        with open(f"/proc/{pid}/stat") as f:
            d["majflt"] = int(f.read().rsplit(")", 1)[1].split()[9])
    except (OSError, ValueError, IndexError):
        pass
    return d


def meminfo():
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            d[k] = int(v.split()[0])
    return d


def smaps(pid):
    d = {}
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if ":" in line and line.split(":")[0] in ("Rss", "Pss_Anon", "Pss_File", "Swap"):
                    k, v = line.split(":", 1)
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


def throttled():
    try:
        return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def sampler(pid, path, stop):
    with open(path, "w") as f:
        f.write("t_s,read_bytes,rchar,majflt,rss_kb,pss_anon_kb,pss_file_kb,swap_kb,mem_available_kb,cached_kb,swap_free_kb,temp_c\n")
        t0 = time.time()
        while not stop.is_set():
            io = proc_io(pid)
            if not io:
                break
            sm = smaps(pid)
            mi = meminfo()
            f.write(",".join(str(x) for x in [round(time.time() - t0, 1), io.get("read_bytes"), io.get("rchar"), io.get("majflt"),
                                                sm.get("Rss"), sm.get("Pss_Anon"), sm.get("Pss_File"), sm.get("Swap"),
                                                mi.get("MemAvailable"), mi.get("Cached"), mi.get("SwapFree"), temp_c()]) + "\n")
            f.flush()
            stop.wait(1.0)


def http(port, path, body=None, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def evict(path):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def sha256_prefix(path, nbytes=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:16]


VMSTAT_KEYS = ("pgmajfault", "pgpgin", "pswpin", "pswpout", "pgscan_kswapd", "pgscan_direct",
               "pgsteal_kswapd", "pgsteal_direct", "workingset_refault_file", "workingset_refault_anon")


def vmstat():
    d = {}
    try:
        with open("/proc/vmstat") as f:
            for line in f:
                k, v = line.split()
                if k in VMSTAT_KEYS:
                    d[k] = int(v)
    except (OSError, ValueError):
        pass
    return d


def block_stat_path(path):
    """/sys/dev/block/<maj:min>/stat of the device (partition) that holds `path`, or None."""
    try:
        st = os.stat(path)
        p = f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}/stat"
        return p if os.path.exists(p) else None
    except OSError:
        return None


def dev_read_bytes(stat_path):
    try:
        with open(stat_path) as f:
            return int(f.read().split()[2]) * 512  # field 3 = sectors read (always 512-byte units)
    except (OSError, ValueError, IndexError, TypeError):
        return None


def counters(pid, dev_stat):
    return {"t": time.time(), "io": proc_io(pid), "vm": vmstat(), "dev": dev_read_bytes(dev_stat)}


def delta(a, b):
    """b - a for two counters() snapshots."""
    d = {"read_bytes": (b["io"].get("read_bytes", 0) - a["io"].get("read_bytes", 0)) if a["io"] and b["io"] else None,
         "majflt": (b["io"].get("majflt", 0) - a["io"].get("majflt", 0)) if a["io"] and b["io"] else None,
         "dev_read_bytes": (b["dev"] - a["dev"]) if a["dev"] is not None and b["dev"] is not None else None,
         "vmstat": {k: b["vm"][k] - a["vm"][k] for k in b["vm"] if k in a["vm"]},
         "wall_s": round(b["t"] - a["t"], 3)}
    return d


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))
    return s[i]


def http_stream(port, body, on_first, timeout=3600):
    """POST /completion with stream=true; returns (events, final_json, text). events = [(t_rel_s, n_tokens)]."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 data=json.dumps(dict(body, stream=True, return_tokens=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    events, final, text, first = [], {}, [], True
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
            now = time.time()
            ntok = len(d.get("tokens") or []) if "tokens" in d else None
            if d.get("stop"):
                final = d
                if d.get("content"):
                    text.append(d["content"])
                    events.append((round(now - t0, 4), ntok))
                break
            if first:
                on_first()
                first = False
            text.append(d.get("content", ""))
            events.append((round(now - t0, 4), ntok))
    return events, final, "".join(text), t0


def legacy_or_canonical(p):
    if p.get("format") == "canonical":
        return f"<|turn>user\n{p['text']}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    return f"<start_of_turn>user\n{p['text']}<end_of_turn>\n<start_of_turn>model\n"


DEFAULT_PROMPTS = [
    {"id": "en_short", "lang": "en", "text": "In three short sentences: why is a mixture-of-experts model faster to run than a dense model of the same size?"},
    {"id": "he_short", "lang": "he", "text": "ענה בעברית בשלושה משפטים קצרים: מה היתרון של מחשב קטן כמו Raspberry Pi?"},
    {"id": "de_short", "lang": "de", "text": "Antworte auf Deutsch in drei kurzen Sätzen: Warum ist ein Raspberry Pi für Schulen interessant?"},
    {"id": "code_short", "lang": "code", "text": "Write a Python function is_prime(n) that returns True if n is prime. Only the code."},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--prompts")
    ap.add_argument("--long-file", help="text file appended as a long-context summarisation prompt")
    ap.add_argument("--extra", default="")
    ap.add_argument("--balloon-gib", type=float, default=0.0)
    ap.add_argument("--cold", action="store_true", help="evict this model file from the page cache first")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--stream", action="store_true",
                    help="stream tokens: record time to first token, inter-token latencies, prefill/decode I/O split")
    ap.add_argument("--out", default=os.path.expanduser("~/pigemma/results"))
    a = ap.parse_args()

    out = os.path.join(a.out, a.label)
    os.makedirs(out, exist_ok=True)
    prompts = json.load(open(a.prompts, encoding="utf-8")) if a.prompts else list(DEFAULT_PROMPTS)
    if a.long_file:
        txt = open(a.long_file, encoding="utf-8").read()
        prompts.append({"id": "long_summary", "lang": "en", "text": "Summarise the following text in three sentences.\n\n" + txt})

    run = {"label": a.label, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": os.uname().nodename,
           "kernel": os.uname().release, "page_size": os.sysconf("SC_PAGE_SIZE"),
           "model": {"path": os.path.abspath(a.model), "bytes": os.path.getsize(a.model), "sha256_first_MiB": sha256_prefix(a.model)},
           "threads": a.threads, "ctx": a.ctx, "n_predict": a.n_predict, "extra": a.extra, "balloon_gib": a.balloon_gib,
           "meminfo_start_kB": {k: meminfo()[k] for k in ("MemTotal", "MemAvailable", "Cached", "SwapFree")},
           "throttled_start": throttled(), "temp_start_c": temp_c(), "requests": []}

    balloon = None
    if a.cold:
        evict(a.model)
        run["cold"] = True
    if a.balloon_gib > 0:
        balloon = subprocess.Popen(["sudo", "-n", "python3", os.path.join(HERE, "balloon.py"), str(a.balloon_gib), "3000"],
                                   stdout=subprocess.PIPE, text=True)
        line = balloon.stdout.readline().strip()
        run["balloon"] = json.loads(line) if line.startswith("{") else {"raw": line}
        if not run["balloon"].get("ok"):
            print("balloon refused:", line)
            sys.exit(3)
        run["meminfo_after_balloon_kB"] = {k: meminfo()[k] for k in ("MemAvailable", "Cached", "Mlocked")}

    cmd = ["nice", "-n", "10", BIN, "-m", a.model, "-nr", "-t", str(a.threads), "-c", str(a.ctx), "-np", "1",
           "--host", "127.0.0.1", "--port", str(a.port)] + shlex.split(a.extra)
    run["command"] = " ".join(shlex.quote(c) for c in cmd)
    run["stream"] = a.stream
    dev_stat = block_stat_path(a.model)
    run["model_block_stat"] = dev_stat
    load_vm0, load_dev0 = vmstat(), dev_read_bytes(dev_stat)
    log = open(os.path.join(out, "server.log"), "w")
    t_start = time.time()
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    stop = threading.Event()
    th = threading.Thread(target=sampler, args=(srv.pid, os.path.join(out, "samples.csv"), stop), daemon=True)
    th.start()
    try:
        while True:
            if srv.poll() is not None:
                raise RuntimeError("server exited during load, see server.log")
            try:
                if http(a.port, "/health", timeout=5).get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(1)
        run["load_s"] = round(time.time() - t_start, 1)
        io0 = proc_io(srv.pid)
        run["load_read_bytes"] = io0.get("read_bytes")
        run["load_majflt"] = io0.get("majflt")
        run["after_load_smaps_kB"] = smaps(srv.pid)
        load_vm1 = vmstat()
        run["load_vmstat_delta"] = {k: load_vm1[k] - load_vm0[k] for k in load_vm1 if k in load_vm0}
        load_dev1 = dev_read_bytes(dev_stat)
        run["load_dev_read_bytes"] = (load_dev1 - load_dev0) if load_dev0 is not None and load_dev1 is not None else None
        for p in prompts:
            text = legacy_or_canonical(p)
            body = {"prompt": text, "n_predict": a.n_predict, "temperature": 0, "cache_prompt": False}
            c0 = counters(srv.pid, dev_stat)
            before = c0["io"]
            t0 = time.time()
            itl = None
            if a.stream:
                mark = {}
                events, r, content, ts = http_stream(a.port, body, lambda: mark.setdefault("c", counters(srv.pid, dev_stat)))
                r = dict(r, content=content)
                c1 = mark.get("c")
                gaps = [round(events[i][0] - events[i - 1][0], 4) for i in range(1, len(events))]
                med = pct(gaps, 50)
                itl = {"ttft_s": events[0][0] if events else None, "n_events": len(events),
                       "n_tokens_reported": sum(n for _, n in events if n) if any(n for _, n in events) else None,
                       "itl_median_s": med, "itl_p90_s": pct(gaps, 90), "itl_p99_s": pct(gaps, 99),
                       "itl_max_s": max(gaps) if gaps else None, "itl_mean_s": round(sum(gaps) / len(gaps), 4) if gaps else None,
                       "stalls_gt_2x_median": sum(1 for g in gaps if med and g > 2 * med),
                       "stalls_gt_1s": sum(1 for g in gaps if g > 1.0)}
                with open(os.path.join(out, f"itl-{p['id']}.json"), "w", encoding="utf-8") as g:
                    json.dump({"id": p["id"], "events_t_s_ntokens": events, "gaps_s": gaps, **itl}, g)
            else:
                c1 = None
                r = http(a.port, "/completion", body)
            wall = time.time() - t0
            c2 = counters(srv.pid, dev_stat)
            after = c2["io"]
            t = r.get("timings", {})
            gen_n = max(1, t.get("predicted_n", 0))
            rec = {"id": p["id"], "lang": p.get("lang"), "format": p.get("format", "legacy"), "wall_s": round(wall, 1),
                   "prompt_n": t.get("prompt_n"), "prompt_tok_s": t.get("prompt_per_second"),
                   "prompt_ms": t.get("prompt_ms"), "gen_ms": t.get("predicted_ms"),
                   "gen_n": t.get("predicted_n"), "gen_tok_s": t.get("predicted_per_second"),
                   "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
                   "read_bytes": after.get("read_bytes", 0) - before.get("read_bytes", 0),
                   "majflt": after.get("majflt", 0) - before.get("majflt", 0),
                   "io_total": delta(c0, c2),
                   "io_prefill": delta(c0, c1) if c1 else None, "io_decode": delta(c1, c2) if c1 else None,
                   "itl": itl,
                   "smaps_kB": smaps(srv.pid), "mem_available_kB": meminfo()["MemAvailable"], "temp_c": temp_c(),
                   "throttled": throttled()}
            rec["read_MB_per_request_token"] = round(rec["read_bytes"] / 1e6 / (gen_n + (t.get("prompt_n") or 0)), 3)
            with open(os.path.join(out, f"gen-{p['id']}.txt"), "w", encoding="utf-8") as g:
                g.write(r.get("content", ""))
            rec["output_head"] = r.get("content", "")[:240]
            run["requests"].append(rec)
            print(json.dumps({k: rec[k] for k in ("id", "prompt_n", "prompt_tok_s", "gen_n", "gen_tok_s", "draft_n",
                                                   "draft_n_accepted", "read_bytes", "majflt", "wall_s")}), flush=True)
    finally:
        stop.set()
        srv.send_signal(signal.SIGTERM)
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
        if balloon is not None:
            subprocess.run(["sudo", "-n", "kill", "-TERM", str(balloon.pid)], check=False)
            # the sudo wrapper forwards SIGTERM to the python child
            try:
                balloon.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        run["throttled_end"] = throttled()
        run["temp_end_c"] = temp_c()
        run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
            json.dump(run, f, ensure_ascii=False, indent=1)
        print("RUN_DONE", os.path.join(out, "run.json"), flush=True)


if __name__ == "__main__":
    main()
