#!/usr/bin/env python3
"""X4 (comparisons workflow, 2026-10-06): resident-bottleneck driver for the real Pi (stdlib only).

This is pi/adhoc/x2/x2_compare.py (the driver X2 ran on the Pi) plus, for X4:
  * prefill / decode split of read_bytes, major faults and CPU time (snapshot at the first streamed token),
    so "zero storage reads during decode" is measured per phase, not per request;
  * optional hardware counters per phase (perf_event_open on every server thread, user space only:
    cycles, instructions, backend-stall cycles, L2/L3 data-cache refills, memory accesses) - best effort,
    a failure is recorded and never stops a block;
  * per-prompt extra request fields ("body_extra"), repeated prompt ids inside one block (files are numbered),
    several model files for the page-cache pre-read and residency check ("model_files", split GGUFs),
    a per-plan cool-down ("cool_below_c": wait until the SoC is below that temperature before a block).
The plan format and every safety guard are unchanged.

--- original X2 description ---
X2 (comparisons workflow, 2026-10-05): same-model runtime comparison driver for the real Pi (stdlib only).

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
Run under the Pi lock:  flock -n ~/pigemma/.bench.lock python3 x4_compare.py --plan PLAN.json --out DIR
"""
import argparse
import ctypes
import json
import mmap
import os
import shlex
import struct
import signal
import subprocess
import sys
import threading
import time
import urllib.request

SERVICES = ["contextfriend-box", "contextfriend-gpu-tunnel", "hailort", "whisper-bench@base"]
EMULATED = os.environ.get("PIGEMMA_EMULATED") == "1"   # desktop / container functional test only
if EMULATED:
    SERVICES = []
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


def residency_all(paths):
    """Page-cache residency summed over several files (split GGUF)."""
    pages = res = 0
    ps = None
    for p in paths:
        r = page_cache_residency(p)
        if r is None:
            return None
        pages += r["pages"]
        res += r["resident_pages"]
        ps = r["page_size"]
    return {"pages": pages, "resident_pages": res, "fraction": round(res / pages, 6) if pages else None, "page_size": ps,
            "n_files": len(paths)}


def read_whole_file(path):
    """Sequential read (fills the page cache); returns seconds."""
    t0 = time.time()
    with open(path, "rb", buffering=0) as f:
        while f.read(16 << 20):
            pass
    return round(time.time() - t0, 1)


class PerfCounters:
    """Hardware counters for every thread of a process (perf_event_open, user space only, inherit on).

    Best effort: any failure leaves self.error set and read() returning None. Counts are scaled by
    time_enabled / time_running (multiplexing); the smallest running/enabled ratio is reported."""
    EVENTS = [("cycles", 0, 0), ("instructions", 0, 1), ("stall_backend", 4, 0x24),
              ("l2d_cache_refill", 4, 0x17), ("l3d_cache_refill", 4, 0x2A), ("mem_access", 4, 0x13)]
    SYSCALL_NR = {"aarch64": 241, "x86_64": 298}

    def __init__(self, pid):
        self.fds, self.error, self.skipped, self.n_threads = {}, None, [], 0
        try:
            with open("/proc/sys/kernel/perf_event_paranoid") as f:
                self.paranoid = int(f.read().strip())
        except (OSError, ValueError):
            self.paranoid = None
        nr = self.SYSCALL_NR.get(os.uname().machine)
        if nr is None:
            self.error = "unsupported machine " + os.uname().machine
            return
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            libc.syscall.restype = ctypes.c_long
            tids = sorted(int(t) for t in os.listdir(f"/proc/{pid}/task"))
            self.n_threads = len(tids)
            for name, typ, cfg in self.EVENTS:
                fds = []
                for tid in tids:
                    attr = bytearray(64)                      # PERF_ATTR_SIZE_VER0
                    struct.pack_into("IIQ", attr, 0, typ, 64, cfg)
                    struct.pack_into("Q", attr, 32, 3)        # read_format: TOTAL_TIME_ENABLED | TOTAL_TIME_RUNNING
                    struct.pack_into("Q", attr, 40, (1 << 1) | (1 << 5) | (1 << 6))  # inherit, exclude_kernel, exclude_hv
                    buf = (ctypes.c_char * 64).from_buffer(attr)
                    fd = libc.syscall(ctypes.c_long(nr), ctypes.byref(buf), ctypes.c_int(tid), ctypes.c_int(-1),
                                      ctypes.c_int(-1), ctypes.c_ulong(0))
                    if fd < 0:
                        self.skipped.append({"event": name, "errno": ctypes.get_errno()})
                        for f2 in fds:
                            os.close(f2)
                        fds = []
                        break
                    fds.append(fd)
                if fds:
                    self.fds[name] = fds
            if not self.fds:
                self.error = f"no event could be opened (perf_event_paranoid={self.paranoid}, skipped={self.skipped[:3]})"
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)

    def read(self):
        if self.error or not self.fds:
            return None
        out, ratio = {}, 1.0
        try:
            for name, fds in self.fds.items():
                tot = 0.0
                for fd in fds:
                    val, ena, run = struct.unpack("QQQ", os.read(fd, 24))
                    if run and ena:
                        tot += val * (ena / run)
                        ratio = min(ratio, run / ena)
                    else:
                        tot += val
                out[name] = tot
            out["_min_running_ratio"] = round(ratio, 4)
            return out
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
            return None

    def close(self):
        for fds in self.fds.values():
            for fd in fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.fds = {}

    @staticmethod
    def delta(a, b):
        if not a or not b:
            return None
        d = {k: round(b[k] - a[k]) for k in b if not k.startswith("_") and k in a}
        if d.get("cycles"):
            if "instructions" in d:
                d["ipc"] = round(d["instructions"] / d["cycles"], 3)
            if "stall_backend" in d:
                d["stall_backend_fraction"] = round(d["stall_backend"] / d["cycles"], 4)
        d["min_running_ratio"] = min(a.get("_min_running_ratio", 1.0), b.get("_min_running_ratio", 1.0))
        return d


def io_delta(a, b, clk):
    if not a or not b:
        return None
    return {"read_bytes": b.get("read_bytes", 0) - a.get("read_bytes", 0),
            "majflt": b.get("majflt", 0) - a.get("majflt", 0),
            "minflt": b.get("minflt", 0) - a.get("minflt", 0),
            "cpu_s": round((b.get("utime_ticks", 0) + b.get("stime_ticks", 0)
                            - a.get("utime_ticks", 0) - a.get("stime_ticks", 0)) / clk, 2),
            "user_s": round((b.get("utime_ticks", 0) - a.get("utime_ticks", 0)) / clk, 2),
            "sys_s": round((b.get("stime_ticks", 0) - a.get("stime_ticks", 0)) / clk, 2)}


def wait_cool(limit_c, max_wait_s=600):
    """Wait until the SoC temperature is below limit_c (or max_wait_s passed); returns (waited_s, temp)."""
    t0 = time.time()
    while True:
        t = temp_c()
        if t is None or t < limit_c or time.time() - t0 > max_wait_s:
            return round(time.time() - t0, 1), t
        time.sleep(5)


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


def completion_stream(port, body, timeout=3600, on_first=None):
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
                if not events and on_first is not None:
                    on_first()
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
    if EMULATED:
        rec["emulated"] = True
    # ---- guards
    others = other_llama_processes(own)
    mi = meminfo()
    sv = services_state()
    vfs = os.statvfs("/")
    files = [os.path.expanduser(p) for p in plan.get("model_files", [])] or [model]
    cool = None
    if plan.get("cool_below_c") is not None:
        cool = wait_cool(float(plan["cool_below_c"]), float(plan.get("cool_max_wait_s", 600)))
    res_pre = residency_all(files)
    prewarm_s = None
    if plan.get("prewarm", True) and res_pre is not None and res_pre["fraction"] < 0.9999 and not others \
            and mi["MemAvailable"] >= MIN_AVAIL_KB:
        prewarm_s = round(sum(read_whole_file(p) for p in files), 1)
        mi = meminfo()
    rec["pre"] = {"residency_before_prewarm": res_pre, "prewarm_read_s": prewarm_s, "port": port,
                  "other_llama_processes": others, "MemAvailable_kB": mi["MemAvailable"], "Cached_kB": mi["Cached"],
                  "SwapFree_kB": mi["SwapFree"], "services": sv, "temp_c": temp_c(), "throttled": throttled(),
                  "root_free_bytes": vfs.f_bavail * vfs.f_frsize,
                  "loadavg": open("/proc/loadavg").read().split()[:3], "cool_down_wait_s__temp_c": cool,
                  "model_files": files, "model_residency": residency_all(files)}
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
    perf = None
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
            body.update(p.get("body_extra") or {})
            if k == 1 and plan.get("perf_counters", True):
                # the worker threads exist after the warm-up request: attach the counters now
                perf = PerfCounters(srv.pid)
                rec["perf"] = {"error": perf.error, "paranoid": perf.paranoid, "events": sorted(perf.fds),
                               "skipped": perf.skipped, "threads_attached": perf.n_threads}
            mid = {}

            def on_first():
                mid["io"], mid["dev"], mid["perf"], mid["t"] = (proc_io_stat(srv.pid), dev_read_bytes(dstat),
                                                               perf.read() if perf else None, time.time())

            a_io, a_dev, a_thr = proc_io_stat(srv.pid), dev_read_bytes(dstat), throttled()
            a_perf = perf.read() if perf else None
            r = completion_stream(port, body, on_first=on_first)
            b_io, b_dev = proc_io_stat(srv.pid), dev_read_bytes(dstat)
            b_perf = perf.read() if perf else None
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
                 "phase_prefill": io_delta(a_io, mid.get("io"), clk), "phase_decode": io_delta(mid.get("io"), b_io, clk),
                 "phase_prefill_dev_read_bytes": (mid["dev"] - a_dev) if a_dev is not None and mid.get("dev") is not None else None,
                 "phase_decode_dev_read_bytes": (b_dev - mid["dev"]) if b_dev is not None and mid.get("dev") is not None else None,
                 "perf_prefill": PerfCounters.delta(a_perf, mid.get("perf")),
                 "perf_decode": PerfCounters.delta(mid.get("perf"), b_perf),
                 "repeat_index": ids[:k].count(pid_),
                 "dev_read_bytes": (b_dev - a_dev) if a_dev is not None and b_dev is not None else None,
                 "throttled_before": a_thr, "throttled_after": throttled(), "window": win,
                 "status_kB_after": proc_status(srv.pid),
                 "stream_token_ids": r["token_ids"] or None, "output_token_ids_retokenized": out_ids,
                 "output": r["text"]}
            rec["requests"].append(q)
            tag = pid_ if ids.count(pid_) == 1 else f"{pid_}.{ids[:k].count(pid_)}"
            with open(os.path.join(out, f"gen-{tag}.txt"), "w", encoding="utf-8") as g:
                g.write(r["text"])
            with open(os.path.join(out, f"events-{tag}.json"), "w") as g:
                json.dump({"id": pid_, "events_t_s__content_chars__n_tokens": ev_all, "n_events_with_text": n_ev,
                           "gaps_s": gaps}, g)
            print(json.dumps({"block": name, "id": pid_, "prompt_n": q["prompt_n"], "pp": q["prompt_tok_s"], "gen_n": q["gen_n"],
                              "tg": q["gen_tok_s"], "ttft": q["ttft_s"], "total": q["total_s"], "itl_max": q["itl_max_s"],
                              "rd": q["read_bytes"], "rd_decode": (q["phase_decode"] or {}).get("read_bytes"),
                              "ipc_decode": (q["perf_decode"] or {}).get("ipc"), "temp": win["temp_max_c"]}), flush=True)
        rec["after_requests"] = {"status_kB": proc_status(srv.pid), "smaps_rollup_kB": smaps_rollup(srv.pid),
                                 "MemAvailable_kB": meminfo()["MemAvailable"], "temp_c": temp_c(), "throttled": throttled()}
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)
        fatal = "GUARD" in str(e)
        print("BLOCK_ERROR", name, repr(e), flush=True)
    finally:
        smp.stop_ev.set()
        try:
            if perf is not None:
                perf.close()
        except Exception:  # noqa: BLE001
            pass
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
    files = [os.path.expanduser(p) for p in plan.get("model_files", [])] or [model]
    summary["model_bytes"] = sum(os.path.getsize(p) for p in files)
    res0 = residency_all(files)
    summary["model_residency_at_start"] = res0
    if plan.get("prewarm", True) and (res0 is None or res0["fraction"] < 0.9999):
        if meminfo()["MemAvailable"] < MIN_AVAIL_KB:
            print("ABORT: MemAvailable < 3 GiB before prewarm", flush=True)
            sys.exit(5)
        summary["prewarm_read_s"] = round(sum(read_whole_file(p) for p in files), 1)
        summary["model_residency_after_prewarm"] = residency_all(files)
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
