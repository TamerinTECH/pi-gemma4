"""pigemma.controlled_cell - ONE controlled-memory measurement cell, run INSIDE its memory cgroup.

Stage P5 of the decision workflow (brief 2026-10-05 sections 2-4). One cell = one method x one
memory budget x one workload set x one repeat, in a FRESH cgroup (a fresh Docker container on the
desktop; a fresh ``systemd-run --scope`` on the Pi once its memory controller is enabled). This
module is the in-cgroup half: it is started as the cgroup's first process, so every helper it
starts (the FUSE throttle, the pinner, llama-server) is charged to the same budget and is started
BEFORE the model loads. The host half (docker flags, locks, host CPU load, paired analysis) is
``pigemma.controlled`` / ``pigemma.controlled_summary``; the Pi wrapper is ``pi/cgroup_cell.sh``.

Methods (identical cache-state procedure for all):
  A  nothing between eviction and llama-server (plain llama.cpp + Linux page cache)
  B  the pinner started with an EMPTY range list (``--allow-empty``): same helper process, same
     whole-file MAP_SHARED mapping, same instrumentation as C/D, nothing locked
  C  pinner locks non-expert ranges + calibration-ranked hot experts
  D  pinner locks non-expert ranges + RANDOM experts with the same page-accounted total as C
Sequence: [storage up] -> evict the model file's pages (targeted posix_fadvise DONTNEED, never a
global drop_caches) -> verify cold with mincore -> method setup (timed) -> llama-server load
(timed) -> fixed warm-up request (timed) -> workload requests (SSE streaming, client-side per-token
timestamps) -> final residency + pinner verification -> teardown.

Storage modes (config ``storage.mode``):
  fuse-throttled    tools/fusethrottle serves the GGUF at ``rate_bps`` (default 91.6 MB/s, the
                    Pi microSD) from the backing file read with O_DIRECT; llama-server and the
                    pinner mmap the FUSE file (its page cache is the only cache of model bytes)
  fuse-unthrottled  the same FUSE path with rate 0 (separately labelled condition)
  direct            mmap the backing file itself (the Pi's real microSD; or unthrottled ext4)

Outputs in ``out_dir`` (= results/controlled/<cell_id>/):
  config.json        the host's request + ``effective`` (what was really used) + status
  requests.jsonl     one line per request (warm-up included, ``kind`` says which)
  outputs/<rid>.json raw text, token ids, per-event timestamps, server timings, prompt sha256
  io.csv             one row per phase/request: I/O, faults, memory deltas (see IO_FIELDS)
  samples.csv        periodic samples (cgroup memory, per-process I/O, FUSE counters, CPU)
  cgroup_start.json / cgroup_end.json / cgroup_phases.json   full cgroup memory snapshots
  server.log, pinner.log, fusethrottle.log, fuse_trace.csv, residency/<tag>.bin.z
Stdlib only (Python 3.12 in the Docker image, 3.13 on the Pi).

Engines: ``config["server"]["bin"]`` is the llama-server binary of the cell; ``config["engine"]``
(id, revision, parent, flag overrides, API notes; written by ``pigemma.controlled`` from
results/comparisons/engines.json) lets the SAME cell run another build (recent upstream, HotPin and
its parent, ik_llama.cpp). ``server.flag_overrides`` replaces or drops single canonical flags of
``SERVER_FLAGS`` for engines that spell them differently; everything else stays identical.

CLI::

    python3 -m pigemma.controlled_cell run --config CONFIG.json
    python3 -m pigemma.controlled_cell verify-throttle --backing FILE --out DIR [--rate-bps N]
    python3 -m pigemma.controlled_cell probe            (JSON: cgroup version + what is readable)
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import hashlib
import json
import os
import random
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

SCHEMA = "pigemma-controlled-cell v1"
STALL_FACTOR = 3.0

# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_text(p: str | Path) -> str | None:
    try:
        with open(p, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def read_int(p: str | Path) -> int | None:
    t = read_text(p)
    if t is None:
        return None
    t = t.strip()
    if t == "max":
        return None
    try:
        return int(t.split()[0])
    except (ValueError, IndexError):
        return None


def kv_file(p: str | Path) -> dict[str, int]:
    out: dict[str, int] = {}
    t = read_text(p)
    if not t:
        return out
    for line in t.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                out[parts[0].rstrip(":")] = int(parts[1])
            except ValueError:
                pass
    return out


def sha256_file(p: str | Path, limit: int | None = None) -> str | None:
    h = hashlib.sha256()
    try:
        with open(p, "rb") as fh:
            left = limit
            while True:
                n = 1 << 20 if left is None else min(1 << 20, left)
                if n <= 0:
                    break
                b = fh.read(n)
                if not b:
                    break
                h.update(b)
                if left is not None:
                    left -= len(b)
    except OSError:
        return None
    return h.hexdigest()


def write_json(p: str | Path, obj: Any) -> None:
    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, p)


def pct(xs: Sequence[float], q: float) -> float | None:
    """Nearest-rank-style percentile on sorted data (q in 0..100), None for empty input."""
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))
    return s[i]


def itl_stats(token_times: Sequence[float], stall_factor: float = STALL_FACTOR) -> dict[str, Any]:
    """Inter-token latency statistics from per-token client timestamps (seconds, same origin).

    Gaps are between consecutive generated tokens (the first token's latency is the TTFT, not an
    ITL). Tokens delivered in the same SSE event have a 0 gap; their count is reported."""
    gaps = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
    if not gaps:
        return {"n_gaps": 0, "itl_median_s": None, "itl_p90_s": None, "itl_p99_s": None,
                "itl_max_s": None, "itl_mean_s": None, "stalls_gt_3x_median": None,
                "stall_threshold_s": None, "zero_gaps": 0}
    med = statistics.median(gaps)
    thr = stall_factor * med if med > 0 else None
    return {
        "n_gaps": len(gaps),
        "itl_median_s": round(med, 6),
        "itl_p90_s": round(pct(gaps, 90) or 0.0, 6),
        "itl_p99_s": round(pct(gaps, 99) or 0.0, 6),
        "itl_max_s": round(max(gaps), 6),
        "itl_mean_s": round(sum(gaps) / len(gaps), 6),
        "stalls_gt_3x_median": sum(1 for g in gaps if thr is not None and g > thr),
        "stall_threshold_s": round(thr, 6) if thr is not None else None,
        "zero_gaps": sum(1 for g in gaps if g == 0.0),
    }


def psi_totals(text: str | None) -> dict[str, int | None]:
    """{"some": total_us, "full": total_us} from a PSI file (memory.pressure / /proc/pressure/memory)."""
    out: dict[str, int | None] = {"some": None, "full": None}
    for line in (text or "").splitlines():
        parts = line.split()
        if parts and parts[0] in out:
            for kv in parts[1:]:
                k, _, v = kv.partition("=")
                if k == "total":
                    try:
                        out[parts[0]] = int(v)
                    except ValueError:
                        pass
    return out


def thermal_mC() -> int | None:
    """SoC temperature in milli-degC (the Pi's thermal_zone0); None where there is no sensor
    (the Docker Desktop VM)."""
    return read_int("/sys/class/thermal/thermal_zone0/temp")


def pi_throttled() -> str | None:
    """`vcgencmd get_throttled` (Raspberry Pi firmware: under-voltage / frequency capping /
    throttling bits, e.g. 'throttled=0x0'); None when the tool is absent."""
    try:
        r = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None


# ----------------------------------------------------------------------------------------------
# cgroup readers (v1 = Docker Desktop VM, v2 = the Pi once enabled)
# ----------------------------------------------------------------------------------------------


class Cgroup:
    """The calling process's memory/cpu/io cgroup, v1 or v2, read-only."""

    def __init__(self) -> None:
        self.version = 2 if os.path.exists("/sys/fs/cgroup/cgroup.controllers") else 1
        self.mem = self.cpu = self.io = None
        cg = read_text("/proc/self/cgroup") or ""
        if self.version == 2:
            rel = ""
            for line in cg.splitlines():
                if line.startswith("0::"):
                    rel = line[3:].strip().lstrip("/")
            base = Path("/sys/fs/cgroup") / rel
            self.mem = self.cpu = self.io = base
            self.path = "/" + rel
        else:
            paths: dict[str, str] = {}
            for line in cg.splitlines():
                parts = line.split(":", 2)
                if len(parts) == 3:
                    for c in parts[1].split(","):
                        paths[c] = parts[2]

            def pick(ctrl: str, mount: str) -> Path | None:
                root = Path("/sys/fs/cgroup") / mount
                if not root.exists():
                    return None
                rel = paths.get(ctrl, "/").lstrip("/")
                cand = root / rel
                # inside a container with a private cgroup namespace the own cgroup IS the root
                return cand if rel and cand.exists() else root
            self.mem = pick("memory", "memory")
            self.cpu = pick("cpuacct", "cpuacct") or pick("cpu", "cpu,cpuacct")
            self.io = pick("blkio", "blkio")
            self.path = paths.get("memory")

    # -- memory --------------------------------------------------------------------------------
    def memory(self) -> dict[str, Any]:
        m = self.mem
        if m is None:
            return {"version": self.version, "available": False}
        if self.version == 2:
            ev = kv_file(m / "memory.events")
            return {
                "version": 2, "path": self.path,
                "max": read_int(m / "memory.max"), "high": read_int(m / "memory.high"),
                "low": read_int(m / "memory.low"), "min": read_int(m / "memory.min"),
                "current": read_int(m / "memory.current"), "peak": read_int(m / "memory.peak"),
                "swap_max": read_int(m / "memory.swap.max"), "swap_current": read_int(m / "memory.swap.current"),
                "stat": kv_file(m / "memory.stat"), "events": ev,
                "pressure": (read_text(m / "memory.pressure") or "").strip() or None,
                "raw_max": (read_text(m / "memory.max") or "").strip(),
                "raw_high": (read_text(m / "memory.high") or "").strip(),
            }
        oc = kv_file(m / "memory.oom_control")
        return {
            "version": 1, "path": self.path,
            # v1 equivalents of the v2 fields asked for (brief section 2), recorded under both names
            "max": read_int(m / "memory.limit_in_bytes"),
            "high": None,  # v1 has no memory.high; soft_limit is the closest (reclaim target under global pressure)
            "soft_limit": read_int(m / "memory.soft_limit_in_bytes"),
            "current": read_int(m / "memory.usage_in_bytes"),
            "peak": read_int(m / "memory.max_usage_in_bytes"),
            "memsw_limit": read_int(m / "memory.memsw.limit_in_bytes"),
            "memsw_current": read_int(m / "memory.memsw.usage_in_bytes"),
            "memsw_peak": read_int(m / "memory.memsw.max_usage_in_bytes"),
            "kmem_current": read_int(m / "memory.kmem.usage_in_bytes"),
            "kmem_peak": read_int(m / "memory.kmem.max_usage_in_bytes"),
            "swappiness": read_int(m / "memory.swappiness"),
            "use_hierarchy": read_int(m / "memory.use_hierarchy"),
            "stat": kv_file(m / "memory.stat"),
            "events": {"failcnt": read_int(m / "memory.failcnt"),
                       "memsw_failcnt": read_int(m / "memory.memsw.failcnt"),
                       "oom_kill": oc.get("oom_kill"), "under_oom": oc.get("under_oom"),
                       "oom_kill_disable": oc.get("oom_kill_disable")},
            "pressure": None,  # PSI unavailable (CONFIG_PSI unset in the WSL2 5.15 kernel; no v1 PSI file)
        }

    def memory_brief(self) -> dict[str, Any]:
        """The few fields sampled every interval."""
        full = self.memory()
        st = full.get("stat") or {}
        if self.version == 2:
            psi = psi_totals(full.get("pressure"))
            return {"usage": full.get("current"), "peak": full.get("peak"),
                    "anon": st.get("anon"), "file": st.get("file"), "file_mapped": st.get("file_mapped"),
                    "unevictable": st.get("unevictable"), "kernel": st.get("kernel"),
                    "pgmajfault": st.get("pgmajfault"), "pgscan": st.get("pgscan"), "pgsteal": st.get("pgsteal"),
                    "workingset_refault_file": st.get("workingset_refault_file"),
                    "pgpgin": None, "pgpgout": None,  # v1 only (charge / uncharge events)
                    "psi_some_us": psi["some"], "psi_full_us": psi["full"],
                    "swap": full.get("swap_current"),
                    "ev_max": (full.get("events") or {}).get("max"),
                    "ev_high": (full.get("events") or {}).get("high"),
                    "ev_oom_kill": (full.get("events") or {}).get("oom_kill")}
        ev = full.get("events") or {}
        # limit hits: memory.failcnt + memory.memsw.failcnt (with memsw == mem the memsw counter is
        # the one that trips first; measured in the 2026-10-05 smoke cell: 0 vs 3,481,753)
        hits = None
        if ev.get("failcnt") is not None or ev.get("memsw_failcnt") is not None:
            hits = (ev.get("failcnt") or 0) + (ev.get("memsw_failcnt") or 0)
        return {"usage": full.get("current"), "peak": full.get("peak"),
                "anon": st.get("rss"), "file": st.get("cache"), "file_mapped": st.get("mapped_file"),
                "unevictable": st.get("unevictable"), "kernel": full.get("kmem_current"),
                "pgmajfault": st.get("pgmajfault"), "pgscan": None, "pgsteal": None,
                "workingset_refault_file": None,
                # v1 reclaim evidence: pgpgin = pages CHARGED to the cgroup (page cache read in or
                # anon mapped), pgpgout = pages UNCHARGED (evicted page cache or freed anon). At
                # the limit their deltas are the cgroup's page turnover (x page size = bytes).
                "pgpgin": st.get("pgpgin"), "pgpgout": st.get("pgpgout"),
                "psi_some_us": None, "psi_full_us": None,  # no PSI in this kernel
                "swap": st.get("swap"),
                "ev_max": hits, "ev_high": None, "ev_oom_kill": ev.get("oom_kill")}

    # -- cpu / io ------------------------------------------------------------------------------
    def cpu_usage_ns(self) -> int | None:
        if self.cpu is None:
            return None
        if self.version == 2:
            st = kv_file(self.cpu / "cpu.stat")
            return st["usage_usec"] * 1000 if "usage_usec" in st else None
        return read_int(self.cpu / "cpuacct.usage")

    def io_stat(self) -> dict[str, dict[str, int]] | None:
        """v2 io.stat {"maj:min": {rbytes, wbytes, rios, wios}}; v1 blkio service bytes if present."""
        if self.io is None:
            return None
        if self.version == 2:
            t = read_text(self.io / "io.stat")
            if t is None:
                return None
            out: dict[str, dict[str, int]] = {}
            for line in t.splitlines():
                parts = line.split()
                if not parts:
                    continue
                d: dict[str, int] = {}
                for kv in parts[1:]:
                    k, _, v = kv.partition("=")
                    try:
                        d[k] = int(v)
                    except ValueError:
                        pass
                out[parts[0]] = d
            return out
        for name in ("blkio.throttle.io_service_bytes", "blkio.io_service_bytes"):
            t = read_text(self.io / name)
            if t is None:
                continue
            out = {}
            for line in t.splitlines():
                parts = line.split()
                if len(parts) == 3:
                    out.setdefault(parts[0], {})[parts[1].lower()] = int(parts[2])
            return {k: {"rbytes": v.get("read", 0), "wbytes": v.get("write", 0)} for k, v in out.items()}
        return None

    def describe(self) -> dict[str, Any]:
        files = {}
        for name, base in (("memory", self.mem), ("cpu", self.cpu), ("io", self.io)):
            try:
                files[name] = sorted(os.listdir(base)) if base is not None else None
            except OSError:
                files[name] = None
        return {"version": self.version, "path": self.path,
                "memory_dir": str(self.mem) if self.mem else None,
                "cpu_dir": str(self.cpu) if self.cpu else None,
                "io_dir": str(self.io) if self.io else None,
                "files": files,
                "psi_available": os.path.exists("/proc/pressure/memory"),
                "io_stat_available": self.io_stat() is not None}


# ----------------------------------------------------------------------------------------------
# process / VM-wide counters
# ----------------------------------------------------------------------------------------------


def proc_io(pid: int | None) -> dict[str, int]:
    if not pid:
        return {}
    d = kv_file(f"/proc/{pid}/io")
    t = read_text(f"/proc/{pid}/stat")
    if t:
        try:
            f = t.rsplit(")", 1)[1].split()
            d["minflt"] = int(f[7])
            d["majflt"] = int(f[9])
            d["utime_ticks"] = int(f[11])
            d["stime_ticks"] = int(f[12])
        except (IndexError, ValueError):
            pass
    return d


def proc_status(pid: int | None) -> dict[str, int]:
    """VmRSS, RssAnon, RssFile, RssShmem, VmLck, VmSwap, VmHWM in kB."""
    if not pid:
        return {}
    keep = ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmLck", "VmSwap", "VmHWM", "VmPin")
    return {k: v for k, v in kv_file(f"/proc/{pid}/status").items() if k in keep}


def smaps_rollup(pid: int | None) -> dict[str, int]:
    if not pid:
        return {}
    keep = ("Rss", "Pss", "Pss_Anon", "Pss_File", "Shared_Clean", "Private_Clean", "Private_Dirty",
            "Anonymous", "Locked", "Swap")
    return {k: v for k, v in kv_file(f"/proc/{pid}/smaps_rollup").items() if k in keep}


VMSTAT_KEYS = ("pgmajfault", "pgfault", "pgpgin", "pswpin", "pswpout", "pgscan_kswapd", "pgscan_direct",
               "pgsteal_kswapd", "pgsteal_direct", "workingset_refault_file", "workingset_refault",
               "workingset_activate_file", "workingset_restore_file", "allocstall_normal", "allocstall_movable",
               "unevictable_pgs_mlocked", "unevictable_pgs_munlocked")


def vmstat() -> dict[str, int]:
    d = kv_file("/proc/vmstat")
    return {k: d[k] for k in VMSTAT_KEYS if k in d}


def diskstats(devs: Iterable[str]) -> dict[str, int]:
    """sectors read * 512 per device name (VM/host-wide: includes unrelated traffic)."""
    want = set(devs)
    out: dict[str, int] = {}
    t = read_text("/proc/diskstats") or ""
    for line in t.splitlines():
        f = line.split()
        if len(f) > 6 and f[2] in want:
            out[f[2]] = int(f[5]) * 512
    return out


def cpu_totals() -> tuple[int, int] | None:
    """(busy, total) jiffies of the whole (V)M from /proc/stat."""
    t = read_text("/proc/stat")
    if not t:
        return None
    f = t.splitlines()[0].split()
    if f[0] != "cpu":
        return None
    vals = [int(x) for x in f[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return sum(vals) - idle, sum(vals)


def meminfo() -> dict[str, int]:
    return kv_file("/proc/meminfo")


def block_dev_of(path: str) -> dict[str, Any]:
    """maj:min, kernel name, read_ahead_kb and rotational flag of the device holding path."""
    try:
        st = os.stat(path)
    except OSError:
        return {}
    mm = f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}"
    out: dict[str, Any] = {"maj_min": mm, "st_dev": st.st_dev}
    sysd = Path(f"/sys/dev/block/{mm}")
    if sysd.exists():
        try:
            real = sysd.resolve()
            out["name"] = real.name
            q = real / "queue"
            if not q.exists():  # partition: queue lives on the parent disk
                q = real.parent / "queue"
                out["parent"] = real.parent.name
            out["read_ahead_kb"] = read_int(q / "read_ahead_kb")
            out["rotational"] = read_int(q / "rotational")
            out["scheduler"] = (read_text(q / "scheduler") or "").strip()
            out["logical_block_size"] = read_int(q / "logical_block_size")
        except OSError:
            pass
    return out


# ----------------------------------------------------------------------------------------------
# page-cache residency (mincore) of the served model file
# ----------------------------------------------------------------------------------------------

_libc = None


def _lib() -> Any:
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        _libc.mmap.restype = ctypes.c_void_p
        _libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
        _libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    return _libc


def residency_vector(path: str) -> bytes | None:
    """One byte per page (1 = in the page cache) for the whole file, via mmap + mincore.

    The mapping faults nothing in. Reliable only when the caller owns the file or has CAP_FOWNER
    (root in the container / Pi sudo); otherwise Linux >= 5.0 reports every page resident."""
    lib = _lib()
    page = os.sysconf("SC_PAGE_SIZE")
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        if size == 0:
            return b""
        addr = lib.mmap(None, size, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED
        if addr in (None, ctypes.c_void_p(-1).value):
            return None
        try:
            n = (size + page - 1) // page
            vec = ctypes.create_string_buffer(n)
            if lib.mincore(ctypes.c_void_p(addr), size, vec) != 0:
                return None
            return vec.raw.translate(_BIT0)
        finally:
            lib.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)


_BIT0 = bytes(b & 1 for b in range(256))


class Categories:
    """Page runs per category (header / nonexpert / expert layer L) from the expert map."""

    def __init__(self, map_path: str | None, page: int, file_size: int) -> None:
        self.page = page
        self.n_pages = (file_size + page - 1) // page
        self.runs: list[tuple[int, int, str]] = []
        self.ok = False
        if not map_path or not os.path.exists(map_path):
            return
        with open(map_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        em = doc.get("expert_map", doc)
        files = em.get("files") or []
        if len(files) != 1 or int(files[0]["size"]) != file_size:
            return
        cat = bytearray(b"\xff") * self.n_pages  # 255 = padding / unowned

        def fill(off: int, n: int, c: int) -> None:
            a, b = off // page, (off + n + page - 1) // page
            cat[a:b] = bytes([c]) * (b - a)

        fill(0, int(files[0]["data_offset"]), 0)
        for x in em["experts"]:
            for r in x["ranges"]:
                fill(int(r["offset"]), int(r["n_bytes"]), 1 + int(x["layer"]))
        for r in em["nonexpert_ranges"]:  # non-expert wins shared pages (it is read every token)
            fill(int(r["offset"]), int(r["length"]), 200)
        names = {0: "header", 200: "nonexpert", 255: "unowned"}
        i = 0
        while i < self.n_pages:
            c = cat[i]
            j = i
            while j < self.n_pages and cat[j] == c:
                j += 1
            self.runs.append((i, j, names.get(c, f"expert_L{c - 1}")))
            i = j
        self.ok = True

    def summarize(self, vec: bytes) -> dict[str, Any]:
        out: dict[str, int] = {}
        for a, b, name in self.runs:
            out[name] = out.get(name, 0) + vec[a:b].count(1)
        experts = sum(v for k, v in out.items() if k.startswith("expert_L"))
        return {"page": self.page, "resident_pages_total": vec.count(1),
                "resident_bytes_total": vec.count(1) * self.page,
                "nonexpert_pages": out.get("nonexpert", 0), "expert_pages": experts,
                "header_pages": out.get("header", 0), "unowned_pages": out.get("unowned", 0),
                "expert_pages_by_layer": {k: v for k, v in sorted(out.items()) if k.startswith("expert_L")}}


# ----------------------------------------------------------------------------------------------
# FUSE throttle control
# ----------------------------------------------------------------------------------------------


class Fuse:
    def __init__(self, cfg: Mapping[str, Any], backing: str, out: Path) -> None:
        self.cfg = cfg
        self.backing = backing
        self.mnt = cfg.get("mountpoint", "/mnt/pigemma-thr")
        self.name = os.path.basename(backing)
        self.path = os.path.join(self.mnt, self.name)
        self.stats_path = os.path.join(self.mnt, ".stats")
        self.out = out
        self.proc: subprocess.Popen[str] | None = None
        self.ready: dict[str, Any] | None = None
        self.exit_stats: dict[str, Any] | None = None
        self._lines: list[str] = []

    def start(self, bin_path: str, rate_bps: float) -> None:
        os.makedirs(self.mnt, exist_ok=True)
        argv = [bin_path, "--backing", self.backing, "--mountpoint", self.mnt, "--rate-bps", f"{rate_bps:.0f}",
                "--op-latency-us", str(self.cfg.get("op_latency_us", 0)),
                "--max-readahead", str(self.cfg.get("max_readahead", 131072))]
        if self.cfg.get("max_read"):
            argv += ["--max-read", str(self.cfg["max_read"])]
        if self.cfg.get("trace", True):
            argv += ["--trace", str(self.out / "fuse_trace.csv")]
        self.argv = argv
        log = open(self.out / "fusethrottle.log", "w", encoding="utf-8")
        self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=log, text=True)

        def reader() -> None:
            assert self.proc is not None and self.proc.stdout is not None
            for line in self.proc.stdout:
                self._lines.append(line.strip())
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("event") == "ready":
                    self.ready = d
                elif d.get("event") == "exit":
                    self.exit_stats = d.get("stats")
        threading.Thread(target=reader, daemon=True).start()
        t0 = time.time()
        while time.time() - t0 < 30:
            if self.proc.poll() is not None:
                raise RuntimeError(f"fusethrottle exited with {self.proc.returncode}; see fusethrottle.log")
            if self.ready and os.path.exists(self.path) and self.stats() is not None:
                return
            time.sleep(0.05)
        raise RuntimeError("fusethrottle did not become ready within 30 s")

    def stats(self) -> dict[str, Any] | None:
        try:
            with open(self.stats_path, "rb", buffering=0) as fh:
                return json.loads(fh.read(4096).decode("utf-8", "replace").strip())
        except (OSError, ValueError):
            return None

    def stop(self) -> None:
        if self.proc is None:
            return
        subprocess.run(["fusermount3", "-u", self.mnt], capture_output=True, timeout=30)
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        time.sleep(0.2)


# ----------------------------------------------------------------------------------------------
# counters snapshot + deltas (io.csv)
# ----------------------------------------------------------------------------------------------


class Probe:
    """Collects one consistent snapshot of every counter the cell records."""

    def __init__(self, cg: Cgroup, fuse: Fuse | None, devs: Sequence[str]) -> None:
        self.cg = cg
        self.fuse = fuse
        self.devs = list(devs)
        self.pids: dict[str, int | None] = {"server": None, "pinner": None, "fuse": None, "self": os.getpid()}

    def snap(self) -> dict[str, Any]:
        t = time.time()
        s: dict[str, Any] = {"t": t, "mono": time.monotonic()}
        s["cg"] = self.cg.memory_brief()
        s["cg_cpu_ns"] = self.cg.cpu_usage_ns()
        s["cg_io"] = self.cg.io_stat()
        s["proc"] = {k: proc_io(p) for k, p in self.pids.items() if p}
        s["status"] = {k: proc_status(p) for k, p in self.pids.items() if p}
        s["fuse"] = self.fuse.stats() if self.fuse and self.fuse.proc else None
        s["disk"] = diskstats(self.devs)
        s["vm"] = vmstat()
        s["cpu"] = cpu_totals()
        return s


IO_FIELDS = [
    "phase", "rid", "t_start", "wall_s",
    "fuse_backing_bytes", "fuse_requests", "fuse_queue_wait_s", "fuse_sleep_s",
    "fuse_daemon_read_bytes", "cg_io_rbytes",
    "disk_read_bytes", "disk_devices",
    "server_read_bytes", "server_rchar", "server_majflt", "server_minflt",
    "pinner_read_bytes", "pinner_rchar", "pinner_majflt",
    "cg_pgmajfault", "vm_pgmajfault", "vm_pgscan", "vm_pgsteal", "vm_pswpin", "vm_pswpout", "vm_refault_file",
    "cg_usage_end", "cg_file_end", "cg_anon_end", "cg_unevictable_end", "cg_failcnt_delta", "cg_oom_kill_delta",
    "cg_cpu_s", "vm_cpu_busy_s", "vm_other_cpu_s",
    # added 2026-10-05 22:45 (appended, older columns unchanged): cgroup reclaim / pressure
    "cg_pgpgin", "cg_pgpgout", "cg_pgscan", "cg_pgsteal", "cg_refault_file",
    "cg_psi_some_s", "cg_psi_full_s", "cg_swap_end", "cg_ev_high_delta",
    "device_read_bytes", "device_read_source",
]


def _d(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None, k: str) -> int | None:
    if not a or not b or a.get(k) is None or b.get(k) is None:
        return None
    return int(b[k]) - int(a[k])


def _dp(pa: Mapping[str, Any], pb: Mapping[str, Any], name: str, key: str) -> int | None:
    """Delta of a per-process counter over a phase. A process that started INSIDE the phase (the
    pinner in `setup`, llama-server in `load`) has no start snapshot: its counter began at 0."""
    b = pb.get(name) or {}
    if b.get(key) is None:
        return None
    a = pa.get(name) or {}
    return int(b[key]) - int(a.get(key) or 0)


def _us(v: int | None) -> float | None:
    return None if v is None else round(v / 1e6, 6)


def io_row(phase: str, rid: str, a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    pa, pb = a.get("proc", {}), b.get("proc", {})
    fa, fb = a.get("fuse") or {}, b.get("fuse") or {}
    va, vb = a.get("vm", {}), b.get("vm", {})
    ca, cb = a.get("cg", {}), b.get("cg", {})
    disk = None
    if a.get("disk") and b.get("disk"):
        disk = sum(b["disk"].get(k, 0) - a["disk"].get(k, 0) for k in b["disk"])
    cgio = None
    if a.get("cg_io") and b.get("cg_io"):
        cgio = sum((b["cg_io"].get(k, {}).get("rbytes", 0) - a["cg_io"].get(k, {}).get("rbytes", 0)) for k in b["cg_io"])
    cpu_busy = vm_other = None
    if a.get("cpu") and b.get("cpu"):
        hz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100  # USER_HZ; Windows: tests only
        cpu_busy = (b["cpu"][0] - a["cpu"][0]) / hz
        if a.get("cg_cpu_ns") is not None and b.get("cg_cpu_ns") is not None:
            vm_other = cpu_busy - (b["cg_cpu_ns"] - a["cg_cpu_ns"]) / 1e9

    def ssum(*keys: str) -> int | None:
        vals = [_d(va, vb, k) for k in keys]
        vals2 = [v for v in vals if v is not None]
        return sum(vals2) if vals2 else None

    # ONE physical-read figure per phase, from the most exact source available (labelled):
    # the emulated device's own counter (FUSE mode) > the cgroup's io.stat (v2, real device) >
    # the cell's processes (/proc/<pid>/io read_bytes: page-fault read-ins of mmap'ed model pages
    # are counted there) > the whole block device (includes unrelated traffic).
    proc_rb = [_dp(pa, pb, k, "read_bytes") for k in ("server", "pinner")]
    proc_rb = [v for v in proc_rb if v is not None]
    if fa and fb and _d(fa, fb, "backing_bytes") is not None:
        dev_bytes, dev_src = _d(fa, fb, "backing_bytes"), "fuse_backing_bytes (emulated device; exact)"
    elif cgio is not None:
        dev_bytes, dev_src = cgio, "cgroup io.stat rbytes"
    elif proc_rb:
        dev_bytes, dev_src = sum(proc_rb), "/proc/<pid>/io read_bytes of server+pinner"
    elif disk is not None:
        dev_bytes, dev_src = disk, "/proc/diskstats (whole device; includes unrelated traffic)"
    else:
        dev_bytes, dev_src = None, None

    return {
        "phase": phase, "rid": rid, "t_start": round(a["t"], 3), "wall_s": round(b["t"] - a["t"], 4),
        "fuse_backing_bytes": _d(fa, fb, "backing_bytes"), "fuse_requests": _d(fa, fb, "requests"),
        "fuse_queue_wait_s": (_d(fa, fb, "queue_wait_ns") or 0) / 1e9 if fa and fb else None,
        "fuse_sleep_s": (_d(fa, fb, "sleep_ns") or 0) / 1e9 if fa and fb else None,
        "fuse_daemon_read_bytes": _dp(pa, pb, "fuse", "read_bytes"),
        "cg_io_rbytes": cgio,
        "disk_read_bytes": disk, "disk_devices": "+".join(sorted((b.get("disk") or {}).keys())) or None,
        "server_read_bytes": _dp(pa, pb, "server", "read_bytes"),
        "server_rchar": _dp(pa, pb, "server", "rchar"),
        "server_majflt": _dp(pa, pb, "server", "majflt"),
        "server_minflt": _dp(pa, pb, "server", "minflt"),
        "pinner_read_bytes": _dp(pa, pb, "pinner", "read_bytes"),
        "pinner_rchar": _dp(pa, pb, "pinner", "rchar"),
        "pinner_majflt": _dp(pa, pb, "pinner", "majflt"),
        "cg_pgmajfault": _d(ca, cb, "pgmajfault"),
        "vm_pgmajfault": _d(va, vb, "pgmajfault"),
        "vm_pgscan": ssum("pgscan_kswapd", "pgscan_direct"),
        "vm_pgsteal": ssum("pgsteal_kswapd", "pgsteal_direct"),
        "vm_pswpin": _d(va, vb, "pswpin"), "vm_pswpout": _d(va, vb, "pswpout"),
        "vm_refault_file": _d(va, vb, "workingset_refault_file") if "workingset_refault_file" in vb else _d(va, vb, "workingset_refault"),
        "cg_usage_end": cb.get("usage"), "cg_file_end": cb.get("file"), "cg_anon_end": cb.get("anon"),
        "cg_unevictable_end": cb.get("unevictable"),
        "cg_failcnt_delta": _d(ca, cb, "ev_max"), "cg_oom_kill_delta": _d(ca, cb, "ev_oom_kill"),
        "cg_cpu_s": round((b["cg_cpu_ns"] - a["cg_cpu_ns"]) / 1e9, 3) if a.get("cg_cpu_ns") is not None and b.get("cg_cpu_ns") is not None else None,
        "vm_cpu_busy_s": round(cpu_busy, 3) if cpu_busy is not None else None,
        "vm_other_cpu_s": round(vm_other, 3) if vm_other is not None else None,
        "cg_pgpgin": _d(ca, cb, "pgpgin"), "cg_pgpgout": _d(ca, cb, "pgpgout"),
        "cg_pgscan": _d(ca, cb, "pgscan"), "cg_pgsteal": _d(ca, cb, "pgsteal"),
        "cg_refault_file": _d(ca, cb, "workingset_refault_file"),
        "cg_psi_some_s": _us(_d(ca, cb, "psi_some_us")), "cg_psi_full_s": _us(_d(ca, cb, "psi_full_us")),
        "cg_swap_end": cb.get("swap"), "cg_ev_high_delta": _d(ca, cb, "ev_high"),
        "device_read_bytes": dev_bytes, "device_read_source": dev_src,
    }


SAMPLE_FIELDS = ["t_s", "phase", "cg_usage", "cg_peak", "cg_file", "cg_anon", "cg_mapped", "cg_unevictable",
                 "cg_kernel", "cg_swap", "cg_pgmajfault", "cg_failcnt",
                 "server_rss_kb", "server_anon_kb", "server_file_kb", "server_majflt", "server_read_bytes",
                 "pinner_rss_kb", "pinner_vmlck_kb", "pinner_majflt",
                 "fuse_rss_kb", "fuse_backing_bytes", "fuse_requests", "fuse_queue_wait_ns",
                 "disk_read_bytes", "vm_pgmajfault", "vm_pgscan", "cg_cpu_ns", "vm_cpu_busy", "vm_cpu_total",
                 "cg_pgpgin", "cg_pgpgout", "temp_mC"]


class Sampler(threading.Thread):
    """Periodic samples -> samples.csv. With ``abort_below`` (bytes) it also enforces the Pi safety
    rule: when the host's MemAvailable drops below it, ``on_abort`` is called once (it stops the
    cell's own llama-server) and ``abort_reason`` is set."""

    def __init__(self, probe: Probe, path: Path, interval: float, phase_ref: list[str],
                 abort_below: int | None = None, on_abort: Callable[[str], None] | None = None) -> None:
        super().__init__(daemon=True)
        self.probe, self.path, self.interval, self.phase_ref = probe, path, interval, phase_ref
        self.stop_ev = threading.Event()
        self.t0 = time.time()
        self.max_usage_by_phase: dict[str, int] = {}
        self.abort_below = abort_below
        self.on_abort = on_abort
        self.abort_reason: str | None = None
        self.min_mem_available: int | None = None

    def run(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(",".join(SAMPLE_FIELDS) + "\n")
            while not self.stop_ev.is_set():
                try:
                    s = self.probe.snap()
                    cg, st, pr, fu = s["cg"], s["status"], s["proc"], s.get("fuse") or {}
                    ph = self.phase_ref[0]
                    if cg.get("usage") is not None:
                        self.max_usage_by_phase[ph] = max(self.max_usage_by_phase.get(ph, 0), cg["usage"])
                    vm = s["vm"]
                    row = [round(s["t"] - self.t0, 2), ph, cg.get("usage"), cg.get("peak"), cg.get("file"), cg.get("anon"),
                           cg.get("file_mapped"), cg.get("unevictable"), cg.get("kernel"), cg.get("swap"),
                           cg.get("pgmajfault"), cg.get("ev_max"),
                           st.get("server", {}).get("VmRSS"), st.get("server", {}).get("RssAnon"),
                           st.get("server", {}).get("RssFile"), pr.get("server", {}).get("majflt"),
                           pr.get("server", {}).get("read_bytes"),
                           st.get("pinner", {}).get("VmRSS"), st.get("pinner", {}).get("VmLck"),
                           pr.get("pinner", {}).get("majflt"),
                           st.get("fuse", {}).get("VmRSS"), fu.get("backing_bytes"), fu.get("requests"),
                           fu.get("queue_wait_ns"), sum((s.get("disk") or {}).values()) if s.get("disk") else None,
                           vm.get("pgmajfault"), (vm.get("pgscan_kswapd", 0) + vm.get("pgscan_direct", 0)) if vm else None,
                           s.get("cg_cpu_ns"), (s.get("cpu") or (None, None))[0], (s.get("cpu") or (None, None))[1],
                           cg.get("pgpgin"), cg.get("pgpgout"), thermal_mC()]
                    fh.write(",".join("" if v is None else str(v) for v in row) + "\n")
                    fh.flush()
                    if self.abort_below is not None:
                        ma = (meminfo().get("MemAvailable") or 0) * 1024
                        self.min_mem_available = ma if self.min_mem_available is None else min(self.min_mem_available, ma)
                        if ma < self.abort_below and self.abort_reason is None:
                            self.abort_reason = f"MemAvailable {ma / 1e9:.2f} GB < {self.abort_below / 1e9:.2f} GB"
                            if self.on_abort:
                                self.on_abort(self.abort_reason)
                except Exception as e:  # never let sampling kill the cell
                    fh.write(f"# sampler error: {e!r}\n")
                self.stop_ev.wait(self.interval)


# ----------------------------------------------------------------------------------------------
# llama-server + streaming client
# ----------------------------------------------------------------------------------------------


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# Canonical llama-server flags of a controlled cell, in command-line order. The templates are the
# a7b94df spelling; an engine profile may replace one (list of argv words, "{name}" placeholders)
# or drop it ([] or None) through server["flag_overrides"] when that build spells it differently or
# lacks it. The effective argv and the overrides used are recorded in config.json "effective".
SERVER_FLAGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("model", ("-m", "{model}")),
    ("no_repack", ("-nr",)),
    ("threads", ("-t", "{threads}", "-tb", "{threads}")),
    ("ctx", ("-c", "{ctx}")),
    ("batch", ("-b", "{batch}", "-ub", "{ubatch}")),
    ("kv_type", ("-ctk", "{ctk}", "-ctv", "{ctv}")),
    ("flash_attn", ("-fa", "{flash_attn}")),
    ("parallel", ("-np", "{parallel}")),
    ("cache_ram", ("--cache-ram", "{cache_ram_mib}")),
    ("ctx_checkpoints", ("--ctx-checkpoints", "{ctx_checkpoints}")),
    ("seed", ("--seed", "{seed}")),
    ("verbosity", ("-lv", "{verbosity}")),
    ("listen", ("--host", "127.0.0.1", "--port", "{port}")),
    ("no_webui", ("--no-webui",)),
    ("warmup_on", ("--warmup",)),
    ("warmup_off", ("--no-warmup",)),
)
# never allowed in a controlled cell (brief section 3: MTP and every speculative method OFF)
SPECULATIVE_FLAGS = ("-md", "--model-draft", "--spec-type", "--draft", "--spec-draft-n-max", "--spec-draft-n-min",
                     "-mdr", "--model-draft-reasoning", "--lookup-cache-static", "--lookup-cache-dynamic",
                     "--spec-replace", "-hfd", "-hfrd", "--hf-repo-draft")


def server_argv(scfg: Mapping[str, Any], model_path: str, port: int) -> list[str]:
    vals = {"model": model_path, "port": port, "threads": scfg.get("threads", 4), "ctx": scfg.get("ctx", 4096),
            "batch": scfg.get("batch", 2048), "ubatch": scfg.get("ubatch", 512),
            "ctk": scfg.get("ctk", "f16"), "ctv": scfg.get("ctv", "f16"),
            "flash_attn": scfg.get("flash_attn", "auto"), "parallel": scfg.get("parallel", 1),
            "cache_ram_mib": scfg.get("cache_ram_mib", 0), "ctx_checkpoints": scfg.get("ctx_checkpoints", 32),
            "seed": scfg.get("seed", 0), "verbosity": scfg.get("verbosity", 4)}
    over = dict(scfg.get("flag_overrides") or {})
    unknown = sorted(set(over) - {k for k, _ in SERVER_FLAGS})
    if unknown:
        raise ValueError(f"unknown flag_overrides keys {unknown}; known: {[k for k, _ in SERVER_FLAGS]}")
    if "model" in over and not over["model"]:
        raise ValueError("flag_overrides may respell the model flag but not drop it")
    warm = bool(scfg.get("warmup", True))
    argv = [scfg["bin"]]
    for key, tmpl in SERVER_FLAGS:
        if (key == "warmup_on" and not warm) or (key == "warmup_off" and warm):
            continue
        words = over[key] if key in over else tmpl
        argv += [str(w).format(**vals) for w in (words or [])]
    argv += [str(x) for x in scfg.get("extra", [])]
    nice = int(scfg.get("nice", 0))
    if nice:
        argv = ["nice", "-n", str(nice)] + argv
    for bad in SPECULATIVE_FLAGS:
        if bad in argv:
            raise ValueError(f"speculative decoding flag {bad} is not allowed in a controlled cell")
    return argv


def http_json(port: int, path: str, body: Mapping[str, Any] | None = None, timeout: float = 30) -> Any:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def parse_server_log(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"buffers_mib": {}}
    for m in re.finditer(r"(\S+)\s+(model|KV|compute|output|RS) buffer size =\s*([\d.]+) MiB", text):
        key = f"{m.group(1)} {m.group(2)}"
        out["buffers_mib"][key] = out["buffers_mib"].get(key, 0.0) + float(m.group(3))
    out["buffer_lines_found"] = bool(out["buffers_mib"])
    out["repack_present"] = "CPU_REPACK" in text
    m = re.search(r"flash_attn\s*=\s*([^\n]+)", text)
    out["flash_attn_effective"] = m.group(1).strip() if m else None
    m = re.search(r"context checkpoints enabled, max = (\d+)", text)
    out["ctx_checkpoints_max"] = int(m.group(1)) if m else None
    m = re.search(r"build:?\s*(\d+)\s*\(([0-9a-f]+)\)", text)
    if m:
        out["build"] = {"number": int(m.group(1)), "commit": m.group(2)}
    out["n_forcing_full_reprocess"] = text.count("forcing full prompt re-processing")
    out["n_checkpoint_restores"] = text.count("restored context checkpoint")
    out["n_checkpoints_created"] = text.count("created context checkpoint")
    out["spec_mentions"] = len(re.findall(r"speculative|draft model", text, re.I))
    out["spec_disabled_confirmed"] = "no implementations specified for speculative decoding" in text
    return out


def stream_completion(port: int, body: Mapping[str, Any], on_first: Callable[[], None],
                      timeout: float = 7200, path: str = "/completion") -> dict[str, Any]:
    """POST /completion stream=true return_tokens=true; client-side timestamps per SSE event.

    Engines whose server does not return token ids (no ``tokens`` array in the events) are timed
    per content-bearing SSE event instead; ``timestamps_per`` says which ("token" / "sse_event")."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(dict(body, stream=True, return_tokens=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    events: list[list[Any]] = []          # [t_rel_s, [token ids], content]
    token_times: list[float] = []
    token_ids: list[int] = []
    event_times: list[float] = []         # content-bearing events (fallback when no ids are returned)
    text: list[str] = []
    final: dict[str, Any] = {}
    first = True
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
            tnow = time.monotonic() - t0
            toks = [int(x) for x in (d.get("tokens") or [])]
            if (toks or d.get("content")) and first:
                on_first()
                first = False
            if toks or d.get("content"):
                events.append([round(tnow, 6), toks, d.get("content", "")])
                event_times.append(tnow)
            for tk in toks:
                token_ids.append(tk)
                token_times.append(tnow)
            text.append(d.get("content", ""))
            if d.get("stop"):
                final = d
                break
    t_end = time.monotonic() - t0
    per = "token"
    if not token_ids and event_times:
        token_times, per = event_times, "sse_event"
    return {"events": events, "token_ids": token_ids, "token_times": token_times, "text": "".join(text),
            "final": final, "t_end": t_end, "timestamps_per": per}


def first_text_diff(a: str, b: str) -> int | None:
    """Index of the first differing character, len(shorter) for a strict prefix, None if equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


# ----------------------------------------------------------------------------------------------
# the cell
# ----------------------------------------------------------------------------------------------


class Cell:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.out = Path(cfg["out_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "outputs").mkdir(exist_ok=True)
        (self.out / "residency").mkdir(exist_ok=True)
        self.cg = Cgroup()
        self.phase = ["init"]
        self.eff: dict[str, Any] = {"started": now_iso(), "phases": {}}
        self.fuse: Fuse | None = None
        self.server: subprocess.Popen[bytes] | None = None
        self.pinner: subprocess.Popen[str] | None = None
        self.pinner_summary: dict[str, Any] | None = None
        self.requests_fh = open(self.out / "requests.jsonl", "w", encoding="utf-8")
        self.io_fh = open(self.out / "io.csv", "w", encoding="utf-8", newline="")
        self.io_csv = csv.writer(self.io_fh, lineterminator="\n")
        self.io_csv.writerow(IO_FIELDS)
        self.cg_phases: list[dict[str, Any]] = []
        self.cats: Categories | None = None
        self.served_path = ""
        self.template_api: bool | None = None  # None = not tried, False = endpoint unavailable
        self.template_checks: list[dict[str, Any]] = []
        self.reuse_checks: list[dict[str, Any]] = []

    # -- bookkeeping ---------------------------------------------------------------------------
    def log(self, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def io(self, phase: str, rid: str, a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
        row = io_row(phase, rid, a, b)
        self.io_csv.writerow(["" if row[k] is None else row[k] for k in IO_FIELDS])
        self.io_fh.flush()
        return row

    def cg_mark(self, tag: str) -> dict[str, Any]:
        m = self.cg.memory()
        m["tag"] = tag
        m["t"] = time.time()
        m["procs"] = {k: {"status": proc_status(p), "smaps_rollup": smaps_rollup(p)}
                      for k, p in self.probe.pids.items() if p}
        self.cg_phases.append(m)
        return m

    def residency(self, tag: str) -> dict[str, Any] | None:
        mode = self.cfg.get("sampling", {}).get("residency", "per_request")
        if mode == "off" or not self.served_path:
            return None
        t0 = time.monotonic()
        vec = residency_vector(self.served_path)
        if vec is None:
            return {"tag": tag, "error": "mincore failed"}
        summ = self.cats.summarize(vec) if self.cats and self.cats.ok else {
            "page": os.sysconf("SC_PAGE_SIZE"), "resident_pages_total": vec.count(1)}
        summ["tag"] = tag
        summ["mincore_s"] = round(time.monotonic() - t0, 4)
        with open(self.out / "residency" / f"{tag}.bin.z", "wb") as fh:
            fh.write(zlib.compress(vec, 6))
        with open(self.out / "residency" / "index.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(summ) + "\n")
        return summ

    # -- steps ---------------------------------------------------------------------------------
    def preflight(self) -> None:
        mem = self.cg.memory()
        mi = meminfo()
        self.eff["cgroup_describe"] = self.cg.describe()
        self.eff["cgroup_limits"] = {k: mem.get(k) for k in ("version", "path", "max", "high", "soft_limit",
                                                             "memsw_limit", "swap_max", "swappiness")}
        self.eff["meminfo_start_kB"] = {k: mi.get(k) for k in ("MemTotal", "MemAvailable", "Cached", "SwapTotal", "SwapFree")}
        budget = int(self.cfg["budget_bytes"])
        lim = mem.get("max")
        self.eff["budget_check"] = {"budget_bytes": budget, "cgroup_limit_bytes": lim,
                                    "limit_matches_budget": lim == budget}
        if lim != budget:
            raise RuntimeError(f"cgroup memory limit {lim} != budget {budget}: not a valid cell")
        swap_ok = (mem.get("memsw_limit") == budget) if mem.get("version") == 1 else (mem.get("swap_max") in (0,))
        self.eff["swap_disabled"] = swap_ok
        if not swap_ok and not self.cfg.get("allow_swap", False):
            raise RuntimeError(f"swap is not disabled for this cgroup (memsw/swap.max = "
                               f"{mem.get('memsw_limit') if mem.get('version') == 1 else mem.get('swap_max')})")
        need = budget + int(self.cfg.get("host_headroom_bytes", 2 * 1024**3))
        avail = (mi.get("MemAvailable") or 0) * 1024
        self.eff["host_mem_available_bytes"] = avail
        if avail < need:
            raise RuntimeError(f"MemAvailable {avail/1e9:.1f} GB < budget + headroom {need/1e9:.1f} GB")
        self.eff["page_size"] = os.sysconf("SC_PAGE_SIZE")
        self.eff["nproc"] = os.cpu_count()
        try:
            self.eff["cpus_allowed"] = sorted(os.sched_getaffinity(0))
        except AttributeError:
            pass
        self.eff["uname"] = list(os.uname())
        lim_line = [l for l in (read_text("/proc/self/limits") or "").splitlines() if l.startswith("Max locked memory")]
        self.eff["memlock_limit_line"] = lim_line[0] if lim_line else None
        caps = [l for l in (read_text("/proc/self/status") or "").splitlines() if l.startswith("CapEff")]
        self.eff["cap_eff"] = caps[0].split()[1] if caps else None
        self.eff["thermal_start"] = {"temp_mC": thermal_mC(), "throttled": pi_throttled(),
                                     "note": "None = no sensor / no vcgencmd (Docker Desktop VM)"}
        self.eff["engine"] = self.engine_identity()

    def engine_identity(self) -> dict[str, Any]:
        """What binary this cell really runs: path, size, sha256 and its own --version text."""
        scfg = self.cfg["server"]
        ecfg = dict(self.cfg.get("engine") or {})
        b = scfg["bin"]
        ident: dict[str, Any] = {"id": ecfg.get("id", "a7b94df"), "revision": ecfg.get("revision"),
                                 "parent": ecfg.get("parent"), "bin": b,
                                 "flag_overrides": scfg.get("flag_overrides") or {},
                                 "extra": list(scfg.get("extra", [])), "env": scfg.get("env") or {}}
        if not os.path.exists(b):
            raise RuntimeError(f"engine binary not found in the cell: {b}")
        ident["bin_size"] = os.path.getsize(b)
        ident["bin_sha256"] = sha256_file(b)
        try:
            r = subprocess.run([b, "--version"], capture_output=True, text=True, timeout=60,
                               env=dict(os.environ, **{str(k): str(v) for k, v in (scfg.get("env") or {}).items()}))
            ident["version_text"] = ((r.stdout or "") + (r.stderr or "")).strip()[-600:]
            ident["version_exit"] = r.returncode
        except (OSError, subprocess.SubprocessError) as e:
            ident["version_text"] = None
            ident["version_error"] = repr(e)
        return ident

    def storage_up(self) -> None:
        m = self.cfg["model"]
        st = self.cfg.get("storage", {})
        backing = m["backing_path"]
        mode = st.get("mode", "fuse-throttled")
        self.eff["storage"] = {"mode": mode, "backing": backing, "backing_device": block_dev_of(backing)}
        if mode == "direct":
            self.served_path = backing
        elif mode in ("fuse-throttled", "fuse-unthrottled"):
            rate = float(st.get("rate_bps", 91.6e6)) if mode == "fuse-throttled" else 0.0
            self.fuse = Fuse(st, backing, self.out)
            self.probe.fuse = self.fuse
            self.fuse.start(st.get("bin", "/work/tools/build-linux-x86/bin/fusethrottle"), rate)
            self.probe.pids["fuse"] = self.fuse.proc.pid if self.fuse.proc else None
            self.served_path = self.fuse.path
            self.eff["storage"].update({"fuse_argv": self.fuse.argv, "fuse_ready": self.fuse.ready,
                                        "fuse_stats_at_start": self.fuse.stats(), "rate_bps": rate})
        else:
            raise ValueError(f"unknown storage.mode {mode!r}")
        size = os.path.getsize(self.served_path)
        if size != int(m.get("size", size)):
            raise RuntimeError(f"served file size {size} != config model.size {m.get('size')}")
        self.cats = Categories(m.get("map"), os.sysconf("SC_PAGE_SIZE"), size)
        self.eff["storage"]["served_path"] = self.served_path
        self.eff["storage"]["categories_from_map"] = self.cats.ok

    def evict(self) -> None:
        """Cold start: targeted posix_fadvise(DONTNEED) of the model file (backing and served)."""
        before = residency_vector(self.served_path)
        paths = [self.cfg["model"]["backing_path"]]
        if self.served_path not in paths:
            paths.append(self.served_path)
        for p in paths:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        after = residency_vector(self.served_path)
        back = residency_vector(self.cfg["model"]["backing_path"]) if self.served_path != self.cfg["model"]["backing_path"] else after
        page = os.sysconf("SC_PAGE_SIZE")
        res = {"served_resident_bytes_before": before.count(1) * page if before is not None else None,
               "served_resident_bytes_after": after.count(1) * page if after is not None else None,
               "backing_resident_bytes_after": back.count(1) * page if back is not None else None}
        self.eff["evict"] = res
        cold_max = int(self.cfg.get("cold_max_resident_bytes", 64 * 1024**2))
        vals = [v for v in (res["served_resident_bytes_after"], res["backing_resident_bytes_after"]) if v is not None]
        if not vals:
            raise RuntimeError("cold state cannot be verified: mincore failed on the model file")
        worst = max(vals)
        res["cold_ok"] = worst <= cold_max
        if not res["cold_ok"]:
            raise RuntimeError(f"cold state not reached: {worst} B of the model still cached after DONTNEED")

    def setup_method(self) -> None:
        method = self.cfg["method"]
        pcfg = self.cfg.get("pinner") or {}
        self.eff["setup"] = {"method": method}
        if method == "A":
            return
        ranges = pcfg.get("ranges")
        if method == "B":
            ranges = str(self.out / "empty.ranges.txt")
            with open(ranges, "w", encoding="ascii") as fh:
                fh.write(f"# pigemma-ranges v1 gguf={os.path.basename(self.served_path)} "
                         f"size={os.path.getsize(self.served_path)} page={os.sysconf('SC_PAGE_SIZE')} budget=0\n")
        if not ranges or not os.path.exists(ranges):
            raise RuntimeError(f"method {method} needs pinner.ranges (got {ranges!r})")
        argv = [pcfg.get("bin", "/work/tools/build-linux-x86/bin/pinner"), "--gguf", self.served_path,
                "--ranges", ranges, "--json", "--verify", "--pidfile", "/tmp/pigemma-pinner.pid"]
        if method == "B":
            argv.append("--allow-empty")
        if pcfg.get("max_bytes"):
            argv += ["--max-bytes", str(int(pcfg["max_bytes"]))]
        argv += [str(x) for x in pcfg.get("extra", [])]
        self.eff["setup"]["pinner_argv"] = argv
        self.eff["setup"]["ranges_sha256"] = sha256_file(ranges)
        log = open(self.out / "pinner.log", "w", encoding="utf-8")
        self.pinner = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=log, text=True)
        self.probe.pids["pinner"] = self.pinner.pid
        assert self.pinner.stdout is not None
        line = self.pinner.stdout.readline()  # the daemon prints its summary after locking, then holds
        try:
            summ = json.loads(line)
        except ValueError:
            rc = self.pinner.poll()
            raise RuntimeError(f"pinner gave no JSON summary (exit {rc}); see pinner.log")
        self.pinner_summary = summ
        self.eff["setup"]["pinner_summary"] = summ
        if summ.get("status") != "ok":
            raise RuntimeError(f"pinner status {summ.get('status')}: {summ.get('reason')}")
        if method in ("C", "D") and int(summ.get("locked_bytes") or 0) != int(summ.get("requested_bytes") or -1):
            raise RuntimeError("pinner locked != requested")

    def start_server(self) -> int:
        scfg = self.cfg["server"]
        port = int(scfg.get("port") or free_port())
        argv = server_argv(scfg, self.served_path, port)
        self.eff["server_argv"] = argv
        self.server_log = open(self.out / "server.log", "wb")
        env = dict(os.environ, **{str(k): str(v) for k, v in (scfg.get("env") or {}).items()})
        self.server = subprocess.Popen(argv, stdout=self.server_log, stderr=subprocess.STDOUT, env=env)
        self.probe.pids["server"] = self.server.pid
        t_limit = float(scfg.get("load_timeout_s", 3600))
        t0 = time.monotonic()
        while True:
            if self.server.poll() is not None:
                raise RuntimeError(f"llama-server exited during load ({self.server.returncode}); see server.log")
            try:
                if http_json(port, "/health", timeout=5).get("status") == "ok":
                    break
            except (urllib.error.URLError, OSError, ValueError):
                pass
            if time.monotonic() - t0 > t_limit:
                raise RuntimeError("llama-server load timeout")
            time.sleep(0.25)
        self.eff["server_ready_s"] = round(time.monotonic() - t0, 3)
        try:
            props = http_json(port, "/props", timeout=10)
            gs = props.get("default_generation_settings", {})
            self.eff["server_props"] = {"n_ctx": gs.get("n_ctx"), "build_info": props.get("build_info"),
                                        "total_slots": props.get("total_slots"),
                                        "model_path": props.get("model_path")}
        except Exception as e:  # noqa: BLE001
            self.eff["server_props"] = {"error": repr(e)}
        return port

    def template_check(self, port: int, rid: str, messages: Sequence[Mapping[str, Any]] | None,
                       prompt: str) -> dict[str, Any] | None:
        """Runtime evidence that the prompt sent to /completion equals the GGUF's OWN chat template
        (thinking disabled) as the server renders it: POST /apply-template and compare. Not timed
        (before the request's first snapshot); no model weights are touched."""
        if messages is None or self.template_api is False or not self.cfg.get("request", {}).get("template_check", True):
            return None
        try:
            r = http_json(port, "/apply-template", {"messages": list(messages),
                                                     "chat_template_kwargs": {"enable_thinking": False}}, timeout=30)
            srv = r.get("prompt") if isinstance(r, Mapping) else None
            if not isinstance(srv, str):
                raise ValueError("no prompt in the /apply-template reply")
        except Exception as e:  # noqa: BLE001 - other engines may not have the endpoint
            self.template_api = False
            self.eff["template_check_unavailable"] = repr(e)[:300]
            return {"rid": rid, "available": False}
        self.template_api = True
        cand = srv[5:] if srv.startswith("<bos>") and not prompt.startswith("<bos>") else srv
        fd = first_text_diff(cand, prompt)
        res = {"rid": rid, "available": True, "match": fd is None, "first_diff": fd,
               "server_len": len(cand), "ours_len": len(prompt), "server_had_bos": srv.startswith("<bos>")}
        if fd is not None:
            res["server_at_diff"] = cand[max(0, fd - 40):fd + 40]
            res["ours_at_diff"] = prompt[max(0, fd - 40):fd + 40]
        self.template_checks.append(res)
        return res

    def expected_reuse(self, port: int, prev_prompt: str, prompt: str) -> dict[str, Any]:
        """What a follow-up CAN reuse: the tokens of the longest common text prefix of the previous
        and the new prompt (BOS included), counted by the server's own tokenizer. Not timed."""
        cp = os.path.commonprefix([prev_prompt, prompt])
        out: dict[str, Any] = {"common_prefix_chars": len(cp), "prev_prompt_chars": len(prev_prompt),
                               "expected_reusable_prefix_tokens": None}
        try:
            tk = http_json(port, "/tokenize", {"content": cp, "add_special": True, "parse_special": True}, timeout=60)
            if isinstance(tk, Mapping) and isinstance(tk.get("tokens"), list):
                out["expected_reusable_prefix_tokens"] = len(tk["tokens"])
        except Exception as e:  # noqa: BLE001
            out["error"] = repr(e)[:200]
        return out

    def request(self, port: int, kind: str, rid: str, prompt: str, n_predict: int, cache_prompt: bool,
                meta: Mapping[str, Any], messages: Sequence[Mapping[str, Any]] | None = None,
                prev_prompt: str | None = None) -> dict[str, Any]:
        rq = self.cfg.get("request", {})
        api = (self.cfg.get("engine") or {}).get("api") or {}
        body = {"prompt": prompt, "n_predict": int(n_predict), "temperature": float(rq.get("temperature", 0.0)),
                "seed": int(rq.get("seed", 0)), "cache_prompt": bool(cache_prompt), "id_slot": 0}
        for k in ("top_k", "top_p", "min_p", "repeat_penalty"):
            if k in rq:
                body[k] = rq[k]
        body.update(api.get("extra_body") or {})
        path = api.get("completion_path", "/completion")
        tcheck = self.template_check(port, rid, messages, prompt)
        reuse = self.expected_reuse(port, prev_prompt, prompt) if (cache_prompt and prev_prompt) else None
        self.phase[0] = f"{kind}:{rid}"
        s0 = self.probe.snap()
        mark: dict[str, Any] = {}
        res = stream_completion(port, body, lambda: mark.setdefault("s1", self.probe.snap()), path=path)
        s2 = self.probe.snap()
        s1 = mark.get("s1") or s2
        fin = res["final"]
        tim = fin.get("timings", {}) if fin else {}
        tt = res["token_times"]
        ids_source = "server (return_tokens)"
        if not res["token_ids"]:
            # engine without return_tokens: ids re-derived from the text (deterministic, good for
            # comparing two runs' texts; NOT the sampled ids) - labelled as such
            ids_source = "unavailable"
            try:
                tk = http_json(port, "/tokenize", {"content": res["text"], "add_special": False,
                                                    "parse_special": True}, timeout=60)
                if isinstance(tk, Mapping) and isinstance(tk.get("tokens"), list):
                    res["token_ids"] = [int(x if not isinstance(x, Mapping) else x.get("id")) for x in tk["tokens"]]
                    ids_source = "re-tokenized from the output text (/tokenize), not the sampled ids"
            except Exception:  # noqa: BLE001
                pass
        n_out = len(tt) if res["timestamps_per"] == "token" else int(tim.get("predicted_n") or len(tt))
        if reuse is not None:
            cn, exp = tim.get("cache_n"), reuse.get("expected_reusable_prefix_tokens")
            reuse.update({"rid": rid, "cache_n": cn, "prompt_n_new": tim.get("prompt_n"),
                          "cache_n_minus_expected": (cn - exp) if cn is not None and exp is not None else None,
                          # a7b94df restores a context checkpoint for SWA models and re-evaluates its
                          # last token (server.log: "restored context checkpoint ... n_tokens = N,
                          # n_past = N-1"), so expected - 1 is the complete reuse of the common prefix
                          "verdict": ("no reuse (full re-prefill)" if not cn else
                                      "expected prefix reused (cache_n = expected)" if exp is not None and cn >= exp else
                                      "expected prefix reused (cache_n = expected - 1)" if exp is not None and cn == exp - 1 else
                                      "partial reuse" if exp is not None else "reused (expected size unknown)")})
            self.reuse_checks.append(reuse)
        ttft = tt[0] if tt else None
        dec_s = (tt[-1] - tt[0]) if len(tt) >= 2 else None
        it = itl_stats(tt)
        pre = self.io(f"{kind}_prefill", rid, s0, s1)
        dec = self.io(f"{kind}_decode", rid, s1, s2)
        tot = self.io(f"{kind}_total", rid, s0, s2)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        def other_cores(row: Mapping[str, Any]) -> float | None:
            # CPU cores busy in the (V)M OUTSIDE this cell's cgroup during the phase (background load)
            if row.get("vm_other_cpu_s") is None or not row.get("wall_s"):
                return None
            return round(max(0.0, float(row["vm_other_cpu_s"])) / float(row["wall_s"]), 3)
        rec = {
            "kind": kind, "rid": rid, **{k: meta[k] for k in meta},
            "endpoint": f"{path} (stream=true, return_tokens=true; prompt rendered with the Gemma 4 chat "
                        f"template by pigemma.workloads.render_gemma4, thinking disabled)",
            "engine": (self.cfg.get("engine") or {}).get("id", "a7b94df"),
            "template_check": tcheck, "reuse_check": reuse,
            "token_ids_source": ids_source, "timestamps_per": res["timestamps_per"],
            "t_start_epoch": round(s0["t"], 3),
            "bg_vm_other_cores_prefill": other_cores(pre), "bg_vm_other_cores_decode": other_cores(dec),
            "cache_prompt": cache_prompt, "n_predict": n_predict, "prompt_sha256": prompt_sha,
            "prompt_chars": len(prompt),
            "input_tokens": tim.get("prompt_n", 0) + tim.get("cache_n", 0) if tim else None,
            "prompt_n_new": tim.get("prompt_n"), "cache_n_reused": tim.get("cache_n"),
            "output_tokens": n_out, "server_predicted_n": tim.get("predicted_n"),
            "stop_type": fin.get("stop_type") if fin else None,
            "ttft_s": round(ttft, 6) if ttft is not None else None,
            "total_s": round(res["t_end"], 6),
            "prefill_tok_s_client": round(tim["prompt_n"] / ttft, 4) if tim.get("prompt_n") and ttft else None,
            "prefill_tok_s_server": tim.get("prompt_per_second"), "prompt_ms_server": tim.get("prompt_ms"),
            "decode_tok_s_client": round((len(tt) - 1) / dec_s, 4) if dec_s and len(tt) >= 2 else None,
            "decode_tok_s_server": tim.get("predicted_per_second"), "predicted_ms_server": tim.get("predicted_ms"),
            **it,
            "io_prefill": pre, "io_decode": dec, "io_total": tot,
            "decode_backing_bytes_per_token": (round(dec["fuse_backing_bytes"] / (len(tt) - 1))
                                               if dec.get("fuse_backing_bytes") is not None and len(tt) >= 2 else None),
            "decode_device_bytes_per_token": (round(dec["device_read_bytes"] / (len(tt) - 1))
                                              if dec.get("device_read_bytes") is not None and len(tt) >= 2 else None),
            "device_read_source": tot.get("device_read_source"),
            "text_sha256": hashlib.sha256(res["text"].encode("utf-8")).hexdigest(),
            "cg_after": s2["cg"],
            "token_ids_sha256": hashlib.sha256(json.dumps(res["token_ids"]).encode()).hexdigest(),
            "text_head": res["text"][:200],
        }
        out = {"rid": rid, "kind": kind, "prompt": prompt, "prompt_sha256": prompt_sha, "text": res["text"],
               "token_ids": res["token_ids"], "token_ids_source": ids_source,
               "timestamps_per": res["timestamps_per"], "token_times_s": [round(x, 6) for x in tt],
               "events": res["events"], "final": {k: v for k, v in (fin or {}).items() if k not in ("content", "tokens")}}
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", rid)
        with open(self.out / "outputs" / f"{safe}.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False)
        self.requests_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.requests_fh.flush()
        self.log(f"{kind} {rid}: in={rec['input_tokens']} (reused {rec['cache_n_reused']}) out={n_out} "
                 f"ttft={rec['ttft_s']} dec={rec['decode_tok_s_client']} tok/s itl_med={rec['itl_median_s']} "
                 f"reads={tot.get('device_read_bytes')} B")
        rec["_text"] = res["text"]
        return rec

    def run_workload(self, port: int) -> None:
        from pigemma import workloads as wl  # stdlib-only module of this repo
        wcfg = self.cfg["workload"]
        convs = wl.load_set(wcfg["set"])
        if wcfg.get("conversations"):
            want = list(wcfg["conversations"])
            by_id = {c["id"]: c for c in convs}
            missing = [x for x in want if x not in by_id]
            if missing:
                raise RuntimeError(f"unknown conversations {missing}")
            convs = [by_id[x] for x in want]
        cap = wcfg.get("n_predict_cap")
        self.eff["workload"] = {"set": wcfg["set"], "n_conversations": len(convs), "n_predict_cap": cap,
                                "order": [c["id"] for c in convs]}
        for conv in convs:
            replies: list[str] = []
            prev_prompt: str | None = None
            n_user = len(wl.user_turns(conv))
            for k in range(n_user):
                msgs = wl.request_messages(conv, k, replies)
                prompt = wl.render_gemma4(msgs, bos=False)
                follow = conv.get("context_reuse") == "follow_up_same_conversation" and k > 0
                n_pred = int(conv["max_tokens"]) if not cap else min(int(conv["max_tokens"]), int(cap))
                rec = self.request(port, "workload", f"{conv['id']}#{k}", prompt, n_pred, follow,
                                   {"conv_id": conv["id"], "turn": k, "set": conv.get("set"),
                                    "lang": conv.get("lang"), "category": conv.get("category"),
                                    "context_reuse": conv.get("context_reuse"), "max_tokens": conv["max_tokens"]},
                                   messages=msgs, prev_prompt=prev_prompt if follow else None)
                prev_prompt = prompt
                replies.append(rec.pop("_text"))
                if self.cfg.get("sampling", {}).get("residency", "per_request") == "per_request":
                    self.residency(f"after_{re.sub(r'[^A-Za-z0-9_.-]', '_', rec['rid'])}")

    def teardown(self) -> None:
        if self.server is not None and self.server.poll() is None:
            self.server.send_signal(signal.SIGTERM)
            try:
                self.server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.server.kill()
        if self.pinner is not None and self.pinner.poll() is None:
            self.pinner.send_signal(signal.SIGTERM)
            try:
                self.eff["pinner_exit_code"] = self.pinner.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.pinner.kill()
                self.eff["pinner_exit_code"] = "killed"
        if self.fuse is not None:
            self.eff["fuse_stats_end"] = self.fuse.stats()
            self.fuse.stop()
            self.eff["fuse_exit_stats"] = self.fuse.exit_stats

    def plan_info(self, requested_union: int | None) -> dict[str, Any] | None:
        """The pin plan written by `pigemma.controlled make-ranges` next to the ranges file
        (<stem>.plan.json): experts, unique pages, nominal tensor bytes and the page-alignment
        overhead, cross-checked against what the pinner says it was asked to lock."""
        rg = (self.cfg.get("pinner") or {}).get("ranges")
        if not rg or not str(rg).endswith(".ranges.txt"):
            return None
        pp = str(rg)[:-len(".ranges.txt")] + ".plan.json"
        try:
            with open(pp, encoding="utf-8") as fh:
                plan = json.load(fh)
        except (OSError, ValueError):
            return None
        keep = ("method", "page", "n_experts", "nonexpert_pages", "expert_pages", "total_unique_pages", "total_bytes",
                "nominal_expert_bytes", "nominal_nonexpert_bytes", "alignment_overhead_bytes", "seed",
                "expert_pin_bytes_budget", "target_total_pages", "diff_pages_vs_target")
        info: dict[str, Any] = {k: plan.get(k) for k in keep}
        info.update({"path": pp, "sha256": sha256_file(pp), "ranking": plan.get("ranking"),
                     "matched_to": plan.get("matched_to"),
                     "plan_total_eq_pinner_requested": (plan.get("total_bytes") == requested_union)
                     if requested_union is not None else None})
        return info

    def lock_accounting(self) -> dict[str, Any]:
        """Requested vs EFFECTIVE locked memory (brief section 3), from four independent views:
        the pinner's own summary, its VmLck, the cgroup's unevictable counter and mincore."""
        page = int(self.eff.get("page_size") or os.sysconf("SC_PAGE_SIZE"))
        ps = self.pinner_summary or {}
        ve = self.eff.get("pinner_verify_end") or {}
        marks = {m.get("tag"): m for m in self.cg_phases}

        def unev(tag: str) -> int | None:
            return ((marks.get(tag) or {}).get("stat") or {}).get("unevictable")

        def lck(tag: str, who: str = "pinner") -> int | None:
            v = ((((marks.get(tag) or {}).get("procs") or {}).get(who) or {}).get("status") or {}).get("VmLck")
            return v * 1024 if v is not None else None
        req = ps.get("requested_bytes")
        acc: dict[str, Any] = {
            "method": self.cfg["method"], "page": page,
            "requested_raw_bytes": ps.get("requested_raw_bytes"),
            "requested_bytes_page_aligned_union": req,
            "requested_unique_pages": (int(req) // page) if req is not None else None,
            "raw_note": "requested_raw_bytes = sum of the range lengths as listed (overlapping ranges counted "
                        "twice); the page-aligned UNION is what is locked and what counts in the budget",
            "plan": self.plan_info(req),
            "pinner_locked_bytes": ps.get("locked_bytes"),
            "pinner_failed_ranges": ps.get("failed"), "pinner_errors": ps.get("errors"),
            "pinner_vm_lck_bytes_at_summary": ps["vm_lck_kb"] * 1024 if isinstance(ps.get("vm_lck_kb"), int) else None,
            "pinner_vm_lck_bytes_after_setup": lck("after_setup"), "pinner_vm_lck_bytes_end": lck("end"),
            "cg_unevictable_bytes_after_setup": unev("after_setup"), "cg_unevictable_bytes_end": unev("end"),
            "mincore_resident_bytes_of_ranges_end": ve.get("resident_bytes"),
            "pinner_lock_s": ps.get("lock_s"),
            # an engine that pins inside llama-server (e.g. HotPin) shows up here and in the cgroup
            # counter, not in the pinner's fields; 0 for the stock builds
            "server_vm_lck_bytes_after_load": lck("after_load", "server"),
            "server_vm_lck_bytes_end": lck("end", "server"),
        }
        if self.cfg["method"] in ("C", "D") and req is not None:
            checks = {"locked_eq_requested": ps.get("locked_bytes") == req,
                      "vm_lck_end_eq_requested": acc["pinner_vm_lck_bytes_end"] == req,
                      "mincore_end_eq_requested": ve.get("resident_bytes") == req,
                      # pages reach the unevictable LRU in per-CPU batches, so this counter can trail
                      # the locked total by a few pages (MEASURED: 4-7 pages right after mlock, 0 at the
                      # end of the cells); VmLck and mincore above are exact
                      "cg_unevictable_end_within_1MiB_of_requested":
                          (acc["cg_unevictable_bytes_end"] or 0) >= int(req) - 1024**2}
            acc["checks"] = checks
            acc["effective_lock_verified"] = all(checks.values())
        else:
            acc["effective_lock_verified"] = None  # A: no pinner; B: nothing requested
        return acc

    def _abort(self, reason: str) -> None:
        """Safety abort from the sampler thread: stop our own server (never anything else)."""
        self.log(f"SAFETY ABORT: {reason}")
        self.eff["safety_abort"] = reason
        if self.server is not None and self.server.poll() is None:
            self.server.send_signal(signal.SIGTERM)

    # -- main ----------------------------------------------------------------------------------
    def run(self) -> int:
        status, failure = "ok", None
        self.probe = Probe(self.cg, None, self.cfg.get("disk_devices") or [])
        write_json(self.out / "cgroup_start.json", self.cg.memory())
        sampler = None
        try:
            self.preflight()
            if not self.probe.devs:
                dev = block_dev_of(self.cfg["model"]["backing_path"]).get("name")
                self.probe.devs = [dev] if dev else []
            self.eff["disk_devices"] = self.probe.devs
            ab = self.cfg.get("abort_mem_available_below_bytes")
            sampler = Sampler(self.probe, self.out / "samples.csv",
                              float(self.cfg.get("sampling", {}).get("interval_s", 0.5)), self.phase,
                              abort_below=int(ab) if ab else None, on_abort=self._abort)
            sampler.start()
            # 1. storage
            self.phase[0] = "storage"
            a = self.probe.snap()
            self.storage_up()
            self.io("storage_up", "-", a, self.probe.snap())
            # 2. cold
            self.phase[0] = "evict"
            a = self.probe.snap()
            self.evict()
            self.io("evict", "-", a, self.probe.snap())
            self.residency("cold")
            self.cg_mark("cold")
            # 3. method setup (timed, its reads counted)
            self.phase[0] = "setup"
            a = self.probe.snap()
            t0 = time.monotonic()
            self.setup_method()
            self.eff["phases"]["setup_s"] = round(time.monotonic() - t0, 3)
            row = self.io("setup", "-", a, self.probe.snap())
            self.eff["phases"]["setup_backing_bytes"] = row["fuse_backing_bytes"]
            self.eff["phases"]["setup_device_read_bytes"] = row["device_read_bytes"]
            self.residency("after_setup")
            self.cg_mark("after_setup")
            # 4. server load (timed)
            self.phase[0] = "load"
            a = self.probe.snap()
            t0 = time.monotonic()
            port = self.start_server()
            self.eff["phases"]["load_s"] = round(time.monotonic() - t0, 3)
            row = self.io("load", "-", a, self.probe.snap())
            self.eff["phases"]["load_backing_bytes"] = row["fuse_backing_bytes"]
            self.eff["phases"]["load_device_read_bytes"] = row["device_read_bytes"]
            self.eff["phases"]["device_read_source"] = row["device_read_source"]
            self.residency("after_load")
            self.cg_mark("after_load")
            # 5. fixed warm-up request (same for every method)
            wu = self.cfg.get("warmup_request") or {}
            t0 = time.monotonic()
            from pigemma import workloads as wl
            wmsgs = [{"role": "user", "content": wu.get("text", "Count from one to five.")}]
            wprompt = wl.render_gemma4(wmsgs, bos=False)
            wrec = self.request(port, "warmup", "warmup", wprompt, int(wu.get("n_predict", 16)), False, {},
                                messages=wmsgs)
            wrec.pop("_text", None)
            self.eff["phases"]["warmup_s"] = round(time.monotonic() - t0, 3)
            self.eff["phases"]["warmup_backing_bytes"] = wrec["io_total"]["fuse_backing_bytes"]
            self.eff["phases"]["warmup_device_read_bytes"] = wrec["io_total"]["device_read_bytes"]
            self.residency("after_warmup")
            self.cg_mark("after_warmup")
            # 6. workload
            t0 = time.monotonic()
            self.run_workload(port)
            self.eff["phases"]["workload_s"] = round(time.monotonic() - t0, 3)
            self.phase[0] = "end"
            self.residency("end")
            self.cg_mark("end")
            # 7. pinner verification (requested vs effective)
            if self.pinner is not None:
                self.eff["pinner_end_status"] = proc_status(self.pinner.pid)
                vargv = [self.eff["setup"]["pinner_argv"][0], "--gguf", self.served_path, "--ranges",
                         self.eff["setup"]["pinner_argv"][4], "--verify-only", "--json"]
                if self.cfg["method"] == "B":
                    vargv.append("--allow-empty")
                vr = subprocess.run(vargv, capture_output=True, text=True, timeout=300)
                try:
                    self.eff["pinner_verify_end"] = json.loads(vr.stdout.strip().splitlines()[-1])
                except (ValueError, IndexError):
                    self.eff["pinner_verify_end"] = {"error": vr.stderr[-500:]}
            try:  # bookkeeping must never turn a completed measurement into a failed cell
                self.eff["lock_accounting"] = self.lock_accounting()
            except Exception as e:  # noqa: BLE001
                self.eff["lock_accounting"] = {"error": repr(e)[:300], "effective_lock_verified": None}
        except Exception as e:  # noqa: BLE001 - the cell records every failure
            status, failure = "fail", f"{type(e).__name__}: {e}"
            self.log(f"CELL FAILED: {failure}")
        finally:
            self.phase[0] = "teardown"
            try:
                self.eff["server_log"] = parse_server_log((self.out / "server.log").read_text("utf-8", "replace")) \
                    if (self.out / "server.log").exists() else None
            except OSError:
                pass
            tc = [t for t in self.template_checks if t.get("available")]
            self.eff["template_check"] = {
                "endpoint": "/apply-template, chat_template_kwargs.enable_thinking=false",
                "available": self.template_api, "n_checked": len(tc),
                "n_match": sum(1 for t in tc if t.get("match")),
                "all_match": (all(t.get("match") for t in tc) if tc else None),
                "mismatches": [t for t in tc if not t.get("match")][:5]}
            self.eff["reuse_check"] = {
                "what": "follow-up requests (cache_prompt=true): server timings.cache_n vs the token count of the "
                        "longest common text prefix with the previous prompt of the conversation",
                "n_followups": len(self.reuse_checks),
                "verdicts": {v: sum(1 for r in self.reuse_checks if r.get("verdict") == v)
                             for v in sorted({str(r.get("verdict")) for r in self.reuse_checks})},
                "details": self.reuse_checks}
            self.eff["thermal_end"] = {"temp_mC": thermal_mC(), "throttled": pi_throttled()}
            if sampler is not None:
                self.eff["max_usage_by_phase_sampled"] = dict(sampler.max_usage_by_phase)
                self.eff["min_host_mem_available_bytes"] = sampler.min_mem_available
                if sampler.abort_reason:
                    status, failure = "fail", f"safety abort: {sampler.abort_reason}"
            try:
                self.teardown()
            except Exception as e:  # noqa: BLE001
                self.eff["teardown_error"] = repr(e)
            if sampler is not None:
                sampler.stop_ev.set()
                sampler.join(timeout=5)
            write_json(self.out / "cgroup_end.json", self.cg.memory())
            write_json(self.out / "cgroup_phases.json", self.cg_phases)
            self.eff["finished"] = now_iso()
            self.eff["meminfo_end_kB"] = {k: meminfo().get(k) for k in ("MemAvailable", "Cached", "SwapFree")}
            cfg_path = self.out / "config.json"
            doc = json.loads(cfg_path.read_text("utf-8")) if cfg_path.exists() else dict(self.cfg)
            doc["effective"] = self.eff
            doc["status"] = status
            doc["failure"] = failure
            write_json(cfg_path, doc)
            self.requests_fh.close()
            self.io_fh.close()
            self.log(f"CELL {status.upper()} -> {self.out}")
        return 0 if status == "ok" else 2


# ----------------------------------------------------------------------------------------------
# throttle verification (does the rate limit apply to mmap page-cache reads? is the backing cached?)
# ----------------------------------------------------------------------------------------------


def _touch_pages(path: str, offsets: Sequence[int]) -> float:
    """mmap path and read one byte at each offset (page faults); returns seconds."""
    import mmap
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
        t0 = time.monotonic()
        acc = 0
        for o in offsets:
            acc ^= mm[o]
        dt = time.monotonic() - t0
        mm.close()
        return dt
    finally:
        os.close(fd)


def verify_throttle(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cg = Cgroup()
    page = os.sysconf("SC_PAGE_SIZE")
    backing = args.backing
    size = os.path.getsize(backing)
    rep: dict[str, Any] = {"schema": "pigemma-throttle-verify v1", "started": now_iso(), "backing": backing,
                           "backing_device": block_dev_of(backing), "page": page, "cgroup": cg.describe(),
                           "rate_bps_requested": args.rate_bps, "tests": []}
    rng = random.Random(1234)
    region = int(args.region_bytes)

    def cache_bytes() -> int | None:
        return cg.memory_brief().get("file")

    def run_case(label: str, rate: float, fn: Callable[[str], dict[str, Any]], use_fuse: bool = True) -> None:
        fd = os.open(backing, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        fuse = None
        path = backing
        if use_fuse:
            fuse = Fuse({"mountpoint": args.mountpoint, "trace": False, "max_readahead": args.max_readahead},
                        backing, out)
            fuse.start(args.bin, rate)
            path = fuse.path
        try:
            c0 = cache_bytes()
            f0 = fuse.stats() if fuse else None
            pio0 = proc_io(fuse.proc.pid) if fuse and fuse.proc else {}
            r = fn(path)
            f1 = fuse.stats() if fuse else None
            pio1 = proc_io(fuse.proc.pid) if fuse and fuse.proc else {}
            c1 = cache_bytes()
            bvec = residency_vector(backing)
            svec = residency_vector(path) if fuse else bvec
            served = (f1["backing_bytes"] - f0["backing_bytes"]) if f0 and f1 else None
            r.update({
                "label": label, "rate_bps": rate, "fuse": use_fuse,
                "fuse_backing_bytes": served,
                "fuse_requests": (f1["requests"] - f0["requests"]) if f0 and f1 else None,
                "fuse_size_hist_end": f1.get("size_hist") if f1 else None,
                "fuse_daemon_read_bytes": _d(pio0, pio1, "read_bytes") if fuse else None,
                "served_rate_MBps": round(served / r["seconds"] / 1e6, 2) if served and r.get("seconds") else None,
                "cg_file_cache_growth_bytes": (c1 - c0) if c0 is not None and c1 is not None else None,
                "backing_resident_bytes": bvec.count(1) * page if bvec is not None else None,
                "served_resident_bytes": svec.count(1) * page if svec is not None else None,
                "fuse_conn": (f1 or {}).get("conn"), "fuse_direct_io": (f1 or {}).get("direct_io"),
            })
            rep["tests"].append(r)
            print(json.dumps({k: r.get(k) for k in ("label", "seconds", "bytes_requested", "fuse_backing_bytes",
                                                    "served_rate_MBps", "app_rate_MBps", "cg_file_cache_growth_bytes",
                                                    "backing_resident_bytes", "served_resident_bytes")}), flush=True)
        finally:
            if fuse:
                fuse.stop()

    start = int(args.offset)

    def seq_read(path: str) -> dict[str, Any]:
        fd = os.open(path, os.O_RDONLY)
        t0 = time.monotonic()
        left, pos = region, start
        while left > 0:
            b = os.pread(fd, min(1 << 20, left), pos)
            if not b:
                break
            pos += len(b)
            left -= len(b)
        dt = time.monotonic() - t0
        os.close(fd)
        return {"kind": "sequential read() 1 MiB", "seconds": round(dt, 3), "bytes_requested": region,
                "app_rate_MBps": round(region / dt / 1e6, 2)}

    def mmap_seq(path: str) -> dict[str, Any]:
        offs = list(range(start + region, start + 2 * region, page))
        dt = _touch_pages(path, offs)
        return {"kind": "mmap sequential page touch", "seconds": round(dt, 3), "bytes_requested": region,
                "app_rate_MBps": round(region / dt / 1e6, 2), "pages_touched": len(offs)}

    def mmap_rand(path: str) -> dict[str, Any]:
        n = int(args.random_pages)
        offs = [rng.randrange(0, size // page) * page for _ in range(n)]
        dt = _touch_pages(path, offs)
        return {"kind": "mmap random page touch", "seconds": round(dt, 3), "pages_touched": n,
                "bytes_requested": n * page, "faults_per_s": round(n / dt, 1)}

    if not args.skip_unthrottled:
        run_case("ext4-direct-mmap-seq (reference, unthrottled, buffered)", 0.0, mmap_seq, use_fuse=False)
        run_case("fuse-unthrottled-seq-read", 0.0, seq_read)
    run_case("fuse-throttled-seq-read", args.rate_bps, seq_read)
    run_case("fuse-throttled-mmap-seq", args.rate_bps, mmap_seq)
    run_case("fuse-throttled-mmap-random", args.rate_bps, mmap_rand)
    # verdicts
    tol = float(args.tolerance)
    verdict: dict[str, Any] = {}
    for t in rep["tests"]:
        if t["rate_bps"] and t.get("served_rate_MBps") is not None:
            ratio = t["served_rate_MBps"] * 1e6 / t["rate_bps"]
            t["served_over_target"] = round(ratio, 4)
            verdict[t["label"]] = {"served_over_target": round(ratio, 4),
                                   "rate_limit_holds": ratio <= 1.0 + tol}
        if t["fuse"]:
            br = t.get("backing_resident_bytes")
            verdict.setdefault(t["label"], {})["backing_not_cached"] = (br is not None and br <= 4 * 1024**2)
            g, s = t.get("cg_file_cache_growth_bytes"), t.get("fuse_backing_bytes")
            if g is not None and s:
                verdict[t["label"]]["cache_growth_over_served"] = round(g / s, 3)
    rep["verdict"] = verdict
    rep["finished"] = now_iso()
    write_json(out / "throttle_verify.json", rep)
    print("VERIFY_DONE", out / "throttle_verify.json", flush=True)
    return 0


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m pigemma.controlled_cell", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one cell from its config.json (inside the cgroup)")
    r.add_argument("--config", required=True)
    v = sub.add_parser("verify-throttle", help="measure whether the FUSE rate limit holds for mmap reads")
    v.add_argument("--backing", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--bin", default="/work/tools/build-linux-x86/bin/fusethrottle")
    v.add_argument("--mountpoint", default="/mnt/pigemma-thr")
    v.add_argument("--rate-bps", type=float, default=91.6e6)
    v.add_argument("--max-readahead", type=int, default=131072)
    v.add_argument("--region-bytes", type=int, default=256 * 1024**2)
    v.add_argument("--offset", type=int, default=2 * 1024**3)
    v.add_argument("--random-pages", type=int, default=1500)
    v.add_argument("--tolerance", type=float, default=0.03)
    v.add_argument("--skip-unthrottled", action="store_true")
    sub.add_parser("probe", help="print what this cgroup exposes (JSON)")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        cfg = json.loads(Path(a.config).read_text("utf-8"))
        if cfg.get("schema") != SCHEMA:
            print(f"config schema {cfg.get('schema')!r} != {SCHEMA!r}", file=sys.stderr)
            return 1
        return Cell(cfg).run()
    if a.cmd == "verify-throttle":
        return verify_throttle(a)
    cg = Cgroup()
    print(json.dumps({"describe": cg.describe(), "memory": cg.memory(), "io_stat": cg.io_stat(),
                      "cpu_ns": cg.cpu_usage_ns()}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
