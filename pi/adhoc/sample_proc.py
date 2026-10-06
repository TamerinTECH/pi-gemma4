#!/usr/bin/env python3
"""Per-second sampler for one process on Linux (stdlib only; runs on the Pi).

Usage: sample_proc.py <pid> <out.csv> [interval_s]

Columns (one row per sample, until the pid exits):
  t_s                 seconds since start
  read_bytes          /proc/<pid>/io read_bytes   - bytes fetched from the storage layer (cache misses)
  rchar               /proc/<pid>/io rchar        - logical read() bytes (mmap faults are NOT counted here)
  majflt              /proc/<pid>/stat field 12   - major page faults (mmap pages read from storage)
  rss_kb, pss_anon_kb, pss_file_kb, swap_kb      - /proc/<pid>/smaps_rollup (file pages = mapped model weights)
  vm_hwm_kb           /proc/<pid>/status VmHWM    - peak resident set
  mem_available_kb, cached_kb, swap_free_kb       - system-wide /proc/meminfo
  temp_c              /sys/class/thermal/thermal_zone0/temp
Note: RSS and the page cache overlap (mapped model pages are counted in both); never add them.
"""
import sys
import time


def read_kv(path, sep=":"):
    out = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if sep in line:
                    k, v = line.split(sep, 1)
                    out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def kb(v):
    try:
        return int(v.split()[0])
    except (AttributeError, IndexError, ValueError):
        return -1


def main():
    pid = int(sys.argv[1])
    out = sys.argv[2]
    interval = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    cols = ["t_s", "read_bytes", "rchar", "majflt", "rss_kb", "pss_anon_kb", "pss_file_kb", "swap_kb",
            "vm_hwm_kb", "mem_available_kb", "cached_kb", "swap_free_kb", "temp_c"]
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        while True:
            io = read_kv(f"/proc/{pid}/io")
            if not io:
                break
            try:
                with open(f"/proc/{pid}/stat", "r") as s:
                    stat = s.read()
                majflt = int(stat.rsplit(")", 1)[1].split()[9])
            except (OSError, IndexError, ValueError):
                break
            sm = read_kv(f"/proc/{pid}/smaps_rollup")
            st = read_kv(f"/proc/{pid}/status")
            mi = read_kv("/proc/meminfo")
            try:
                with open("/sys/class/thermal/thermal_zone0/temp") as tf:
                    temp = int(tf.read().strip()) / 1000.0
            except (OSError, ValueError):
                temp = -1
            row = [round(time.time() - t0, 2), io.get("read_bytes", "-1"), io.get("rchar", "-1"), majflt,
                   kb(sm.get("Rss")), kb(sm.get("Pss_Anon")), kb(sm.get("Pss_File")), kb(sm.get("Swap")),
                   kb(st.get("VmHWM")), kb(mi.get("MemAvailable")), kb(mi.get("Cached")), kb(mi.get("SwapFree")),
                   temp]
            f.write(",".join(str(x) for x in row) + "\n")
            f.flush()
            time.sleep(interval)


if __name__ == "__main__":
    main()
