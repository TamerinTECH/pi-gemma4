#!/usr/bin/env python3
"""Memory balloon: pin N GiB of anonymous RAM so the page cache has less room (emulates a smaller board).

The Pi kernel boots with cgroup_disable=memory, so cgroup limits are unavailable; a locked balloon creates
real global memory pressure instead (closer to a genuinely smaller RAM than a per-cgroup limit).

Usage (needs CAP_IPC_LOCK, i.e. sudo): sudo python3 balloon.py <GiB> <max_seconds> [<pidfile>]
Safety:
  - refuses if MemAvailable - GiB would leave less than 2.5 GiB (services + inference process need room)
  - exits by itself after max_seconds (hard cap 3600) and on SIGTERM/SIGINT; the memory is freed on exit
  - prints one JSON line when the balloon is fully resident
"""
import ctypes
import json
import mmap
import os
import signal
import sys
import time

GIB = 1 << 30


def mem_available():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def main():
    size = int(float(sys.argv[1]) * GIB)
    max_s = min(int(sys.argv[2]), 3600)
    pidfile = sys.argv[3] if len(sys.argv) > 3 else None
    avail = mem_available()
    if avail - size < int(2.5 * GIB):
        print(json.dumps({"ok": False, "reason": "would leave < 2.5 GiB", "mem_available": avail, "size": size}))
        sys.exit(3)
    stop = {"flag": False}

    def handler(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    buf = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    page = mmap.PAGESIZE
    for off in range(0, size, page):
        buf[off] = 1  # touch every page so it is really allocated
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    addr = ctypes.c_void_p.from_buffer(buf)
    rc = libc.mlock(ctypes.c_void_p(ctypes.addressof(addr)), ctypes.c_size_t(size))
    locked = rc == 0
    if pidfile:
        with open(pidfile, "w") as f:
            f.write(str(os.getpid()))
    print(json.dumps({"ok": True, "size_bytes": size, "mlocked": locked, "errno": 0 if locked else ctypes.get_errno(),
                      "mem_available_after": mem_available(), "max_seconds": max_s}), flush=True)
    t0 = time.time()
    while not stop["flag"] and time.time() - t0 < max_s:
        time.sleep(1)
    del addr
    buf.close()


if __name__ == "__main__":
    main()
