#!/usr/bin/env python3
"""Write the sha256 manifest of the raw results kept in this repository.

  make_checksums.py            results/pi-real/** and results/desktop-fast/** -> results/pi-real/SHA256SUMS-2026-10-06.txt

The manifest lists path, size and sha256 for every file (the manifest itself excluded), sorted by path.
"""
import hashlib
import os
import sys

ROOTS = ("results/pi-real", "results/desktop-fast")
OUT = "results/pi-real/SHA256SUMS-2026-10-06.txt"


def main():
    rows = []
    for root in ROOTS:
        for d, _, files in os.walk(root):
            for f in files:
                path = os.path.join(d, f).replace("\\", "/")
                if path == OUT:
                    continue
                h = hashlib.sha256()
                with open(path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                rows.append((path, os.path.getsize(path), h.hexdigest()))
    rows.sort()
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("# sha256  bytes  path   (raw results of 2026-10-05/06; Pi copies: ~/pigemma/results/ on context-friend-1)\n")
        for path, size, digest in rows:
            f.write(f"{digest}  {size:>10}  {path}\n")
    print(f"{len(rows)} files, {sum(r[1] for r in rows) / 1e6:.1f} MB -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
