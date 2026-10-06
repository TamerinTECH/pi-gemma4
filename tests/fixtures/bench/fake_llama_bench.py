"""Test double for llama-bench (stdlib only): prints a captured ``-o json`` output.

Env: FAKE_BENCH_JSON (file to print, default llama_bench_tiny.json next to this file),
FAKE_RC (exit code, default 0), FAKE_BENCH_STDERR (file copied to stderr, default none).
Echoes its argv to stderr so tests can check the exact flags.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if "--version" in sys.argv:
    print("version: 0.5.0-dev (build 11398, commit a7b94df2c)")
    sys.exit(0)

here = Path(__file__).resolve().parent
sys.stderr.write("fake-argv: " + " ".join(sys.argv[1:]) + "\n")
se = os.environ.get("FAKE_BENCH_STDERR")
if se:
    sys.stderr.write(Path(se).read_text(encoding="utf-8", errors="replace"))
sys.stdout.write(Path(os.environ.get("FAKE_BENCH_JSON") or here / "llama_bench_tiny.json")
                 .read_text(encoding="utf-8"))
sys.stdout.flush()
sys.exit(int(os.environ.get("FAKE_RC", "0")))
