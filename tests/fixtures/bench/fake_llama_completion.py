"""Test double for llama-completion (stdlib only).

Replays a captured stderr log (env FAKE_STDERR, default completion_tiny_lv4.stderr.txt next to
this file) around a streamed generation, the way the real tool interleaves them: everything up
to the 'generate: n_ctx' line, a pause, the generated text on stdout in chunks, then the
'common_perf_print' block. Knobs (env): FAKE_STDOUT (text, default an English sentence),
FAKE_RC (exit code, default 0), FAKE_HANG=1 (sleep 60 s after the generate line), FAKE_MEM_MB
(bytes to touch so the memory sampler sees something, default 40). Echoes argv to stderr.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

if "--version" in sys.argv:
    print("version: 0.5.0-dev (build 11398, commit a7b94df2c)")
    sys.exit(0)

here = Path(__file__).resolve().parent
lines = Path(os.environ.get("FAKE_STDERR") or here / "completion_tiny_lv4.stderr.txt") \
    .read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
i_gen = next(i for i, ln in enumerate(lines) if "generate: n_ctx" in ln)
i_perf = next(i for i, ln in enumerate(lines) if "common_perf_print" in ln)
err = sys.stderr
err.write("fake-argv: " + " ".join(sys.argv[1:]) + "\n")
if "-f" in sys.argv:
    prompt = Path(sys.argv[sys.argv.index("-f") + 1]).read_bytes().decode("utf-8", "replace")
    err.write("fake-prompt: " + json.dumps(prompt) + "\n")
mem = bytearray(int(float(os.environ.get("FAKE_MEM_MB", "40")) * 1_000_000))
for i in range(0, len(mem), 4096):
    mem[i] = 1
for ln in lines[: i_gen + 1]:
    err.write(ln)
err.flush()
if os.environ.get("FAKE_HANG") == "1":
    time.sleep(60)
time.sleep(0.15)
text = os.environ.get("FAKE_STDOUT",
                      "Experts are chosen per token by a router, so only a few run. "
                      "That saves compute and memory bandwidth.")
out = sys.stdout.buffer
for i in range(0, len(text), 16):
    out.write(text[i:i + 16].encode("utf-8"))
    out.flush()
    time.sleep(0.01)
out.write(b"\n")
out.flush()
time.sleep(0.1)
for ln in lines[i_perf:]:
    err.write(ln)
err.flush()
sys.exit(int(os.environ.get("FAKE_RC", "0")))
