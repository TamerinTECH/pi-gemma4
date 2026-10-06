"""Test double for llama-server (stdlib only): a ThreadingHTTPServer with /health and /completion.

Accepts the llama-server flags pigemma.bench passes (-m, -t, -c, -np, --host, --port, -md,
--spec-type, --spec-draft-n-max, anything else ignored) and answers like the real server at
b11398 (same response keys as tests/fixtures/bench/server_moe_*.json). Behaviour knobs (env):

  FAKE_SERVER_MODE   ok (default) | crash (exit 1 before listening) | never-healthy (503 forever)
  FAKE_LOAD_S        seconds of 503 "Loading model" before /health says ok (default 0.1)
  FAKE_DRAFT_N       draft_n per request when a draft or --spec-type is configured (default 12)
  FAKE_DRAFT_ACC     draft_n_accepted per request (default 9); FAKE_DRAFT_N=0 omits both keys
  FAKE_INFERRED      speculative.types reported for -md without --spec-type (default 'draft-dflash')
  FAKE_DIVERGE       substring: prompts containing it get a different answer when drafting
  FAKE_LOG           file that receives one JSON line per /completion request body
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if "--version" in sys.argv:
    print("version: 0.5.0-dev (build 11398, commit a7b94df2c)")
    print("built with fake for tests")
    sys.exit(0)


def arg(name: str, default: str | None = None) -> str | None:
    a = sys.argv
    return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else default


MODE = os.environ.get("FAKE_SERVER_MODE", "ok")
HOST = arg("--host", "127.0.0.1") or "127.0.0.1"
PORT = int(arg("--port", "8080") or 8080)
DRAFT = arg("-md")
SPEC = arg("--spec-type")
LOAD_S = float(os.environ.get("FAKE_LOAD_S", "0.1"))
T0 = time.monotonic()

print(f"fake llama-server: model={arg('-m')} draft={DRAFT} spec={SPEC} port={PORT}", flush=True)
if MODE == "crash":
    print("0.00.100.000 E llama_init_from_model: failed to initialize the context: fake crash",
          flush=True)
    sys.exit(1)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a: object) -> None:  # quiet
        pass

    def _send(self, code: int, obj: object) -> None:
        b = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send(404, {"error": "not found"})
            return
        if MODE == "never-healthy" or time.monotonic() - T0 < LOAD_S:
            self._send(503, {"error": {"code": 503, "message": "Loading model",
                                       "type": "unavailable_error"}})
            return
        self._send(200, {"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n).decode("utf-8"))
        lp = os.environ.get("FAKE_LOG")
        if lp:
            with open(lp, "a", encoding="utf-8") as f:
                f.write(json.dumps(body, ensure_ascii=False) + "\n")
        prompt = str(body.get("prompt", ""))
        n_pred = int(body.get("n_predict", 16))
        drafting = bool(DRAFT or SPEC)
        answer = "In short: " + " ".join(f"w{i}" for i in range(min(n_pred, 40)))
        div = os.environ.get("FAKE_DIVERGE")
        if drafting and div and div in prompt:
            answer += " (diverged)"
        content = "<|channel>thought\n<channel|>" + answer
        timings = {"cache_n": 0, "prompt_n": max(1, len(prompt) // 4), "prompt_ms": 20.0,
                   "prompt_per_token_ms": 1.0, "prompt_per_second": 50.0 * (len(prompt) // 4 or 1) / 1.0,
                   "predicted_n": n_pred, "predicted_ms": 10.0 * n_pred,
                   "predicted_per_token_ms": 10.0 * n_pred / max(n_pred - 1, 1),
                   "predicted_per_second": (n_pred - 1) / (0.01 * n_pred)}
        dn = int(os.environ.get("FAKE_DRAFT_N", "12"))
        if drafting and dn > 0:
            timings["draft_n"] = dn
            timings["draft_n_accepted"] = int(os.environ.get("FAKE_DRAFT_ACC", "9"))
        if SPEC:
            types = f"none,{SPEC}"
        elif DRAFT:
            types = os.environ.get("FAKE_INFERRED", "draft-dflash")
        else:
            types = "none"
        self._send(200, {
            "index": 0, "content": content, "tokens": [], "id_slot": 0, "stop": True,
            "model": arg("-m"), "tokens_predicted": n_pred, "tokens_evaluated": timings["prompt_n"],
            "generation_settings": {"temperature": body.get("temperature"), "n_predict": n_pred,
                                    "speculative.types": types},
            "prompt": "<bos>" + prompt, "has_new_line": False, "truncated": False,
            "stop_type": "limit", "stopping_word": "", "tokens_cached": 0, "timings": timings,
        })


srv = ThreadingHTTPServer((HOST, PORT), H)
print(f"0.00.500.000 I srv  llama_server: listening on http://{HOST}:{PORT}", flush=True)
threading.Thread(target=srv.serve_forever, daemon=True).start()
try:
    while True:
        time.sleep(0.2)
except KeyboardInterrupt:
    pass
