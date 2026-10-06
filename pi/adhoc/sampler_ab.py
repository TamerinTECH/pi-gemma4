#!/usr/bin/env python3
"""How much of a decode step is the server's sampler chain?  One llama-server, same prompts, greedy, alternating
request variants (default chain vs minimal chains).  Reports the server's own decode tok/s and whether the text is
identical to the default-chain text.  stdlib only.

Usage: sampler_ab.py --bin <llama-server> --model M.gguf [--extra "..."] [--max-tokens 96] [--rounds 2]
"""
import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import time
import urllib.request

PROMPTS = [
    "Explain in a few sentences how a refrigerator keeps food cold.",
    "Give five practical tips for learning a new language as an adult, as a numbered list.",
]
VARIANTS = {
    "default": {},
    "topk1": {"samplers": ["top_k"], "top_k": 1},
    "temp_only": {"samplers": ["temperature"]},
}


def http(port, path, body=None, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--port", type=int, default=18170)
    ap.add_argument("--out", default=os.path.expanduser("~/pigemma/results/sampler-ab"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cmd = ["nice", "-n", "10", a.bin, "-m", a.model, "-t", "4", "-c", "4096", "-np", "1", "--host", "127.0.0.1",
           "--port", str(a.port)] + shlex.split(a.extra)
    log = open(os.path.join(a.out, "server.log"), "w")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    rows = []
    try:
        while True:
            if srv.poll() is not None:
                raise RuntimeError("server exited during load")
            try:
                if http(a.port, "/health", timeout=5).get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(1)
        http(a.port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 8, "temperature": 0})
        ref = {}
        for rnd in range(a.rounds):
            for name, extra in VARIANTS.items():
                for pi, p in enumerate(PROMPTS):
                    body = {"messages": [{"role": "user", "content": p}], "max_tokens": a.max_tokens, "temperature": 0,
                            "cache_prompt": False, **extra}
                    try:
                        w0 = time.time()
                        r = http(a.port, "/v1/chat/completions", body)
                        wall = time.time() - w0
                    except Exception as e:  # noqa: BLE001
                        rows.append({"variant": name, "prompt": pi, "error": str(e)})
                        print(json.dumps(rows[-1]), flush=True)
                        continue
                    t = r.get("timings") or {}
                    msg = (r.get("choices") or [{}])[0].get("message", {})
                    text = (msg.get("content") or "") + "|" + (msg.get("reasoning_content") or "")
                    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
                    if name == "default" and pi not in ref:
                        ref[pi] = h
                    row = {"round": rnd, "variant": name, "prompt": pi, "gen_n": t.get("predicted_n"),
                           "gen_tok_s": round(t.get("predicted_per_second") or 0, 3),
                           "ms_per_tok": round(t.get("predicted_per_token_ms") or 0, 2),
                           "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
                           "wall_s": round(wall, 2), "same_as_default": h == ref.get(pi)}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        srv.send_signal(signal.SIGTERM)
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
        with open(os.path.join(a.out, "rows.json"), "w") as f:
            json.dump(rows, f, indent=1)
    print("SAMPLER_AB_DONE", flush=True)


if __name__ == "__main__":
    main()
