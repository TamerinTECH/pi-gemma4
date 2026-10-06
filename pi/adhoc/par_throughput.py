#!/usr/bin/env python3
"""Aggregate throughput of N parallel generation streams against one llama-server (stdlib only).

Starts its own server (nice 10, 127.0.0.1), waits until healthy, then for each N in --streams fires N
simultaneous /completion requests (distinct prompts, greedy, n_predict tokens each) and reports:
  total generated tokens / wall seconds  (aggregate tok/s),
  per-stream decode tok/s as reported by the server timings (mean/min/max),
  prompt tok/s (mean).
A single stream (N=1) is the interactive speed; N>1 shows what batched decoding buys in TOTAL throughput
(each user still sees only the per-stream speed).

Usage: par_throughput.py --bin <llama-server> --model M.gguf --label NAME [--threads 4] [--streams 1,2,4,8,16]
                         [--n-predict 96] [--ctx-per-slot 1024] [--extra "..."] [--out ~/pigemma/results]
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import threading
import time
import urllib.request

PROMPTS = [
    "Explain in simple words how a refrigerator keeps food cold.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Describe the water cycle step by step for a ten year old.",
    "Give five practical tips for learning a new language as an adult.",
    "Explain what a prime number is and list the first ten primes with a remark on each.",
    "Describe a walk through a forest in autumn, with sounds and smells.",
    "Explain how a bicycle stays upright while moving.",
    "Write a friendly email inviting a colleague to a team lunch next Friday.",
    "Explain the difference between weather and climate with two examples.",
    "Describe how bread is baked, from flour to a finished loaf.",
    "Explain why the sky is blue during the day and red at sunset.",
    "Write a short dialogue between a customer and a baker early in the morning.",
    "Explain how a library organises its books so people can find them.",
    "Describe the rules of chess for a complete beginner.",
    "Explain what an operating system does on a computer.",
    "Write a short speech thanking volunteers after a school event.",
]


def http(port, path, body=None, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000.0
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--streams", default="1,2,4,8,16")
    ap.add_argument("--n-predict", type=int, default=96)
    ap.add_argument("--ctx-per-slot", type=int, default=1024)
    ap.add_argument("--extra", default="")
    ap.add_argument("--port", type=int, default=18090)
    ap.add_argument("--out", default=os.path.expanduser("~/pigemma/results"))
    a = ap.parse_args()
    streams = [int(x) for x in a.streams.split(",")]
    nmax = max(streams)
    out = os.path.join(a.out, a.label)
    os.makedirs(out, exist_ok=True)
    cmd = ["nice", "-n", "10", a.bin, "-m", a.model, "-t", str(a.threads), "-c", str(a.ctx_per_slot * nmax),
           "-np", str(nmax), "--host", "127.0.0.1", "--port", str(a.port)] + shlex.split(a.extra)
    run = {"label": a.label, "command": " ".join(shlex.quote(c) for c in cmd), "model_bytes": os.path.getsize(a.model),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_predict": a.n_predict, "results": []}
    log = open(os.path.join(out, "server.log"), "w")
    t0 = time.time()
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        while True:
            if srv.poll() is not None:
                raise RuntimeError("server exited during load; see server.log")
            try:
                if http(a.port, "/health", timeout=5).get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(1)
        run["load_s"] = round(time.time() - t0, 1)
        # one warm-up request
        http(a.port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 8, "temperature": 0})
        for n in streams:
            res = [None] * n

            def worker(i):
                # Chat endpoint: the server applies the GGUF's own Gemma 4 template (older runs used Gemma-3 markers).
                try:
                    res[i] = http(a.port, "/v1/chat/completions", {
                        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
                        "max_tokens": a.n_predict, "temperature": 0, "cache_prompt": False})
                except Exception as e:  # noqa: BLE001
                    res[i] = {"error": str(e)}

            ths = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            w0 = time.time()
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            wall = time.time() - w0
            ok = [r for r in res if r and "error" not in r and ("usage" in r or "timings" in r)]
            def gen_n(r):
                return (r.get("usage") or {}).get("completion_tokens") or (r.get("timings") or {}).get("predicted_n", 0)
            def prm_n(r):
                return (r.get("usage") or {}).get("prompt_tokens") or (r.get("timings") or {}).get("prompt_n", 0)
            gen = sum(gen_n(r) for r in ok)
            pn = sum(prm_n(r) for r in ok)
            per = [r["timings"].get("predicted_per_second", 0) for r in ok if r.get("timings")]
            pps = [r["timings"].get("prompt_per_second", 0) for r in ok if r.get("timings")]
            rec = {"streams": n, "ok": len(ok), "wall_s": round(wall, 2), "generated_tokens": gen, "prompt_tokens": pn,
                   "aggregate_gen_tok_s": round(gen / wall, 2) if wall > 0 else None,
                   "aggregate_all_tok_s": round((gen + pn) / wall, 2) if wall > 0 else None,
                   "per_stream_gen_tok_s_mean": round(sum(per) / len(per), 2) if per else None,
                   "per_stream_gen_tok_s_min": round(min(per), 2) if per else None,
                   "per_stream_gen_tok_s_max": round(max(per), 2) if per else None,
                   "prompt_tok_s_mean": round(sum(pps) / len(pps), 2) if pps else None,
                   "temp_c": temp_c(), "errors": [r.get("error") for r in res if r and "error" in r][:3]}
            run["results"].append(rec)
            print(json.dumps(rec), flush=True)
            if n == 1 and ok:
                with open(os.path.join(out, "sample_output.txt"), "w", encoding="utf-8") as g:
                    g.write(json.dumps(ok[0].get("choices", [{}])[0].get("message", {}), ensure_ascii=False))
    finally:
        srv.send_signal(signal.SIGTERM)
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
        run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
            json.dump(run, f, ensure_ascii=False, indent=1)
        print("RUN_DONE", os.path.join(out, "run.json"), flush=True)


if __name__ == "__main__":
    main()
