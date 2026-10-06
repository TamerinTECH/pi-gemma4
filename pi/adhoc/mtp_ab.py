#!/usr/bin/env python3
"""Single-stream A/B: no drafter vs Gemma 4 MTP drafter (speculative decoding) on the Pi (stdlib only).

For each configuration in the given order: start llama-server (nice 10, 127.0.0.1), one warm-up request, then the
same prompts sequentially through the CHAT endpoint (GGUF template), greedy; record per-request timings from the
server, wall time, and the drafter statistics the server prints ("draft acceptance = ... mean len = ...").
Configurations are passed as NAME=EXTRA_ARGS; run them alternated (e.g. base mtp3 base mtp3) for a paired comparison.

Usage: mtp_ab.py --bin <llama-server> --model M.gguf --label NAME --config base= --config "mtp3=-md D.gguf --spec-type draft-mtp"
                 [--order base,mtp3,base,mtp3] [--max-tokens 128] [--threads 4] [--out ~/pigemma/results]
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import time
import urllib.request

PROMPTS = [
    {"id": "en_explain", "lang": "en", "text": "Explain in a few sentences how a refrigerator keeps food cold."},
    {"id": "en_story", "lang": "en", "text": "Write a short story (about 100 words) about a lighthouse keeper who finds a message in a bottle."},
    {"id": "he_explain", "lang": "he", "text": "הסבר בכמה משפטים פשוטים איך פועל מקרר."},
    {"id": "de_explain", "lang": "de", "text": "Erkläre in einigen einfachen Sätzen, wie ein Kühlschrank funktioniert."},
    {"id": "code_py", "lang": "code", "text": "Write a Python function that returns the n-th Fibonacci number iteratively, with a short docstring. Only the code."},
    {"id": "list_tips", "lang": "en", "text": "Give five practical tips for learning a new language as an adult, as a numbered list."},
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
    ap.add_argument("--config", action="append", required=True, help="NAME=EXTRA_ARGS (EXTRA may be empty)")
    ap.add_argument("--order", default="")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--port", type=int, default=18110)
    ap.add_argument("--out", default=os.path.expanduser("~/pigemma/results"))
    ap.add_argument("--prompts-file", default="", help="JSON list of {id, lang, text[, max_tokens]} replacing the built-in prompts")
    ap.add_argument("--common", default="", help="server arguments added to EVERY configuration")
    ap.add_argument("--request-extra", default="", help="JSON object merged into every request body")
    a = ap.parse_args()
    prompts = PROMPTS
    if a.prompts_file:
        with open(a.prompts_file, encoding="utf-8") as f:
            prompts = json.load(f)
    req_extra = json.loads(a.request_extra) if a.request_extra else {}
    configs = dict(c.split("=", 1) for c in a.config)
    order = a.order.split(",") if a.order else list(configs)
    out = os.path.join(a.out, a.label)
    os.makedirs(out, exist_ok=True)
    result = {"label": a.label, "model": a.model, "model_bytes": os.path.getsize(a.model), "max_tokens": a.max_tokens,
              "threads": a.threads, "configs": configs, "order": order, "runs": []}
    for k, name in enumerate(order):
        port = a.port + k
        cmd = ["nice", "-n", "10", a.bin, "-m", a.model, "-t", str(a.threads), "-c", str(a.ctx), "-np", "1",
               "--host", "127.0.0.1", "--port", str(port)] + shlex.split(a.common) + shlex.split(configs[name])
        logp = os.path.join(out, f"server_{k}_{name}.log")
        run = {"index": k, "config": name, "command": " ".join(shlex.quote(c) for c in cmd), "requests": []}
        with open(logp, "w") as log:
            t0 = time.time()
            srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    if srv.poll() is not None:
                        raise RuntimeError(f"server exited during load; see {logp}")
                    try:
                        if http(port, "/health", timeout=5).get("status") == "ok":
                            break
                    except Exception:
                        pass
                    time.sleep(1)
                run["load_s"] = round(time.time() - t0, 1)
                http(port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 8, "temperature": 0})
                for p in prompts:
                    w0 = time.time()
                    r = http(port, "/v1/chat/completions", {"messages": [{"role": "user", "content": p["text"]}],
                                                             "max_tokens": p.get("max_tokens", a.max_tokens), "temperature": 0,
                                                             "cache_prompt": False, **req_extra})
                    wall = time.time() - w0
                    t = r.get("timings") or {}
                    u = r.get("usage") or {}
                    ch0 = (r.get("choices") or [{}])[0]
                    msg = ch0.get("message", {})
                    full = (msg.get("content") or "") + "\x00" + (msg.get("reasoning_content") or "")
                    rec = {"id": p["id"], "lang": p["lang"], "wall_s": round(wall, 2),
                           "prompt_n": t.get("prompt_n", u.get("prompt_tokens")), "gen_n": t.get("predicted_n", u.get("completion_tokens")),
                           "gen_tok_s": t.get("predicted_per_second"), "prompt_tok_s": t.get("prompt_per_second"),
                           "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
                           "wall_tok_s": round((t.get("predicted_n") or u.get("completion_tokens") or 0) / wall, 2) if wall > 0 else None,
                           "temp_c": temp_c(), "content": msg.get("content") or "",
                           "reasoning": msg.get("reasoning_content") or "", "finish": ch0.get("finish_reason"),
                           "text_sha256": hashlib.sha256(full.encode("utf-8")).hexdigest()}
                    run["requests"].append(rec)
                    print(json.dumps({"config": name, **{x: rec[x] for x in ("id", "gen_n", "gen_tok_s", "wall_tok_s", "draft_n", "draft_n_accepted", "temp_c", "finish")}}), flush=True)
            except Exception as e:  # noqa: BLE001 - keep the other configurations; the error is recorded
                run["error"] = str(e)
                print(json.dumps({"config": name, "error": str(e)}), flush=True)
            finally:
                srv.send_signal(signal.SIGTERM)
                try:
                    srv.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    srv.kill()
        txt = open(logp, encoding="utf-8", errors="replace").read()
        run["draft_log"] = re.findall(r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)", txt)
        run["spec_lines"] = [ln.strip()[:200] for ln in txt.splitlines() if re.search(r"speculative|draft model|spec_type|adding speculative", ln)][:8]
        result["runs"].append(run)
        with open(os.path.join(out, "ab.json"), "w", encoding="utf-8") as f:  # rewritten after every configuration
            json.dump(result, f, ensure_ascii=False, indent=1)
        time.sleep(20)  # cool-down between server starts
    with open(os.path.join(out, "ab.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("AB_DONE", os.path.join(out, "ab.json"), flush=True)


if __name__ == "__main__":
    main()
