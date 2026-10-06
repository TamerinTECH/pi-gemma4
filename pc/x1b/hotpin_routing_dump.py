#!/usr/bin/env python3
"""X1b: run HotPin's OWN routing-profiling tool on our calibration set (runs INSIDE the container).

HotPin's procedure (third_party/hotpin-llm/README.md, "Generate a hot file for a new model"):
    llama-simple-chat -m model.gguf --dump-routing routing.bin -n 200 --no-color   (type prompts)
    python tools/make_hot_file.py --routing routing.bin --budget-gb B --model model.gguf --out hot.txt

This driver feeds the patched llama-simple-chat (built from HotPin's unmodified patches) with the
requests of data/workloads/calibration.jsonl ONLY, through the tool's own --batch protocol:
  * prompt text = pigemma.workloads.render_gemma4(request_messages(conv, k, replies), bos=False)
    (the canonical Gemma 4 template, thinking disabled, BOS added by the tool's tokenizer), sent
    with --raw because llama-simple-chat cannot apply the GGUF's Jinja template (HotPin README);
  * greedy (--temp 0, as in HotPin's benchmark scripts), -n = the conversation's max_tokens
    (one process per distinct max_tokens value because -n is a process-wide flag);
  * follow-up requests re-send the whole conversation with the model's own earlier replies
    (--batch clears the KV state per prompt, so earlier turns are prefilled - and dumped - again).
Nothing of HotPin is modified or copied here; this file only starts its binary and logs its I/O.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, "/work")
from pigemma import workloads as wl  # noqa: E402

READY = b"===READY===\n"
TIMINGS = re.compile(r"\[timings\] prompt=(\d+) tok, TTFT=([\d.]+) ms, gen=(\d+) tok, eval=([\d.]+) ms, "
                     r"([\d.]+) tok/s, total=([\d.]+) ms")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Chat:
    def __init__(self, argv: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
        self.argv = argv
        self.err_lines: list[str] = []
        self.log = open(log_path, "wb")
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  env=dict(os.environ, **(env or {})))
        self.t = threading.Thread(target=self._drain, daemon=True)
        self.t.start()

    def _drain(self) -> None:
        assert self.p.stderr is not None
        for raw in iter(self.p.stderr.readline, b""):
            self.log.write(raw)
            self.log.flush()
            self.err_lines.append(raw.decode("utf-8", "replace"))

    def read_until_ready(self, timeout: float) -> bytes:
        assert self.p.stdout is not None
        buf = b""
        t0 = time.monotonic()
        fd = self.p.stdout.fileno()
        while not buf.endswith(READY):
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"no READY after {timeout} s; got {len(buf)} bytes")
            chunk = os.read(fd, 65536)
            if not chunk:
                raise RuntimeError(f"llama-simple-chat exited (rc={self.p.poll()}) before READY; stderr tail: "
                                   + "".join(self.err_lines[-8:]))
            buf += chunk
        return buf[:-len(READY)]

    def ask(self, prompt: str, timeout: float) -> tuple[str, dict]:
        assert self.p.stdin is not None
        n_err = len(self.err_lines)
        t0 = time.monotonic()
        self.p.stdin.write(prompt.encode("utf-8") + b"\n===END===\n")
        self.p.stdin.flush()
        out = self.read_until_ready(timeout)
        wall = time.monotonic() - t0
        # the tool prints the reply, then a newline, then READY
        text = out.decode("utf-8", "replace")
        if text.endswith("\n"):
            text = text[:-1]
        tm: dict = {"wall_s": round(wall, 3)}
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            for ln in self.err_lines[n_err:]:
                m = TIMINGS.search(ln)
                if m:
                    tm.update(prompt_tokens=int(m.group(1)), ttft_ms=float(m.group(2)), gen_tokens=int(m.group(3)),
                              eval_ms=float(m.group(4)), tok_s_functional_only=float(m.group(5)),
                              total_ms=float(m.group(6)))
                    return text, tm
            time.sleep(0.05)
        return text, tm

    def close(self, timeout: float = 120) -> int:
        assert self.p.stdin is not None
        try:
            self.p.stdin.close()  # EOF -> the tool leaves its loop and closes the dump file
        except OSError:
            pass
        try:
            rc = self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.p.terminate()  # our own child only
            rc = self.p.wait(timeout=30)
        self.t.join(timeout=10)
        self.log.close()
        return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="patched llama-simple-chat")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="calibration", choices=["calibration"],
                    help="ONLY the calibration set may rank experts")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--tag", default="original")
    ap.add_argument("--limit-conversations", type=int, default=0, help="debug only")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    convs = wl.load_set(a.set)
    if a.limit_conversations:
        convs = convs[:a.limit_conversations]
    groups: dict[int, list[dict]] = {}
    for c in convs:
        groups.setdefault(int(c["max_tokens"]), []).append(c)
    meta: dict = {"schema": "pigemma-x1b-hotpin-routing-dump v1", "tag": a.tag,
                  "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "exe": a.exe, "exe_sha256": sha256_file(Path(a.exe)), "model": a.model,
                  "model_bytes": os.path.getsize(a.model), "set": a.set,
                  "set_sha256": sha256_file(Path("/work/data/workloads") / f"{a.set}.jsonl"),
                  "threads": a.threads, "ctx": a.ctx, "decoding": "--temp 0 (greedy), --raw, --batch",
                  "repack": "tool default (llama_model_default_params: use_extra_bufts=true; "
                            "llama-simple-chat has no -nr)",
                  "nproc": os.cpu_count(), "limit_conversations": a.limit_conversations,
                  "groups": [], "requests": []}
    meta_path = out / f"routing_{a.tag}_meta.json"
    parts: list[Path] = []
    for n_predict in sorted(groups):
        part = out / f"routing_{a.tag}_n{n_predict}.bin"
        argv = [a.exe, "-m", a.model, "-c", str(a.ctx), "-n", str(n_predict), "-t", str(a.threads),
                "--temp", "0", "--raw", "--batch", "--dump-routing", str(part)]
        print("==", " ".join(argv), flush=True)
        t0 = time.monotonic()
        chat = Chat(argv, out / f"simple_chat_{a.tag}_n{n_predict}.stderr.log")
        chat.read_until_ready(1800)
        load_s = round(time.monotonic() - t0, 2)
        print(f"   loaded in {load_s} s", flush=True)
        for c in groups[n_predict]:
            replies: list[str] = []
            n_user = len(wl.user_turns(c))
            for k in range(n_user):
                msgs = wl.request_messages(c, k, replies)
                prompt = wl.render_gemma4(msgs, bos=False)
                text, tm = chat.ask(prompt, 3600)
                replies.append(text)
                rec = {"id": c["id"], "k": k, "lang": c.get("lang"), "category": c.get("category"),
                       "max_tokens": n_predict, "group_file": part.name,
                       "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                       "prompt_chars": len(prompt), "reply": text, **tm}
                meta["requests"].append(rec)
                print(f"   {c['id']}#{k}: prompt={tm.get('prompt_tokens')} gen={tm.get('gen_tokens')} "
                      f"wall={tm.get('wall_s')} s", flush=True)
                json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        rc = chat.close()
        tensors = None
        for ln in chat.err_lines:
            m = re.search(r"\[routing-dump\] (\d+) tensors written", ln)
            if m:
                tensors = int(m.group(1))
        meta["groups"].append({"n_predict": n_predict, "argv": argv, "file": part.name, "rc": rc, "load_s": load_s,
                               "bytes": part.stat().st_size if part.exists() else None,
                               "sha256": sha256_file(part) if part.exists() else None,
                               "tool_reported_tensors_written": tensors,
                               "pinned_or_warn_lines": [ln.strip() for ln in chat.err_lines
                                                        if re.search(r"expert-cache|warning|error", ln, re.I)][:20]})
        parts.append(part)
    # one routing file for make_hot_file.py: plain concatenation (the format is a flat record stream)
    cat = out / f"routing_{a.tag}.bin"
    with open(cat, "wb") as w:
        for p in parts:
            with open(p, "rb") as r:
                for b in iter(lambda: r.read(8 << 20), b""):
                    w.write(b)
    meta["routing_file"] = cat.name
    meta["routing_bytes"] = cat.stat().st_size
    meta["routing_sha256"] = sha256_file(cat)
    meta["concat_order"] = [p.name for p in parts]
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["totals"] = {"requests": len(meta["requests"]),
                      "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in meta["requests"]),
                      "gen_tokens": sum(r.get("gen_tokens") or 0 for r in meta["requests"])}
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("DUMP_DONE", json.dumps(meta["totals"]), cat, meta["routing_bytes"], flush=True)
    return 0 if all(g["rc"] == 0 for g in meta["groups"]) else 2


if __name__ == "__main__":
    sys.exit(main())
