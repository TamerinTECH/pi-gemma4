#!/usr/bin/env python3
"""X4: write the prompt file and the plan files for the Pi-side resident-bottleneck runs (desktop, no model).

Outputs (next to this script):
  x4_prompts.json        tuning prompts tun-01 / tun-06 and the warm-up, byte-identical to pi/adhoc/x2/x2_prompts.json
                         (so X4 numbers are comparable with X2's -t 3 / -t 4 tuning blocks), plus one long-generation
                         diagnostic prompt that belongs to no workload set.
  x4_plan_threads.json   UD-Q3_K_M, -nr, threads 4,1,3,2,2,3,1,4 (two interleaved passes), short + 687-token prompt
  x4_plan_longgen.json   UD-Q3_K_M, -nr, one generation of up to 640 tokens at 4 and at 3 threads
  x4_plan_repack.json    UD-Q3_K_M, -nr vs llama.cpp default repack (same binary), repeated requests until settled
  x4_plan_qat.json       Google QAT Q4_0 (5 split parts on the USB stick), -nr, 4 and 3 threads, repeated requests
  x4_plan_udref.json     UD-Q3_K_M, -nr, 4 threads again after the QAT block (same-session reference)
  x4_plan_desktop_smoke.json   FUNCTIONAL test of the driver in the desktop Docker image (not a measurement)

No held-out item is used: thread count and repack are settings (tuning set), the long generation is a diagnostic.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
from pigemma.workloads import render_gemma4  # noqa: E402

UD = "~/pigemma/models/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf"
QAT_DIR = "/mnt/usb/pigemma/models/26b-qat-q4_0"
QAT_PARTS = [f"{QAT_DIR}/gemma-4-26B_q4_0-it-{i:05d}-of-00005.gguf" for i in range(1, 6)]
COMMON = ["-c", "4096", "-np", "1", "-b", "2048", "-ub", "512", "--cache-ram", "0"]
SERVER = "~/pigemma/llama.cpp/build/bin/llama-server"

ENGINES = {
    "ours": {"what": "ggml-org/llama.cpp a7b94df, native Pi build, weights memory-mapped (-nr)",
             "bin": SERVER, "args": ["-nr"] + COMMON},
    "ours-repack": {"what": "same binary WITHOUT -nr: llama.cpp default repack (eligible tensors are copied into "
                            "anonymous CPU_REPACK memory)", "bin": SERVER, "args": list(COMMON)},
}
REQUEST = {"endpoint": "POST /completion", "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": False, "stream": True,
           "mtp": "off (no -md / --spec-type)"}

LONG_USER = ("Write a detailed, well-structured explanation of how bread is made, from choosing the flour to cooling "
             "the finished loaf. Cover mixing, kneading, the role of yeast and gluten, both rises, shaping, scoring, "
             "baking temperature and steam, and common mistakes. Write at least eight paragraphs and do not use lists.")


def blk(label, engine, threads, prompts):
    return {"label": label, "engine": engine, "config": f"t{threads}", "args": ["-t", str(threads)], "prompts": prompts}


def plan(model, purpose, blocks, **kw):
    p = {"schema": "x4-plan/1 (x2-plan/1 + model_files, cool_below_c, perf_counters, body_extra, repeated prompt ids)",
         "model": model, "prompts_file": "~/pigemma/x4/x4_prompts.json", "warmup_prompt_id": "warmup", "port": 18190,
         "gap_s": 45, "prewarm": True, "load_timeout_s": 900, "cool_below_c": 62, "cool_max_wait_s": 420,
         "perf_counters": True, "engines": ENGINES, "common_request": REQUEST, "purpose": purpose, "blocks": blocks}
    p.update(kw)
    return p


def main():
    with open(os.path.join(ROOT, "pi", "adhoc", "x2", "x2_prompts.json"), encoding="utf-8") as fh:
        x2 = json.load(fh)
    keep = [p for p in x2["prompts"] if p["id"] in ("warmup", "tun-01", "tun-06")]
    assert len(keep) == 3
    msgs = [{"role": "user", "content": LONG_USER}]
    long_p = {"id": "x4-long", "set": "x4-diagnostic (no workload set)", "lang": "en", "category": "long_generation",
              "max_tokens": 640, "user_text": LONG_USER,
              "prompt": render_gemma4(msgs, add_generation_prompt=True, enable_thinking=False, bos=False),
              "messages": msgs}
    assert long_p["prompt"].endswith(keep[0]["prompt"].split("<turn|>\n", 1)[1]), "generation-prompt suffix differs from X2"
    prompts = {"schema": "x4-prompts/1",
               "note": "warmup / tun-01 / tun-06 are byte-identical to pi/adhoc/x2/x2_prompts.json (pre-rendered with "
                       "pigemma.workloads.render_gemma4: GGUF Gemma 4 chat template, thinking disabled, bos=False; the "
                       "server adds BOS). x4-long is an X4 diagnostic prompt outside every workload set.",
               "source_sha256": x2.get("source_sha256"), "prompts": keep + [long_p]}
    files = {"x4_prompts.json": prompts}

    order = [4, 1, 3, 2, 2, 3, 1, 4]
    seen = {}
    blocks = []
    for t in order:
        seen[t] = seen.get(t, 0) + 1
        blocks.append(blk(f"thr-t{t}-r{seen[t]}", "ours", t, ["tun-01", "tun-06"]))
    files["x4_plan_threads.json"] = plan(
        UD, "Thread scaling 1/2/3/4 for prefill and decode, resident (no memory pressure), our a7b94df -nr build, the two "
            "X2 tuning requests (30-token chat, 687-token extraction), two interleaved passes (4,1,3,2 then 2,3,1,4). "
            "-tb is left at its default (= -t).", blocks)

    files["x4_plan_longgen.json"] = plan(
        UD, "One long generation (up to 640 tokens) at 4 threads and at 3 threads: storage reads and major faults during "
            "DECODE over a long generation (phase split at the first token), inter-token latency against position "
            "(context growth, thermal drift), temperature / frequency / throttle flags every second.",
        [blk("long-t4", "ours", 4, ["x4-long"]), blk("long-t3", "ours", 3, ["x4-long"])])

    rep6 = ["tun-06", "tun-01", "tun-06", "tun-01", "tun-06", "tun-01", "tun-06", "tun-01"]
    files["x4_plan_repack.json"] = plan(
        UD, "-nr (memory-mapped weights) against llama.cpp's default repack with the SAME binary, 4 threads. X2 saw that "
            "the repack run is not resident right after load (6.9 GiB of anonymous copies push file pages out; 1.1-2.3 GB "
            "re-read per request). Here the repack block repeats the two requests four times; only requests with zero "
            "storage reads count as resident. MemAvailable stayed above 8.5 GB in X2's repack block (floor 3 GiB).",
        [blk("nr-t4-a", "ours", 4, ["tun-01", "tun-06"]),
         blk("repack-t4", "ours-repack", 4, rep6),
         blk("nr-t4-b", "ours", 4, ["tun-01", "tun-06"])])

    qrep = ["tun-06", "tun-01", "tun-06", "tun-01", "tun-06", "tun-01"]
    files["x4_plan_qat.json"] = plan(
        QAT_PARTS[0],
        "Google QAT Q4_0 26B (5 split parts on the FAT32 USB stick, 14.44 GB), -nr, 4 threads then 3 threads. The file "
        "is at the page-cache limit of this board, so every part is read once before the block and the requests are "
        "repeated; only requests with zero storage reads count as resident. Loading it pushes part of the UD-Q3_K_M "
        "file out of the page cache (no drop_caches is used); x4_plan_udref.json re-reads it afterwards.",
        [blk("qat-t4", "ours", 4, qrep), blk("qat-t3", "ours", 3, ["tun-06", "tun-01", "tun-06", "tun-01"])],
        model_files=QAT_PARTS)

    files["x4_plan_udref.json"] = plan(
        UD, "UD-Q3_K_M -nr -t 4 again right after the QAT block: same-session reference for the QAT / UD ratio.",
        [blk("udref-t4", "ours", 4, ["tun-06", "tun-01", "tun-06", "tun-01"])])

    smoke = plan("/models/unsloth__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf",
                 "FUNCTIONAL test of x4_compare.py in the desktop Docker image (x86, PIGEMMA_EMULATED=1). Not a measurement.",
                 [blk("smoke-t4", "ours", 4, ["tun-01", "tun-01"])],
                 prompts_file="/work/pi/adhoc/x4/x4_prompts_smoke.json", prewarm=False, cool_below_c=None, gap_s=1)
    smoke["engines"] = {"ours": {"what": "desktop x86 Docker build of a7b94df", "args": ["-nr"] + COMMON,
                                 "bin": "/work/third_party/llama.cpp/build-linux-x86/bin/llama-server"}}
    files["x4_plan_desktop_smoke.json"] = smoke
    sm_prompts = json.loads(json.dumps(prompts))
    for p in sm_prompts["prompts"]:
        p["max_tokens"] = min(int(p["max_tokens"]), 12)
    files["x4_prompts_smoke.json"] = sm_prompts

    for name, obj in files.items():
        with open(os.path.join(HERE, name), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=1)
        print("wrote", name)
    print("x4-long prompt chars:", len(long_p["prompt"]))


if __name__ == "__main__":
    main()
