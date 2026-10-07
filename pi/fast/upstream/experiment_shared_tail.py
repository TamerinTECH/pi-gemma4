#!/usr/bin/env python3
"""Throw-away experiment for the padded-tail patch on master: is the per-thread preparation of the padded block
the reason why the upstream-style version is slower than the experiment-local one?

  experiment_shared_tail.py <repack.cpp>            rewrite the file: thread 0 prepares one shared block
  experiment_shared_tail.py <repack.cpp> --restore  put the saved original back

The shared variant has the data race described in README.md and is NOT a candidate for upstream.
"""
import shutil
import sys

path = sys.argv[1]
backup = path + ".pt2"
if "--restore" in sys.argv:
    shutil.copyfile(backup, path)
    print("restored")
    sys.exit(0)
shutil.copyfile(path, backup)
code = open(path, encoding="utf-8").read()


def rep(old, new):
    global code
    if code.count(old) != 1:
        raise SystemExit(f"expected one occurrence, found {code.count(old)}: {old[:70]}")
    code = code.replace(old, new)


rep("        static thread_local std::vector<char>  tail_q;\n", "        static std::vector<char> tail_q; // EXPERIMENT: shared\n")
rep("        if (pad_tail) {\n", "        if (pad_tail && ith == 0) {\n")
rep("        const char * src1_tail = pad_tail ? tail_q.data() : nullptr;\n", "")
rep("        ggml_barrier(params->threadpool);\n\n        // The first chunk comes from our thread_id",
    "        ggml_barrier(params->threadpool);\n        const char * src1_tail = pad_tail ? tail_q.data() : nullptr;\n\n        // The first chunk comes from our thread_id")
open(path, "w", encoding="utf-8", newline="\n").write(code)
print("shared-buffer variant written")
