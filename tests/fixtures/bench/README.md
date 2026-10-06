# Fixtures for tests/test_bench_parse.py

Every capture below came from a real binary at llama.cpp a7b94df (b11398). Only the
scratchpad directory path was replaced with `<scratch>`. Nothing was edited by hand.

| File | Source |
|---|---|
| `llama_bench_tiny.json` | `llama-bench.exe -m tiny-q4_0.gguf -t 2,4 -p 32 -n 16 -r 3 -o json` (Windows official build, 2026-10-05). The model is a 76 KB random-weight llama-arch GGUF with Q4_0 matrices, written by gguf-py, so the tok/s values mean nothing. |
| `llama_bench_fail.stdout.txt` / `.stderr.txt` | The same tool with `-fa on -ctk q8_0 -ctv q8_0` on a tiny model with head_dim 16. It failed to create the context, exited with 1, and stdout was only `[`. This is a real truncated array. |
| `completion_tiny_lv4.stderr.txt` / `.stdout.txt` | `llama-completion.exe -m tiny-q4_0.gguf -f p.txt -n 24 --temp 0 -t 2 -c 512 -no-cnv --no-display-prompt -lv 4` (default variant). The stderr has `load_tensors: CPU_Mapped` and `CPU_REPACK` model buffer lines, which only print at `-lv 4`. |
| `completion_tiny_norepack.stderr.txt` | A `pigemma.bench completion --chat --variant norepack` run (`-nr`) on the tiny model. It has a CPU_Mapped line and no CPU_REPACK line. |
| `completion_26b_smoke.stderr.txt` | A copy of `results/smoke-26b-moe.stderr.txt`: a real 26B-A4B Q4_0 `llama-completion` run at default verbosity, so it has no buffer lines. |
| `completion_pi_31b.stderr.txt` | A copy of `results/pi-real/31b-iq3xxs/gen-en.stderr`: a real Raspberry Pi 5 16 GB run, 31B UD-IQ3_XXS, 4 threads. |
| `server_moe_plain.json` / `server_moe_mtp.json` / `server_moe_dflash.json` | Copies of `results/specsrv-moe-{baseline,mtp,dflash}.json`. These are real llama-server `/completion` responses for the 26B-A4B at 6 threads with 200 greedy tokens: the plain run, `-md` MTP (no `draft_n`, so it never drafted) and `-md` DFlash (321 drafted / 91 accepted). |
| `membw_sample.json` | `membw --size-mb 64 --threads 1,2 --json` from tools/build-linux-x86/bin in the Docker image (pi-gemma4-build, WSL2 kernel), 2026-10-05. |
| `fake_llama_server.py`, `fake_llama_completion.py`, `fake_llama_bench.py` | Stdlib test doubles that mimic the tools' CLI and output, so the subprocess, HTTP and lock paths of `pigemma.bench` can be tested without a model. |

The DFlash response's visible text differs from the plain run's after about 10 tokens (difflib
similarity 0.51 with `autojunk=False`). The MTP response is byte-identical to plain because MTP never drafted. The
test suite pins both facts through `compare_to_plain`.
