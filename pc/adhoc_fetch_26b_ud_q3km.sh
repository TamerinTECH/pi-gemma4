#!/usr/bin/env bash
# P4: download the primary MoE comparison file (the exact file used on the Pi) to the desktop and verify it.
# unsloth/gemma-4-26B-A4B-it-GGUF @ c099eb48e663fd284577b04978a94ffccb261841, gemma-4-26B-A4B-it-UD-Q3_K_M.gguf
# expected 12,728,497,888 bytes, sha256 811882f6df81c8ea204a5d382f196d6bd601245091786912d4c70df41cdfa052.
# The HF token is read from HF_TOKEN_FILE and never printed. Re-running is safe (skips a verified file).
set -euo pipefail
cd REPO_ROOT
export HF_TOKEN="$(sed -n 's/^HUGGING_FACE_TOKEN=//p' HF_TOKEN_FILE | tr -d '\r\n')"
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0   # no xet chunk cache on C:
echo "[$(date +%T)] free space before download:"; df -h /c /d | tail -2
python - <<'PY'
import os, hashlib, time
from huggingface_hub import hf_hub_download
REPO, REV = "unsloth/gemma-4-26B-A4B-it-GGUF", "c099eb48e663fd284577b04978a94ffccb261841"
FILE, SIZE = "gemma-4-26B-A4B-it-UD-Q3_K_M.gguf", 12728497888
SHA = "811882f6df81c8ea204a5d382f196d6bd601245091786912d4c70df41cdfa052"
dest = "models/unsloth__gemma-4-26B-A4B-it-GGUF"
t0 = time.time()
p = hf_hub_download(REPO, FILE, revision=REV, token=os.environ["HF_TOKEN"], local_dir=dest)
t1 = time.time()
size = os.path.getsize(p)
print(f"downloaded {p} {size} bytes in {t1-t0:.0f} s", flush=True)
assert size == SIZE, (size, SIZE)
h = hashlib.sha256()
with open(p, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
d = h.hexdigest()
print(f"sha256 {d} ({time.time()-t1:.0f} s)", flush=True)
assert d == SHA, d
with open(os.path.join(dest, "SHA256SUMS"), "w", encoding="ascii", newline="\n") as f:
    f.write(f"{SHA}  {FILE}\n")
print("VERIFIED_OK", flush=True)
PY
echo "[$(date +%T)] free space after download:"; df -h /c /d | tail -2
echo FETCH_DONE
