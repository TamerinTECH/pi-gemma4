#!/usr/bin/env bash
# Desktop -> Pi: unsloth 31B UD-IQ3_XXS (download, sha256) + split copy to the USB stick + 31B MTP drafter.
set -euo pipefail
cd REPO_ROOT
export HF_TOKEN="$(sed -n 's/^HUGGING_FACE_TOKEN=//p' HF_TOKEN_FILE | tr -d '\r\n')"
python - <<'PY'
import os, hashlib
from huggingface_hub import hf_hub_download
p = hf_hub_download("unsloth/gemma-4-31B-it-GGUF", "gemma-4-31B-it-UD-IQ3_XXS.gguf", token=os.environ["HF_TOKEN"],
                    local_dir="models/unsloth__gemma-4-31B-it-GGUF")
h = hashlib.sha256()
with open(p, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
assert h.hexdigest() == "beb8b6e69da5d3778d781664d376f94962528c84982e1bf5aa7453d9b2d739b8", h.hexdigest()
print("downloaded+verified", p, os.path.getsize(p))
PY
src=models/unsloth__gemma-4-31B-it-GGUF/gemma-4-31B-it-UD-IQ3_XXS.gguf
tmp=models/_split_tmp/gemma-4-31B-it-UD-IQ3_XXS; mkdir -p "$tmp"
if ! ls "$tmp"/*-00001-of-*.gguf >/dev/null 2>&1; then
  third_party/llama.cpp-win/llama-gguf-split.exe --split --split-max-size 3500M "$src" "$tmp/gemma-4-31B-it-UD-IQ3_XXS" > "$tmp/split.log" 2>&1
fi
rm -f "$src"   # D: is tight; re-download in ~3 min if the desktop needs it (sha256 beb8b6e6...)
df -h /d | tail -1
bash pc/adhoc_copy_split_to_pi.sh "$src" /mnt/usb/pigemma/models/31b-iq3xxs 3500M
PI="-o BatchMode=yes -o ConnectTimeout=30 -o HostKeyAlias=context-friend-1"
scp -q $PI models/unsloth__gemma-4-31B-it-qat-GGUF/MTP/mtp-gemma-4-31B-it-Q4_0.gguf pi@PI_HOST:/mnt/usb/pigemma/models/31b-iq3xxs/mtp-gemma-4-31B-it-Q4_0.gguf
ssh $PI pi@PI_HOST "stat -c %s /mnt/usb/pigemma/models/31b-iq3xxs/mtp-gemma-4-31B-it-Q4_0.gguf"
echo ALL_31B_DONE
