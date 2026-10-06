#!/usr/bin/env bash
# pi-gemma4 ad-hoc: guarded GGUF download for the real Pi 5.
# Usage: fetch_gguf.sh <hf_repo> <file> <size_bytes> <sha256> [min_free_after_bytes]
# - refuses if free space after the download would drop below min_free_after (default 2 GiB)
# - resumes partial downloads (curl -C -), idle I/O priority, verifies size and SHA-256
set -euo pipefail
repo=$1; file=$2; size=$3; sha=$4; min_free=${5:-2147483648}
dir=~/pigemma/models; mkdir -p "$dir"
dst="$dir/$file"; part="$dst.part"
if [ -f "$dst" ] && [ "$(stat -c %s "$dst")" = "$size" ]; then echo "already present: $dst"; exit 0; fi
have=0; [ -f "$part" ] && have=$(stat -c %s "$part")
free=$(df -B1 --output=avail "$dir" | tail -1)
need=$(( size - have ))
after=$(( free - need ))
echo "free=$free need=$need free_after=$after min_free_after=$min_free"
if [ "$after" -lt "$min_free" ]; then echo "REFUSED: not enough disk (would leave $after bytes)"; exit 3; fi
url="https://huggingface.co/$repo/resolve/main/$file"
start=$(date +%s)
# Resume loop: one curl per attempt, each resuming from the CURRENT .part size.
# (curl's own --retry restarts from the offset of the first attempt and can truncate the file.)
for attempt in $(seq 1 60); do
  cur=0; [ -f "$part" ] && cur=$(stat -c %s "$part")
  [ "$cur" = "$size" ] && break
  ionice -c3 nice -n 10 curl -fL -C - -o "$part" -sS --connect-timeout 30 --speed-limit 100000 --speed-time 60 "$url" && break
  echo "attempt $attempt failed at $(stat -c %s "$part" 2>/dev/null || echo 0) bytes; retrying in 10 s"
  sleep 10
done
end=$(date +%s)
got=$(stat -c %s "$part")
echo "downloaded_bytes=$got seconds=$((end-start)) avg_MBps=$(( (got-have) / ((end-start)>0?(end-start):1) / 1000000 ))"
[ "$got" = "$size" ] || { echo "SIZE MISMATCH $got != $size"; exit 4; }
echo "verifying sha256..."
calc=$(ionice -c3 nice -n 10 sha256sum "$part" | cut -d' ' -f1)
[ "$calc" = "$sha" ] || { echo "SHA MISMATCH $calc"; exit 5; }
mv "$part" "$dst"
echo "OK $dst free_now=$(df -B1 --output=avail "$dir" | tail -1)"
echo FETCH_DONE
