#!/usr/bin/env bash
# pi-gemma4 ad-hoc: download one GGUF (< 4 GiB, FAT32 limit) from Hugging Face straight onto the USB stick.
# Usage: fetch_to_usb.sh <hf_repo> <file> <size_bytes> <sha256> <dest_dir_on_usb>
# Resumes with a loop of separate `curl -C -` calls (curl --retry can truncate), idle I/O priority, verifies sha256.
set -euo pipefail
repo=$1; file=$2; size=$3; sha=$4; dir=$5
[ "$size" -lt 4294967296 ] || { echo "REFUSED: $size bytes exceeds the FAT32 single-file limit; split it on the desktop"; exit 3; }
findmnt "$(dirname "$dir")" >/dev/null 2>&1 || findmnt /mnt/usb >/dev/null || { echo "REFUSED: /mnt/usb is not mounted"; exit 3; }
mkdir -p "$dir"; dst="$dir/$file"; part="$dst.part"
if [ -f "$dst" ] && [ "$(stat -c %s "$dst")" = "$size" ]; then echo "already present: $dst"; echo FETCH_DONE; exit 0; fi
free=$(df -B1 --output=avail "$dir" | tail -1)
[ "$free" -gt $(( size + 1073741824 )) ] || { echo "REFUSED: not enough space on the stick ($free)"; exit 3; }
url="https://huggingface.co/$repo/resolve/main/$file"
start=$(date +%s)
for attempt in $(seq 1 60); do
  cur=0; [ -f "$part" ] && cur=$(stat -c %s "$part")
  [ "$cur" = "$size" ] && break
  ionice -c3 nice -n 10 curl -fL -C - -o "$part" -sS --connect-timeout 30 --speed-limit 100000 --speed-time 60 "$url" && break
  echo "attempt $attempt interrupted at $(stat -c %s "$part" 2>/dev/null || echo 0) bytes; retrying in 10 s"; sleep 10
done
end=$(date +%s); got=$(stat -c %s "$part")
echo "downloaded_bytes=$got seconds=$((end-start))"
[ "$got" = "$size" ] || { echo "SIZE MISMATCH $got != $size"; exit 4; }
calc=$(ionice -c3 nice -n 10 sha256sum "$part" | cut -d' ' -f1)
[ "$calc" = "$sha" ] || { echo "SHA MISMATCH $calc"; exit 5; }
mv "$part" "$dst"; echo "OK $dst"; echo FETCH_DONE
