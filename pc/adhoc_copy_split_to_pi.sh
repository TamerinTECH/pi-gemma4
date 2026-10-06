#!/usr/bin/env bash
# Split a GGUF into < 4 GB parts (FAT32 limit on the Pi's USB stick) and copy them to the Pi.
# Usage: pc/adhoc_copy_split_to_pi.sh <src.gguf> <remote_dir> [max_size e.g. 3500M]
# llama.cpp loads split GGUFs natively: pass the first part (-00001-of-0000N.gguf) to -m.
set -euo pipefail
src=$1; rdir=$2; max=${3:-3500M}
cd REPO_ROOT
PI="-o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -o HostKeyAlias=context-friend-1"
HOST=pi@PI_HOST
stem=$(basename "$src" .gguf)
tmp=models/_split_tmp/$stem; mkdir -p "$tmp"
if ls "$tmp"/*-00001-of-*.gguf >/dev/null 2>&1; then echo "[$(date +%T)] reusing existing split parts in $tmp"; else
  echo "[$(date +%T)] splitting $src -> $tmp (max $max)"
  third_party/llama.cpp-win/llama-gguf-split.exe --split --split-max-size "$max" "$src" "$tmp/$stem" > "$tmp/split.log" 2>&1
fi
ls -la "$tmp"
ssh $PI $HOST "mkdir -p '$rdir'"
for f in "$tmp"/*.gguf; do
  n=$(basename "$f"); sz=$(stat -c %s "$f")
  for attempt in $(seq 1 40); do
    rsz=$(ssh $PI $HOST "stat -c %s '$rdir/$n' 2>/dev/null || echo 0") || { sleep 15; continue; }
    [ "$rsz" = "$sz" ] && break
    # resumable: append the missing tail to <name>.part (a Wi-Fi reset only loses the bytes in flight)
    have=$(ssh $PI $HOST "stat -c %s '$rdir/$n.part' 2>/dev/null || echo 0") || { sleep 15; continue; }
    if [ "$have" -gt "$sz" ]; then ssh $PI $HOST "rm -f '$rdir/$n.part'"; have=0; fi
    echo "[$(date +%T)] copy $n from byte $have of $sz (attempt $attempt)"
    t0=$(date +%s)
    if tail -c +$((have+1)) "$f" | ssh $PI $HOST "cat >> '$rdir/$n.part'"; then :; else echo "transfer interrupted"; sleep 10; fi
    have=$(ssh $PI $HOST "stat -c %s '$rdir/$n.part' 2>/dev/null || echo 0") || { sleep 15; continue; }
    t1=$(date +%s)
    if [ "$have" = "$sz" ]; then
      ssh $PI $HOST "mv '$rdir/$n.part' '$rdir/$n'" && echo "[$(date +%T)] $n done ($((t1-t0)) s for this attempt)"
    fi
  done
  rsz=$(ssh $PI $HOST "stat -c %s '$rdir/$n' 2>/dev/null || echo 0")
  [ "$rsz" = "$sz" ] || { echo "FAILED $n remote=$rsz local=$sz"; exit 4; }
done
echo "[$(date +%T)] verifying sha256 of parts"
( cd "$tmp" && sha256sum *.gguf ) > "$tmp/SHA256SUMS"
scp -q $PI "$tmp/SHA256SUMS" "$HOST:$rdir/SHA256SUMS"
ssh $PI $HOST "cd '$rdir' && nice -n 10 sha256sum -c SHA256SUMS" && rm -rf "$tmp" && echo "COPY_DONE $rdir"
