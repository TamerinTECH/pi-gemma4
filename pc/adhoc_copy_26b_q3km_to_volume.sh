#!/usr/bin/env bash
# P4: copy the verified 26B UD-Q3_K_M GGUF into Docker volume pigemma-models (ext4 on the Docker VM's /dev/sdc 8:32)
# and verify sha256 inside the VM. Heavy I/O window is logged so concurrent timing runs can be cross-checked.
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
REPO_W="$(cd "$REPO" && (pwd -W 2>/dev/null || pwd))"
cd "$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
echo "[$(date '+%F %T')] HEAVY_IO_START copy to volume; free space:"; df -h /c /d /f | tail -3
docker volume inspect pigemma-models --format '{{.Name}} {{.Mountpoint}} {{.Labels}}'
MSYS_NO_PATHCONV=1 docker run --rm -v pigemma-models:/models -v "$REPO_W/models:/src:ro" \
  pi-gemma4-build:latest bash -c '
set -euo pipefail
d=/models/unsloth__gemma-4-26B-A4B-it-GGUF; f=gemma-4-26B-A4B-it-UD-Q3_K_M.gguf
mkdir -p $d; df -h /models | tail -1
t0=$(date +%s.%N)
if [ -f $d/$f ] && [ "$(stat -c %s $d/$f)" = 12728497888 ] && [ -f $d/SHA256SUMS ]; then echo "already present"; else
  cp /src/unsloth__gemma-4-26B-A4B-it-GGUF/$f $d/$f.tmp && mv $d/$f.tmp $d/$f; fi
t1=$(date +%s.%N); sync
echo "copy_s=$(awk -v a=$t0 -v b=$t1 "BEGIN{printf \"%.1f\", b-a}")"
cd $d; echo "811882f6df81c8ea204a5d382f196d6bd601245091786912d4c70df41cdfa052  $f" > SHA256SUMS
t2=$(date +%s.%N); sha256sum -c SHA256SUMS; t3=$(date +%s.%N); echo "sha256_s=$(awk -v a=$t2 -v b=$t3 "BEGIN{printf \"%.1f\", b-a}")"
ls -la $d; stat -c "size=%s blocks512=%b ino=%i dev=%D" $d/$f
df -h /models | tail -1; grep " /models " /proc/self/mountinfo
'
echo "[$(date '+%F %T')] HEAVY_IO_END; free space:"; df -h /c /d /f | tail -3
echo VOLUME_COPY_DONE
