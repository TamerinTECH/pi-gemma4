#!/usr/bin/env bash
# pigemma-fast: durable work area for the patched source and build trees (replaces /tmp/llama-fast).
# The USB stick is FAT32 (no symlinks, 4 GiB file limit), so the trees live in an ext4 image file on the stick,
# loop-mounted at /mnt/pigemma-fast. Nothing is written to /etc/fstab; run this once after every boot.
# Reversible: sudo umount /mnt/pigemma-fast   (the image file stays on the stick)
set -euo pipefail
stick=/mnt/usb
img=$stick/pigemma/fast/pigemma-work.ext4
mnt=/mnt/pigemma-fast
size=${PIGEMMA_WORK_SIZE:-3600M}   # below the FAT32 4 GiB file limit
findmnt "$stick" > /dev/null || sudo mount -o rw,uid=1000,gid=1000,umask=022,noatime /dev/sda1 "$stick"
export PATH="$PATH:/usr/sbin:/sbin"
if [ ! -f "$img" ]; then
  mkdir -p "$(dirname "$img")"
  truncate -s "$size" "$img"
fi
if ! blkid "$img" 2>/dev/null | grep -q 'TYPE="ext4"'; then
  mkfs.ext4 -q -F -L pigemma-work "$img"
  echo "created ext4 file system in $img ($(stat -c %s "$img") bytes)"
fi
if ! findmnt "$mnt" > /dev/null; then
  sudo mkdir -p "$mnt"
  sudo mount -o loop,noatime "$img" "$mnt"
  sudo chown "$(id -u):$(id -g)" "$mnt"
fi
df -h "$mnt" | tail -1
echo WORK_MOUNTED
