#!/usr/bin/env bash
# X4: copy the Pi-side results back and run the analysis. Usage: pc/x4/fetch_results_from_pi.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PI="${PI_HOST:?Set PI_HOST=user@host}"
SSHO="-o BatchMode=yes -o ConnectTimeout=20 -o HostKeyAlias=context-friend-1"
DST="$REPO/results/pi-real/x4-resident-bottleneck/pi"
mkdir -p "$DST"
# shellcheck disable=SC2086
SRC=$(ssh $SSHO "$PI" 'if [ -d /mnt/usb/pigemma/x4/results ]; then echo /mnt/usb/pigemma/x4/results; else echo /dev/shm/pigemma-x4-results; fi')
echo "source on the Pi: $SRC"
# shellcheck disable=SC2086
scp -r $SSHO "$PI:$SRC/." "$DST/"
# shellcheck disable=SC2086
scp $SSHO "$PI:/dev/shm/x4_all.out" "$DST/x4_all.out" || true
cd "$REPO" && PYTHONUTF8=1 python pi/adhoc/x4/x4_analyze_pi.py --pi-dir "$DST"
