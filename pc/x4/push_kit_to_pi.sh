#!/usr/bin/env bash
# X4: copy the resident-bottleneck kit to the real Pi and print the launch command (does NOT start anything).
# Kit on the Pi: ~/pigemma/x4 (scripts, plans, membw.c: < 1 MB on /), /dev/shm/pigemma-x4-bin/kernbench (tmpfs).
# Usage (Git Bash on the desktop): pc/x4/push_kit_to_pi.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PI="${PI_HOST:?Set PI_HOST=user@host}"
SSHO="-o BatchMode=yes -o ConnectTimeout=20 -o HostKeyAlias=context-friend-1"
# shellcheck disable=SC2086
ssh $SSHO "$PI" 'set -e; mkdir -p ~/pigemma/x4 /dev/shm/pigemma-x4-bin; uptime;
  for s in contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base; do echo "$s $(systemctl is-active $s)"; done;
  pgrep -a "llama|powerinfer" || echo "no llama/powerinfer process";
  df -B1 --output=avail / | tail -1; grep MemAvailable /proc/meminfo; mountpoint /mnt/usb || true'
# shellcheck disable=SC2086
scp $SSHO "$REPO/pi/adhoc/x4/x4_run_pi.sh" "$REPO/pi/adhoc/x4/x4_compare.py" "$REPO/pi/adhoc/x4/x4_prompts.json" \
  "$REPO"/pi/adhoc/x4/x4_plan_threads.json "$REPO"/pi/adhoc/x4/x4_plan_longgen.json "$REPO"/pi/adhoc/x4/x4_plan_repack.json \
  "$REPO"/pi/adhoc/x4/x4_plan_qat.json "$REPO"/pi/adhoc/x4/x4_plan_udref.json "$REPO/tools/membw/membw.c" "$PI:pigemma/x4/"
if [ -f "$REPO/tools/build-pi-arm64/kernbench" ]; then
  # shellcheck disable=SC2086
  scp $SSHO "$REPO/tools/build-pi-arm64/kernbench" "$PI:/dev/shm/pigemma-x4-bin/kernbench"
  # shellcheck disable=SC2086
  ssh $SSHO "$PI" 'chmod +x /dev/shm/pigemma-x4-bin/kernbench; sha256sum /dev/shm/pigemma-x4-bin/kernbench; /dev/shm/pigemma-x4-bin/kernbench --version'
else
  echo "NOTE: tools/build-pi-arm64/kernbench not built (pc/x4/build_kernbench_arm64.sh) - the kernbench step will be skipped"
fi
# shellcheck disable=SC2086
ssh $SSHO "$PI" 'chmod +x ~/pigemma/x4/x4_run_pi.sh; ls -la ~/pigemma/x4; python3 -m py_compile ~/pigemma/x4/x4_compare.py && echo DRIVER_OK; bash -n ~/pigemma/x4/x4_run_pi.sh && echo RUNNER_OK'
cat <<'MSG'
Kit is on the Pi. Start everything (about 1.5-2 h, resident runs only, Pi bench lock held for the whole time):
  ssh -o BatchMode=yes -o ConnectTimeout=20 -o HostKeyAlias=context-friend-1 ${PI_HOST:?Set PI_HOST=user@host} \
    'nohup bash ~/pigemma/x4/x4_run_pi.sh all > /dev/shm/x4_all.out 2>&1 < /dev/null & echo started'
Poll:   ssh ... 'tail -5 /dev/shm/x4_all.out'      (ends with "X4_ALL_DONE rc=0 results=<dir>")
Fetch:  pc/x4/fetch_results_from_pi.sh
MSG
