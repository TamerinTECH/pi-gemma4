#!/usr/bin/env bash
# Desktop CPU lock helper (CLAUDE.md "Brief UPDATE 2 + coordination rules"):
#   results/.cpu.lock = JSON {owner, purpose, start}; created atomically (noclobber), removed when done,
#   stale after 6 h. Usage: cpu_lock.sh acquire <owner> <purpose> [max_wait_s] | release <owner> | status
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LOCK="$REPO/results/.cpu.lock"
cmd="${1:-status}"
case "$cmd" in
  status)
    if [ -f "$LOCK" ]; then echo "HELD: $(cat "$LOCK")"; exit 1; else echo "FREE"; exit 0; fi ;;
  acquire)
    owner="${2:?owner}"; purpose="${3:?purpose}"; max_wait="${4:-7200}"; waited=0
    while :; do
      if ( set -o noclobber; printf '{"owner": "%s", "purpose": "%s", "start": "%s", "pid_note": "git-bash wrapper, see pc/x1a"}\n' \
             "$owner" "$purpose" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK" ) 2>/dev/null; then
        echo "ACQUIRED $LOCK: $(cat "$LOCK")"; exit 0
      fi
      # stale after 6 h (file mtime)
      now=$(date +%s); mt=$(stat -c %Y "$LOCK" 2>/dev/null || echo "$now")
      if [ $((now - mt)) -gt 21600 ]; then
        echo "STALE lock (> 6 h): $(cat "$LOCK" 2>/dev/null) - not removing automatically; report it" >&2; exit 3
      fi
      if [ "$waited" -ge "$max_wait" ]; then echo "TIMEOUT waiting for $LOCK: $(cat "$LOCK" 2>/dev/null)" >&2; exit 4; fi
      echo "waiting: lock held: $(cat "$LOCK" 2>/dev/null)" >&2
      sleep 60; waited=$((waited + 60))
    done ;;
  release)
    owner="${2:?owner}"
    if [ -f "$LOCK" ]; then
      if grep -q "\"owner\": \"$owner\"" "$LOCK"; then rm -f "$LOCK"; echo "RELEASED"; else echo "NOT MINE: $(cat "$LOCK")" >&2; exit 5; fi
    else echo "no lock"; fi ;;
  *) echo "usage: $0 acquire <owner> <purpose> [max_wait_s] | release <owner> | status" >&2; exit 2 ;;
esac
