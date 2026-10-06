#!/usr/bin/env bash
set -euo pipefail
base=/mnt/usb/pigemma/cf-engine
bash "$base/pi_operator.sh" operator-2
bash "$base/pi_operator.sh" operator-3
