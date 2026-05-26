#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WAN_PYTHON="${WAN_PYTHON:-experiments/wan_baseline/envs/lightx_overlay/bin/python}"

dry_run=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    dry_run=1
  fi
done

if [[ "$dry_run" -eq 0 ]]; then
  "$WAN_PYTHON" experiments/wan22_ti2v5b/scripts/preflight_wan22_ti2v5b.py \
    --config experiments/wan22_ti2v5b/configs/baseline_720p_ti2v5b.json
fi

exec "$WAN_PYTHON" experiments/wan22_ti2v5b/scripts/run_wan22_batch.py \
  --python "$WAN_PYTHON" \
  --config experiments/wan22_ti2v5b/configs/baseline_720p_ti2v5b.json \
  "$@"
