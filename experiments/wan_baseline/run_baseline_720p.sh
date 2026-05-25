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
  "$WAN_PYTHON" experiments/wan_baseline/scripts/preflight_baseline.py \
    --config experiments/wan_baseline/configs/baseline_720p_t2v14b.json
fi

exec "$WAN_PYTHON" experiments/wan_baseline/scripts/run_batch_inference.py \
  --python "$WAN_PYTHON" \
  --config experiments/wan_baseline/configs/baseline_720p_t2v14b.json \
  "$@"
