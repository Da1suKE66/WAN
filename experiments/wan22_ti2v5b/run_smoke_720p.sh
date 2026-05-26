#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WAN_PYTHON="${WAN_PYTHON:-experiments/wan_baseline/envs/lightx_overlay/bin/python}"

"$WAN_PYTHON" experiments/wan22_ti2v5b/scripts/preflight_wan22_ti2v5b.py \
  --config experiments/wan22_ti2v5b/configs/smoke_720p_ti2v5b.json

exec "$WAN_PYTHON" experiments/wan22_ti2v5b/scripts/run_wan22_batch.py \
  --python "$WAN_PYTHON" \
  --config experiments/wan22_ti2v5b/configs/smoke_720p_ti2v5b.json \
  "$@"
