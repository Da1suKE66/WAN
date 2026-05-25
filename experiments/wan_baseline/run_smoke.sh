#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WAN_PYTHON="${WAN_PYTHON:-experiments/wan_baseline/envs/lightx_overlay/bin/python}"
exec "$WAN_PYTHON" experiments/wan_baseline/scripts/run_batch_inference.py \
  --python "$WAN_PYTHON" \
  --config experiments/wan_baseline/configs/smoke_1p3b_480p_f1.json \
  "$@"
