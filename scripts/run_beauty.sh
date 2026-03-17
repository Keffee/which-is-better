#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT"

python -m src.stage121.prepare_eval_data \
  --config "$REPO_ROOT/configs/beauty/stage121.yaml"

python -m src.stage121.evaluate_dynamic_alpha \
  --config "$REPO_ROOT/configs/beauty/stage121.yaml" \
  --variant predicted_main \
  --output_prefix beauty_4b_stage121_predicted_main

