#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT"

python -m src.stage17.prepare_candidates \
  --config "$REPO_ROOT/configs/yelp/stage17.yaml"

python -m src.stage19.prepare_memory_graph \
  --config "$REPO_ROOT/configs/yelp/stage19.yaml"

python -m src.stage103.evaluate_memory_graph \
  --config "$REPO_ROOT/configs/yelp/stage103.yaml" \
  --variant conflict_calibrated_memory_head_local \
  --output_prefix yelp_4b_conflict_calibrated_memory_head
