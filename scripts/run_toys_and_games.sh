#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT"

python -m src.stage17.prepare_candidates \
  --config "$REPO_ROOT/configs/toys_and_games/stage17.yaml"

python -m src.stage19.prepare_memory_graph \
  --config "$REPO_ROOT/configs/toys_and_games/stage19.yaml"

python -m src.stage41.evaluate_memory_graph \
  --config "$REPO_ROOT/configs/toys_and_games/stage41.yaml" \
  --variant adaptive_memory_type_selector \
  --output_prefix toys_4b_memory_graph_adaptive_selector

