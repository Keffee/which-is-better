# Paper 1 Main Experiment 4B Release

This repository is the anonymous code release for the **Paper 1 main experiment** built around the `Qwen3.5-4B` line.

It keeps only the method-critical path needed to understand and reproduce the accepted 4B results on:

- Amazon Beauty
- Amazon Movies and TV
- Amazon Toys and Games
- Yelp

## What Is Included

- dataset-specific SFT config templates
- stage-121 dynamic-alpha code used by the accepted Beauty line
- stage-17 candidate construction
- stage-19 structured memory graph construction
- stage-41 adaptive memory selector
- stage-103 conflict-calibrated memory head
- paper-ready metric tables for the accepted 4B endpoints

## What Is Not Included

- raw datasets
- model checkpoints
- private logs and experiment farm scripts
- remote monitoring and recovery tooling
- unrelated 9B and baseline branches

## Accepted Endpoints

- `beauty`: `121 predicted_main`
- `movies_and_tv`: `41 adaptive_memory_type_selector`
- `toys_and_games`: `41 adaptive_memory_type_selector`
- `yelp`: `103 conflict_calibrated_memory_head_local`

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare local data under the layout described in [docs/datasets.md](docs/datasets.md), then run:

```bash
bash scripts/run_beauty.sh
bash scripts/run_movies_and_tv.sh
bash scripts/run_toys_and_games.sh
bash scripts/run_yelp.sh
```

The accepted `Qwen3.5-4B` paper table is stored in:

- [paper1_4b_main_metrics.md](paper_results/paper1_4b_main_metrics.md)
- [paper1_4b_main_metrics.csv](paper_results/paper1_4b_main_metrics.csv)

This package focuses on the method-critical code and public configuration templates. Generic SFT trainer infrastructure is not duplicated here; the exact dataset-local SFT settings are exposed via `configs/<dataset>/sft.yaml`.
