# Which is Better: A Selective Recommendation Correction Framework via Case-Conditioned Challenger Routing

This repository is the anonymous code release for the **Which is Better: A Selective Recommendation Correction Framework via Case-Conditioned Challenger Routing** built around the `Qwen3.5-4B` line.

It keeps only the method-critical path needed to understand and reproduce the accepted 4B results on:

- Amazon Beauty
- Amazon Movies and TV
- Amazon Toys and Games
- Yelp

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

This package focuses on the method-critical code and public configuration templates. Generic SFT trainer infrastructure is not duplicated here; the exact dataset-local SFT settings are exposed via `configs/<dataset>/sft.yaml`.
