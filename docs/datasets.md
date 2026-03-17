# Dataset Download And Local Layout

This release does **not** redistribute any dataset files.

## Official Sources

Amazon review datasets:

- Main index: [Amazon Review Data (McAuley Lab)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)
- Beauty reviews: [reviews_Beauty_5.json.gz](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz)
- Movies and TV reviews: [reviews_Movies_and_TV_5.json.gz](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz)
- Toys and Games reviews: [reviews_Toys_and_Games_5.json.gz](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz)

Yelp:

- Official page: [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)

## Expected Local Layout

```text
data/
├── processed/
│   ├── beauty/
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   ├── test.jsonl
│   │   └── item_texts.json
│   ├── movies_and_tv/
│   ├── toys_and_games/
│   └── yelp/
└── cache/
    ├── beauty/
    │   └── sft_best_full/
    │       ├── item_embeddings.pt
    │       ├── train_features.pt
    │       ├── val_features.pt
    │       └── test_features.pt
    ├── movies_and_tv/
    ├── toys_and_games/
    └── yelp/
```

`item_texts.json` should contain rows with `item_id` and `condensed_title`.

The processed `*.jsonl` files should expose the dataset-local history fields used by the stage scripts, including:

- `user_id`
- `target_item_id`
- `history_titles`
- `history_ratings`

For the structured-memory stages, rows may also include `history_reviews`.

