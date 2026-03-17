# Paper 1 4B Main Chain

This package exposes the accepted `Qwen3.5-4B` chain for Paper 1.

## Shared Backbone Boundary

All four datasets start from the same high-level boundary:

1. fine-tune a `Qwen3.5-4B` backbone with the dataset-local SFT configuration
2. build aligned backbone cache artifacts
3. run the accepted downstream endpoint for that dataset

The SFT settings are kept in `configs/<dataset>/sft.yaml`.

## Dataset-Specific Accepted Endpoints

- Beauty: `121 predicted_main`
- Movies and TV: `41 adaptive_memory_type_selector`
- Toys and Games: `41 adaptive_memory_type_selector`
- Yelp: `103 conflict_calibrated_memory_head_local`

The accepted endpoint is the trustworthy branch-local best result for that dataset, not a forced uniform stage.

