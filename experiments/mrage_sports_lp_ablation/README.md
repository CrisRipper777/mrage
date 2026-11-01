# MRAGE Sports LP ablations

This study uses the `sports-copurchase` dataset, which is LP-only in this
project. It runs the previous `single_filter` component ablation together with
the four capability variants defined for NC5 and adapted to sampled LP:
`physical_only`, `semantic_only`, `no_cross_modal_collaboration`, and
`feature_only`.

The LP implementation combines the existing ablation operators with
`src/models/mrage_lp.py`, preserving the project's sampled MRAGE adapter. The
recommended MRAGE LP configuration is loaded from
`experiments/lp_recommended_20261031/mrage_lp_default_h256.yaml`; it uses the
official fixed `lp-edge-split.pt`, seeds 42/43/44, shared `[5, 5, 5]` loader
fanouts, Adam at `1e-3`, weight decay `1e-5`, best Validation MRR checkpoint
selection, and one independent Final Test evaluation.

The launcher defaults to 15 seed-runs and does not train unless `--execute` is
passed. Results go to `outputs/mrage_sports_lp_ablation`, separate from the
existing MRAGE LP and NC ablation outputs. The existing Full MRAGE Sports
result can serve as a reference; the launcher does not rerun Full. It supports
`--stage validation` followed by `--stage all --resume`; the existing runner resumes only from a
completed Validation checkpoint, not from an interrupted epoch's optimizer
state.

`physical_only` uses only the standard sampled physical subgraph; semantic
KNN candidates are not allowed to expand the batch's physical context.
`feature_only` skips semantic closure and extra physical-context expansion.
`semantic_only`, `single_filter`, and `no_cross_modal_collaboration` use the
existing MRAGE LP semantic closure and sampled physical context as required by
their retained paths. The standard LP supervision loader remains the same for
all variants.

## Dry run

```bash
python scripts/run_mrage_sports_lp_ablation.py --dry-run
```

## Execute all 15 runs

```bash
python scripts/run_mrage_sports_lp_ablation.py --device cuda:0 --stage all --execute
```

## Validation first, then Final Test

```bash
python scripts/run_mrage_sports_lp_ablation.py --device cuda:0 --stage validation --execute
python scripts/run_mrage_sports_lp_ablation.py --device cuda:0 --stage all --resume --execute
```
