# MRAGE NC ablation runs

This runner uses the existing NC training, validation checkpoint selection, and
single Final Test path in `scripts/run_paper_benchmark.py`. It uses the adopted
Full MRAGE configuration independently for each dataset. The five datasets use
the fixed MAGB seed-42 splits or the existing official ele-fashion split, and
the default training seeds are 42, 43, and 44.

The matrix contains only `single_filter`, `unshared_experts`,
`no_semantic_propagation`, `no_topological_collaboration`, and
`no_semantic_refinement`. The internal `full` parity-test mode is intentionally
not accepted by the launcher.

## Dry run

One dataset, one variant, one seed:

```bash
python scripts/run_mrage_ablation_nc5.py --datasets Movies --variants single_filter --seeds 42 --device cuda:0 --stage all --dry-run
```

The default is also a dry run. The launcher prints the exact delegated command
and asks `run_paper_benchmark.py` to print each resolved config hash and output
directory. Add `--execute` only after reviewing those paths.

## Execute runs

One variant across all five datasets and three seeds:

```bash
python scripts/run_mrage_ablation_nc5.py --datasets Movies Toys Grocery ele-fashion Reddit-S --variants single_filter --seeds 42 43 44 --device cuda:0 --stage all --execute
```

All 75 seed-runs:

```bash
python scripts/run_mrage_ablation_nc5.py --datasets Movies Toys Grocery ele-fashion Reddit-S --variants single_filter unshared_experts no_semantic_propagation no_topological_collaboration no_semantic_refinement --seeds 42 43 44 --device cuda:0 --stage all --execute
```

To prepare validation-selected checkpoints first, then continue into the
single Final Test phase:

```bash
python scripts/run_mrage_ablation_nc5.py --datasets Movies Toys Grocery ele-fashion Reddit-S --variants single_filter unshared_experts no_semantic_propagation no_topological_collaboration no_semantic_refinement --seeds 42 43 44 --device cuda:0 --stage validation --execute
python scripts/run_mrage_ablation_nc5.py --datasets Movies Toys Grocery ele-fashion Reddit-S --variants single_filter unshared_experts no_semantic_propagation no_topological_collaboration no_semantic_refinement --seeds 42 43 44 --device cuda:0 --stage all --resume --execute
```

`--resume` follows the existing runner: it continues from
`validation_complete`, skips completed runs, and rejects states that do not
have a validated checkpoint. The existing runner does not resume optimizer
state from a mid-training interruption.

## Output and status

The default output root is `outputs/mrage_ablation_nc5`. Every seed has its own
config-hashed directory containing `best_validation.pt`, `train.log`,
`validation_metrics.json`, `final_test_metrics.json`, `run_metadata.json`, and
`status.json`. The metadata records the split fingerprint, semantic cache
identity/fingerprints, source commit, parameter counts, best validation epoch,
checkpoint hash, and final test metrics.

List run states and find failures/logs:

```bash
find outputs/mrage_ablation_nc5 -name status.json -print
rg -n '"status": "failed"|"error"|FAILED' outputs/mrage_ablation_nc5
find outputs/mrage_ablation_nc5 -name train.log -print
```

Only Validation Accuracy selects the checkpoint. Final Test Accuracy and Macro
F1 are calculated once from that checkpoint. This framework does not retrain
Full MRAGE or generate result tables before actual runs exist.
