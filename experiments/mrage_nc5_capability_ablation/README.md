# MRAGE NC5 capability-level ablation

This second-round study adds four capability-level variants to the first-round
component ablations (`single_filter`, `unshared_experts`,
`no_semantic_propagation`, `no_topological_collaboration`, and
`no_semantic_refinement`). Round one isolates individual architecture
components; this round tests the larger information capabilities supplied by
physical topology, semantic KNN graphs, graph-stage cross-modal collaboration,
and graph propagation as a whole. It reuses existing Full MRAGE and Single
Filter results and writes only below `outputs/mrage_capability_nc5`; first-round
results remain under `outputs/mrage_ablation_nc5`.

The launcher delegates training and validation-selected, single-use Final Test
to `scripts/run_paper_benchmark.py`. It reads each dataset's current Full
configuration from `experiments/nc5_test_exploration_20261031/adopted/<dataset>.yaml`.

## Variants and research questions

| Variant | Question | Disabled paths | Retained paths |
|---|---|---|---|
| `physical_only` | Does the semantic KNN add value beyond observed topology? | Private Semantic filtering and all Semantic-neighbor attention in Late Refinement | Private Physical filtering, Early Topological Cross, Same-node/None Late candidates, JointViewContribution, Fusion, NC auxiliary CE |
| `semantic_only` | Does observed topology add value beyond modality semantic graphs? | Private and Collaborative Physical filter responses; Early Topological Cross | Private Semantic filtering, Semantic-neighbor/Same-node/None Late Refinement, JointViewContribution, Fusion, NC auxiliary CE |
| `no_cross_modal_collaboration` | Is graph-stage cross-modal collaboration better than private graph encoders followed by fusion? | Early Topological Cross and all Late cross-modal additions | All four Private Physical/Semantic relations, Shared Experts, Router, JointViewContribution, final Text–Visual Fusion, NC auxiliary CE |
| `feature_only` | How much does graph propagation add beyond multimodal node features? | All graph filters, Router, Balance Loss, Early Cross, and all Late Cross candidates | Original Text/Visual projectors, original fusion stack, Unimodal Heads, NC objective and auxiliary CE |

All variants preserve the six logical filter-call balance average. Disabled
filter calls contribute zero; `feature_only` returns zero balance and freezes
unused graph parameters, so the logged trainable parameter count reflects its
actual feature/fusion path. The objective weight is never changed to amplify an
ablation.

### Path details

- `physical_only` uses an independent Same-node-only Late path. That Late path
  never reads Semantic KNN edge indices and never calls sparse Semantic
  attention. The inherited forward still constructs the private semantic
  filter inputs, but the variant's `_filter` immediately returns zero for those
  two calls, so neither those operators nor their Balance Contributions can
  affect predictions. It reuses MRAGE's existing source selector, cross
  projection, channel gate, `cross_lambda`, `tau`, norm control, and dropout.
  With Same-node disabled, the selector safely falls back to None-only.
- `semantic_only` zeroes all physical responses and Early Cross. Therefore the
  existing zero-context fallback gives zero Collaborative Physical response,
  and Collaborative State equals Private State. The physical graph can still
  be loaded and used for the cache identity; it cannot affect the encoder's
  output after the cache has been attached.
- `no_cross_modal_collaboration` zeros Early Cross and returns the
  Collaborative State unchanged from Late. Paired RNG replay and the existing
  fallback keep Collaborative and Private states equal while final Fusion
  continues to merge both modalities.
- `feature_only` directly computes `H_t`, `H_v`, and the original Fusion. It
  ignores the physical edge argument and semantic operator buffers. Projectors,
  Fusion, and Unimodal Heads remain trainable; graph/cross modules are frozen.

All four variants inherit the per-dataset model and task hyperparameters from
the current adopted Full configurations at
`experiments/nc5_test_exploration_20261031/adopted/<dataset>.yaml`; these were
checked against the resolved first-round run configs. There is no
variant-specific tuning. Dataset splits remain fixed (MAGB split seed 42 and
official ele-fashion split).
Training seeds default to 42, 43, and 44. Checkpoint selection remains best
Validation Accuracy, followed by one independent Final Test evaluation.

The saved adopted task configuration uses `macro_f1_labels: all_classes` for
ele-fashion, yielding a raw 12-class metric that includes the empty class 5.
Keep that raw value with the run. For an 11-observed-class report, transform
each seed by `F1_11 = F1_12 * 12/11`; do not rerun Final Test solely for this
conversion. Accuracy is unchanged.

## Dry run

The launcher never executes training unless `--execute` is explicitly present.
One dataset, one variant, one seed:

```bash
python scripts/run_mrage_capability_nc5.py --datasets Movies --variants physical_only --seeds 42 --device cuda:0 --stage all --dry-run
```

The default is also a dry run. It prints the delegated command and lets the
existing runner print the exact config hash and run directory.

## Execute

One dataset, one variant, three seeds:

```bash
python scripts/run_mrage_capability_nc5.py --datasets Movies --variants physical_only --seeds 42 43 44 --device cuda:0 --stage all --execute
```

All four variants on all five datasets (60 seed-runs):

```bash
python scripts/run_mrage_capability_nc5.py --datasets Movies Toys Grocery ele-fashion Reddit-S --variants physical_only semantic_only no_cross_modal_collaboration feature_only --seeds 42 43 44 --device cuda:0 --stage all --execute
```

For validation first and Final Test afterwards:

```bash
python scripts/run_mrage_capability_nc5.py --datasets Movies Toys Grocery ele-fashion Reddit-S --variants physical_only semantic_only no_cross_modal_collaboration feature_only --seeds 42 43 44 --device cuda:0 --stage validation --execute
python scripts/run_mrage_capability_nc5.py --datasets Movies Toys Grocery ele-fashion Reddit-S --variants physical_only semantic_only no_cross_modal_collaboration feature_only --seeds 42 43 44 --device cuda:0 --stage all --resume --execute
```

Resume follows the existing Runner: it can continue from `validation_complete`
and skips completed cells. It does not restore optimizer state from an
interruption in the middle of training.

## Artifacts and audit fields

The default output root is `outputs/mrage_capability_nc5`, isolated from the
first-round output root. Each run is stored at
`<output-root>/<dataset>/nc/mrage_capability_ablation/<config-hash>/seed-<seed>/`;
the variant is part of the resolved config hash, so variants and datasets have
separate directories while the three seeds for one dataset/variant share a
config hash. Every seed directory stores the resolved config, training log,
best Validation checkpoint, Validation/Test metrics, status, and run metadata.
The runner records dataset, variant in the resolved model config, seed, source
commit/tree fingerprint, resolved config/hash, split path/SHA, semantic graph
identity/candidate/feature/V7B fingerprints, effective optimizer and actual
learning rate/weight decay, trainable model/head parameter counts, best epoch,
Validation Accuracy/Macro-F1, Test Accuracy/Macro-F1, checkpoint SHA, and Test
evaluation count. `train.log` and Hydra phase logs sit in that same seed
directory, so the log location is directly derivable from each run directory.

```bash
find outputs/mrage_capability_nc5 -name status.json -print
rg -n '"status": "failed"|"error"|FAILED' outputs/mrage_capability_nc5
find outputs/mrage_capability_nc5 -name train.log -print
```

The internal `full` mode exists only for synthetic parity testing and is not a
valid launcher choice. The first-round variants, Full results, checkpoints,
logs, and adopted configuration files are not changed by this framework.
