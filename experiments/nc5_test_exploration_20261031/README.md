# MRAGE NC5 Test-guided exploration

This study explicitly uses Test metrics to select hyperparameters at the owner's request. These are exploratory results, separate from the prior Validation-selected formal Benchmark.

Default model/task files remain unchanged. All runs use GPU1, the fixed official NC splits and training seeds42/43/44. Checkpoints still use best Validation Accuracy.

Budget: at most65 training runs. See plan.json for candidates and prior evidence.

## Current adopted scheme

The owner adopted the original Movies default run outputs/2026-10-31/13-05-49.
See ADOPTED_RESULTS.md and adopted_selection.json; current configs are in adopted/.
selection.json and selected/ retain the historical search winners. No new training.
