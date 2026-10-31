# MRAGE NC5 tuning, 2026-10-30

Scope: Movies, Toys, Grocery, ele-fashion, Reddit-S, MRAGE only. The original
`configs/model/mrage.yaml` is retained. Base model/task snapshots and separate
candidate overlays reproduce the study without changing the default config.

The target is stronger final Test Accuracy. Candidate and checkpoint selection
use Validation Accuracy. Search trials mask Test labels and do not evaluate
Test. Test is evaluated only after the configuration is selected and the
selection is recorded. MAGB NC uses the official seed42 split; ele-fashion uses
its existing official split.pt. Training seeds are 42, 43, and 44.

The owner's latest strategy supersedes the initial capacity-first queue:

1. Use Movies, Grocery and ele-fashion with seed42 as representatives.
2. Sequentially tune learning rate, then weight decay and dropout.
3. Sequentially tune gamma_init, beta_init and rho_init.
4. Tune router_hidden_dim, attention_dim, router_embed_dim and channel_gate_hidden.
5. Tune auxiliary loss weights, expert_bottleneck and expert_eta.
6. Consider hidden_dim last, based on convergence/underfitting evidence.
7. Extend a small shortlist to all five datasets and seeds42/43/44. Use mean
   Validation Accuracy first and Macro-F1 second; Test does not select settings.
8. Record the choice before evaluating final Test results.

Each sweep changes one parameter relative to its recorded anchor. The sweep
winner becomes the next anchor; this is coordinate search, not a Cartesian
parameter grid. Parameter values are listed in study.yaml. The completed NC5
default baseline and the extra Toys learning-rate run remain in the ledger;
they are not used to expand the representative screening set.

Each trial retains resolved config, train log, best Validation checkpoint,
Validation metrics, split fingerprints, run metadata, status, and elapsed time.
The study ledger is `trials.csv` / `trials.json`; detailed run files are under
`outputs/nc5_tuning_20261030/`. `selection.json` will document the final choice
before any Test results are read. Final metrics are exploratory tuning results,
not evidence of improvement until the relevant runs have completed.

The initial queue used physical GPU1 while GPU0 was occupied by a pre-existing
job. That unrelated job has now finished. Subsequent representative searches
can use both GPUs after checking that each has no existing compute process.
OMP_NUM_THREADS and MKL_NUM_THREADS are set to 8 for these trial processes.
The owner reserved GPU0 after its active trial completed (hardware_schedule.json).
Remaining training and Final Test use physical GPU1 only. Final Test is sequential.
For checkpoints trained on logical cuda:0, the wrapper uses CUDA_VISIBLE_DEVICES=1
so their saved logical cuda:0 maps to physical GPU1; the saved config is untouched.
The launcher ledger records this physical/logical distinction.
The observed sandbox NVIDIA error does not indicate a host driver failure:
host-level nvidia-smi works. Dataset and original research files are read only.

The initial launcher preflight rejected an unsupported task.torch_threads
overlay before any training began. Its candidate files and ledger are retained
under preflight_failed/. CPU thread limits now use environment variables.

Completed: 136 Validation training runs over 33 configurations; all three finalists
confirmed on all five datasets with seeds42/43/44. Selection frozen in selection.json.
30 final Test evaluations (baseline + selected common config) completed on physical
GPU1. See REPORT.md, selected_common.yaml, validation_summary.csv and
final_test_summary.csv. Original model/task configs and MRAGE NC source unchanged.
