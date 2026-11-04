# MRAGE Figure 3: Shared Adaptive Graph Experts

This directory contains the read-only mechanism analysis for **Figure 3 —
Mechanistic Analysis of Shared Adaptive Graph Experts**.

- Baseline branch: `codex/mrage-sports-lp-ablation`
- Baseline commit: `dce34f8`
- Analysis branch: `codex/mrage-figure3-mechanism`
- Analysis type: frozen Full MRAGE checkpoints, `model.eval()` and
  `torch.inference_mode()` only
- No retraining, no hyperparameter search, no official Test evaluation, and no
  modification of model weights, checkpoints, semantic caches, or training runs.

## Checkpoints

The analysis uses the adopted Full MRAGE checkpoints recorded in
`experiments/nc5_test_exploration_20261031/adopted_selection.json`:

| Dataset | Seeds | Adopted configuration | Checkpoint root |
|---|---|---|---|
| Grocery | 42, 43, 44 | `tuned128` | `outputs/nc5_test_exploration_20261031/tuned128/Grocery/nc/mrage/a1f872eff7a7c1a9/seed-*/best_validation.pt` |
| Toys | 42, 43, 44 | `tuned128` | `outputs/nc5_test_exploration_20261031/tuned128/Toys/nc/mrage/8a883402bdb6a07e/seed-*/best_validation.pt` |
| Reddit-S | 42, 43, 44 | `reddit_aux01` | `outputs/nc5_test_exploration_20261031/reddit_aux01/Reddit-S/nc/mrage/700dd52266746cb9/seed-*/best_validation.pt` |
| ele-fashion | 42, 43, 44 | `fashion_gate64` | `outputs/nc5_test_exploration_20261031/fashion_gate64/ele-fashion/nc/mrage/91edd92176df744a/seed-*/best_validation.pt` |

Movies is omitted: the adopted historical Movies Full run has no saved
checkpoint (`save_ckpt_path=null`). No Movies model was retrained for this
analysis.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/mrage_figure3_common.py` | Checkpoint loading, semantic-cache attachment, context capture, routing statistics, router-row shuffling |
| `scripts/analyze_mrage_experts.py` | Figure 3(a): effective coefficients, initialization change, expert-output cosine |
| `scripts/analyze_mrage_router.py` | Figure 3(b): six-context Top-2 routing patterns and diagnostics |
| `scripts/analyze_mrage_router_shuffle.py` | Figure 3(c): frozen Validation router-shuffle intervention |
| `scripts/plot_mrage_figure3.py` | Reads the CSVs and renders Figure 3 and individual panels |
| `scripts/build_mrage_figure3_manifest.py` | Writes `manifest.json` with checkpoint hashes and analysis configuration |
| `tests/test_mrage_figure3_analysis.py` | Unit checks for coefficient formulas, routing statistics, and JSD |

## Quick start

Single-checkpoint smoke test:

```bash
conda run -n yhf_env env PYTHONPATH=. python scripts/analyze_mrage_experts.py \
  --dataset Grocery --seeds 42 --device cuda:0 \
  --output-dir experiments/mrage_figure3_mechanism/data/debug --max-nodes 1024

conda run -n yhf_env env PYTHONPATH=. python scripts/analyze_mrage_router.py \
  --dataset Grocery --seeds 42 --device cuda:0 \
  --output-dir experiments/mrage_figure3_mechanism/data/debug

conda run -n yhf_env env PYTHONPATH=. python scripts/analyze_mrage_router_shuffle.py \
  --dataset Grocery --seeds 42 --device cuda:0 \
  --output-dir experiments/mrage_figure3_mechanism/data/debug --repeats 2
```

Full analysis (12 checkpoints):

```bash
conda run -n yhf_env env PYTHONPATH=. python scripts/analyze_mrage_experts.py \
  --all --seeds 42 43 44 --device cuda:0 \
  --output-dir experiments/mrage_figure3_mechanism/data --max-nodes 4096

conda run -n yhf_env env PYTHONPATH=. python scripts/analyze_mrage_router.py \
  --all --seeds 42 43 44 --device cuda:0 \
  --output-dir experiments/mrage_figure3_mechanism/data

conda run -n yhf_env env PYTHONPATH=. python scripts/analyze_mrage_router_shuffle.py \
  --all --seeds 42 43 44 --device cuda:0 \
  --output-dir experiments/mrage_figure3_mechanism/data --repeats 10

conda run -n yhf_env env PYTHONPATH=. python scripts/plot_mrage_figure3.py
conda run -n yhf_env env PYTHONPATH=. python scripts/build_mrage_figure3_manifest.py
```

Run the focused tests:

```bash
conda run -n yhf_env env PYTHONPATH=. python -m pytest -q tests/test_mrage_figure3_analysis.py
```

The batch analysis was executed on `cuda:0` while the Sports LP training job
was using the same GPU; it used about 2.7 GB host RSS and completed without an
OOM. The analysis is sequential across checkpoints.

## Outputs

- `data/expert_coefficients.csv`
- `data/expert_diversity.csv`
- `data/routing_patterns.csv`
- `data/routing_diagnostics.csv`
- `data/routing_pairwise_jsd.csv`
- `data/router_shuffle_trials.csv`
- `data/router_shuffle_summary.csv`
- `data/router_shuffle_dataset_summary.csv`
- `figures/figure3.png`, `figures/figure3.pdf`
- `figures/figure3a.png/.pdf`, `figures/figure3b.png/.pdf`, `figures/figure3c.png/.pdf`
- `figures/figure3b_by_dataset.png/.pdf`
- `figure3_mechanism_report.md`
- `manifest.json`

The main Figure 3 panels (a) and (b) use Grocery seeds 42–44 as the layout
prototype; all four datasets and all seeds are retained in the CSVs and in the
supplementary per-dataset routing figure. Panel (c) includes all four datasets.

## Important interpretation boundary

The router-shuffle analysis is a **frozen Validation diagnostic**, not a
training ablation. It can show that the learned node–expert assignment is
predictively used by the frozen model; it cannot by itself establish that the
routing mechanism causes better Test performance or that an alternative router
would improve the model. No Test result is used anywhere in this directory.
