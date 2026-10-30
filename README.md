# MRAGE

MRAGE (Modality–Relation Adaptive Graph Experts) is the cleaned, reproducible
research release of the frozen MR-CGE V1.3-A Final model. This repository
contains the NC and sampled LP pipelines, ten selected baselines, and an
automatic, provenance-aware semantic KNN cache. The source migration is pinned
to commit `8f1f9c490907583b003beb73b72e873d62f16c3d`; see
[`PROVENANCE.md`](PROVENANCE.md).

## Environment

The validated environment used Python 3.12, PyTorch 2.4.0+cu121, PyG 2.7.0,
Hydra 1.3.2, FAISS CPU 1.15.1, DGL 2.4.0+cu121, NumPy 2.4.3, and
scikit-learn 1.8.0. `requirements.txt` records package versions. Install a
PyTorch build and matching `pyg-lib`/`torch-sparse` wheels for your CUDA or CPU
runtime before installing the remaining requirements. DGL is needed by the
MAGB graph loader; FAISS is needed when building a semantic cache.

## Data paths

MAGB data and fixed splits are read from `/hdd1/DataInHere/YHF/data` by default.
Override the roots without editing configs:

```bash
export MRAGE_DATA_ROOT=/path/to/data
export MRAGE_SPLIT_ROOT=/path/to/MAGB_split
```

Dataset and task paths are in `configs/dataset/`. Existing split files are
required by default; `auto_generate_splits` is false in the supplied dataset
configs, so a missing split produces an error instead of silently replacing
the official split.

## Run NC

```bash
conda run -n yhf_env python -m src.main \
  dataset=Movies task=nc model=mrage device=cuda:0
```

Direct script execution is also supported:

```bash
conda run -n yhf_env python src/main.py \
  dataset=Movies task=nc model=mrage device=cuda:0
```

The same entry point can run a selected baseline, for example:

```bash
conda run -n yhf_env python -m src.main \
  dataset=Movies task=nc model=gcn device=cuda:0
```

MAGB NC always reads the official `*_nc_seed42_*` split, independently of the
training seed. The ordinary config runs training seeds 42, 43, and 44 on that
same split. NC selects by Validation Accuracy and reports Macro-F1 over all
configured classes (`zero_division=0`).
LP keeps its existing official edge-split path selection, including its current
seed interpolation where configured; the NC split-seed change does not alter it.

For formal, isolated paper runs use the paper runner below. It masks NC Test
labels during training, saves the Validation-selected checkpoint, then restores
it in a separate process, checks Validation again, and evaluates Test once.
The direct `src.main` entry point remains useful for ordinary experiments and
can evaluate Test in the same process.

## Formal paper benchmark

The runner creates a separate directory for every dataset/task/model/config
hash/training-seed cell. It records the resolved configuration, split hash,
effective optimizer, checkpoint hash, source commit, Validation metrics, and
single-use Test status. Existing runs are never overwritten; `--resume` only
continues from `validation_complete`.

```bash
conda run -n yhf_env python scripts/run_paper_benchmark.py \
  --dataset Movies --task nc --models mrage gcn --device cuda:0 \
  --stage all --override model.hidden_dim=512 --override task.lr=1e-4

conda run -n yhf_env python scripts/run_paper_benchmark.py \
  --dataset sports-copurchase --task lp --models mrage gcn --seeds 42 43 44 \
  --device cuda:0 --experiment-config experiments/my_lp.yaml --stage validation

conda run -n yhf_env python scripts/collect_results.py --root outputs/paper
```

Use `--stage validation` to stop before Test or `--stage final-test --resume`
to finish a saved Validation run. `--dry-run` prints the planned run paths.

## Model parameters and optimizer precedence

The formal MRAGE defaults are `hidden_dim=256`, four experts, and Top-2
routing. Positive integer hidden dimensions such as 128, 256, and 512 are
supported. The four-expert/Top-2 structure and the validated graph settings
(`physical_hops=3`, `semantic_hops=1`, KNN K=8) remain fixed.

The optimizer name comes from `task.optimizer`. For learning rate and weight
decay, an explicit `model.lr` or `model.weight_decay` takes precedence over
`task.lr` or `task.weight_decay` to preserve baseline presets. MRAGE has no
model-level LR preset, so `task.lr=1e-4` changes its effective learning rate.
The trainer logs the effective optimizer and every parameter group's LR/decay.

## Run sampled LP

```bash
conda run -n yhf_env python -m src.main \
  dataset=sports-copurchase task=lp model=mrage device=cuda:0
```

LP uses `LinkNeighborLoader`, three configured physical fanouts, the training
physical graph, global edge IDs for positive supervision-edge masking, and the
project's link predictor and ranking metrics. MRAGE also restores the complete
cached semantic candidate neighborhoods. Physical context added for semantic
source nodes uses the explicit `[1,1,1]` train-graph fanout approximation in
`configs/model/mrage.yaml`. Baselines with verified sampled LP support are
`mlp`, `gcn`, `sage`, `mmgcn`, `mgat`, `dgf`, and `dmgc`. `dip`, `rolemag`, and
`lgmrec` are NC-only and fail with a clear unsupported-task error for LP.
For sampled LP, DGF's ten graph-filter terms request ten sampler fanouts; the
last configured fanout is repeated when `task.num_neighbors` is shorter.

LP Validation/Test ranking runs without Autograd and scores candidate edges in
bounded batches, with pessimistic ties retained. Evaluation embeddings stay on
GPU only when the free-memory check leaves a safety reserve; projection can be
computed in chunks. The trainer no longer clears the CUDA allocator cache after
every epoch. Optional phase timings are enabled with `task.profile_timing=true`;
they synchronize CUDA only while profiling is enabled. Loader worker counts
remain configurable and do not change negative sampling or fanout defaults.

## Semantic graph cache

MRAGE does not require a model-level cache path. After the dataset split and
allowed physical graph are loaded, the manager searches or builds a V7B.1
Text/Visual K=8 graph and restores separate modality operators. Cache files are
stored under `artifacts/semantic_graphs/<dataset>/<task>/<identity>/` and are
excluded from version control. Identity includes row-ordered feature content,
the task, physical topology, KNN rule version, and FAISS settings. Files are
locked, atomically replaced, checked on load, and bound into the model
checkpoint. LP construction receives only the official Train physical graph;
labels are not passed to the builder.

Modes can be set with Hydra overrides:

```text
semantic_graph.mode=auto       # load a valid matching cache or build one
semantic_graph.mode=existing   # fail if no valid matching cache exists
semantic_graph.mode=rebuild    # rebuild this exact identity
semantic_graph.cache_dir=/path/to/cache
```

The formal graph rule is validated only at `knn_k=8`. Cache identity, source
flags, propagation operators, and checkpoint validation retain their existing
V7B.1 compatibility fingerprints.

Independent Ablation and Analysis implementations are not included in this
release; they remain planned follow-up work.

## Models

The ten migrated baselines are `mlp`, `gcn`, `sage`, `mmgcn`, `mgat`, `dip`,
`dgf`, `dmgc`, `rolemag`, and `lgmrec`. `model=mrage` selects the formal NC
encoder or sampled LP adapter based on `task.name`.

## Outputs and experiments

Hydra writes logs and `results.json` under `outputs/<date>/<time>/`; model
checkpoints are written only when `task.save_ckpt_path` is set. Semantic caches
are kept separately under `artifacts/semantic_graphs/`. Both are ignored by
Git.

`experiments/README.md` records the planned ablation and analysis extensions.
Those experiment-specific implementations are not yet included: the formal
model is intentionally free of ablation switches and research diagnostic
interfaces. No result or checkpoint from the source research repository was
migrated.

For an explicitly chosen run matrix, `scripts/run_benchmark.py` runs each
seed/model combination sequentially. It does not define a paper benchmark
recipe; supply the protocol's settings as Hydra overrides. For example:

```bash
python scripts/run_benchmark.py --dataset Movies --task nc \
  --models mrage gcn --seeds 42 43 --device cuda:0 \
  --override task.epochs=300
python scripts/collect_results.py --root outputs
```

## Validation

Run the focused checks in the project environment:

```bash
PYTHONPATH=. python -m pytest -q tests
```

The pytest suite covers semantic-cache equivalence and identity isolation,
one-epoch CPU NC/LP and GCN training smokes, plus one-step NC and sampled-LP
forward/update checks for all baselines and each declared LP-capable baseline.
To run the separate frozen
parity comparison, make the read-only source checkout available and pass its
path (it defaults to `../model_research`):

```bash
PYTHONPATH=. python tests/test_frozen_parity.py ../model_research
```

That script compares frozen forward outputs, NC losses, gradients, one update,
and LP sampled context/predictions. A CUDA training benchmark is deliberately
not part of the smoke checks.
