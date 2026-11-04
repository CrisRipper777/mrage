# MRAGE 实验结果汇总

> 本文件汇总 **Table 1（NC 主表）**、**Table 2（LP 主表）** 与 **Table 4（能力消融）**。
> 所有路径以项目根 `/hdd1/DataInHere/YHF/mrage` 为基准；尚未完成的单元格留空。

## 统一说明

- **指标单位**：表中 Accuracy / Macro-F1 / MRR / Hits@k 均为百分数（%），括号内为三个训练种子 42/43/44 的**总体标准差**（`ddof=0`）。
- **NC 协议**：固定 split seed 42（MAGB 为 `*_nc_seed42_*`，ele-fashion 为官方 split）；全图训练；按 **best Validation Accuracy** 选 checkpoint；Test 标签训练期 mask；Final Test 独立进程只评估一次。
- **LP 协议**：官方固定 `lp-edge-split.pt`；`LinkNeighborLoader` fanout `[5,5,5]`；按 **best Validation MRR** 选 checkpoint；Final Test 候选固定，只评估一次。
- **运行目录**：每个 `seed-<N>/` 子目录包含 `config_resolved.yaml`、`train.log`、`best_validation.pt`、`validation_metrics.json`、`final_test_metrics.json`、`run_metadata.json`、`status.json` 与 `hydra/`。
- **未完成标记**：空白单元格表示尚未得到完整三种子结果；具体已完成部分见各表脚注。

## Table 1. NC 主表

五个 NC 数据集上的 Test Accuracy / Test Macro-F1。**MRAGE 行为当前 adopted 参考结果**（Movies 为历史 default 运行，其余四个数据集为 Test 参与选参的探索结果），不是与 baseline 同批次的独立对照。

### Movies

| Model | Test Acc (%) | Test Macro-F1 (%) | 运行目录 |
|---|---:|---:|---|
| **MRAGE (adopted)** | 55.04 ± 0.22 | 47.95 ± 0.30 | `outputs/2026-10-31/13-05-49/`（3 seeds 汇总于同一 `results.json`，无 checkpoint） |
| mlp | 51.80 ± 0.13 | 40.26 ± 1.21 | `outputs/baseline_benchmark_full_method_nc_20261031/Movies/nc/mlp/7f1998f0692c0068/seed-{42,43,44}/` |
| gcn | 53.59 ± 0.25 | 42.50 ± 1.19 | `outputs/baseline_benchmark_full_method_nc_20261031/Movies/nc/gcn/cefa99659fcfeceb/seed-{42,43,44}/` |
| sage | 54.67 ± 0.23 | 44.79 ± 1.88 | `outputs/baseline_benchmark_full_method_nc_20261031/Movies/nc/sage/592f325e6fbb926a/seed-{42,43,44}/` |
| mmgcn | 54.67 ± 0.12 | 46.19 ± 1.44 | `outputs/baseline_benchmark_full_method_nc_20261031/Movies/nc/mmgcn/d9156460ef7ea03a/seed-{42,43,44}/` |
| mgat | 53.24 ± 0.24 | 35.06 ± 4.09 | `outputs/baseline_benchmark_full_method_nc_20261031/Movies/nc/mgat/20271c9c02b870d6/seed-{42,43,44}/` |
| dip | 55.36 ± 0.25 | 47.50 ± 0.89 | `outputs/baseline_benchmark_full_method_nc_20261031/Movies/nc/dip/b1f3a0b6951470e0/seed-{42,43,44}/` |
| dgf | 52.63 ± 0.25 | 34.43 ± 1.07 | `outputs/baseline_benchmark_full_method_nc_20261031/Movies/nc/dgf/bdb76b079eb4fb39/seed-{42,43,44}/` |
| dmgc | 47.96 ± 0.35 | 26.98 ± 2.90 | `outputs/dmgc_nc_lambda_cm_screen_20261031/lambda_cm_0.3/Movies/nc/dmgc/2e6abc1579f8946d/seed-{42,43,44}/` |
| rolemag | 54.72 ± 0.47 | 44.59 ± 1.01 | `outputs/baseline_benchmark_full_method_nc_20261031/Movies/nc/rolemag/60f63e7af0788721/seed-{42,43,44}/` |
| lgmrec | 55.33 ± 0.58 | 49.46 ± 0.27 | `outputs/baseline_benchmark_full_method_nc_20261031/Movies/nc/lgmrec/8e53fcc516ff3f19/seed-{42,43,44}/` |
| graphmore | 55.09 ± 0.31 | 46.26 ± 1.29 | `outputs/baseline_benchmark/Movies/nc/graphmore/9a5e7bd358ca9f65/seed-{42,43,44}/` |
| ntsformer | 54.21 ± 0.15 | 43.28 ± 1.24 | `outputs/baseline_benchmark/Movies/nc/ntsformer/2e6a3e24d7775ff1/seed-{42,43,44}/` |

### Toys

| Model | Test Acc (%) | Test Macro-F1 (%) | 运行目录 |
|---|---:|---:|---|
| **MRAGE (adopted)** | 79.03 ± 0.39 | 76.29 ± 0.48 | `outputs/nc5_test_exploration_20261031/tuned128/Toys/nc/mrage/8a883402bdb6a07e/seed-{42,43,44}/` |
| mlp | 74.78 ± 0.17 | 71.85 ± 0.17 | `outputs/baseline_benchmark_full_method_nc_20261031/Toys/nc/mlp/d66894c4287dc68b/seed-{42,43,44}/` |
| gcn | 78.38 ± 0.08 | 75.01 ± 0.18 | `outputs/baseline_benchmark_full_method_nc_20261031/Toys/nc/gcn/fd532e054d71d45b/seed-{42,43,44}/` |
| sage | 78.74 ± 0.18 | 75.83 ± 0.15 | `outputs/baseline_benchmark_full_method_nc_20261031/Toys/nc/sage/2e693b26b0ee5d11/seed-{42,43,44}/` |
| mmgcn | 77.23 ± 0.04 | 73.83 ± 0.17 | `outputs/baseline_benchmark_full_method_nc_20261031/Toys/nc/mmgcn/06d20c0ea2281545/seed-{42,43,44}/` |
| mgat | 77.62 ± 0.28 | 73.21 ± 0.93 | `outputs/baseline_benchmark_full_method_nc_20261031/Toys/nc/mgat/c2ba026fd0b76ba9/seed-{42,43,44}/` |
| dip | 78.99 ± 0.43 | 76.46 ± 0.32 | `outputs/baseline_benchmark_full_method_nc_20261031/Toys/nc/dip/f2d33f8f48f65e92/seed-{42,43,44}/` |
| dgf | 78.01 ± 0.32 | 74.12 ± 0.56 | `outputs/baseline_benchmark_full_method_nc_20261031/Toys/nc/dgf/0edf988d1b4ab7a0/seed-{42,43,44}/` |
| dmgc | 74.19 ± 0.54 | 70.74 ± 0.95 | `outputs/dmgc_nc_lambda_cm_screen_20261031/lambda_cm_0.3/Toys/nc/dmgc/35478af2baba3bda/seed-{42,43,44}/` |
| rolemag | 78.46 ± 0.41 | 75.70 ± 0.64 | `outputs/baseline_benchmark_full_method_nc_20261031/Toys/nc/rolemag/0e9fa3ca7a6e4507/seed-{42,43,44}/` |
| lgmrec | 79.18 ± 0.02 | 76.52 ± 0.13 | `outputs/baseline_benchmark_full_method_nc_20261031/Toys/nc/lgmrec/073b01f08c5a1f5e/seed-{42,43,44}/` |
| graphmore | 78.67 ± 0.14 | 75.77 ± 0.21 | `outputs/baseline_benchmark/Toys/nc/graphmore/a369a1e9563b950f/seed-{42,43,44}/` |
| ntsformer | 78.88 ± 0.21 | 75.76 ± 0.43 | `outputs/baseline_benchmark/Toys/nc/ntsformer/b1c279bb4a761089/seed-{42,43,44}/` |

### Grocery

| Model | Test Acc (%) | Test Macro-F1 (%) | 运行目录 |
|---|---:|---:|---|
| **MRAGE (adopted)** | 83.26 ± 0.19 | 74.69 ± 0.51 | `outputs/nc5_test_exploration_20261031/tuned128/Grocery/nc/mrage/a1f872eff7a7c1a9/seed-{42,43,44}/` |
| mlp | 78.46 ± 0.19 | 70.39 ± 0.60 | `outputs/baseline_benchmark_full_method_nc_20261031/Grocery/nc/mlp/59bdf449300309a9/seed-{42,43,44}/` |
| gcn | 79.75 ± 0.01 | 68.09 ± 0.15 | `outputs/baseline_benchmark_full_method_nc_20261031/Grocery/nc/gcn/0c0f54d3b2338e21/seed-{42,43,44}/` |
| sage | 81.38 ± 0.13 | 71.70 ± 0.58 | `outputs/baseline_benchmark_full_method_nc_20261031/Grocery/nc/sage/ec3ee56e32bc78b0/seed-{42,43,44}/` |
| mmgcn | 79.12 ± 0.44 | 67.90 ± 0.88 | `outputs/baseline_benchmark_full_method_nc_20261031/Grocery/nc/mmgcn/f5cf09c745cc283e/seed-{42,43,44}/` |
| mgat | 80.79 ± 0.43 | 67.71 ± 1.10 | `outputs/baseline_benchmark_full_method_nc_20261031/Grocery/nc/mgat/9bd5ee1a4577dc77/seed-{42,43,44}/` |
| dip | 83.64 ± 0.16 | 75.51 ± 0.38 | `outputs/baseline_benchmark_full_method_nc_20261031/Grocery/nc/dip/40b16ad1d3a329ff/seed-{42,43,44}/` |
| dgf | 79.89 ± 0.32 | 65.22 ± 0.39 | `outputs/baseline_benchmark_full_method_nc_20261031/Grocery/nc/dgf/75a3c919507313c4/seed-{42,43,44}/` |
| dmgc | 76.70 ± 3.76 | 64.20 ± 4.49 | `outputs/dmgc_nc_lambda_cm_screen_20261031/lambda_cm_0.3/Grocery/nc/dmgc/0cb6e260b95fa7fd/seed-{42,43,44}/` |
| rolemag | 82.16 ± 0.17 | 72.71 ± 0.30 | `outputs/baseline_benchmark_full_method_nc_20261031/Grocery/nc/rolemag/fb839745d34594f4/seed-{42,43,44}/` |
| lgmrec | 82.88 ± 0.08 | 75.33 ± 0.21 | `outputs/baseline_benchmark_full_method_nc_20261031/Grocery/nc/lgmrec/0bc44276d95f3908/seed-{42,43,44}/` |
| graphmore | 81.81 ± 0.22 | 70.91 ± 0.42 | `outputs/baseline_benchmark/Grocery/nc/graphmore/2aa4be15d1a3f604/seed-{42,43,44}/` |
| ntsformer | 82.89 ± 0.26 | 74.01 ± 0.72 | `outputs/baseline_benchmark/Grocery/nc/ntsformer/2e610e6c0bdaf8b6/seed-{42,43,44}/` |

### Reddit-S

| Model | Test Acc (%) | Test Macro-F1 (%) | 运行目录 |
|---|---:|---:|---|
| **MRAGE (adopted)** | 96.53 ± 0.08 | 91.71 ± 0.46 | `outputs/nc5_test_exploration_20261031/reddit_aux01/Reddit-S/nc/mrage/700dd52266746cb9/seed-{42,43,44}/` |
| mlp | 92.74 ± 0.05 | 87.23 ± 0.29 | `outputs/baseline_benchmark_full_method_nc_20261031/Reddit-S/nc/mlp/fbf870ceeb77e9eb/seed-{42,43,44}/` |
| gcn | 93.74 ± 0.07 | 89.01 ± 0.18 | `outputs/baseline_benchmark_full_method_nc_20261031/Reddit-S/nc/gcn/07e32d0ff68eadb0/seed-{42,43,44}/` |
| sage | 94.12 ± 0.14 | 89.44 ± 0.28 | `outputs/baseline_benchmark_full_method_nc_20261031/Reddit-S/nc/sage/559bb1808bcea0d2/seed-{42,43,44}/` |
| mmgcn | 94.23 ± 0.12 | 89.47 ± 0.31 | `outputs/baseline_benchmark_full_method_nc_20261031/Reddit-S/nc/mmgcn/0541598bb93e3638/seed-{42,43,44}/` |
| mgat | 95.72 ± 0.08 | 90.76 ± 0.51 | `outputs/baseline_benchmark_full_method_nc_20261031/Reddit-S/nc/mgat/101d37abf4913a23/seed-{42,43,44}/` |
| dip | 96.41 ± 0.07 | 92.42 ± 0.21 | `outputs/baseline_benchmark_full_method_nc_20261031/Reddit-S/nc/dip/f7a8d65caa1ca11c/seed-{42,43,44}/` |
| dgf | 95.87 ± 0.08 | 90.05 ± 0.20 | `outputs/baseline_benchmark_full_method_nc_20261031/Reddit-S/nc/dgf/2252badbb0eb105e/seed-{42,43,44}/` |
| dmgc | 94.56 ± 0.11 | 89.17 ± 0.18 | `outputs/dmgc_nc_lambda_cm_screen_20261031/lambda_cm_0.3/Reddit-S/nc/dmgc/16ddffbd81c2c7aa/seed-{42,43,44}/` |
| rolemag | 95.28 ± 0.12 | 90.78 ± 0.28 | `outputs/baseline_benchmark_full_method_nc_20261031/Reddit-S/nc/rolemag/edc205e75b7fd90a/seed-{42,43,44}/` |
| lgmrec | 95.99 ± 0.04 | 91.89 ± 0.13 | `outputs/baseline_benchmark_full_method_nc_20261031/Reddit-S/nc/lgmrec/44d2ca80f606f13e/seed-{42,43,44}/` |
| graphmore | 94.78 ± 0.12 | 89.67 ± 0.36 | `outputs/baseline_benchmark/Reddit-S/nc/graphmore/13299f7ef8443ceb/seed-{42,43,44}/` |
| ntsformer | 95.93 ± 0.32 | 91.47 ± 0.63 | `outputs/baseline_benchmark/Reddit-S/nc/ntsformer/c0a859b20c52c3c9/seed-{42,43,44}/` |

### ele-fashion

| Model | Test Acc (%) | Test Macro-F1 (%) | 运行目录 |
|---|---:|---:|---|
| **MRAGE (adopted)** | 87.63 ± 0.05 | 75.20 ± 0.34 | `outputs/nc5_test_exploration_20261031/fashion_gate64/ele-fashion/nc/mrage/91edd92176df744a/seed-{42,43,44}/` |
| mlp | 88.12 ± 0.07 | 77.82 ± 0.32 | `outputs/baseline_benchmark_full_method_nc_20261031/ele-fashion/nc/mlp/2618394f51abcd22/seed-{42,43,44}/` |
| gcn | 82.97 ± 0.04 | 67.27 ± 0.11 | `outputs/baseline_benchmark_full_method_nc_20261031/ele-fashion/nc/gcn/379df89d069a6674/seed-{42,43,44}/` |
| sage | 87.09 ± 0.15 | 74.60 ± 0.93 | `outputs/baseline_benchmark_full_method_nc_20261031/ele-fashion/nc/sage/1cc1e4429ddffb41/seed-{42,43,44}/` |
| mmgcn | 86.54 ± 0.07 | 72.66 ± 0.14 | `outputs/baseline_benchmark_full_method_nc_20261031/ele-fashion/nc/mmgcn/672b5450afead04f/seed-{42,43,44}/` |
| mgat | 87.02 ± 0.12 | 72.94 ± 0.62 | `outputs/baseline_benchmark_full_method_nc_20261031/ele-fashion/nc/mgat/a881238dcd1d95cf/seed-{42,43,44}/` |
| dip | 88.09 ± 0.13 | 77.08 ± 0.64 | `outputs/baseline_benchmark_full_method_nc_20261031/ele-fashion/nc/dip/3671c346a85c191b/seed-{42,43,44}/` |
| dgf | 86.33 ± 0.06 | 68.58 ± 0.80 | `outputs/baseline_benchmark_full_method_nc_20261031/ele-fashion/nc/dgf/77914dce705b4bdc/seed-{42,43,44}/` |
| dmgc | 87.23 ± 0.03 | 72.69 ± 1.12 | `outputs/dmgc_nc_lambda_cm_screen_20261031/lambda_cm_0.3/ele-fashion/nc/dmgc/73988b94af3de82d/seed-{42,43,44}/` |
| rolemag | 87.46 ± 0.09 | 75.92 ± 0.72 | `outputs/baseline_benchmark_full_method_nc_20261031/ele-fashion/nc/rolemag/c6d419e6171b7961/seed-{42,43,44}/` |
| lgmrec | 85.68 ± 0.31 | 71.29 ± 1.52 | `outputs/baseline_benchmark_full_method_nc_20261031/ele-fashion/nc/lgmrec/8768695d43312f30/seed-{42,43,44}/` |
| graphmore | 85.36 ± 0.28 | 72.41 ± 0.51 | `outputs/baseline_benchmark/ele-fashion/nc/graphmore/11ebcccf44eb432e/seed-{42,43,44}/` |
| ntsformer | 87.48 ± 0.13 | 75.77 ± 0.94 | `outputs/baseline_benchmark/ele-fashion/nc/ntsformer/10fbe8570fc2b15e/seed-{42,43,44}/` |

### Table 1 备注

- **MRAGE 各数据集当前配置**：Movies = `default`；Toys = `tuned128`；Grocery = `tuned128`；Reddit-S = `reddit_aux01`；ele-fashion = `fashion_gate64`。完整覆盖配置位于 `experiments/nc5_test_exploration_20261031/adopted/<dataset>.yaml`。
- **MRAGE 不是同批次对照**：Movies 使用 `outputs/2026-10-31/13-05-49` 的历史 default 三 seed 运行；其余四个数据集来自 `outputs/nc5_test_exploration_20261031/` 的 65-run 探索，且 Test 参与了超参选择。引用为“adopted reference”更准确。
- **另有 validation-only 正式 benchmark**：`outputs/mrage_nc_benchmark_20261031/` 使用一套统一 tuned 配置（validation 选参，Test 不参与），其 MRAGE 结果为 Movies 54.68 / 47.92、Toys 79.19 / 76.35、Grocery 83.11 / 74.78、Reddit-S 96.53 / 92.20、ele-fashion 87.58 / 69.43（12 类 raw）。该批数据与上表的 adopted 行来自不同批次，不能拼接。
- **ele-fashion Macro-F1 口径**：baseline 已按 11 个实际有标签类别计算；MRAGE 的磁盘原始值是 12 类口径 **68.93 ± 0.31**（class 5 全空，F1=0 纳入分母）。本表为与 baseline 对齐，按 `F1_11 = F1_12 × 12/11` 换算为 **75.20 ± 0.34**，未重新推理 Test、未改动 Accuracy。
- **DMGC**：表中 DMGC 行使用 `outputs/dmgc_nc_lambda_cm_screen_20261031/lambda_cm_0.3/` 下按 Validation Accuracy 选出的 `lambda_cm=0.3` 运行，不是 `outputs/baseline_benchmark_full_method_nc_20261031/.../dmgc/` 下的旧批次。
- **GraphMoRE / NTSFormer**：追加 baseline 的运行目录在 `outputs/baseline_benchmark/<Dataset>/nc/<model>/<config-hash>/seed-{42,43,44}/`。

## Table 2. LP 主表

两个 LP 数据集上的 Test MRR / Hits@1 / Hits@3 / Hits@10。NTSFormer 为 NC-only，不跑 LP。

### sports-copurchase

| Model | Test MRR (%) | H@1 (%) | H@3 (%) | H@10 (%) | N | 运行目录 |
|---|---:|---:|---:|---:|---:|---|
| **MRAGE** | 37.34 ± 0.17 | 21.25 ± 0.19 | 42.78 ± 0.42 | 73.76 ± 0.24 | 3/3 | `outputs/lp_mrage_baselines_20261101/sports-copurchase/lp/mrage/d96c7a8791fad593/seed-{42,43,44}/` |
| mlp † |  |  |  |  | 0/3 | 未运行 |
| gcn |  |  |  |  | 0/3 | 未运行 |
| sage |  |  |  |  | 0/3 | 未运行 |
| mmgcn |  |  |  |  | 0/3 | 未运行 |
| mgat |  |  |  |  | 0/3 | 未运行 |
| dip |  |  |  |  | 0/3 | 未运行 |
| dgf |  |  |  |  | 0/3 | 未运行 |
| dmgc |  |  |  |  | 0/3 | 未运行 |
| rolemag |  |  |  |  | 0/3 | 未运行 |
| lgmrec |  |  |  |  | 0/3 | 未运行 |
| graphmore |  |  |  |  | 0/3 | 未运行 |

### cloth-copurchase

| Model | Test MRR (%) | H@1 (%) | H@3 (%) | H@10 (%) | N | 运行目录 |
|---|---:|---:|---:|---:|---:|---|
| **MRAGE** |  |  |  |  | 1/3（仅 seed42 完成） | `outputs/lp_mrage_baselines_20261101/cloth-copurchase/lp/mrage/1ead5dffc7d3576f/seed-{42,43,44}/` |
| mlp † |  |  |  |  | 0/3 | 未运行 |
| gcn |  |  |  |  | 0/3 | 未运行 |
| sage |  |  |  |  | 0/3 | 未运行 |
| mmgcn |  |  |  |  | 0/3 | 未运行 |
| mgat |  |  |  |  | 0/3 | 未运行 |
| dip |  |  |  |  | 0/3 | 未运行 |
| dgf |  |  |  |  | 0/3 | 未运行 |
| dmgc |  |  |  |  | 0/3 | 未运行 |
| rolemag |  |  |  |  | 0/3 | 未运行 |
| lgmrec |  |  |  |  | 0/3 | 未运行 |
| graphmore |  |  |  |  | 0/3 | 未运行 |

### Table 2 备注

- **sports-copurchase MRAGE（3 seeds）**：Test MRR 37.34 ± 0.17，H@1 21.25 ± 0.19，H@3 42.78 ± 0.42，H@10 73.76 ± 0.24；Val MRR 40.46 ± 0.16。运行目录：`outputs/lp_mrage_baselines_20261101/sports-copurchase/lp/mrage/d96c7a8791fad593/seed-{42,43,44}/`。
- **cloth-copurchase MRAGE（仅 seed42 完成）**：seed42 Test MRR 28.80 / H@1 15.85 / H@3 31.70 / H@10 56.54；Val MRR 32.20。seed43 正在训练，seed44 待跑；因此三种子均值暂留空。运行目录：`outputs/lp_mrage_baselines_20261101/cloth-copurchase/lp/mrage/1ead5dffc7d3576f/seed-{42,43,44}/`。
- **LP baselines 尚未运行**：当前只完成了 MRAGE 的 Sports 与 Cloth（Cloth 部分）。按当前 MRAGE 批次 output root，计划目录为 `outputs/lp_mrage_baselines_20261101/<dataset>/lp/<model>/<config-hash>/seed-{42,43,44}/`；若使用 `scripts/run_lp_baseline_benchmark.py` 的默认 output root，则为 `outputs/baseline_benchmark/...`。
- **† MLP**：项目官方 `run_lp_baseline_benchmark.py` 当前把 MLP 与 NTSFormer 都排除在 LP 矩阵外；本表按“仅排除 NTSFormer”列出 MLP 行，若需要纳入需单独补跑。
- **LP 主表结果文件**：每个已完成 run 的指标在对应 `seed-*/final_test_metrics.json`；汇总可用 `scripts/collect_results.py --root outputs/lp_mrage_baselines_20261101`。

## Table 4. 能力消融（capability ablation）

四个能力级变体在五个 NC 数据集上的 Test Accuracy / Test Macro-F1，共 60 个正式 run 单元（4 变体 × 5 数据集 × 3 seeds），全部完成。**Full 行为 adopted 参考，不是本轮重跑**；Δ 为相对 Full 的描述性差值（百分点）。

### Movies

| Variant | Test Acc (%) | Test Macro-F1 (%) | ΔAcc vs Full (pp) | ΔF1 vs Full (pp) | 运行目录 |
|---|---:|---:|---:|---:|---|
| **Full (adopted ref)** | 55.04 ± 0.22 | 47.95 ± 0.30 | — | — | `outputs/2026-10-31/13-05-49/`（3 seeds 汇总于同一 `results.json`） |
| physical_only | 54.30 ± 0.26 | 45.64 ± 0.99 | -0.74 | -2.31 | `outputs/mrage_capability_nc5/Movies/nc/mrage_capability_ablation/02029fe666d965d4/seed-{42,43,44}/` |
| semantic_only | 51.67 ± 0.23 | 42.42 ± 0.47 | -3.37 | -5.53 | `outputs/mrage_capability_nc5/Movies/nc/mrage_capability_ablation/0897efbee952c4e5/seed-{42,43,44}/` |
| no_cross_modal_collaboration | 54.50 ± 0.28 | 47.05 ± 0.80 | -0.54 | -0.90 | `outputs/mrage_capability_nc5/Movies/nc/mrage_capability_ablation/614054f11870be33/seed-{42,43,44}/` |
| feature_only | 51.02 ± 0.35 | 40.45 ± 0.26 | -4.02 | -7.50 | `outputs/mrage_capability_nc5/Movies/nc/mrage_capability_ablation/3c8c0663c2627a39/seed-{42,43,44}/` |

### Toys

| Variant | Test Acc (%) | Test Macro-F1 (%) | ΔAcc vs Full (pp) | ΔF1 vs Full (pp) | 运行目录 |
|---|---:|---:|---:|---:|---|
| **Full (adopted ref)** | 79.03 ± 0.39 | 76.29 ± 0.48 | — | — | `outputs/nc5_test_exploration_20261031/tuned128/Toys/nc/mrage/8a883402bdb6a07e/seed-{42,43,44}/` |
| physical_only | 78.68 ± 0.40 | 75.80 ± 0.37 | -0.35 | -0.49 | `outputs/mrage_capability_nc5/Toys/nc/mrage_capability_ablation/763dd462429e2caa/seed-{42,43,44}/` |
| semantic_only | 74.91 ± 0.18 | 72.23 ± 0.36 | -4.12 | -4.07 | `outputs/mrage_capability_nc5/Toys/nc/mrage_capability_ablation/11ea0e4a6895c1e1/seed-{42,43,44}/` |
| no_cross_modal_collaboration | 79.02 ± 0.59 | 76.08 ± 0.71 | -0.01 | -0.22 | `outputs/mrage_capability_nc5/Toys/nc/mrage_capability_ablation/d7bd86bcefa0bb88/seed-{42,43,44}/` |
| feature_only | 73.62 ± 0.37 | 70.99 ± 0.42 | -5.40 | -5.30 | `outputs/mrage_capability_nc5/Toys/nc/mrage_capability_ablation/c05c7b416380c605/seed-{42,43,44}/` |

### Grocery

| Variant | Test Acc (%) | Test Macro-F1 (%) | ΔAcc vs Full (pp) | ΔF1 vs Full (pp) | 运行目录 |
|---|---:|---:|---:|---:|---|
| **Full (adopted ref)** | 83.26 ± 0.19 | 74.69 ± 0.51 | — | — | `outputs/nc5_test_exploration_20261031/tuned128/Grocery/nc/mrage/a1f872eff7a7c1a9/seed-{42,43,44}/` |
| physical_only | 83.06 ± 0.15 | 74.83 ± 0.20 | -0.20 | +0.14 | `outputs/mrage_capability_nc5/Grocery/nc/mrage_capability_ablation/939c7b213c70bbc1/seed-{42,43,44}/` |
| semantic_only | 78.54 ± 0.09 | 69.20 ± 0.48 | -4.72 | -5.49 | `outputs/mrage_capability_nc5/Grocery/nc/mrage_capability_ablation/0d251f3522e9beb8/seed-{42,43,44}/` |
| no_cross_modal_collaboration | 82.89 ± 0.31 | 73.30 ± 0.29 | -0.37 | -1.38 | `outputs/mrage_capability_nc5/Grocery/nc/mrage_capability_ablation/8545feb3352c7e93/seed-{42,43,44}/` |
| feature_only | 77.28 ± 0.28 | 67.61 ± 0.20 | -5.98 | -7.07 | `outputs/mrage_capability_nc5/Grocery/nc/mrage_capability_ablation/ee6ccc67e3ee774f/seed-{42,43,44}/` |

### Reddit-S

| Variant | Test Acc (%) | Test Macro-F1 (%) | ΔAcc vs Full (pp) | ΔF1 vs Full (pp) | 运行目录 |
|---|---:|---:|---:|---:|---|
| **Full (adopted ref)** | 96.53 ± 0.08 | 91.71 ± 0.46 | — | — | `outputs/nc5_test_exploration_20261031/reddit_aux01/Reddit-S/nc/mrage/700dd52266746cb9/seed-{42,43,44}/` |
| physical_only | 96.61 ± 0.06 | 92.76 ± 0.29 | +0.08 | +1.05 | `outputs/mrage_capability_nc5/Reddit-S/nc/mrage_capability_ablation/8e357180e8dfe25a/seed-{42,43,44}/` |
| semantic_only | 93.01 ± 0.10 | 87.64 ± 0.34 | -3.52 | -4.07 | `outputs/mrage_capability_nc5/Reddit-S/nc/mrage_capability_ablation/0c4a8631a9e85ffd/seed-{42,43,44}/` |
| no_cross_modal_collaboration | 96.49 ± 0.08 | 92.72 ± 0.03 | -0.04 | +1.01 | `outputs/mrage_capability_nc5/Reddit-S/nc/mrage_capability_ablation/fdcce74014a5db1d/seed-{42,43,44}/` |
| feature_only | 93.08 ± 0.16 | 87.67 ± 0.31 | -3.45 | -4.04 | `outputs/mrage_capability_nc5/Reddit-S/nc/mrage_capability_ablation/76f29462a7ab26c5/seed-{42,43,44}/` |

### ele-fashion

| Variant | Test Acc (%) | Test Macro-F1 (%) | ΔAcc vs Full (pp) | ΔF1 vs Full (pp) | 运行目录 |
|---|---:|---:|---:|---:|---|
| **Full (adopted ref)** | 87.63 ± 0.05 | 75.20 ± 0.34 | — | — | `outputs/nc5_test_exploration_20261031/fashion_gate64/ele-fashion/nc/mrage/91edd92176df744a/seed-{42,43,44}/` |
| physical_only | 87.25 ± 0.02 | 75.20 ± 0.51 | -0.39 | 0.00 | `outputs/mrage_capability_nc5/ele-fashion/nc/mrage_capability_ablation/9a651f6328e5fac4/seed-{42,43,44}/` |
| semantic_only | 87.31 ± 0.17 | 74.90 ± 0.51 | -0.33 | -0.30 | `outputs/mrage_capability_nc5/ele-fashion/nc/mrage_capability_ablation/5380fff49cce00a2/seed-{42,43,44}/` |
| no_cross_modal_collaboration | 87.58 ± 0.06 | 75.44 ± 0.80 | -0.06 | +0.24 | `outputs/mrage_capability_nc5/ele-fashion/nc/mrage_capability_ablation/807dc69d4c2ff539/seed-{42,43,44}/` |
| feature_only | 87.27 ± 0.10 | 74.84 ± 0.15 | -0.37 | -0.36 | `outputs/mrage_capability_nc5/ele-fashion/nc/mrage_capability_ablation/b41c6d73b89b4440/seed-{42,43,44}/` |

### Table 4 备注

- **完整性**：`outputs/mrage_capability_nc5` 下 60/60 个 seed-run 均为 `final_test_complete`，每个 Test 评估次数为 1；20/20 个 dataset×variant 单元均含三个种子。
- **协议**：`unified_full_graph_nc_v1`；按 best Validation Accuracy 选 checkpoint；MAGB split seed 42，ele-fashion 官方 split。变体继承各数据集 adopted Full 配置，不做变体专属调参。
- **ele-fashion 的 11 类口径**：本表 Macro-F1 已按 `F1_11 = F1_12 × 12/11` 换算。12 类 raw 值分别为：Full 68.93 ± 0.31，physical_only 68.93 ± 0.47，semantic_only 68.65 ± 0.47，no_cross_modal_collaboration 69.15 ± 0.73，feature_only 68.61 ± 0.14。Accuracy 不受口径影响。
- **读数提示**：`physical_only` 的 Accuracy 与 Full 很接近（各数据集 ΔAcc 约 −0.39~+0.08pp），说明 physical graph 是主要贡献、semantic KNN 在 physical 之上增量很小；`semantic_only` 与 `feature_only` 在 Movies/Toys/Grocery/Reddit-S 掉约 3.4–6pp；`no_cross_modal_collaboration` 的差异通常 ≤~1.4pp（Reddit-S 上 F1 反而 +1.01pp）；ele-fashion 上所有图机制的影响都只有约 0.4pp 量级。
- **Full reference 目录**：与 Table 1 的 MRAGE adopted 行相同。Movies 为 `outputs/2026-10-31/13-05-49/`；其余为 `outputs/nc5_test_exploration_20261031/<candidate>/<Dataset>/nc/mrage/<hash>/seed-{42,43,44}/`。

## 附：未纳入本文件的相关实验

- **NC 组件消融（Table 3 候选，本文件未展开）**：`outputs/mrage_ablation_nc5/`，75/75 个 seed-run 完成；汇总见 `outputs/mrage_ablation_nc5/summary/nc5_ablation_summary.md`。
- **Sports LP 消融（非 Table 2 主表）**：`outputs/mrage_sports_lp_ablation/`，目前 7/15 完成（single_filter 3/3、physical_only 3/3、semantic_only seed42），semantic_only seed43 运行中，其余待跑。
- **Cloth LP 与 LP baselines**：Cloth MRAGE seed42 完成，seed43 训练中、seed44 待跑；LP baselines 尚未启动。完整三种子结果出现后再填入 Table 2。

## 数据来源与复现入口

| 内容 | 文件 / 目录 |
|---|---|
| Table 1 baseline 聚合值 | `outputs/baseline_benchmark_full_method_nc_20261031/nc_baseline_summary.csv` |
| Table 1 baseline 逐 seed 值与 run_dir | `outputs/baseline_benchmark_full_method_nc_20261031/nc_baseline_per_seed_metrics.csv` |
| Table 1 MRAGE adopted 清单 | `experiments/nc5_test_exploration_20261031/adopted_selection.json`、`adopted_results_percent.csv`、`ADOPTED_RESULTS.md` |
| 正式 validation-only NC benchmark（另一批） | `outputs/mrage_nc_benchmark_20261031/summary/nc_three_seed_report.md` |
| Table 2 MRAGE 指标 | `outputs/lp_mrage_baselines_20261101/{sports-copurchase,cloth-copurchase}/lp/mrage/*/seed-*/final_test_metrics.json` |
| Table 4 聚合值 | `outputs/mrage_capability_nc5/summary/capability_summary.csv` |
| Table 4 逐 seed 值与 run_dir | `outputs/mrage_capability_nc5/summary/capability_seed_metrics.csv` |
| 组件消融（本文件未包含） | `outputs/mrage_ablation_nc5/summary/nc5_ablation_summary.md` |

```bash
# NC baseline 主表（5 数据集 × 12 baselines，seed 42/43/44）
python scripts/run_nc_baseline_benchmark.py --device cuda:0

# 追加 GraphMoRE / NTSFormer
python scripts/run_nc_additional_baseline_benchmark.py --device cuda:0

# MRAGE 正式 NC benchmark（同一套 tuned 配置，validation-only 选参）
python scripts/run_nc_mrage_benchmark.py --device cuda:0

# LP：先完成 MRAGE Sports/Cloth，再跑 10 个 LP baselines
python scripts/run_lp_baseline_benchmark.py --device cuda:1 --resume
```

---

*生成时间：2026-11-04。表中数值按项目现有 completed 运行汇总；未完成 run 未参与任何均值。*
