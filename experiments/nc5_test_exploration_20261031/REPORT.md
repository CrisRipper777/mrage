# MRAGE NC5：Test 参与选参的有界探索

> 当前采用方案已更新：Movies采用用户既有默认运行（2026-10-31/13-05-49）。当前采用清单见 [ADOPTED_RESULTS.md](ADOPTED_RESULTS.md)，配置位于 adopted/。本报告及 selected/ 保留原65次搜索阶段的数值排序与同批次对照，不代表Movies当前采用方案。

本轮按用户明确要求，以三种子平均 Test Accuracy 为主，平均 Test Macro-F1 在 Accuracy 完全相同时破同分，标准差辅助描述稳定性。Test 已参与参数选择，因此本报告是探索性结果，不能视作独立留出集上的无偏泛化验证。与原 Validation 选参正式 Benchmark 分开保留。

固定官方 NC 划分，训练种子42/43/44；单卡物理GPU1；每次训练仍按最佳Validation Accuracy保存Checkpoint。每次保存模型只正式评估一次Test。Metric仍采用all_classes，ele-fashion空类别ID5的口径没有单独修改。

## 与本轮默认参数的三种子对照

均值 ± 总体标准差（ddof=0），指标单位 %，差值单位百分点。下表最优仅指本轮已测、已补齐三种子的候选。

| 数据集 | 选中候选 | 默认 Test Acc | 选中 Test Acc | Acc 差值 | 默认 Macro-F1 | 选中 Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| Movies | tuned256 | 54.60 ± 0.35 | 54.86 ± 0.59 | +0.260 | 47.01 ± 1.39 | 47.60 ± 0.65 |
| Toys | tuned128 | 78.67 ± 0.37 | 79.03 ± 0.39 | +0.362 | 75.67 ± 0.38 | 76.29 ± 0.48 |
| Grocery | tuned128 | 82.90 ± 0.14 | 83.26 ± 0.19 | +0.361 | 74.85 ± 0.52 | 74.69 ± 0.51 |
| Reddit-S | reddit_aux01 | 96.37 ± 0.26 | 96.53 ± 0.08 | +0.157 | 92.43 ± 0.79 | 91.71 ± 0.46 |
| ele-fashion | fashion_gate64 | 87.34 ± 0.07 | 87.63 ± 0.05 | +0.295 | 68.30 ± 0.33 | 68.93 ± 0.31 |

五数据集等权平均 Test Accuracy：默认 79.9759%，逐数据集选中配置 80.2631%，差值 +0.2872pp。该数不是合并所有节点后的Accuracy。

## 三套基础配置的完整复跑

| 数据集 | 默认 Acc / F1 | 原调优256 Acc / F1 | 原调优128 Acc / F1 |
|---|---:|---:|---:|
| Movies | 54.60 ± 0.35 / 47.01 ± 1.39 | 54.86 ± 0.59 / 47.60 ± 0.65 | 54.73 ± 0.20 / 47.61 ± 0.61 |
| Toys | 78.67 ± 0.37 / 75.67 ± 0.38 | 78.75 ± 0.32 / 75.76 ± 0.57 | 79.03 ± 0.39 / 76.29 ± 0.48 |
| Grocery | 82.90 ± 0.14 / 74.85 ± 0.52 | 83.25 ± 0.32 / 75.97 ± 0.34 | 83.26 ± 0.19 / 74.69 ± 0.51 |
| Reddit-S | 96.37 ± 0.26 / 92.43 ± 0.79 | 96.30 ± 0.19 / 92.23 ± 0.51 | 96.52 ± 0.06 / 92.19 ± 0.20 |
| ele-fashion | 87.34 ± 0.07 / 68.30 ± 0.33 | 87.41 ± 0.09 / 68.74 ± 0.10 | 87.58 ± 0.18 / 69.39 ± 0.53 |

三套基础配置按每个数据集三种子完整复跑，共45次训练。统一配置按五数据集等权平均Test Accuracy选择，Macro-F1破完全同分。

| 统一候选 | 五数据集平均 Test Acc (%) | 五数据集平均 Macro-F1 (%) |
|---|---:|---:|
| default | 79.9759 | 71.6556 |
| tuned256 | 80.1160 | 72.0597 |
| tuned128 | 80.2244 | 72.0327 |

本轮最佳统一候选为 `tuned128`，完整配置：selected/common.yaml。

## 各数据集选中参数相对默认值的变化

所有未列参数维持默认。完整Model/Task配置保存在selected/<dataset>.yaml，原configs/model/mrage.yaml未改。

### Movies：tuned256

```yaml
model.expert_eta: 0.2
model.dropout: 0.3
model.gamma_init: 0.16
model.beta_init: 0.05
task.weight_decay: 1.0e-05
```

### Toys：tuned128

```yaml
model.hidden_dim: 128
model.expert_eta: 0.2
model.dropout: 0.3
model.gamma_init: 0.16
model.beta_init: 0.05
task.weight_decay: 1.0e-05
```

### Grocery：tuned128

```yaml
model.hidden_dim: 128
model.expert_eta: 0.2
model.dropout: 0.3
model.gamma_init: 0.16
model.beta_init: 0.05
task.weight_decay: 1.0e-05
```

### Reddit-S：reddit_aux01

```yaml
model.hidden_dim: 128
model.expert_eta: 0.2
model.dropout: 0.3
model.gamma_init: 0.16
model.beta_init: 0.05
model.nc_unimodal_weight: 0.1
task.weight_decay: 1.0e-05
```

### ele-fashion：fashion_gate64

```yaml
model.dropout: 0.3
model.gamma_init: 0.16
model.beta_init: 0.05
model.channel_gate_hidden: 64
task.weight_decay: 1.0e-05
```

## 定向初筛和确认记录

每个数据集只试两个预先记录理由的候选，seed42初筛，然后只给该数据集初筛Accuracy较优者补43/44。未进入确认的候选只有n=1，不能与n=3结果当作同等证据。

| 数据集 | 定向候选 | n | Test Acc | Test Macro-F1 |
|---|---|---:|---:|---:|
| Movies | movies_gamma016 | 3 | 54.58 ± 0.23 | 48.15 ± 1.13 |
| Movies | movies_balanced | 1 | 53.04 ± 0.00 | 46.56 ± 0.00 |
| Toys | toys_aux01 | 3 | 78.87 ± 0.42 | 75.91 ± 0.63 |
| Toys | toys_aux05 | 1 | 78.50 ± 0.00 | 75.90 ± 0.00 |
| Grocery | grocery_wd1e5 | 1 | 83.07 ± 0.00 | 75.85 ± 0.00 |
| Grocery | grocery_lr3e4 | 3 | 83.20 ± 0.23 | 75.41 ± 0.04 |
| Reddit-S | reddit_dropout02 | 1 | 96.16 ± 0.00 | 90.91 ± 0.00 |
| Reddit-S | reddit_aux01 | 3 | 96.53 ± 0.08 | 91.71 ± 0.46 |
| ele-fashion | fashion_gate64 | 3 | 87.63 ± 0.05 | 68.93 ± 0.31 |
| ele-fashion | fashion_balanced | 1 | 87.37 ± 0.00 | 69.84 ± 0.00 |

## Accuracy 接近时的实际取舍

| 数据集 | Accuracy前两名 | 均值差 (pp) | 三次累计正确预测数差 | 前两名Macro-F1均值 (%) |
|---|---|---:|---:|---:|
| Movies | tuned256 / tuned128 | 0.1299 | 13 | 47.60 / 47.61 |
| Toys | tuned128 / toys_aux01 | 0.1611 | 20 | 76.29 / 75.91 |
| Grocery | tuned128 / tuned256 | 0.0098 | 1 | 74.69 / 75.97 |
| Reddit-S | reddit_aux01 / tuned128 | 0.0105 | 1 | 91.71 / 92.19 |
| ele-fashion | fashion_gate64 / tuned128 | 0.0534 | 47 | 68.93 / 69.39 |

正确预测数差按相同Test节点在三次运行中的累计次数计算。数值最优不等于统计显著更好；尤其当Acc只高一个累计正确预测、却损失较多Macro-F1时，必须保留这个取舍，不能笼统称两项都提升。

具体候选继承和理由见plan.json。movies_gamma016和fashion_gate64为历史完整配置的复跑，不是相对本轮tuned256只改一个参数；不据此单独归因某个机制。其他定向候选按计划相对指定锚点改变一个参数。

## 逐种子结果

| 数据集 | 配置 | Seed | Test Acc (%) | Test Macro-F1 (%) | 最佳epoch |
|---|---|---:|---:|---:|---:|
| Movies | default | 42 | 54.5427 | 47.8605 | 72 |
| Movies | default | 43 | 55.0525 | 48.1307 | 60 |
| Movies | default | 44 | 54.2129 | 45.0525 | 72 |
| Movies | tuned256 | 42 | 55.3223 | 46.7502 | 74 |
| Movies | tuned256 | 43 | 55.2324 | 48.3413 | 54 |
| Movies | tuned256 | 44 | 54.0330 | 47.7093 | 77 |
| Toys | default | 42 | 78.8355 | 75.7576 | 65 |
| Toys | default | 43 | 79.0046 | 76.0874 | 53 |
| Toys | default | 44 | 78.1590 | 75.1770 | 51 |
| Toys | tuned128 | 42 | 78.4972 | 75.6187 | 54 |
| Toys | tuned128 | 43 | 79.1737 | 76.5974 | 72 |
| Toys | tuned128 | 44 | 79.4153 | 76.6654 | 60 |
| Grocery | default | 42 | 82.7233 | 75.5789 | 113 |
| Grocery | default | 43 | 82.8990 | 74.6151 | 78 |
| Grocery | default | 44 | 83.0747 | 74.3671 | 83 |
| Grocery | tuned128 | 42 | 83.4261 | 75.0870 | 78 |
| Grocery | tuned128 | 43 | 83.3675 | 73.9638 | 65 |
| Grocery | tuned128 | 44 | 82.9868 | 75.0045 | 83 |
| Reddit-S | default | 42 | 96.0365 | 91.4280 | 31 |
| Reddit-S | default | 43 | 96.4140 | 92.5251 | 53 |
| Reddit-S | default | 44 | 96.6656 | 93.3469 | 91 |
| Reddit-S | reddit_aux01 | 42 | 96.4140 | 91.0637 | 46 |
| Reddit-S | reddit_aux01 | 43 | 96.5712 | 91.9891 | 49 |
| Reddit-S | reddit_aux01 | 44 | 96.6027 | 92.0795 | 54 |
| ele-fashion | default | 42 | 87.4361 | 68.6664 | 146 |
| ele-fashion | default | 43 | 87.3133 | 68.3713 | 119 |
| ele-fashion | default | 44 | 87.2690 | 67.8695 | 99 |
| ele-fashion | fashion_gate64 | 42 | 87.6236 | 68.7661 | 118 |
| ele-fashion | fashion_gate64 | 43 | 87.5793 | 68.6634 | 106 |
| ele-fashion | fashion_gate64 | 44 | 87.7020 | 69.3691 | 152 |

## 稳定性、限制与保留材料

当前运行仍使用项目原有CUDA执行方式，没有在本轮修改稀疏算子或启用新的确定性训练协议。此前已发现同seed重跑会有差异，因此小于波动幅度的均值增益不等同于稳定或统计显著提升。没有因为本轮结果不好而挑掉某个seed，也没有将历史较好的单次运行混入本轮平均。

作为历史参照，用户的默认运行Movies（outputs/2026-10-31/13-05-49）为55.04±0.22% Test Acc；该数与本轮默认值分开记录，不参与拼接均值。ele-fashion对应用户默认目录为outputs/2026-10-31/13-11-08。

Checkpoint仍按Validation Accuracy选择；选参使用了Test，所以本轮所选参数日后若写作独立泛化改进结论，需要另行设计独立验证。这里不把Test选参收益冒称为未见Test时的提升。

实际完成训练数：65，预算上限：65。每次保留config_resolved.yaml、best_validation.pt、train.log、validation_metrics.json、final_test_metrics.json、run_metadata.json、status.json。

- plan.json：先验候选、搜索理由和上限；focused_decisions.json：定向确认的选择记录。
- trials.csv/json：逐次指标、seed、耗时、配置/划分哈希、运行目录；summary.csv/json：均值与标准差。
- selection.json：Test选参口径、统一配置与逐数据集选择。
- selected/*.yaml：可直接供正式Runner使用的完整覆盖配置。
- outputs/nc5_test_exploration_20261031/：本轮全部原始运行，独立于旧调参和正式Benchmark目录。

复现（以Movies为例，使用新输出目录）：

```bash
conda run --no-capture-output -n yhf_env python scripts/run_paper_benchmark.py \
  --dataset Movies --task nc --models mrage --seeds 42 43 44 --split-seed 42 \
  --device cuda:1 --stage all \
  --experiment-config experiments/nc5_test_exploration_20261031/selected/Movies.yaml \
  --output-root outputs/nc5_test_selected_reproduction
```
