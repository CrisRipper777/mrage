# MRAGE 五数据集 NC 调参结果

配置选择仅使用 Validation Accuracy，Macro-F1 辅助；最终 Test 在 selection.json 冻结后统一评估，不参与调参。

统一配置：`a14_hidden_dim_128`，完整模型与 Task 配置在 `selected_common.yaml`。原 `configs/model/mrage.yaml` 保持不变。

## 最终 Test 结果

固定官方 NC 划分、训练种子 42/43/44。均值 ± 总体标准差（ddof=0），单位为百分数；差值单位为百分点。这里只呈现基准与冻结后选定的配置，未选中的候选不评估 Test。

| 数据集 | 原始 Test Acc | 调优 Test Acc | Acc 差值 | 原始 Test Macro-F1 | 调优 Test Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Movies | 54.89 ± 0.03 | 54.57 ± 0.20 | -0.32 | 47.12 ± 0.66 | 47.68 ± 0.55 |
| Toys | 78.67 ± 0.42 | 79.15 ± 0.33 | +0.48 | 75.58 ± 0.43 | 76.21 ± 0.53 |
| Grocery | 83.06 ± 0.12 | 83.15 ± 0.18 | +0.10 | 74.89 ± 0.53 | 74.51 ± 0.84 |
| ele-fashion | 87.43 ± 0.08 | 87.51 ± 0.12 | +0.08 | 68.94 ± 0.16 | 68.77 ± 0.83 |
| Reddit-S | 96.34 ± 0.30 | 96.52 ± 0.06 | +0.18 | 92.39 ± 0.85 | 92.19 ± 0.20 |

五数据集等权平均 Test Accuracy：基准 80.0761%，调优 80.1803%，差值 +0.1042 个百分点。该平均数不是将五个数据集节点混合后的 Accuracy。

## Validation 选择依据

| 候选 | Movies | Toys | Grocery | ele-fashion | Reddit-S | 五数据集均值 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 55.55 ± 0.26 | 79.97 ± 0.09 | 83.64 ± 0.06 | 87.33 ± 0.12 | 96.16 ± 0.07 | 80.5301 |
| a13_expert_eta_0p2 | 56.01 ± 0.25 | 80.06 ± 0.13 | 83.26 ± 0.28 | 87.45 ± 0.01 | 96.10 ± 0.08 | 80.5756 |
| a14_hidden_dim_128 | 55.98 ± 0.30 | 80.43 ± 0.05 | 83.45 ± 0.09 | 87.54 ± 0.12 | 96.34 ± 0.04 | 80.7468 |

| 数据集 | Val Acc 差值 (pp) | 逐种子 Acc 提升数 | 原始 Val Macro-F1 | 调优 Val Macro-F1 |
|---|---:|---:|---:|---:|
| Movies | +0.430 | 3/3 | 47.57 ± 0.54 | 48.21 ± 1.39 |
| Toys | +0.459 | 3/3 | 76.84 ± 0.21 | 77.57 ± 0.05 |
| Grocery | -0.195 | 0/3 | 76.82 ± 0.34 | 75.78 ± 1.35 |
| ele-fashion | +0.211 | 3/3 | 67.84 ± 0.62 | 68.09 ± 0.70 |
| Reddit-S | +0.178 | 3/3 | 92.77 ± 0.39 | 92.61 ± 0.35 |

Test Accuracy 均值下降的数据集：Movies。这些 Test 结果不用于重新选择参数。

最终选择对五个数据集赋相同权重。初筛阶段只使用 Movies、Grocery、ele-fashion 的 seed42；每阶段候选相对锚点仅改变一个参数。阶段获胜者成为下一阶段锚点，没有运行全组合网格搜索。最终三种子结果优先于单种子排序。

## 选定参数

```yaml
task:
  optimizer: adamw
  lr: 0.001
  weight_decay: 1.0e-05
  epochs: 300
  patience: 30
  early_stop_min_epoch: 30
  early_stop_min_delta: 0.0001
  grad_clip: 1.0
  scheduler: null
model:
  hidden_dim: 128
  dropout: 0.3
  late_cross_dropout: 0.15
  fusion_dropout: 0.2
  gamma_init: 0.16
  beta_init: 0.05
  rho_init: 0.25
  router_hidden_dim: 256
  attention_dim: 32
  router_embed_dim: 8
  channel_gate_hidden: 128
  nc_unimodal_weight: 0.3
  balance_weight: 0.01
  expert_bottleneck: 64
  expert_eta: 0.2
  num_experts: 4
  top_k: 2
  physical_hops: 3
  semantic_hops: 1
  semantic_knn_k: 8
```

其他 Task/Model 参数完整保存在 selected_common.yaml。rho/gamma/beta 均为可训练系数的初始化值，不代表训练后的固定权重。四专家、Top-2、Physical 3-hop、Semantic 1-hop、KNN K=8 保持原样。

## 收敛与效率

以下按相同数据集三个训练种子的训练日志估算，每 epoch 耗时不包含首次数据加载、语义图准备及进程启动。CPU 线程数 8；硬件为 RTX 3090。部分阶段使用 GPU0/1 双卡各跑一次训练，用户预约 GPU0 后，余下训练与 Final Test 全部使用物理 GPU1。并发条件发生过变化，这些是本机观测，不是独立的严格速度基准。

| 数据集 | 基准参数量 | 调优参数量 | 基准秒/epoch | 调优秒/epoch | 基准总轮数均值 | 调优总轮数均值 |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 2062795 | 920011 | 0.359 | 0.230 | 97.7 | 118.3 |
| Toys | 2061767 | 919495 | 0.396 | 0.261 | 87.3 | 95.7 |
| Grocery | 2062795 | 920011 | 0.350 | 0.236 | 109.7 | 122.7 |
| ele-fashion | 1796539 | 786875 | 1.498 | 0.964 | 151.7 | 178.7 |
| Reddit-S | 2062795 | 920011 | 0.362 | 0.232 | 88.7 | 83.0 |

全部完成 136 次 Validation 训练、30 次 Final Test。训练失败记录 0，达到 300 epoch 上限的训练 0。

本次只是有界单变量搜索；没有覆盖参数间的全部交互，也没有为 512 维重新搜索学习率。三种子的波动用于描述稳定性，不能据此宣称统计显著。未把不同数据集或训练种子作为新的官方划分。

## 可复现记录

- `study.yaml`：搜索空间和所有候选覆盖参数；`axis_decisions.md/json`：15 个轴的完整选择记录。
- `trials.csv/json`：逐次运行路径、配置哈希、划分哈希、种子、指标、耗时；每个运行目录保留解析配置、日志、最佳 Validation Checkpoint 和结果。
- `validation_summary.csv` / `final_test_summary.csv`：三种子均值、标准差、逐种子 Accuracy 与 Macro-F1。
- `convergence.csv`：实际训练轮数、最佳 epoch 与速度；`selection.json`：看 Test 前冻结的选择及 Validation 证据。
- `study_metadata.json`：起始提交、原 mrage.yaml 哈希；`latest_audit.json`：末次运行检查。
- `preflight_failed/`：早期不支持的 task.torch_threads 配置在训练启动前被拒绝的记录；随后改用环境变量，未计入训练结果。
- 原始默认配置快照：`base_model.yaml`、`base_task_nc.yaml`。

数据划分：Movies/Toys/Grocery/Reddit-S 读取各自官方 MAGB seed42 NC 划分；ele-fashion 读取原有官方 split.pt。各数据集全部训练种子的 split_id 一致。没有修改官方划分或生成新的 Split Seed。

复现统一配置（以 Movies 为例，输出到新目录）：

```bash
conda run --no-capture-output -n yhf_env python scripts/run_paper_benchmark.py \
  --dataset Movies --task nc --models mrage --seeds 42 43 44 \
  --device cuda:1 --stage validation \
  --experiment-config experiments/nc5_tuning_20261030/selected_common.yaml \
  --output-root outputs/nc5_selected_reproduction
```

原 mrage.yaml 未修改，MRAGE 核心实现未修改；保留工作区其他 Baseline/LP 修改，未执行 git push。
