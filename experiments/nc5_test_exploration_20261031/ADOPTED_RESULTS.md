# MRAGE 五数据集当前采用方案

根据用户最新决定，Movies采用原始默认配置及其既有运行结果；其余四个数据集沿用65次探索后选中的配置。

所有指标均为训练种子42/43/44的均值 ± 总体标准差（ddof=0），单位 %。Movies与其他四个数据集来自不同批次，未拼接不同运行的seed，也不将这个采用表描述为同批次对照实验。

| 数据集 | 当前采用配置 | Test Accuracy | Test Macro-F1 | 来源 |
|---|---|---:|---:|---|
| Movies | default | 55.04 ± 0.22 | 47.95 ± 0.30 | 用户原始默认运行 2026-10-31/13-05-49 |
| Toys | tuned128 | 79.03 ± 0.39 | 76.29 ± 0.48 | Test参与选参的65次探索，三种子确认 |
| Grocery | tuned128 | 83.26 ± 0.19 | 74.69 ± 0.51 | Test参与选参的65次探索，三种子确认 |
| ele-fashion | fashion_gate64 | 87.63 ± 0.05 | 68.93 ± 0.31 | Test参与选参的65次探索，三种子确认 |
| Reddit-S | reddit_aux01 | 96.53 ± 0.08 | 91.71 ± 0.46 | Test参与选参的65次探索，三种子确认 |

Movies不再采用搜索阶段的tuned256及54.86±0.59%那批结果；采用的是用户原来完整的三个seed汇总55.04±0.22% Accuracy、47.95±0.30% Macro-F1。未将其中较好的seed挑出来与其他批次混合。

## Movies原始默认参数

```yaml
model:
  hidden_dim: 256
  dropout: 0.2
  gamma_init: 0.08
  beta_init: 0.25
  rho_init: 0.25
  expert_eta: 0.1
  nc_unimodal_weight: 0.3
task:
  optimizer: adamw
  lr: 0.001
  weight_decay: 0.0001
  loss:
    class_balanced_main_ce: false
```

其余参数完整保存于adopted/Movies.yaml，直接取自原运行的.hydra/config.yaml中的Model/Task配置，已与研究中的default候选核对一致。原configs/model/mrage.yaml未修改。

## 来源和保留方式

- adopted_selection.json：当前采用清单、准确指标、配置位置和逐项来源。
- adopted/*.yaml：五个数据集当前采用的完整Model/Task覆盖配置。
- adopted/sources/Movies_default_20261031_130549/：原Movies运行的完整副本，含日志、results.json和.hydra配置；文件哈希已逐一核对。
- adopted_results_percent.csv：本采用表的数值CSV（百分数）。
- selection.json、selected/*.yaml、REPORT.md：保留65次探索的原始数值排序、配置和同批次对照结果，属于历史搜索记录，不再代表Movies当前采用方案。

原Movies运行save_ckpt_path=null，没有保存Checkpoint。此处保留的是原始结果、日志与可复现参数，本次没有重新训练或覆盖原结果。原日志只打印到两位小数的逐seed指标，精确三seed汇总以原results.json为准。

其他四个数据集仍属于Test参与选参的探索结果；本次只调整采用决定，不改变原实验性质。原65次探索对照仍按原批次保存，当前采用表不计算跨批次对照收益。
