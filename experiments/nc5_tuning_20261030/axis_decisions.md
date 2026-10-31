# 单变量初筛记录

固定官方 NC 划分，代表数据集 Movies/Grocery/ele-fashion，训练 seed42；按三数据集等权平均 Validation Accuracy 选择，Macro-F1 破同分。分数为百分数。每行候选均相对该阶段锚点只改变一个参数。

| 参数 | 搜索值 | 选中值 | 选中候选 | 平均 Val Acc |
|---|---|---:|---|---:|
| task.lr | [0.001, 0.0006, 0.0003] | 0.001 | baseline | 75.5417 |
| task.weight_decay | [1e-05, 0.0001, 0.001] | 1e-05 | a01_weight_decay_1em05 | 75.5956 |
| model.dropout | [0.1, 0.2, 0.3] | 0.3 | a02_dropout_0p3 | 75.6202 |
| model.gamma_init | [0.04, 0.08, 0.16, 0.25] | 0.16 | a03_gamma_init_0p16 | 75.6748 |
| model.beta_init | [0.05, 0.15, 0.25, 0.4] | 0.05 | a04_beta_init_0p05 | 75.7382 |
| model.rho_init | [0.15, 0.25, 0.4] | 0.25 | a04_beta_init_0p05 | 75.7382 |
| model.router_hidden_dim | [128, 256, 512] | 256 | a04_beta_init_0p05 | 75.7382 |
| model.attention_dim | [16, 32, 64] | 32 | a04_beta_init_0p05 | 75.7382 |
| model.router_embed_dim | [4, 8, 16] | 8 | a04_beta_init_0p05 | 75.7382 |
| model.channel_gate_hidden | [64, 128, 256] | 128 | a04_beta_init_0p05 | 75.7382 |
| model.nc_unimodal_weight | [0.1, 0.3, 0.5] | 0.3 | a04_beta_init_0p05 | 75.7382 |
| model.balance_weight | [0.001, 0.01, 0.03] | 0.01 | a04_beta_init_0p05 | 75.7382 |
| model.expert_bottleneck | [32, 64, 128] | 64 | a04_beta_init_0p05 | 75.7382 |
| model.expert_eta | [0.05, 0.1, 0.2] | 0.2 | a13_expert_eta_0p2 | 75.8001 |
| model.hidden_dim | [128, 256, 512] | 128 | a14_hidden_dim_128 | 75.8004 |

hidden_dim=512 在选定的 lr=0.001 下平均表现下降；没有额外对该维度重新扫描学习率，因此该结论仅适用于本次条件比较。128 与 256 在代表数据集 seed42 上几乎持平，需要五数据集三训练种子确认。
