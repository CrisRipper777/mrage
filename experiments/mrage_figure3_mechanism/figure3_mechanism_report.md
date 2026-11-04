# Figure 3 机制分析报告

**Mechanistic Analysis of Shared Adaptive Graph Experts**

- 分析分支：`codex/mrage-figure3-mechanism`
- 基线分支 / Commit：`codex/mrage-sports-lp-ablation` / `dce34f8`
- 分析对象：冻结的 Full MRAGE Checkpoint（5 个数据集 × 3 个训练种子 = 15 个）
- 分析模式：`model.eval()` + `torch.inference_mode()`，只读
- 评估数据：仅 Validation；未调用正式 Test 评估器
- 原始 adopted 运行：未重训、未修改 Checkpoint、语义缓存或原始数据
- Movies：历史 adopted 运行没有 Checkpoint；本轮按 adopted `default` 配置新增了 **validation-only** 的 3 个 Full MRAGE Checkpoint，仅用于机制分析

---

## 1. 执行摘要

本轮成功分析了 **5 个数据集 × 3 个训练种子 = 15 个 Full MRAGE Checkpoint**，全部完成 Figure 3(a)(b)(c)，没有 Checkpoint 加载失败：

| 数据集 | 种子 | 采用的 Full 配置 | 完成情况 |
|---|---|---|---|
| Grocery | 42, 43, 44 | `tuned128` | 3/3 |
| Toys | 42, 43, 44 | `tuned128` | 3/3 |
| Reddit-S | 42, 43, 44 | `reddit_aux01` | 3/3 |
| ele-fashion | 42, 43, 44 | `fashion_gate64` | 3/3 |
| Movies | 42, 43, 44 | adopted `default` | 3/3（新增 validation-only Checkpoint） |

共完成：

- **Figure 3(a)**：15 个 Checkpoint 的有效滤波系数、初始化变化、专家两两系数距离、真实 Expert Output 余弦相似度。
- **Figure 3(b)**：15 个 Checkpoint 的六类 Router 上下文、四专家 Top-2 路由权重与选择频率、节点级路由多样性、上下文 JSD。
- **Figure 3(c)**：15 个 Checkpoint × 4 个主干预分组 × 10 次 Shuffle = **600 次主分组试验**；另完成 6 个单独上下文 × 15 × 10 = **900 次单上下文试验**；合计 **1,500 次真实 Shuffle 试验**。
- 15 次 Identity-Permutation 检查全部通过（prediction mismatch = 0；最大 embedding 差异 ≤ `2.2 × 10⁻⁶`）。

所有 15 个 Checkpoint 的模型参数与 `edge_index` 校验和在分析前后一致。图 3(c) 的 1,500 次干预全部产生有限指标，无失败。

运行资源：`cuda:0`，与正在运行的 Sports LP 训练共享 GPU；主机峰值 RSS 约 2.7 GB，无 OOM。Movies 三个种子的 validation-only 训练约 6 分 28 秒；批量分析约：专家 39 秒、Router 39 秒、Shuffle（含单上下文）10 分 22 秒。

---

## 2. 使用的 Checkpoint

完整路径、SHA256、resolved config 与逐 Checkpoint 的保存/重算 Validation 指标见 `manifest.json` 和 `data/manifest_experts.json`。

| Dataset | Seed | 最佳 epoch | Saved Val Acc (%) | Recomputed Val Acc (%) | Saved Val Macro-F1 (%) | Recomputed Val Macro-F1 (%) |
|---|---:|---:|---:|---:|---:|---:|
| Grocery | 42 | 78 | 83.46 | 83.46 | 75.39 | 75.39 |
| Grocery | 43 | 65 | 83.31 | 83.31 | 74.16 | 74.16 |
| Grocery | 44 | 83 | 83.28 | 83.28 | 76.61 | 76.61 |
| Toys | 42 | 54 | 80.43 | 80.43 | 77.02 | 77.02 |
| Toys | 43 | 72 | 80.50 | 80.50 | 77.81 | 77.81 |
| Toys | 44 | 60 | 80.50 | 80.50 | 77.43 | 77.43 |
| Reddit-S | 42 | 46 | 96.26 | 96.26 | 92.01 | 92.01 |
| Reddit-S | 43 | 49 | 96.54 | 96.54 | 92.99 | 92.99 |
| Reddit-S | 44 | 54 | 96.32 | 96.32 | 92.06 | 92.06 |
| ele-fashion | 42 | 118 | 87.74 | 87.74 | 68.38 | 68.38 |
| ele-fashion | 43 | 106 | 87.61 | 87.61 | 67.66 | 67.66 |
| ele-fashion | 44 | 152 | 87.43 | 87.43 | 68.47 | 68.47 |
| Movies | 42 | 84 | 55.76 | 55.76 | 47.34 | 47.34 |
| Movies | 43 | 73 | 55.25 | 55.25 | 49.06 | 49.06 |
| Movies | 44 | 79 | 55.73 | 55.73 | 47.58 | 47.58 |

重算值与保存值在显示精度内一致（最大差异约 `1.2 × 10⁻⁷`）。Movies 的 3 个 Checkpoint 位于：

`outputs/mrage_figure3_movies_full/Movies/nc/mrage/413afa0d5c05843e/seed-{42,43,44}/best_validation.pt`

其训练使用 adopted Movies `default` 配置、固定官方 NC split、`--stage validation`，`test_evaluation_count = 0`。历史 adopted Movies 运行（`outputs/2026-10-31/13-05-49`）没有保存权重，因此本轮新增训练没有覆盖任何原始运行。

---

## 3. 方法

### 3.1 Figure 3(a)：有效滤波系数与专家输出

正式 MRAGE 的四个专家共享同一个 `filter_bank.theta`，形状 `[4, 4]`。对物理关系，`max_order = 3`；对语义关系，`max_order = 1`。有效系数严格复现源码：

```text
mask = (order <= max_order)
raw = theta * mask
coefficient = raw / (|raw|.sum(-1, keepdim=True) + 1e-8)
```

语义系数只取前两列，分母也只使用前两阶的绝对值之和；**没有**用物理四阶的归一化结果代替语义系数。初始化系数直接使用 `SharedExpertBank.__init__` 中的字面量 `theta`。

另外，在 Eval Forward 中通过临时 Hook 捕获 `SharedExpertBank` 的真实专家输出 `[4, N, H]`，对每个传播上下文的最多 4096 个有效节点计算专家两两余弦相似度。实际采样数量记录在 `data/expert_diversity.csv` 的 `n_sampled` 字段。

### 3.2 Figure 3(b)：六类路由上下文

正式 `Model.forward` 中 Router 共被调用六次，顺序固定：

1. Private Text–Physical
2. Private Text–Semantic
3. Private Visual–Physical
4. Private Visual–Semantic
5. Collaborative Text–Physical
6. Collaborative Visual–Physical

分析不是仅凭 `modality_id/relation_id` 识别上下文（Private Physical 与 Collaborative Physical 的 ID 相同），而是在 `_filter` 包装器中按实际调用顺序打标签，并校验 `(modality_id, relation_id)`。每次 Router 调用记录 `[N, 4]` logits，再按正式模型的 Top-2 规则计算：

- 每个节点的 Top-2 专家；
- Top-2 内重新归一化的 Routing Weight；
- 每个专家的平均 Top-2 weight（每行和约 1）；
- 每个专家的 Top-2 选中频率（每行和约 2）；
- Top-2 组合数量、最常见组合占比、节点间 Routing Weight 标准差；
- 六类上下文的平均路由分布与两两 JSD。

Physical 上下文使用 Physical Support；Semantic 上下文使用对应模态的 Semantic Support；Collaborative Physical 使用 Physical Support。由于本模型中 `cross_context` 几乎对所有物理有效节点都非零（见 4.2），额外记录的 `cross_nonzero` 子集与 Physical Support 基本相同。

### 3.3 Figure 3(c)：冻结 Validation Router-Shuffle

对于选定的上下文，取得原始 Router logits `[N, 4]`，在当前上下文的有效节点集合内随机置乱**整个四维 logit 向量**：

```text
shuffled_logits[active] = logits[active][permutation(active)]
```

然后继续使用原模型的 Top-2、Softmax、专家混合和分类头。没有置乱 Expert ID、输入特征、物理图、语义图、专家参数、Router 参数或分类头参数。

干预分组：

- **主分组（Figure 3(c) 主图）**：Private Physical（C1+C3）、Private Semantic（C2+C4）、Collaborative Physical（C5+C6）、All Contexts（C1–C6）。
- **单上下文补充**：C1–C6 各自单独置乱，用于定位哪个具体上下文最敏感。

每个 Checkpoint、每个分组重复 10 次，固定 `shuffle_seed` 并记录到 `router_shuffle_trials.csv`。每个 Checkpoint 先对 10 次 Shuffle 求均值和总体标准差，再按训练种子汇总；**没有**把 30 次 Shuffle 当作 30 次独立模型训练。

评价指标仅使用 Validation：

- Validation Cross-Entropy；
- Validation Accuracy；
- Prediction Disagreement Rate；
- Validation Macro-F1（辅助）。

`Delta CE = Shuffle CE - Original CE`；`Delta Acc = Original Acc - Shuffle Acc`。

---

## 4. 结果

### 4.1 Figure 3(a)：专家具有不同有效滤波配置和真实输出

#### 4.1.1 有效系数

Grocery seeds 42–44 的均值如下（完整逐 Checkpoint 数值见 `data/expert_coefficients.csv`）。

**Physical（4×4，orders 0–3）**

| Expert | Order 0 | Order 1 | Order 2 | Order 3 |
|---|---:|---:|---:|---:|
| E1 | 0.550 | -0.214 | 0.175 | -0.060 |
| E2 | -0.238 | 0.510 | -0.072 | 0.179 |
| E3 | 0.193 | 0.146 | -0.413 | 0.248 |
| E4 | -0.111 | -0.137 | 0.338 | 0.414 |

**Semantic（4×2，orders 0–1）**

| Expert | Order 0 | Order 1 |
|---|---:|---:|
| E1 | 0.720 | -0.280 |
| E2 | -0.318 | 0.682 |
| E3 | 0.569 | 0.431 |
| E4 | -0.447 | -0.553 |

四个专家的符号模式明显不同（例如 E1 的 order-0 为正、E2 的 order-0 为负；E3/E4 的高阶项方向相反）。该结构在 Toys、Reddit-S、ele-fashion、Movies 上高度一致。Movies 的均值例如：Physical E1 `[0.532, -0.232, 0.163, -0.073]`、E4 `[-0.104, -0.163, 0.338, 0.395]`；Semantic E1 `[0.697, -0.303]`、E4 `[-0.391, -0.609]`。完整矩阵见 CSV。

#### 4.1.2 初始化 vs 训练

| 统计量 | Physical | Semantic |
|---|---:|---:|
| 初始化到训练后 L2 变化（全部 60 个专家-关系） | 0.0496（范围 0.0091–0.1030） | 0.0494（范围 0.0001–0.1554） |
| 初始化时专家两两系数 L2 距离（6 对） | 0.8601（范围 0.6702–1.1254） | 1.1498（范围 0.7339–1.4159） |
| 训练后专家两两系数 L2 距离（90 个 Checkpoint-对） | 0.8534（范围 0.6264–1.1466） | 1.1521（范围 0.7205–1.4353） |
| 训练后专家两两系数余弦（90 对） | -0.1302（范围 -0.8898–0.3659） | -0.2917（范围 -0.9998–0.5388） |

**判断：在滤波系数层面，专家差异主要来自初始化，训练只做了 modest refinement。** 训练引起的单个专家 L2 位移约 0.05；但初始化时专家两两距离已经是 0.860（Physical）/1.150（Semantic），训练后几乎不变（0.853 / 1.152）。也就是说，训练**没有**把专家系数“推得更开”；它只是沿初始化附近的小范围调整。不能声称“训练创造了专家滤波多样性”；更准确的说法是：固定的四组初始滤波配置已经提供了不同的归纳偏置，训练在初始化附近做了小幅 refinement。

需要区分两个层级：Expert Output 还会经过训练后的 bottleneck MLP，因此输出层面的差异不能完全归因于初始化。本报告分别报告“滤波系数”和“真实专家输出”两个层级；上面的初始化结论只适用于前者。

#### 4.1.3 真实 Expert Output 差异

Eval Forward 中，专家真实输出的两两余弦相似度（所有 Checkpoint、所有专家对、该上下文的均值）：

| 上下文 | 平均余弦 | 范围 |
|---|---:|---:|
| Private Text–Physical | 0.3384 | -0.7132–0.8239 |
| Private Text–Semantic | -0.1133 | -0.9979–0.9619 |
| Private Visual–Physical | 0.3274 | -0.7617–0.8333 |
| Private Visual–Semantic | -0.1157 | -0.9981–0.9597 |
| Collaborative Text–Physical | 0.3310 | -0.7134–0.8031 |
| Collaborative Visual–Physical | 0.3210 | -0.7607–0.8302 |

物理上下文的专家输出平均只有中等相似度（约 0.32–0.34），语义上下文平均接近不相关（约 -0.11）。但范围非常宽：某些专家对在部分 Checkpoint/上下文上高度相似（接近 +0.96），也有接近完全相反（接近 -1.00）。因此“四个专家总是彼此不同”过强；更准确的结论是：**专家输出在平均意义上是不同的，但不同专家对之间存在明显的上下文/种子依赖。**

### 4.2 Figure 3(b)：路由随关系、Private/Collaborative 和节点变化

#### 4.2.1 Grocery 六类上下文的平均 Top-2 路由权重

| Context | E1 | E2 | E3 | E4 |
|---|---:|---:|---:|---:|
| Private Text–Physical | 0.10 | 0.17 | 0.10 | **0.63** |
| Private Text–Semantic | 0.19 | 0.28 | 0.22 | 0.32 |
| Private Visual–Physical | 0.07 | 0.27 | 0.05 | **0.61** |
| Private Visual–Semantic | 0.09 | **0.44** | 0.12 | 0.34 |
| Collaborative Text–Physical | 0.06 | **0.58** | 0.06 | 0.30 |
| Collaborative Visual–Physical | 0.04 | **0.55** | 0.02 | 0.38 |

每行四个权重之和约 1.00。最明显的变化是：**Private Physical 由 E4 主导（Grocery/Toys）或 E2/E4 混合（Reddit-S），而 Collaborative Physical 一致转为 E2 主导**。Grocery、Toys、Reddit-S 都呈现“E2 上升、E4 下降”的 Private→Collaborative 重分配；ele-fashion 的差异较弱。

#### 4.2.2 逐数据集路由摘要

| Dataset | Private Physical 主导 | Collaborative Physical 主导 | 主要观察 |
|---|---|---|---|
| Grocery | E4 (0.63 / 0.61) | E2 (0.58 / 0.55) | 最强的 Private→Collaborative E4→E2 重分配 |
| Toys | E4 (0.65 / 0.67) | E2 (0.47 / 0.52) | 同方向，E2 权重略低 |
| Reddit-S | E2/E4 (0.44/0.40, 0.43/0.25) | E2 (0.81 / 0.71) | Collaborative 几乎完全转向 E2 |
| ele-fashion | E1/E4 (0.40/0.26, 0.20/0.42) | E1/E2/E4 (0.36/0.22/0.33, 0.24/0.24/0.37) | 路由分化最弱，与已有消融中“图机制影响小”一致 |
| Movies | Text: E4 (0.49); Visual: E2/E4 (0.37/0.37) | Text: **E1 (0.43)**; Visual: E2 (0.51) | Text 协同上下文出现独有的 E1 主导 |

Movies 的 Collaborative Text–Physical 路由由 E1 主导（`0.434`），而不是其他数据集常见的 E2；Collaborative Visual–Physical 仍由 E2 主导（`0.507`）。这说明协同路由的专家偏好具有数据集依赖性，不能用一个全局模板概括。

#### 4.2.3 上下文距离（JSD）

六类上下文的平均 Top-2 路由分布之间的 JSD（均值，范围）：

| 比较 | 平均 JSD | 范围 |
|---|---:|---:|
| Text Physical vs Text Semantic | 0.0477 | 0.0052–0.1137 |
| Visual Physical vs Visual Semantic | 0.0284 | 0.0012–0.0827 |
| Text Physical vs Visual Physical | 0.0199 | 0.0025–0.0510 |
| Private Text Physical vs Collaborative Text Physical | 0.0711 | 0.0032–0.1896 |
| Private Visual Physical vs Collaborative Visual Physical | 0.0292 | 0.0023–0.0557 |

**判断：**

- **关系差异（Physical vs Semantic）** 和 **Private vs Collaborative 差异** 大于 **模态差异（Text vs Visual）**。
- 模态路由差异是存在的，但幅度较小（JSD 约 0.02–0.03）；不能声称 Text 和 Visual 使用完全不同的专家配置。
- Private → Collaborative 的差异在 Text 上最明显（0.0711，Movies 的 E1 路由拉高了上限），Visual 上较小（0.0292）。
- ele-fashion 的 JSD 普遍偏小，路由上下文区分度最弱。

#### 4.2.4 节点级路由多样性

| 指标 | 六类上下文平均 |
|---|---:|
| 出现的 Top-2 组合数（最大 12） | 11.47–11.93 |
| 最常见 Top-2 组合的节点占比 | 0.262–0.312 |
| 节点间 Routing Weight 标准差（每专家平均） | 0.247–0.267 |

这说明路由不是“整个上下文只有一个全局专家选择”：几乎全部 12 种有序 Top-2 组合都在节点层面出现，且最常见组合也只覆盖约 26%–31% 的节点。**节点级路由异质性是真实存在的。**

#### 4.2.5 Collaborative Cross Context 的一个限制

原计划比较“所有 Physical 有效节点”和“Cross Context 非零节点”。实际检查发现本模型的 `cross_context` 几乎是稠密的：

| Dataset | Collaborative 上下文的 Cross Nonzero Ratio |
|---|---:|
| Grocery | 1.0000 |
| Toys | 0.9995 |
| Reddit-S | 1.0000 |
| ele-fashion | 1.0000 |
| Movies | 1.0000 |

因此 `cross_nonzero` 子集与 Physical Support 基本相同，二者路由分布的 JSD 为 0。该比较没有区分力，不能用来声称“只有真正有跨模态信息的节点才改变路由”。Private vs Collaborative 的差异仍然有效，因为它来自 Router 输入中是否包含跨模态上下文，而不是来自节点子集筛选。

### 4.3 Figure 3(c)：路由分配对 Validation 预测具有可测影响，但分层明显

#### 4.3.1 主分组：逐数据集结果

表中为三个训练种子的数据集级均值（每个种子先对 10 次 Shuffle 求均值；标准差为三个种子均值的总体标准差）。

| Dataset | Intervention | Δ Val CE | Δ Val Acc | Prediction Disagreement |
|---|---|---:|---:|---:|
| Grocery | Private Physical | -0.0000 ± 0.0000 | -0.0002 ± 0.0003 | 0.0016 |
| Grocery | Private Semantic | +0.0108 ± 0.0029 | +0.0051 ± 0.0007 | 0.0300 |
| Grocery | Collaborative Physical | +0.0206 ± 0.0028 | +0.0100 ± 0.0012 | 0.0497 |
| Grocery | All Contexts | **+0.0349 ± 0.0061** | **+0.0147 ± 0.0009** | **0.0627** |
| Toys | Private Physical | +0.0001 ± 0.0001 | +0.0000 ± 0.0002 | 0.0012 |
| Toys | Private Semantic | +0.0057 ± 0.0006 | +0.0037 ± 0.0009 | 0.0204 |
| Toys | Collaborative Physical | +0.0162 ± 0.0039 | +0.0072 ± 0.0013 | 0.0401 |
| Toys | All Contexts | **+0.0230 ± 0.0053** | **+0.0104 ± 0.0028** | **0.0480** |
| Reddit-S | Private Physical | -0.0000 ± 0.0000 | -0.0002 ± 0.0001 | 0.0004 |
| Reddit-S | Private Semantic | +0.0010 ± 0.0006 | +0.0011 ± 0.0007 | 0.0059 |
| Reddit-S | Collaborative Physical | +0.0051 ± 0.0008 | +0.0030 ± 0.0012 | 0.0084 |
| Reddit-S | All Contexts | **+0.0068 ± 0.0013** | **+0.0035 ± 0.0011** | **0.0113** |
| ele-fashion | Private Physical | +0.0005 ± 0.0001 | +0.0002 ± 0.0001 | 0.0025 |
| ele-fashion | Private Semantic | +0.0094 ± 0.0018 | +0.0028 ± 0.0008 | 0.0176 |
| ele-fashion | Collaborative Physical | +0.0204 ± 0.0019 | +0.0080 ± 0.0015 | 0.0295 |
| ele-fashion | All Contexts | **+0.0316 ± 0.0053** | **+0.0106 ± 0.0016** | **0.0373** |
| Movies | Private Physical | -0.0002 ± 0.0003 | +0.0002 ± 0.0007 | 0.0083 |
| Movies | Private Semantic | +0.0038 ± 0.0033 | +0.0010 ± 0.0007 | 0.0262 |
| Movies | Collaborative Physical | +0.0466 ± 0.0065 | +0.0192 ± 0.0009 | 0.1488 |
| Movies | All Contexts | **+0.0482 ± 0.0093** | **+0.0190 ± 0.0008** | **0.1542** |

#### 4.3.2 五数据集 pooled 结果

| Intervention | Mean Δ Val CE | Mean Δ Val Acc | Mean Disagreement |
|---|---:|---:|---:|
| Private Physical | +0.0001 | +0.0000 | 0.0028 |
| Private Semantic | +0.0061 | +0.0028 | 0.0200 |
| Collaborative Physical | +0.0218 | +0.0095 | 0.0553 |
| All Contexts | **+0.0289** | **+0.0116** | **0.0627** |

#### 4.3.3 单上下文补充结果

六个上下文分别置乱，pooled 五数据集均值如下（完整逐数据集结果见 `router_shuffle_dataset_summary.csv` 和 `figure3c_single_contexts.pdf`）：

| Single context | Mean Δ Val CE | Mean Δ Val Acc | Mean Disagreement |
|---|---:|---:|---:|
| Private Text–Physical | +0.0000 | +0.0000 | 0.0011 |
| Private Text–Semantic | +0.0025 | +0.0016 | 0.0112 |
| Private Visual–Physical | +0.0000 | -0.0000 | 0.0023 |
| Private Visual–Semantic | +0.0038 | +0.0018 | 0.0149 |
| Collaborative Text–Physical | +0.0061 | +0.0038 | 0.0281 |
| Collaborative Visual–Physical | **+0.0142** | **+0.0067** | **0.0410** |

- 单独置乱 **Private Text/Visual Physical** 仍然几乎无影响。
- 单独置乱 **Collaborative Visual–Physical** 是六个单上下文中影响最大的（pooled ΔCE +0.0142）。
- Movies 的单上下文结果最突出：Collaborative Visual–Physical ΔCE `+0.0391`、ΔAcc `+0.0181`、disagreement `0.1246`；Collaborative Text–Physical ΔCE `+0.0064`、disagreement `0.0636`。Movies 的协同敏感性主要来自 Visual 协同上下文。

#### 4.3.4 正确性检查

| 检查 | 结果 |
|---|---|
| 不加 Shuffle 的基线 | 正常输出，重算 Validation 与 Checkpoint 保存值一致 |
| Identity Permutation | 15/15 Checkpoint 的 prediction mismatch = 0；最大 embedding 差异 ≤ `2.2 × 10⁻⁶` |
| Shuffle 是否真的交换了 Logits | 每次试验记录 `n_changed_total`；Private/All 分组几乎全部有效节点都发生交换 |
| 固定 Shuffle Seed 是否可复现 | 是；每个 trial 的 `shuffle_seed` 已写入 CSV |
| 指标是否有限 | 1,500/1,500 次试验均为有限值 |
| 模型权重 / 图是否被永久修改 | 15/15 Checkpoint 的参数校验和、`edge_index` 校验和分析前后一致 |
| 是否调用 Test | 否；所有指标只使用 `data.val_idx` |

#### 4.3.5 判断

- **支持**：冻结模型中已学习的 Router 分配对 Validation 预测有可测影响；破坏节点—专家匹配会增加 CE、降低 Accuracy，并导致部分节点预测改变。
- **分层结果**：`All Contexts > Collaborative Physical > Private Semantic > Private Physical` 的顺序在五个数据集上一致。
- **不支持**：Private Physical 单独 Shuffle 几乎不改变预测（pooled ΔCE ≈ +0.0001，ΔAcc ≈ 0.0000，disagreement 0.28%）。因此不能笼统声称“所有 Router 调用都同等重要”。
- **数据集差异**：Reddit-S 最不敏感（All Contexts ΔCE +0.0068），Movies 最敏感（+0.0482）；Grocery（+0.0349）、ele-fashion（+0.0316）、Toys（+0.0230）居中。
- **单上下文定位**：Collaborative Visual–Physical 是最敏感的单上下文；Private Text/Visual Physical 单独几乎没有影响。
- **不能推出的结论**：Frozen Shuffle 只能说明“已训练模型对路由分配有预测敏感性”，不能证明路由机制带来 Test 性能提升，也不能代替训练消融或正式 Test 对照。

---

## 5. 哪些论文主张得到支持，哪些证据不足

### 得到支持

1. **四个共享专家具有不同的有效滤波配置。** Physical 4×4 与 Semantic 4×2 系数矩阵的符号/幅度模式不同，且在实际 Expert Output 上平均余弦相似度显著低于 1。
2. **路由是 context-dependent 的。** 关系（Physical vs Semantic）与 Private vs Collaborative 的路由分布差异，大于模态（Text vs Visual）差异。
3. **路由具有明显的节点级异质性。** 几乎全部 12 种 Top-2 组合都会出现，最常见组合只占约 26%–31%。
4. **冻结 Router 分配对 Validation 预测有作用。** Shuffle 后在 Collaborative Physical 和 All Contexts 上稳定增加 CE、降低 Acc；五数据集方向一致；单上下文分析定位到 Collaborative Visual–Physical 最敏感。

### 证据不足 / 反例

1. **“训练塑造了专家滤波系数的多样性”证据不足。** 训练引起的单专家系数 L2 变化约 0.05，而初始化时专家两两距离已经是 0.860（Physical）/1.150（Semantic）；训练后几乎不变（0.853 / 1.152）。滤波系数的多样性主要来自固定初始化，训练是 refinement。该结论不直接适用于 Expert Output 层面，因为输出还经过训练后的 bottleneck MLP。
2. **“Text 与 Visual 使用不同路由”证据较弱。** 模态 JSD 约 0.02–0.03，小于关系/协同差异；只能说存在小幅差异，不能说完全分化。
3. **“Private Physical 路由重要”不成立。** 单独 Shuffle 的影响接近零；单上下文 Private Text/Visual Physical 也接近零。
4. **“Cross Context 非零节点才改变路由”不可检验。** 本模型的 cross context 几乎稠密（ratio ≈ 1.0），cross-only 与 all-physical 子集相同。
5. **不能声称因果性能提升。** 本分析没有重训练、没有 Test、没有替代 Router 的训练对照；Shuffle 是敏感性诊断，不是因果消融。
6. **Movies 的协同路由有数据集特异性。** Movies Collaborative Text–Physical 由 E1 主导，而其他数据集多为 E2；不能把 Grocery/Toys/Reddit-S 的 E2 模板直接外推到所有数据集。

---

## 6. 限制与注意事项

- **Movies 使用新增的 validation-only Checkpoint**：历史 adopted Movies 运行没有保存权重；本轮为分析新增训练了 3 个 Full MRAGE Checkpoint（adopted `default` 配置，seeds 42–44），只跑到 `validation_complete`，`test_evaluation_count=0`。它们不是历史 adopted 运行的权重，不能与历史 Movies Test 结果混用。
- **仅 Validation**：没有调用正式 Final Test；所有结论不能直接外推到 Test 泛化。
- **主图 (a)(b) 使用 Grocery seeds 42–44 作为布局原型**；五个数据集的完整结果在 CSV、`figure3b_by_dataset.pdf` 和单上下文补充图中。主图 (c) 包含五个数据集。
- **Cross Context 稠密**：Collaborative 的 `cross_nonzero` 子集与 Physical Support 基本相同，相关 JSD 为 0；这不是实现错误，而是该模型的实际行为。
- **CUDA 非确定性**：Identity Permutation 的最大 embedding 差异 ≤ `2.2 × 10⁻⁶`，但预测完全不变；Shuffle 的微小 ΔCE 可能包含这一级别的数值噪声。所有指标在 10 次重复上汇总，未做显著性检验（按要求）。
- **资源竞争**：分析在 `cuda:0` 上与正在运行的 Sports LP 训练共享 GPU；未 OOM，但运行时间不是独立基准。
- **专家输出相似度采样**：每个上下文最多 4096 个有效节点；实际 `n_sampled` 记录在 `data/expert_diversity.csv`。路由统计使用全部有效节点，不采样。
- **未分析组件消融 / Single Filter / Unshared Experts**：本轮只使用正式 Full MRAGE Checkpoint。

---

## 7. 复现与文件

### 7.1 Movies validation-only Checkpoint 训练

```bash
conda run -n yhf_env env PYTHONPATH=. python scripts/run_paper_benchmark.py \
  --dataset Movies --task nc --models mrage --seeds 42 43 44 --split-seed 42 \
  --device cuda:0 \
  --experiment-config experiments/nc5_test_exploration_20261031/adopted/Movies.yaml \
  --override semantic_graph.cache_dir=/hdd1/DataInHere/YHF/mrage/artifacts/semantic_graphs \
  --override semantic_graph.mode=existing \
  --output-root /hdd1/DataInHere/YHF/mrage/outputs/mrage_figure3_movies_full \
  --stage validation
```

### 7.2 完整批量分析

```bash
conda run -n yhf_env env PYTHONPATH=. python scripts/analyze_mrage_experts.py \
  --all --seeds 42 43 44 --device cuda:0 \
  --output-dir experiments/mrage_figure3_mechanism/data --max-nodes 4096

conda run -n yhf_env env PYTHONPATH=. python scripts/analyze_mrage_router.py \
  --all --seeds 42 43 44 --device cuda:0 \
  --output-dir experiments/mrage_figure3_mechanism/data

conda run -n yhf_env env PYTHONPATH=. python scripts/analyze_mrage_router_shuffle.py \
  --all --seeds 42 43 44 --device cuda:0 \
  --output-dir experiments/mrage_figure3_mechanism/data --repeats 10 \
  --include-single-contexts

conda run -n yhf_env env PYTHONPATH=. python scripts/plot_mrage_figure3.py
conda run -n yhf_env env PYTHONPATH=. python scripts/build_mrage_figure3_manifest.py
```

### 7.3 单 Checkpoint 调试命令

```bash
conda run -n yhf_env env PYTHONPATH=. python scripts/analyze_mrage_experts.py \
  --dataset Grocery --seeds 42 --device cuda:0 \
  --output-dir experiments/mrage_figure3_mechanism/data/debug --max-nodes 1024

conda run -n yhf_env env PYTHONPATH=. python scripts/analyze_mrage_router.py \
  --dataset Grocery --seeds 42 --device cuda:0 \
  --output-dir experiments/mrage_figure3_mechanism/data/debug

conda run -n yhf_env env PYTHONPATH=. python scripts/analyze_mrage_router_shuffle.py \
  --dataset Grocery --seeds 42 --device cuda:0 \
  --output-dir experiments/mrage_figure3_mechanism/data/debug --repeats 2 \
  --include-single-contexts
```

### 7.4 测试

```bash
conda run -n yhf_env env PYTHONPATH=. python -m pytest -q tests/test_mrage_figure3_analysis.py
```

### 7.5 结果文件

| 文件 | 内容 |
|---|---|
| `data/expert_coefficients.csv` | 15 Checkpoint × 2 关系 × 4 专家 × 阶数的初始化/训练后有效系数 |
| `data/expert_diversity.csv` | 系数 L2 变化、初始化/训练后专家两两距离/余弦、真实输出余弦与平均范数 |
| `data/routing_patterns.csv` | 六类上下文 × 四专家的 Top-2 权重、选择频率、有效节点数 |
| `data/routing_diagnostics.csv` | Top-2 组合多样性、最常见组合、节点路由标准差、上下文 JSD |
| `data/routing_pairwise_jsd.csv` | 全部上下文两两 JSD，以及 Collaborative 的 support-vs-cross 比较 |
| `data/router_shuffle_trials.csv` | 1,500 次 Shuffle 的逐 trial 指标、seed、n_active、n_changed、正确性标记 |
| `data/router_shuffle_summary.csv` | 每个 Checkpoint × 分组的 10 次 Shuffle 汇总 |
| `data/router_shuffle_dataset_summary.csv` | 每个数据集 × 分组（4 主分组 + 6 单上下文）的三种子汇总 |
| `manifest.json` | 基线 Commit、分析配置、Checkpoint 路径/SHA256、代码哈希、Identity 检查 |
| `figures/figure3.png/.pdf` | 三子图组合 |
| `figures/figure3a.png/.pdf` | (a) 两个系数热图 |
| `figures/figure3b.png/.pdf` | (b) 6×4 路由热图 |
| `figures/figure3c.png/.pdf` | (c) ΔCE / ΔAcc 分组柱状图 |
| `figures/figure3b_by_dataset.png/.pdf` | 五数据集路由热图补充 |
| `figures/figure3c_single_contexts.png/.pdf` | 六类单上下文 × 五数据集的 ΔCE / ΔAcc 热图 |

---

## 8. Figure 3 图注（建议）

> **Figure 3. Mechanistic analysis of the shared adaptive graph experts.**
> (a) Effective polynomial filter coefficients of the four shared experts for the physical (orders 0–3) and semantic (orders 0–1) relations; values are the mean of Grocery training seeds 42–44. The fixed initialization already gives distinct coefficient patterns; training changes them only modestly (mean L2 change from initialization: physical 0.050, semantic 0.049).
> (b) Mean Top-2 routing weight across the six router contexts (Grocery, seeds 42–44). Rows sum to one. Private physical contexts are dominated by E4, whereas collaborative physical contexts shift toward E2.
> (c) Frozen Validation router-shuffle sensitivity on five datasets (Grocery, Toys, Reddit-S, ele-fashion, Movies; seeds 42–44). Bars show the mean change in Validation cross-entropy and accuracy after permuting complete four-dimensional router-logit rows within the active nodes; error bars show the population std across training seeds. Private physical routing alone has near-zero effect, while collaborative and all-context interventions consistently increase CE and reduce accuracy. Movies uses newly trained validation-only Full MRAGE checkpoints; no Test was run. **Frozen Validation Diagnostic — No Retraining.**

Supplementary: single-context shuffle results for the six contexts are shown in `figures/figure3c_single_contexts.pdf`.

---

## 9. 最终判断

Figure 3 的三段式证据链在本轮数据上得到**部分支持**：

- **Expert Diversity**：成立，但滤波系数的多样性主要来自初始化；训练只做 modest refinement。
- **Context-dependent Routing**：成立；关系与 Private/Collaborative 差异强于模态差异；节点级路由高度异质。Movies 的协同路由具有数据集特异性（Text 协同由 E1 主导）。
- **Predictive Sensitivity**：成立，但分层明显；Collaborative Physical 与 All Contexts 有稳定影响，单上下文分析定位到 Collaborative Visual–Physical 最敏感；Private Physical 单独几乎没有影响。

因此，论文中应避免“所有路由都重要”或“训练学习出了专家滤波多样性”这类过强表述。更稳妥的表述是：

> The fixed expert initialization provides distinct filter bases; training refines them locally. The learned router is strongly context- and node-dependent, and freezing/shuffling its node–expert assignment reliably degrades Validation predictions in collaborative and combined contexts, while private-physical routing alone is largely redundant. The strongest single-context sensitivity is the collaborative visual–physical router.

这些结论仅限冻结模型的 Validation 诊断；不替代训练消融或正式 Test 比较。
