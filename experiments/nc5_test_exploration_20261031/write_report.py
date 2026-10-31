"""Render the completed Test-guided study without running training or inference."""
import csv,hashlib,json,re,statistics as st
from pathlib import Path
from omegaconf import OmegaConf
S=Path(__file__).resolve().parent;ROOT=S.parents[1]
plan=json.loads((S/'plan.json').read_text());selection=json.loads((S/'selection.json').read_text());rows=json.loads((S/'trials.json').read_text())['runs'];groups=json.loads((S/'summary.json').read_text())['groups']
def group(c,d):return next(g for g in groups if g['candidate']==c and g['dataset']==d)
def fmt(g,m):return f"{g[m+'_mean']*100:.2f} ± {g[m+'_std_ddof0']*100:.2f}"
lines=['# MRAGE NC5：Test 参与选参的有界探索','','本轮按用户明确要求，以三种子平均 Test Accuracy 为主，平均 Test Macro-F1 在 Accuracy 完全相同时破同分，标准差辅助描述稳定性。Test 已参与参数选择，因此本报告是探索性结果，不能视作独立留出集上的无偏泛化验证。与原 Validation 选参正式 Benchmark 分开保留。','',
'固定官方 NC 划分，训练种子42/43/44；单卡物理GPU1；每次训练仍按最佳Validation Accuracy保存Checkpoint。每次保存模型只正式评估一次Test。Metric仍采用all_classes，ele-fashion空类别ID5的口径没有单独修改。','',
'## 与本轮默认参数的三种子对照','','均值 ± 总体标准差（ddof=0），指标单位 %，差值单位百分点。下表最优仅指本轮已测、已补齐三种子的候选。','',
'| 数据集 | 选中候选 | 默认 Test Acc | 选中 Test Acc | Acc 差值 | 默认 Macro-F1 | 选中 Macro-F1 |','|---|---|---:|---:|---:|---:|---:|']
for d in plan['datasets']:
 c=selection['by_dataset'][d];b=group('default',d);g=group(c,d)
 assert b['n']==g['n']==3
 lines.append(f"| {d} | {c} | {fmt(b,'test_acc')} | {fmt(g,'test_acc')} | {(g['test_acc_mean']-b['test_acc_mean'])*100:+.3f} | {fmt(b,'test_macro_f1')} | {fmt(g,'test_macro_f1')} |")
base_acc=st.fmean(group('default',d)['test_acc_mean'] for d in plan['datasets']);selected_acc=st.fmean(group(selection['by_dataset'][d],d)['test_acc_mean'] for d in plan['datasets'])
lines+=['',f'五数据集等权平均 Test Accuracy：默认 {base_acc*100:.4f}%，逐数据集选中配置 {selected_acc*100:.4f}%，差值 {(selected_acc-base_acc)*100:+.4f}pp。该数不是合并所有节点后的Accuracy。','',
'## 三套基础配置的完整复跑','','| 数据集 | 默认 Acc / F1 | 原调优256 Acc / F1 | 原调优128 Acc / F1 |','|---|---:|---:|---:|']
for d in plan['datasets']:
 cells=[f"{fmt(group(c,d),'test_acc')} / {fmt(group(c,d),'test_macro_f1')}" for c in plan['global_candidates']]
 lines.append('| '+d+' | '+' | '.join(cells)+' |')
lines+=['','三套基础配置按每个数据集三种子完整复跑，共45次训练。统一配置按五数据集等权平均Test Accuracy选择，Macro-F1破完全同分。','',
'| 统一候选 | 五数据集平均 Test Acc (%) | 五数据集平均 Macro-F1 (%) |','|---|---:|---:|']
for g in selection['common_comparison']:lines.append(f"| {g['candidate']} | {g['mean_test_acc']*100:.4f} | {g['mean_test_macro_f1']*100:.4f} |")
lines+=['',f"本轮最佳统一候选为 `{selection['best_common']['candidate']}`，完整配置：selected/common.yaml。",'',
'## 各数据集选中参数相对默认值的变化','','所有未列参数维持默认。完整Model/Task配置保存在selected/<dataset>.yaml，原configs/model/mrage.yaml未改。']
def differences(base,new,prefix=''):
 out={}
 for k,v in new.items():
  key=f'{prefix}.{k}' if prefix else k
  if isinstance(v,dict):out.update(differences(base.get(k,{}),v,key))
  elif v!=base.get(k):out[key]=v
 return out
base=OmegaConf.to_container(OmegaConf.load(S/'candidates/default.yaml'),resolve=False)
changes={}
for d in plan['datasets']:
 c=selection['by_dataset'][d];cfg=OmegaConf.to_container(OmegaConf.load(S/f'candidates/{c}.yaml'),resolve=False);changes[d]=differences(base,cfg)
 lines+=['',f'### {d}：{c}','','```yaml',OmegaConf.to_yaml(OmegaConf.create(changes[d])).strip() if changes[d] else '{} # 保留默认参数','```']
(S/'selected/changes_from_default.yaml').write_text(OmegaConf.to_yaml(OmegaConf.create(changes)))
lines+=['','## 定向初筛和确认记录','','每个数据集只试两个预先记录理由的候选，seed42初筛，然后只给该数据集初筛Accuracy较优者补43/44。未进入确认的候选只有n=1，不能与n=3结果当作同等证据。','',
'| 数据集 | 定向候选 | n | Test Acc | Test Macro-F1 |','|---|---|---:|---:|---:|']
for d in plan['datasets']:
 for c in plan['focused_candidates'][d]:
  g=group(c,d);lines.append(f"| {d} | {c} | {g['n']} | {fmt(g,'test_acc')} | {fmt(g,'test_macro_f1')} |")
lines+=['','## Accuracy 接近时的实际取舍','','| 数据集 | Accuracy前两名 | 均值差 (pp) | 三次累计正确预测数差 | 前两名Macro-F1均值 (%) |','|---|---|---:|---:|---:|']
for d in plan['datasets']:
 ranked=sorted([g for g in groups if g['dataset']==d and g['n']==3],key=lambda g:(g['test_acc_mean'],g['test_macro_f1_mean']),reverse=True)
 a,b=ranked[:2]
 source=next(r for r in rows if r['dataset']==d)
 log=(Path(source['run_dir'])/'train.log').read_text()
 ntest=int(re.search(r'NC split: train=\d+ \| val=\d+ \| test=(\d+)',log).group(1))
 correct=lambda c:sum(round(r['test_acc']*ntest) for r in rows if r['candidate']==c and r['dataset']==d)
 lines.append(f"| {d} | {a['candidate']} / {b['candidate']} | {(a['test_acc_mean']-b['test_acc_mean'])*100:.4f} | {correct(a['candidate'])-correct(b['candidate'])} | {a['test_macro_f1_mean']*100:.2f} / {b['test_macro_f1_mean']*100:.2f} |")
lines+=['','正确预测数差按相同Test节点在三次运行中的累计次数计算。数值最优不等于统计显著更好；尤其当Acc只高一个累计正确预测、却损失较多Macro-F1时，必须保留这个取舍，不能笼统称两项都提升。']
lines+=['','具体候选继承和理由见plan.json。movies_gamma016和fashion_gate64为历史完整配置的复跑，不是相对本轮tuned256只改一个参数；不据此单独归因某个机制。其他定向候选按计划相对指定锚点改变一个参数。','',
'## 逐种子结果','','| 数据集 | 配置 | Seed | Test Acc (%) | Test Macro-F1 (%) | 最佳epoch |','|---|---|---:|---:|---:|---:|']
for d in plan['datasets']:
 for c in dict.fromkeys(['default',selection['by_dataset'][d]]):
  for r in sorted([r for r in rows if r['dataset']==d and r['candidate']==c],key=lambda r:r['seed']):
   lines.append(f"| {d} | {c} | {r['seed']} | {r['test_acc']*100:.4f} | {r['test_macro_f1']*100:.4f} | {r['best_epoch']} |")
lines+=['','## 稳定性、限制与保留材料','','当前运行仍使用项目原有CUDA执行方式，没有在本轮修改稀疏算子或启用新的确定性训练协议。此前已发现同seed重跑会有差异，因此小于波动幅度的均值增益不等同于稳定或统计显著提升。没有因为本轮结果不好而挑掉某个seed，也没有将历史较好的单次运行混入本轮平均。','',
'作为历史参照，用户的默认运行Movies（outputs/2026-10-31/13-05-49）为55.04±0.22% Test Acc；该数与本轮默认值分开记录，不参与拼接均值。ele-fashion对应用户默认目录为outputs/2026-10-31/13-11-08。','',
'Checkpoint仍按Validation Accuracy选择；选参使用了Test，所以本轮所选参数日后若写作独立泛化改进结论，需要另行设计独立验证。这里不把Test选参收益冒称为未见Test时的提升。','',
f'实际完成训练数：{len(rows)}，预算上限：{plan["maximum_training_runs"]}。每次保留config_resolved.yaml、best_validation.pt、train.log、validation_metrics.json、final_test_metrics.json、run_metadata.json、status.json。','',
'- plan.json：先验候选、搜索理由和上限；focused_decisions.json：定向确认的选择记录。','- trials.csv/json：逐次指标、seed、耗时、配置/划分哈希、运行目录；summary.csv/json：均值与标准差。','- selection.json：Test选参口径、统一配置与逐数据集选择。','- selected/*.yaml：可直接供正式Runner使用的完整覆盖配置。','- outputs/nc5_test_exploration_20261031/：本轮全部原始运行，独立于旧调参和正式Benchmark目录。','',
'复现（以Movies为例，使用新输出目录）：','','```bash','conda run --no-capture-output -n yhf_env python scripts/run_paper_benchmark.py \\','  --dataset Movies --task nc --models mrage --seeds 42 43 44 --split-seed 42 \\','  --device cuda:1 --stage all \\','  --experiment-config experiments/nc5_test_exploration_20261031/selected/Movies.yaml \\','  --output-root outputs/nc5_test_selected_reproduction','```']
if (S/'adopted_selection.json').exists():
 lines[2:2]=['> 当前采用方案已更新：Movies采用用户既有默认运行（2026-10-31/13-05-49）。当前采用清单见 [ADOPTED_RESULTS.md](ADOPTED_RESULTS.md)，配置位于 adopted/。本报告及 selected/ 保留原65次搜索阶段的数值排序与同批次对照，不代表Movies当前采用方案。','']
(S/'REPORT.md').write_text('\n'.join(lines)+'\n')
export=[]
for d in plan['datasets']:
 for c in dict.fromkeys(['default',selection['by_dataset'][d]]):
  g=group(c,d);export.append({'dataset':d,'candidate':c,'n':g['n'],**{k:g[k]*100 for k in g if k.endswith('_mean') or k.endswith('_std_ddof0')}})
with (S/'selected_vs_default_percent.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=export[0]);w.writeheader();w.writerows(export)
print('wrote',S/'REPORT.md','meanAcc',base_acc*100,selected_acc*100)
