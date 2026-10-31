#!/usr/bin/env python3
"""Bounded, explicitly Test-guided MRAGE NC5 exploration on physical GPU1."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'experiments/nc5_test_exploration_20261031'
OUTPUT=ROOT/'outputs/nc5_test_exploration_20261031'


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)


def read_rows():
    p=STUDY/'trials.json'
    return json.loads(p.read_text())['runs'] if p.exists() else []


def save_record(row):
    rows=read_rows();key=(row['candidate'],row['dataset'],row['seed'])
    rows=[r for r in rows if (r['candidate'],r['dataset'],r['seed'])!=key]+[row]
    write_json(STUDY/'trials.json',{'runs':rows})
    fields=['candidate','dataset','seed','phase','status','test_acc','test_macro_f1','val_acc','val_macro_f1','best_epoch','wall_seconds','config_hash','split_id','device','run_dir','error']
    with (STUDY/'trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    summarize(rows)


def summarize(rows):
    groups={}
    for r in rows:
        if r['status']=='final_test_complete':groups.setdefault((r['candidate'],r['dataset']),[]).append(r)
    summary=[]
    for (candidate,dataset),items in groups.items():
        row={'candidate':candidate,'dataset':dataset,'n':len(items),'seeds':','.join(str(r['seed']) for r in sorted(items,key=lambda r:r['seed']))}
        for metric in ['test_acc','test_macro_f1','val_acc','val_macro_f1']:
            row[metric+'_mean']=statistics.fmean(r[metric] for r in items)
            row[metric+'_std_ddof0']=statistics.pstdev(r[metric] for r in items)
        summary.append(row)
    if summary:
        with (STUDY/'summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
    write_json(STUDY/'summary.json',{'test_guided_selection':True,'groups':summary})
    return summary


def collect(candidate,dataset,seed):
    paths=list((OUTPUT/candidate/dataset/'nc/mrage').glob(f'*/seed-{seed}/status.json'))
    if not paths:return None
    if len(paths)!=1:raise RuntimeError(f'ambiguous config: {candidate}/{dataset}/{seed}')
    directory=paths[0].parent;status=json.loads(paths[0].read_text());metadata=json.loads((directory/'run_metadata.json').read_text())
    if status['status']!='final_test_complete':return None
    result=json.loads((directory/'final_test_metrics.json').read_text())
    if status.get('test_evaluation_count')!=1 or result.get('test_evaluation_count')!=1:raise RuntimeError('Test count is not one')
    return {'candidate':candidate,'dataset':dataset,'seed':seed,'status':'final_test_complete',
            **result['metrics'],'best_epoch':metadata['best_epoch'],'config_hash':metadata['config_hash'],
            'split_id':metadata['split_id'],'device':metadata['device'],'run_dir':str(directory),
            'wall_seconds':(directory/'final_test_metrics.json').stat().st_mtime-metadata['created_at_unix'],'error':''}


def run(candidate,dataset,seed,phase):
    if (STUDY/'STOP_AFTER_CURRENT').exists():raise InterruptedError('STOP_AFTER_CURRENT requested')
    old=collect(candidate,dataset,seed)
    if old:
        old['phase']=phase;save_record(old);print('KEEP',candidate,dataset,seed,flush=True);return
    while True:
        query=subprocess.run(['nvidia-smi','-i','1','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
        if not query.stdout.strip():break
        write_json(STUDY/'progress.json',{'status':'waiting_for_gpu1','candidate':candidate,'dataset':dataset,'seed':seed,'phase':phase,'completed_runs':len(read_rows()),'at_unix':time.time()})
        if (STUDY/'STOP_AFTER_CURRENT').exists():raise InterruptedError('STOP_AFTER_CURRENT requested')
        time.sleep(15)
    cfg=STUDY/f'candidates/{candidate}.yaml'
    command=[sys.executable,'scripts/run_paper_benchmark.py','--dataset',dataset,'--task','nc','--models','mrage','--seeds',str(seed),'--split-seed','42','--device','cuda:1','--stage','all','--resume','--experiment-config',str(cfg),'--output-root',str(OUTPUT/candidate)]
    log=OUTPUT/'launchers'/candidate/dataset/f'seed-{seed}.log';log.parent.mkdir(parents=True,exist_ok=True)
    write_json(STUDY/'progress.json',{'status':'running','candidate':candidate,'dataset':dataset,'seed':seed,'phase':phase,'completed_runs':len(read_rows()),'at_unix':time.time()})
    start=time.monotonic();print('START',phase,candidate,dataset,seed,flush=True)
    with log.open('a') as f:
        f.write(json.dumps({'command':command,'at_unix':time.time(),'test_used_for_selection':True})+'\n');f.flush()
        result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=False,env={**os.environ,'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8'})
    row=collect(candidate,dataset,seed)
    if result.returncode or not row:raise RuntimeError(f'run failed {candidate}/{dataset}/{seed}; inspect {log}')
    row.update(phase=phase,wall_seconds=time.monotonic()-start);save_record(row)
    print('DONE',candidate,dataset,seed,'TestAcc',row['test_acc'],'TestF1',row['test_macro_f1'],flush=True)


def finish(plan):
    rows=read_rows();summary=summarize(rows);selected={};comparisons=[]
    for dataset in plan['datasets']:
        available=[g for g in summary if g['dataset']==dataset and g['n']==3 and g['seeds']=='42,43,44']
        winner=max(available,key=lambda g:(g['test_acc_mean'],g['test_macro_f1_mean']))
        selected[dataset]=winner['candidate'];base=next(g for g in available if g['candidate']=='default')
        comparisons.append({'dataset':dataset,'selected_candidate':winner['candidate'],'test_acc_gain_pp':(winner['test_acc_mean']-base['test_acc_mean'])*100,'test_macro_f1_gain_pp':(winner['test_macro_f1_mean']-base['test_macro_f1_mean'])*100,'selected':winner,'default':base})
    common=[]
    for candidate in plan['global_candidates']:
        groups=[g for g in summary if g['candidate']==candidate and g['n']==3]
        if len(groups)==5:common.append({'candidate':candidate,'mean_test_acc':statistics.fmean(g['test_acc_mean'] for g in groups),'mean_test_macro_f1':statistics.fmean(g['test_macro_f1_mean'] for g in groups)})
    best_common=max(common,key=lambda g:(g['mean_test_acc'],g['mean_test_macro_f1']))
    write_json(STUDY/'selection.json',{'test_used_for_selection':True,'selection_metric':plan['selection'],'training_seeds':[42,43,44],'by_dataset':selected,'best_common':best_common,'common_comparison':common,'dataset_comparison':comparisons,'at_unix':time.time()})
    exported=STUDY/'selected';exported.mkdir(exist_ok=True)
    for dataset,candidate in selected.items():(exported/f'{dataset}.yaml').write_bytes((STUDY/f'candidates/{candidate}.yaml').read_bytes())
    (exported/'common.yaml').write_bytes((STUDY/f"candidates/{best_common['candidate']}.yaml").read_bytes())
    write_json(STUDY/'progress.json',{'status':'complete','completed_runs':len(rows),'test_used_for_selection':True,'at_unix':time.time()})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    plan=json.loads((STUDY/'plan.json').read_text())
    if args.dry_run:
        for dataset in plan['datasets']:
            print(dataset,'global:',plan['global_candidates'],'seeds42/43/44; focused:',plan['focused_candidates'][dataset],'seed42; focused winner seeds43/44')
        return 0
    for dataset in plan['datasets']:
        for candidate in plan['global_candidates']:
            for seed in [42,43,44]:run(candidate,dataset,seed,'global_confirmation')
    for dataset in plan['datasets']:
        for candidate in plan['focused_candidates'][dataset]:run(candidate,dataset,42,'focused_screen')
    decisions=[]
    for dataset in plan['datasets']:
        candidates=[r for r in read_rows() if r['dataset']==dataset and r['seed']==42 and r['candidate'] in plan['focused_candidates'][dataset]]
        winner=max(candidates,key=lambda r:(r['test_acc'],r['test_macro_f1']))
        decisions.append({'dataset':dataset,'candidate':winner['candidate'],'screen_test_acc':winner['test_acc'],'screen_test_macro_f1':winner['test_macro_f1'],'candidates':[{k:r[k] for k in ['candidate','test_acc','test_macro_f1']} for r in candidates]})
        write_json(STUDY/'focused_decisions.json',decisions)
        for seed in [43,44]:run(winner['candidate'],dataset,seed,'focused_confirmation')
    assert len(read_rows())<=plan['maximum_training_runs'];finish(plan)
    print('COMPLETE',len(read_rows()),'training runs; exploratory Test-guided selection saved.',flush=True)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        write_json(STUDY/'progress.json',{'status':'stopped' if isinstance(exc,InterruptedError) else 'failed','error':str(exc),'completed_runs':len(read_rows()),'at_unix':time.time()})
        raise
