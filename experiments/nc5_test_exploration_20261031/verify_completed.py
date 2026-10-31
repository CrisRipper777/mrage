"""Check persisted study artifacts after completion; no training or metric selection."""
import datetime as dt,hashlib,json,math
from pathlib import Path
from omegaconf import OmegaConf
S=Path(__file__).resolve().parent;ROOT=S.parents[1];O=ROOT/'outputs/nc5_test_exploration_20261031'
plan=json.loads((S/'plan.json').read_text());rows=json.loads((S/'trials.json').read_text())['runs'];decisions=json.loads((S/'focused_decisions.json').read_text());selection=json.loads((S/'selection.json').read_text())
expected={(c,d,k) for c in plan['global_candidates'] for d in plan['datasets'] for k in [42,43,44]}
expected|={(c,d,42) for d,cs in plan['focused_candidates'].items() for c in cs}
expected|={(d['candidate'],d['dataset'],k) for d in decisions for k in [43,44]}
assert len(rows)==len(expected)==65
assert {(r['candidate'],r['dataset'],r['seed']) for r in rows}==expected
assert selection['test_used_for_selection'] is True
hashes={};splits={}
for r in rows:
 p=Path(r['run_dir']);assert p.is_relative_to(O)
 status=json.loads((p/'status.json').read_text());md=json.loads((p/'run_metadata.json').read_text());result=json.loads((p/'final_test_metrics.json').read_text());cfg=OmegaConf.load(p/'config_resolved.yaml');candidate=OmegaConf.load(S/f"candidates/{r['candidate']}.yaml")
 assert status['status']==r['status']=='final_test_complete'
 assert status['test_evaluation_count']==md['test_evaluation_count']==result['test_evaluation_count']==1
 assert md['model']=='mrage' and md['task']=='nc' and md['device']=='cuda:1'
 assert md['model_config']==OmegaConf.to_container(candidate.model,resolve=True)
 assert md['effective_learning_rate']==float(candidate.task.lr) and md['effective_weight_decay']==float(candidate.task.weight_decay)
 assert cfg.task.loss.class_balanced_main_ce==candidate.task.loss.class_balanced_main_ce
 assert cfg.task.loss.unimodal_weight==candidate.task.loss.unimodal_weight
 assert cfg.task.macro_f1_labels=='all_classes' and cfg.dataset_split_seed==42
 assert all(math.isfinite(x) for x in result['metrics'].values())
 assert all(result['metrics'][key]==r[key] for key in ['test_acc','test_macro_f1','val_acc','val_macro_f1'])
 hashes.setdefault((r['candidate'],r['dataset']),set()).add(md['config_hash']);splits.setdefault(r['dataset'],set()).add(md['split_id'])
assert all(len(x)==1 for x in hashes.values()) and all(len(x)==1 for x in splits.values())
assert hashlib.sha256((ROOT/'configs/model/mrage.yaml').read_bytes()).hexdigest()==plan['original_mrage_sha256']
audit={'checked_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'completed_training_runs':65,'all_runs_on_gpu1':True,'all_test_counts_one':True,'all_selected_parameters_match':True,'fixed_official_split_ids':{d:next(iter(v)) for d,v in splits.items()},'same_config_across_seeds':True,'original_mrage_yaml_unchanged':True,'test_guided_exploratory_selection':True}
(S/'completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
