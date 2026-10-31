#!/usr/bin/env python3
"""Refresh descriptive summaries from recorded NC5 trials; never select using Test."""
import csv
import datetime as dt
import hashlib
import json
import re
import statistics as stats
from pathlib import Path

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
DATASETS = ['Movies', 'Toys', 'Grocery', 'ele-fashion', 'Reddit-S']
SEEDS = [42, 43, 44]
CANDIDATES = ['baseline', 'a13_expert_eta_0p2', 'a14_hidden_dim_128']


def save_csv(name, rows):
    if not rows:
        return
    with (STUDY / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    runs = json.loads((STUDY / 'trials.json').read_text())['runs']
    results = {}
    for phase, prefix, status in [('validation', 'val', 'validation_complete'),
                                   ('final-test', 'test', 'final_test_complete')]:
        rows = []
        for candidate in dict.fromkeys(CANDIDATES + [r['candidate'] for r in runs]):
            for dataset in DATASETS:
                group = sorted([r for r in runs if r['phase'] == phase and
                                r['status'] == status and r['candidate'] == candidate and
                                r['dataset'] == dataset], key=lambda r: r['seed'])
                if not group:
                    continue
                assert len({r['seed'] for r in group}) == len(group)
                assert len({r['split_id'] for r in group}) == 1
                row = {'candidate': candidate, 'dataset': dataset, 'n': len(group),
                       'seeds': ','.join(str(r['seed']) for r in group)}
                for metric in [f'{prefix}_acc', f'{prefix}_macro_f1']:
                    values = [r[metric] * 100 for r in group]
                    row[metric + '_mean_pct'] = stats.fmean(values)
                    row[metric + '_std_pct_ddof0'] = stats.pstdev(values)
                    for seed in SEEDS:
                        row[f'{metric}_seed{seed}_pct'] = next(
                            (r[metric] * 100 for r in group if r['seed'] == seed), '')
                row['best_epoch_mean'] = stats.fmean(r['best_epoch'] for r in group)
                row['wall_seconds_mean'] = stats.fmean(r['wall_seconds'] for r in group)
                row['model_parameters'] = group[0]['model_parameters']
                row['split_id'] = group[0]['split_id']
                rows.append(row)
        save_csv(f'{phase.replace("-", "_")}_summary.csv', rows)
        results[phase] = rows

    convergence = []
    for r in runs:
        if r['phase'] != 'validation' or r['status'] != 'validation_complete':
            continue
        lines = (Path(r['run_dir']) / 'train.log').read_text().splitlines()
        epochs = []
        times = []
        for line in lines:
            match = re.search(r'^\[([^]]+)\].*Epoch (\d+) \| Train Loss', line)
            if match:
                epochs.append(int(match[2]))
                times.append(dt.datetime.fromisoformat(match[1]))
        count = max(epochs) if epochs else 0
        convergence.append({'candidate': r['candidate'], 'dataset': r['dataset'],
                            'seed': r['seed'], 'epochs_completed': count,
                            'best_epoch': r['best_epoch'], 'budget_reached': count >= 300,
                            'train_loop_seconds': (times[-1]-times[0]).total_seconds() if len(times)>1 else '',
                            'mean_epoch_seconds': (times[-1]-times[0]).total_seconds()/(len(times)-1) if len(times)>1 else '',
                            'wall_seconds': r['wall_seconds']})
    save_csv('convergence.csv', convergence)
    metadata = json.loads((STUDY / 'study_metadata.json').read_text())
    statuses = [json.loads(p.read_text()) for p in (ROOT / 'outputs/nc5_tuning_20261030').rglob('status.json')]
    audit = {'at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
             'completed_validation_trials': len(convergence),
             'final_test_trials': len([r for r in runs if r['phase'] == 'final-test' and r['status'] == 'final_test_complete']),
             'original_mrage_unchanged': hashlib.sha256((ROOT/'configs/model/mrage.yaml').read_bytes()).hexdigest() == metadata['original_mrage_sha256'],
             'test_evaluation_counts': sorted({s.get('test_evaluation_count', 0) for s in statuses}),
             'budget_reached': sum(r['budget_reached'] for r in convergence),
             'failed_ledger_rows': sum(r['status'] == 'failed' for r in runs),
             'three_seed_complete_cells': sum(r['n'] == 3 and r['candidate'] in CANDIDATES for r in results['validation'])}
    (STUDY / 'latest_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))
    for candidate in CANDIDATES:
        cells = [r for r in results['validation'] if r['candidate']==candidate]
        if len(cells)==5:
            print(candidate, 'n=', [r['n'] for r in cells], 'val mean=', round(stats.fmean(r['val_acc_mean_pct'] for r in cells),4))
    return results


if __name__ == '__main__':
    summarize()
