#!/usr/bin/env python3
"""Validate actual paired episode records and compute one prespecified comparison.

This utility does not reconstruct episodes from aggregate manuscript counts and
cannot validate robot execution. It accepts the adjacent empty CSV template.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

REQUIRED = ('experiment_id', 'cohort_id', 'comparison_group', 'pair_id', 'episode_id',
            'condition', 'configuration_id', 'calibration_id', 'task_id',
            'initial_state_id', 'environment_seed', 'policy_seed', 'phase',
            'success', 'termination_reason', 'fault_profile_id', 'support_protocol',
            'evaluation_update_mode')
MATCHED = ('experiment_id', 'cohort_id', 'task_id', 'initial_state_id',
           'environment_seed', 'policy_seed', 'fault_profile_id')


def exact_mcnemar(fixed, broken):
    n = fixed + broken
    return 1.0 if not n else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(fixed, broken) + 1)) / 2**n)


def percentile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    i = int(x)
    return values[i] + (x-i) * (values[min(i+1, len(values)-1)]-values[i])


def load_records(path):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError('Missing CSV columns: ' + ', '.join(sorted(missing)))
        rows = list(reader)
    if not rows:
        raise ValueError('No episode records: the empty template is a format, not experimental evidence.')
    seen = set()
    seen_episodes = set()
    for line, row in enumerate(rows, start=2):
        if row['phase'] not in {'healthy_calibration', 'sensitivity_probe', 'fault_development', 'prior_identification', 'evaluation'}:
            raise ValueError(f'Row {line}: unrecognized phase {row["phase"]!r}.')
        if row['phase'] != 'evaluation':
            continue
        blank = [name for name in REQUIRED if not row[name].strip()]
        if blank:
            raise ValueError(f'Row {line}: blank required evaluation fields: {blank}.')
        if row['success'] not in {'0', '1'}:
            raise ValueError(f'Row {line}: success must be exactly 0 or 1.')
        key = (row['comparison_group'], row['condition'], row['pair_id'])
        if key in seen:
            raise ValueError(f'Row {line}: duplicate comparison/condition/pair {key}.')
        seen.add(key)
        episode_key = (row['comparison_group'], row['condition'], row['episode_id'])
        if episode_key in seen_episodes:
            raise ValueError(f'Row {line}: episode reused within the same comparison arm: {episode_key}.')
        seen_episodes.add(episode_key)
    return rows


def summarize(rows, group, baseline, treatment, bootstrap_samples, seed):
    arms = {baseline: {}, treatment: {}}
    for row in rows:
        if row['phase'] == 'evaluation' and row['comparison_group'] == group and row['condition'] in arms:
            arms[row['condition']][row['pair_id']] = row
    if not arms[baseline] or not arms[treatment]:
        raise ValueError('Both requested arms need evaluation records in the specified comparison group.')
    if set(arms[baseline]) != set(arms[treatment]):
        raise ValueError('Unmatched pair IDs. Supply complete paired arms; this utility does not discard missing pairs.')
    counts = {'both_success': 0, 'fixed': 0, 'broken': 0, 'both_failure': 0}
    tasks = defaultdict(list)
    configs = {baseline: set(), treatment: set()}
    for pair in sorted(arms[baseline]):
        left, right = arms[baseline][pair], arms[treatment][pair]
        mismatches = [name for name in MATCHED if left[name] != right[name]]
        if mismatches:
            raise ValueError(f'Pair {pair}: unmatched metadata {mismatches}. Use a distinct group for another comparison design.')
        b, t = int(left['success']), int(right['success'])
        counts[{(1,1): 'both_success', (0,1): 'fixed', (1,0): 'broken', (0,0): 'both_failure'}[(b,t)]] += 1
        tasks[left['task_id']].append(t-b)
        configs[baseline].add(left['configuration_id'])
        configs[treatment].add(right['configuration_id'])
    n = len(arms[baseline])
    task_stats = {task: {'n_pairs': len(ds), 'paired_difference': sum(ds)/len(ds)} for task, ds in sorted(tasks.items())}
    cluster_ci = None
    if len(tasks) >= 2:
        rng = random.Random(seed)
        clusters = list(tasks.values())
        draws = []
        for _ in range(bootstrap_samples):
            selected = [clusters[rng.randrange(len(clusters))] for _ in clusters]
            draws.append(sum(sum(c) for c in selected)/sum(len(c) for c in selected))
        cluster_ci = [percentile(draws, .025), percentile(draws, .975)]
    return {
        'comparison_group': group, 'baseline': baseline, 'treatment': treatment,
        'n_pairs': n, 'n_tasks': len(tasks), 'counts': counts,
        'baseline_success': counts['both_success']+counts['broken'],
        'treatment_success': counts['both_success']+counts['fixed'],
        'paired_difference': (counts['fixed']-counts['broken'])/n,
        'exact_mcnemar_p': exact_mcnemar(counts['fixed'], counts['broken']),
        'paired_difference_95pct_task_cluster_bootstrap': cluster_ci,
        'bootstrap_samples': bootstrap_samples, 'bootstrap_seed': seed,
        'per_task': task_stats,
        'configurations': {arm: sorted(ids) for arm, ids in configs.items()},
        'interpretation': [
            'Differences are proportions; multiply by 100 for percentage points.',
            'McNemar treats episode pairs as independent; do not interpret it as task-population evidence when episodes are clustered.',
            'The bootstrap resamples whole tasks and retains all paired episodes within each sampled task; its interval targets an episode-weighted difference under task resampling.',
            'Task-cluster uncertainty is unstable with few tasks and unavailable here for a single task.',
            'The ledger checks pairing only. It cannot verify implementation fidelity, sample selection, or whether an omitted run exists.',
            'This is one requested comparison; any multiplicity plan must be declared separately.'
        ]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ledger', type=Path)
    parser.add_argument('--group', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--treatment', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260909)
    args = parser.parse_args()
    if args.baseline == args.treatment:
        parser.error('Baseline and treatment must be distinct.')
    if args.bootstrap_samples < 1000:
        parser.error('Use at least 1000 bootstrap samples.')
    try:
        result = summarize(load_records(args.ledger), args.group, args.baseline, args.treatment, args.bootstrap_samples, args.seed)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'n_pairs': result['n_pairs'], 'paired_difference': result['paired_difference'], 'exact_mcnemar_p': result['exact_mcnemar_p']}))

if __name__ == '__main__':
    main()
