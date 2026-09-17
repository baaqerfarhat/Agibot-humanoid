#!/usr/bin/env python3
"""Retrospective reference envelopes in the registered simulator ee_pos metric.

For Q=.05 I and 50-step paths, R1(r+v)=R1(r)+2<Delta,v>_Q and
S(r+v)=S(r)-2<H,v>_Q. The reported radii bound ||v||_Q by sqrt(2.5)*a;
a per-step displacement bound a is sufficient. These fixed-trajectory
sensitivity envelopes are neither confidence nor closed-loop certificates.
Empirical duplicate references require full-key identity, a matched 30-step
command/physical prefix, and the SAME recorded ee_pos coordinate as the cost.
Telemetry position is not silently substituted for simulator ee_pos.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

DT = .05
HORIZON = 50
RADII_MM = (1., 2.5, 5., 10.)
PHYSICAL_TOL = 1e-8


def read_json(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith('.gz') else data)


def qnorm2(x):
    return float(DT * np.sum(np.asarray(x) ** 2))


def qdot(x, y):
    return float(DT * np.sum(np.asarray(x) * np.asarray(y)))


def key(row):
    return tuple(int(row[x]) for x in ('task', 'init', 'sampler_seed'))


def outcome_keys(path):
    data = read_json(path)
    assert len(data['arms']) == 1, 'provide a single-arm source result'
    keys = [key(r) for r in next(iter(data['arms'].values()))['per_ep']]
    assert len(set(keys)) == len(keys)
    assert len(set(k[:2] for k in keys)) == len(keys), 'legacy telemetry lacks seed; task/init must be unique'
    return keys


def telemetry(path, keys):
    mapping = {k[:2]: k for k in keys}
    groups = {k: [] for k in keys}
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('type') == 'step' and row.get('phase') == 'rollout':
                k = mapping[(int(row['task']), int(row['init']))]
                if 'sampler_seed' in row:
                    assert int(row['sampler_seed']) == k[2]
                groups[k].append(row)
    for rows in groups.values():
        assert rows
        rows.sort(key=lambda s: s['t'])
        assert [s['t'] for s in rows] == list(range(10, 10 + len(rows)))
    return groups


def quat_angle(a, b):
    a, b = np.asarray(a), np.asarray(b)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(4 * np.arcsin(min(1., min(np.linalg.norm(a-b), np.linalg.norm(a+b)) / 2)))


def eq_task(values, tasks, indices):
    values, tasks = np.asarray(values), np.asarray(tasks)
    means = np.array([values[tasks == t].mean() for t in np.unique(tasks)])
    assert indices.shape[1] == len(means)
    return dict(estimate=float(means.mean()), ci95=np.quantile(means[indices].mean(1), [.025, .975]).tolist())


def duplicate_references(args, valid_keys):
    original_keys = outcome_keys(args.ref_results)
    duplicate_keys = outcome_keys(args.dup_results)
    assert original_keys == duplicate_keys, 'source arm full-key order differs'
    original = telemetry(args.ref_src, original_keys)
    duplicate = telemetry(args.dup_src, duplicate_keys)
    rows = []
    for run_key in valid_keys:
        k = key(run_key['key'])
        prefix = run_key['prefix']
        assert prefix == 30
        ref, dup = original[k], duplicate[k]
        row = dict(task=k[0], init=k[1], sampler_seed=k[2], full_key_matched=True,
                   original_steps=len(ref), duplicate_steps=len(dup))
        long_enough = min(len(ref), len(dup)) >= prefix + HORIZON
        row['window_available'] = long_enough
        if min(len(ref), len(dup)) >= prefix:
            row['prefix_gaps'] = {
                field: max(float(np.max(np.abs(np.asarray(a[field])-np.asarray(b[field]))))
                           for a, b in zip(ref[:prefix], dup[:prefix]))
                for field in ('raw_action', 'position', 'joint_position')}
            row['prefix_gaps']['quaternion_angle'] = max(quat_angle(a['quaternion'], b['quaternion']) for a, b in zip(ref[:prefix], dup[:prefix]))
            row['prefix_matched'] = bool(row['prefix_gaps']['raw_action'] == 0 and
                                         all(row['prefix_gaps'][f] <= PHYSICAL_TOL for f in ('position', 'joint_position', 'quaternion_angle')))
        else:
            row['prefix_gaps'] = None
            row['prefix_matched'] = False
        row['same_metric_retained'] = bool(long_enough and all('ee_pos' in s for s in ref[prefix:prefix+HORIZON] + dup[prefix:prefix+HORIZON]))
        row['qualified'] = bool(long_enough and row['prefix_matched'] and row['same_metric_retained'])
        reasons = []
        if not long_enough:
            reasons.append('source window shorter than prefix+horizon')
        if not row['prefix_matched']:
            reasons.append('command/physical prefix mismatch')
        if not row['same_metric_retained']:
            reasons.append('source telemetry does not retain ee_pos for both windows')
        row['unavailable_reasons'] = reasons
        if row['qualified']:
            branches = run_key['branches']
            ref_run = np.asarray([s['ee_pos'] for s in branches['ref']])
            ref_live = np.asarray([s['ee_pos'] for s in ref[prefix:prefix+HORIZON]])
            row['original_reference_max_gap'] = float(np.max(np.abs(ref_run-ref_live)))
            if row['original_reference_max_gap'] > PHYSICAL_TOL:
                row['qualified'] = False
                row['unavailable_reasons'].append('replayed and live original ee_pos reference mismatch')
            else:
                alt = np.asarray([s['ee_pos'] for s in dup[prefix:prefix+HORIZON]])
                paths = {c: np.asarray([s['ee_pos'] for s in branches[c]]) for c in ('J00', 'J10', 'J11')}
                js = {c: qnorm2(p-alt) for c, p in paths.items()}
                row.update(distance_to_original_ref_qnorm=float(np.sqrt(qnorm2(alt-ref_run))),
                           distinct=bool(np.max(np.abs(alt-ref_run)) > PHYSICAL_TOL),
                           R1_alt=js['J10']-js['J11'], S_alt=js['J00']-2*js['J10']+js['J11'])
        rows.append(row)
    qualified = [r for r in rows if r['qualified']]
    output = dict(n_candidates=len(rows), n_prefix_matched=sum(r['prefix_matched'] for r in rows),
                  n_same_metric_retained=sum(r['same_metric_retained'] for r in rows),
                  n_qualified=len(qualified), n_distinct=sum(r.get('distinct', False) for r in qualified),
                  metric='simulator ee_pos; telemetry position is not substituted',
                  prefix_action_tolerance=0., prefix_physical_tolerance=PHYSICAL_TOL,
                  qualification='full task/init/sampler_seed identity, 30-step command and observed physical prefix, 50-step same-coordinate reference',
                  R1_alt_equal_task=None, S_alt_equal_task=None, rows=rows)
    if qualified:
        tasks = np.array([r['task'] for r in qualified])
        indices = np.random.default_rng(args.seed).integers(0, len(np.unique(tasks)), size=(args.boot, len(np.unique(tasks))))
        for metric in ('R1', 'S'):
            output[metric+'_alt_equal_task'] = eq_task([r[metric+'_alt'] for r in qualified], tasks, indices)
    return output


def main():
    parser = argparse.ArgumentParser()
    for flag in ('run', 'out'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--label', required=True)
    for flag in ('ref-src', 'dup-src', 'ref-results', 'dup-results'):
        parser.add_argument('--'+flag, type=Path)
    parser.add_argument('--boot', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260919)
    args = parser.parse_args()
    duplicate_inputs = [args.ref_src, args.dup_src, args.ref_results, args.dup_results]
    assert not any(duplicate_inputs) or all(duplicate_inputs), 'duplicate references require both telemetry and both outcome files'
    assert 'analysis_offline' not in args.out.parts, 'preserve the historical analysis_offline directory'
    data = read_json(args.run)
    keys = [k for k in data['keys'] if k['fidelity']['valid']]
    assert keys and len({key(k['key']) for k in keys}) == len(keys)
    rows = []
    for run_key in keys:
        branches = run_key['branches']
        assert all(len(branches[c]) == HORIZON for c in ('ref', 'J00', 'J10', 'J11'))
        paths = {c: np.asarray([s['ee_pos'] for s in branches[c]]) for c in ('ref', 'J00', 'J10', 'J11')}
        assert all(p.shape == (HORIZON, 3) and np.all(np.isfinite(p)) for p in paths.values())
        ref, p00, p10, p11 = (paths[c] for c in ('ref', 'J00', 'J10', 'J11'))
        j00, j10, j11 = (qnorm2(paths[c]-ref) for c in ('J00', 'J10', 'J11'))
        r1, stream = j10-j11, j00-2*j10+j11
        delta, h = p11-p10, p00-2*p10+p11
        k = key(run_key['key'])
        identity_errors = []
        shifted_j00 = []
        random_shift = np.random.default_rng(k[0]*100+k[1]).normal(scale=.002, size=ref.shape)
        for shift in (random_shift, np.broadcast_to(np.array([.002, -.001, .003]), ref.shape)):
            j0, j1, j2 = (qnorm2(paths[c]-ref-shift) for c in ('J00', 'J10', 'J11'))
            identity_errors.extend([abs(j1-j2-(r1+2*qdot(delta, shift))), abs(j0-2*j1+j2-(stream-2*qdot(h, shift)))])
            shifted_j00.append(j0)
        assert max(identity_errors) < 1e-12 and min(shifted_j00) > 0
        row = dict(task=k[0], init=k[1], seed=k[2], n=HORIZON, J00=j00, J10=j10, J11=j11,
                   R1=r1, S=stream, normDelta=float(np.sqrt(qnorm2(delta))), normH=float(np.sqrt(qnorm2(h))),
                   identity_ok=True, identity_checks=2, identity_max_error=max(identity_errors), min_shifted_J00=min(shifted_j00))
        for radius in RADII_MM:
            vq = np.sqrt(HORIZON*DT)*radius*.001
            for metric, norm in (('R1', row['normDelta']), ('S', row['normH'])):
                row[f'{metric}_lo_{radius}'] = row[metric]-2*norm*vq
                row[f'{metric}_hi_{radius}'] = row[metric]+2*norm*vq
        rows.append(row)
    tasks = np.array([r['task'] for r in rows])
    unique = np.unique(tasks)
    indices = np.random.default_rng(args.seed).integers(0, len(unique), size=(args.boot, len(unique)))
    aggregate = lambda values: eq_task(values, tasks, indices)
    out = dict(label=args.label, run=str(args.run), n_sources=len(rows), n_tasks=len(unique),
               radii_mm=list(RADII_MM), vq_factor='||v_i||_Q <= sqrt(50 dt) a; a uniform per-step displacement bound a is sufficient',
               metric='ee_pos', horizon=HORIZON, dt=DT, boot=args.boot, seed=args.seed,
               scope='retrospective fixed-trajectory sensitivity; point-mean thresholds are not confidence/stability/task-recovery certificates; registered decisions remain unchanged',
               original={m: aggregate([r[m] for r in rows]) for m in ('R1', 'S')},
               radius={}, identities_checked=len(rows), identity_displacements_per_source=2,
               point_mean_sign_envelope_threshold_mm={})
    for radius in RADII_MM:
        entry = {}
        for metric in ('R1', 'S'):
            for side, name in (('lo', 'lower'), ('hi', 'upper')):
                entry[metric+'_'+name] = aggregate([r[f'{metric}_{side}_{radius}'] for r in rows])
            entry['sources_'+metric+'_sign_preserved'] = int(sum((r[metric] > 0 and r[f'{metric}_lo_{radius}'] > 0) or (r[metric] < 0 and r[f'{metric}_hi_{radius}'] < 0) for r in rows))
        out['radius'][str(radius)] = entry
    for metric, norm in (('R1', 'normDelta'), ('S', 'normH')):
        slope = 2*aggregate([r[norm] for r in rows])['estimate']*np.sqrt(HORIZON*DT)
        out['point_mean_sign_envelope_threshold_mm'][metric] = abs(out['original'][metric]['estimate']) / slope * 1000 if slope else None
    if all(duplicate_inputs):
        out['duplicate_reference'] = duplicate_references(args, keys)
    source_paths = [args.run] + [p for p in duplicate_inputs if p]
    out['source_files_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    out['producer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'reference_sensitivity.json').write_text(json.dumps(out, indent=2)+'\n')
    np.save(args.out/'task_bootstrap_indices.npy', indices)
    with (args.out/'per_source.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: out[k] for k in ('label', 'n_sources', 'n_tasks', 'point_mean_sign_envelope_threshold_mm')}))
    if 'duplicate_reference' in out:
        print(json.dumps({k: v for k, v in out['duplicate_reference'].items() if k != 'rows'}))


if __name__ == '__main__':
    main()
