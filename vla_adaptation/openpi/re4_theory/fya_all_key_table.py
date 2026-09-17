#!/usr/bin/env python3
"""Join every assigned full key to task outcomes and explicitly named matrix effects.

Missing physical effects remain empty. Task discordance compares the first two
arms in the supplied order; it is not a causal attribution or a paired matrix
contrast. Use --effect-prefix NT_ for NT-matrix effects alongside NT/DOB task
outcomes. Historical analysis_offline outputs are protected from replacement.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def read_json(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith('.gz') else data)


def key(row):
    return int(row['task']), int(row['init']), int(row.get('sampler_seed', row.get('seed')))


def indexed(rows):
    out = {key(row): row for row in rows}
    assert len(out) == len(rows), 'duplicate full key'
    return out


def csv_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True)
    for flag in ('manifest', 'source-keys', 'run', 'effects', 'costs', 'out'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--arm', action='append', required=True, help='name=result.json; first=task reference, second=task test')
    parser.add_argument('--effect-matrix', required=True, help='controller and matrix used for the physical effects')
    parser.add_argument('--effect-prefix', default='', help='physical-column prefix, e.g. NT_ for the P3 NT-only matrix')
    args = parser.parse_args()
    assert 'analysis_offline' not in args.out.parts, 'preserve historical outputs'
    manifest = read_json(args.manifest)
    assigned = indexed(manifest['scenarios'])
    arms, paths = {}, [args.manifest, args.source_keys, args.run, args.effects, args.costs]
    for spec in args.arm:
        name, path = spec.split('=', 1)
        assert name not in arms
        result = read_json(path)
        assert len(result['arms']) == 1
        outcomes = indexed(next(iter(result['arms'].values()))['per_ep'])
        assert set(outcomes) == set(assigned), 'source outcome cohort differs from full assigned keys'
        arms[name] = outcomes
        paths.append(Path(path))
    order = list(arms)
    assert len(order) >= 2
    source = indexed(csv_rows(args.source_keys))
    assert set(source) == set(assigned)
    run = read_json(args.run)
    fidelity = {key(r['key']): r['fidelity'] for r in run['keys']}
    assert len(fidelity) == len(run['keys']) and set(fidelity) <= set(assigned)
    effects = indexed(csv_rows(args.effects))
    assert set(effects) == {k for k, f in fidelity.items() if f['valid']}
    costs = {}
    for row in csv_rows(args.costs):
        k = key(row)
        assert row['cell'] not in costs.setdefault(k, {})
        costs[k][row['cell']] = row
    assert set(costs) == set(effects)
    assert all(set(cells) == {'J00', 'J10', 'J01', 'J11'} for cells in costs.values())
    rows = []
    for k, scenario in assigned.items():
        row = dict(campaign=args.campaign, suite=manifest.get('suite'), task=k[0], init=k[1], sampler_seed=k[2],
                   physical_effect_matrix=args.effect_matrix)
        for name in order:
            row['success_'+name] = bool(arms[name][k]['ok'])
        reference, test = (bool(arms[name][k]['ok']) for name in order[:2])
        row.update(discordant=reference != test,
                   discordance=('rescue' if test else 'regression') if reference != test else '',
                   task_reference_arm=order[0], task_test_arm=order[1],
                   eligible=source[k]['eligible'], exclusion_reason=source[k].get('reasons', ''),
                   replay_valid=fidelity[k]['valid'] if k in fidelity else None,
                   fidelity_failed_checks=';'.join(c for c, v in fidelity[k]['checks'].items() if not v) if k in fidelity else '')
        for metric in ('D0', 'R1', 'T', 'I', 'S', 'D0_r', 'R1_r', 'T_r'):
            row[args.effect_prefix+metric] = effects.get(k, {}).get(metric, '')
        for cell in ('J00', 'J10', 'J01', 'J11'):
            cost = costs.get(k, {}).get(cell, {})
            for metric in ('endpoint_p', 'endpoint_r', 'done_step'):
                row[args.effect_prefix+metric+'_'+cell] = cost.get(metric, '')
        rows.append(row)
    discordant = [r for r in rows if r['discordant']]
    valid_discordant = sorted((r for r in discordant if r['replay_valid']), key=lambda r: (r['task'], r['init'], r['sampler_seed']))
    selection = {f'first_valid_{kind}': next(([r['task'], r['init'], r['sampler_seed']] for r in valid_discordant if r['discordance'] == kind), None)
                 for kind in ('rescue', 'regression')}
    audit = dict(campaign=args.campaign, assigned=len(rows), eligible=sum(r['eligible'] == 'True' for r in rows),
                 replay_valid=sum(bool(r['replay_valid']) for r in rows), n_discordant=len(discordant),
                 n_discordant_with_effects=len(valid_discordant), task_reference_arm=order[0], task_test_arm=order[1],
                 physical_effect_matrix=args.effect_matrix, physical_effect_column_prefix=args.effect_prefix,
                 physical_effect_scope='single named crossed matrix; not a paired contrast between the task-outcome arms',
                 discordant_keys=discordant, selection_rule='lexicographically first fidelity-valid rescue and regression by task/init/sampler_seed',
                 selection=selection, empty_categories=[k for k, v in selection.items() if v is None],
                 source_files_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                 producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'all_keys.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.out/'inclusion_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k: audit[k] for k in ('campaign', 'assigned', 'eligible', 'replay_valid', 'n_discordant', 'n_discordant_with_effects', 'selection')}))


if __name__ == '__main__':
    main()
