"""Recompute the primary effect interval from published discordant counts.

No robot trials are run, and no episode identities, task labels, or seed records
are synthesized. The normal interval is explicitly approximate under an iid
model for episode pairs. It does not account for task-level clustering.
"""
from pathlib import Path
import csv
import json
import math
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
with (ROOT/'audit/results_ledger.csv').open(newline='') as stream:
    ledger = list(csv.DictReader(stream))
primary = [row for row in ledger if row['dataset_id'].startswith('headline_libero_')]
assert len(primary) == 4, [row['dataset_id'] for row in primary]
n = sum(int(row['n']) for row in primary)
frozen = sum(int(row['frozen_success']) for row in primary)
corrected = sum(int(row['corrected_success']) for row in primary)
b = sum(int(row['fixed']) for row in primary)
c = sum(int(row['broken']) for row in primary)
assert (n, frozen, corrected, b, c) == (120, 28, 78, 51, 1)
assert corrected-frozen == b-c
both_success = corrected-b
both_failure = n-b-c-both_success
assert (both_success, both_failure) == (27, 41)
delta = (b-c)/n
sample_variance = (b+c-n*delta**2)/(n-1)
se = math.sqrt(sample_variance/n)
z = NormalDist().inv_cdf(.975)
ci = [delta-z*se, delta+z*se]
p_exact = min(1., 2*sum(math.comb(b+c, j) for j in range(min(b,c)+1))/2**(b+c))
# An independent second-moment calculation uses the four aggregate outcome cells.
m2 = (b*(1-delta)**2+c*(-1-delta)**2+(both_success+both_failure)*delta**2)/(n-1)
assert math.isclose(sample_variance, m2, rel_tol=1e-14)
assert math.isclose(p_exact, 2.353672812205332e-14, rel_tol=1e-12)
result = {
    'source': 'results_ledger.csv; four primary rows',
    'n_pairs': n, 'frozen_success': frozen, 'corrected_success': corrected,
    'fixed': b, 'broken': c, 'both_success': both_success, 'both_failure': both_failure,
    'paired_effect': delta, 'paired_standard_error': se,
    'paired_normal95_interval': ci, 'exact_mcnemar_two_sided_p': p_exact,
    'assumptions': 'Independent identically distributed episode pairs; normal approximation',
    'limitations': 'Not an exact confidence interval. Does not account for task clustering. No task bootstrap is identifiable from these aggregate counts.',
}
output=ROOT/'audit/paired_effect_checks.json'
output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
