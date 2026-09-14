"""Redraw reported success bars from the supplied table-level ledger.

No rollout trajectories or additional trials are synthesized. Every plotted numerator,
denominator, and condition is read from results_ledger.csv. Run from any directory.
"""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
with (ROOT/'audit/results_ledger.csv').open(newline='') as stream:
    rows = {row['dataset_id']: row for row in csv.DictReader(stream)}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
colors = ('#8A929C', '#206F9C')
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55), sharey=True)
fig.subplots_adjust(left=.065, right=.99, bottom=.27, top=.76, wspace=.16)
interval_records = []


def wilson(k, n, z=1.959963984540054):
    """Marginal binomial score interval, independent of paired-effect uncertainty."""
    p = np.asarray(k, dtype=float) / n
    denom = 1 + z*z/n
    middle = (p + z*z/(2*n))/denom
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return np.maximum(0., middle-radius), np.minimum(1., middle+radius)


families = (
    ('translation', 'Translation', 'translation correction'),
    ('rotation', 'Rotation', 'rotation correction'),
    ('uniform_6axis', 'Uniform six-axis', 'rotation correction'),
)
for ax, (family, title, mask) in zip(axes, families):
    selected = [rows[f'map_{family}_{magnitude}'] for magnitude in ('0.05','0.10','0.15')]
    assert all(int(row['n']) == 20 for row in selected)
    x = np.arange(3)
    for index, field in enumerate(('frozen_success', 'corrected_success')):
        counts = np.array([int(row[field]) for row in selected])
        percentages = 100 * counts / 20
        centers = x + (index-.5)*.36
        ax.bar(centers, percentages, width=.33, color=colors[index], zorder=3)
        low, high = wilson(counts, 20)
        ax.errorbar(centers, percentages,
                    yerr=np.vstack((percentages-100*low, 100*high-percentages)),
                    fmt='none', ecolor='#222F3A', elinewidth=.8, capsize=2,
                    capthick=.8, zorder=4)
        for row, center, value, lo, hi in zip(selected, centers, percentages, low, high):
            ax.text(center, 100*hi+2.0, f'{value:.0f}%', ha='center', va='bottom', fontsize=6.7)
            interval_records.append({'dataset_id': row['dataset_id'], 'arm': field,
                                     'successes': int(row[field]), 'n': 20,
                                     'estimate': value/100, 'wilson95': [float(lo),float(hi)]})
    ax.set_title(f'{title}\n{mask}; n = 20', fontsize=8.5, pad=8)
    ax.set_ylim(0, 110)
    ax.set_yticks([0,25,50,75,100])
    ax.set_xticks(x, ['0.05','0.10','0.15'])
    ax.set_xlabel('Offset magnitude', labelpad=4)
    ax.grid(axis='y', color='#DCE1E6', linewidth=.6, zorder=0)
    ax.tick_params(axis='both', length=2.5, labelsize=7)
axes[0].set_ylabel('Task success (%)')
fig.legend(handles=[Patch(color=c, label=l) for c,l in zip(colors, ('Faulted frozen','Adaptive correction'))],
           loc='lower center', bbox_to_anchor=(.53,.005), ncol=2, frameon=False, fontsize=8)
output=ROOT/'paper/fig_map.pdf'
fig.savefig(output, bbox_inches='tight', pad_inches=.04)
fig.savefig(ROOT/'audit/fig_map_wilson_preview.png', dpi=180,
            bbox_inches='tight', pad_inches=.04)
plt.close(fig)
(ROOT/'audit/wilson_severity_intervals.json').write_text(json.dumps({
    'method': 'Marginal 95% Wilson intervals under an episode-binomial model',
    'scope': 'Not paired-effect intervals; no task-clustering or raw-trial data assumed',
    'intervals': interval_records,
}, indent=2)+'\n')
print(f'Redrew {output.name}: nine exact count pairs, n=20 each; marginal Wilson95 whiskers.')
