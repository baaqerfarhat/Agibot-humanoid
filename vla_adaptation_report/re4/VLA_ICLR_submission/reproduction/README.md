# Reproduction guide

These files reproduce constructed mathematical checks and arithmetic from reported aggregate counts. They are not a VLA simulator implementation or recovered episode logs.

## Dependencies and commands

Use Python 3 with NumPy, SciPy, and Matplotlib. From this `reproduction` directory:

```bash
mkdir -p paper
python audit/verify_execution_scope.py
python audit/verify_history_partition.py
python audit/verify_contraction_tracking.py
python audit/plot_main_servo_panel.py
python audit/verify_contraction_sampled.py
python audit/verify_schedule_bounds.py
python audit/verify_paired_effect.py
python audit/redraw_empirical_figures.py
```

The `paper/` directory created here is a plotting output directory. These commands do not overwrite the manuscript's supplied figure assets. Other included `verify_*.py` scripts are additional constructed checks of the continuous/sampled analysis. Some retain earlier local variable names; the manuscript's notation change does not alter their arithmetic.

Precomputed `*_checks.json` files document constructed runs, including historical checks. This editorial revision preserved the data and inspected the scripts; it does not relabel every saved check as a new experiment. The principal servo data are in `audit/contraction_tracking_trajectories.npz`; `constructed_ood_trajectories.npz` belongs to the separate scalar illustration. Neither is a measured VLA trajectory.

## Empirical and logging files

- `audit/results_ledger.csv`: 66 transcribed aggregate outcome rows, with source locators. These neutral historical paths are provenance references, not required input files for the included scripts.
- `audit/restored_outcomes.csv`: additional transcribed aggregate records.
- `audit/run_configuration_template.json`: schema 3.1; unknown evidence fields are null, including the newly required physical-state/contact, event, forcing, and command-comparison information.
- `audit/episode_record_template.csv`: empty episode schema, not fabricated outcomes.
- `audit/audit_episode_records.py --help`: command-line utility for future real episode records. Aggregate counts cannot reconstruct those records.

The physical certificate concerns the nominal response to the same supplied VLA commands. For a reduced model, omitted history/contact effects require uniform bounds. Hybrid impacts require explicit event/transition treatment. Any error-dependent forcing belongs in the coupled analysis, and realization already included in interface mismatch must not be counted twice. Independent adapters may induce different command streams; their physical comparison needs the additional nominal-response divergence or fixed-command replay.

The empirical convergence figure is included as a vector asset. Figure 1 is
an editable TikZ mechanism diagram and carries no empirical cohort. Raw
convergence traces were not supplied. The figure utilities included here
reconstruct the supported numerical panels; they do not reconstruct
unavailable rollout evidence.
