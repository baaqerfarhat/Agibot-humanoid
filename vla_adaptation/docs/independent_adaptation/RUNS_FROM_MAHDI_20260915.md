# Runs from Mahdi's side, 2026-09-15 (for the queue's author)

All four GPU items are done; outcomes are appended to your preregistrations in `prereg_records/`
and the raw runs are under `results/collab_q/` (result.json, episodes.csv, run_configuration.json,
telemetry.jsonl.gz, timing.jsonl, record.log per run). Scores: `results/collab_q/score_*.json`.

| item | result | prereg verdict |
|---|---|---|
| Q1 healthy control | already run 14 Sep: spatial healthy 20 → 19 (record 54; the lost episode is task 4, not the 0.078-phantom episode) | prediction 3 holds |
| Q2 pose tracking | r_z pose-excess ratio 0.815 [0.70, 0.94], r_x 0.91 [0.80, 1.05]; r_y untouched; healthy 18/20 = 18/20; success +2/60, +3/40 | P1 neither confirmed nor refuted; P2, P3, P4 hold |
| Q5 six-channel oracle | 38/40 (rotation-only 22/40; 17 to 1 paired); frozen 0/40 | both hold: the mask caps libero_10 |
| Q6 pinned FIR/ARX/DC | 10 / 8 / 14 of 80; r_y settle 0.30 / 0.34 / 0.72 | P1 holds; P2 holds (0.715); P3 direction holds, unresolved |

Protocol notes you should know: Q2 ran on GPU 0 shared with another user's training (policy
inference on GPU 0, MuJoCo rendering on GPU 1 because GPU 0's EGL path is dead here); the Q2
healthy keys are the first two per task of the E2 manifest and the joint-5 keys the first four
per task of the spatial manifest, both declared in `scripts/re4/collab_q_chain.sh`. The Q2
primary endpoint was recomputed offline with your tracker's definition (`openpi/re4_theory/
q2_score.py`). Your branch is merged into main unchanged except AGENTS.md.
