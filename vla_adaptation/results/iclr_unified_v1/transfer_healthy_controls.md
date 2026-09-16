# Matched healthy-adaptive controls for the transfer table (2026-09-16, from stored runs)

For `paper_integrated/experiments.tex`, Table "Transfer with frozen policies": the column "Healthy" can
carry the matched healthy-adaptive arm (frozen healthy → adaptive healthy on the same scenarios)
instead of the uncorrected healthy reference, with the caption's caveat removed.

| policy / robot | healthy frozen → healthy adaptive | file | protocol of the healthy arm |
|---|---|---|---|
| OpenVLA-OFT / Panda | 20/20 → 19/20 | `results/oft/oft_healthy.json` | rotation mask, online from zero |
| GR00T N1.7 / Panda | 18/20 → 18/20 (1 gained, 1 lost) | `results/groot/groot_healthy.json` | rotation mask, online from zero |
| π0 / ALOHA | 17/40 → 14/40 | `results/aloha/healthy_identify1_hold_n40.json` | identify-then-hold with a zero fault (the identification episode's phantom is held) |
| GR00T N1.5 / GR1, plate to plate | 18/30 → 19/30 | `results/gr1/p2p_null_hold_s160.json` | held estimate identified on a zero fault |
| GR00T N1.5 / GR1, tray to plate | 15/30 → 16/30 | `results/gr1/t2p_null_hold_s160.json` | same |
| GR00T N1.7 / WidowX | 15/20 → 16/20 | `results/widowx/null_tra000.json` | held estimate identified on a zero fault |

Notes. The Panda rows are online-from-zero controls (the same protocol as their faulted cells);
the joint-space rows are identify-then-hold controls, i.e. what a deployment that identifies on a
healthy robot and holds would do — the ALOHA control (−3 of 40) is the one row with a visible
healthy cost, from a held phantom (record 43: 17 → 14). The healthy frozen rates differ from the
"Healthy" reference column of the current table where that column was the uncorrected policy on
a different cohort (ALOHA 17/40 is the same cohort; GR1 22/30 and 17/30 were the faulted cells'
healthy references on their own scenarios).
