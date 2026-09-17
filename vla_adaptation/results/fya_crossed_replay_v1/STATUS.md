# Crossed command-stream replay v1: status

Registration `prereg_records/PREREG_FYA_CROSSED_REPLAY_V1.md`.

| Item | State |
|---|---|
| Extraction (read-only, 20 keys) | complete: 18 eligible, 2 excluded (task 0 states 13, 14) |
| Configuration (deployed header arrays) | frozen |
| Pilot (task-0 keys, diagonal cells) | complete: exact reproduction (all gaps 0.0) |
| Registration | committed before collection |
| Collection (18 keys × 7 continuations) | complete (re-collected after the angle-check fix); 15 valid, 3 excluded by live-vs-replay gaps |
| Scoring / figures | complete: I unresolved (2.3e-5 [−2.7e-5, 7.6e-5]); D0 8.9e-5, T 5.0e-4 (prereg §8); `figures/crossed_*.pdf` |
| Optional matrices (delayed NT, innovation, healthy control) | complete (prereg §10): delay D0 5.2e-5, innovation D0 8.9e-5 / R1 5.0e-4, healthy D0 −2.4e-5 and R1 −3.6e-4 on every key; I unresolved everywhere |
