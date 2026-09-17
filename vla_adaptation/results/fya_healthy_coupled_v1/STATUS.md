# Healthy coupled replication v1: status

Registration `prereg_records/PREREG_FYA_HEALTHY_COUPLED_V1.md`.

| Item | State |
|---|---|
| Sources healthy_off / healthy_off_dup (GPU 1 server) | complete, 40/40 each; duplicate identical on 37/40 keys (3 keys diverge, max raw-action gap .127) |
| Source healthy_nt on the GPU 0 server | complete (39/40) but **not coupled**: 0/40 prefixes match healthy_off (cross-device numerics); kept as `healthy_nt_gpu0.*` |
| Amendment 1: healthy_nt on the GPU 1 server | complete (39/40) but bit-identical to the GPU 0 run and again 0/40 coupled: process-level, not device-level |
| Amendment 2: all three arms on ONE server process (`sources_v3/`) | complete: coupled 39/40, duplicate exact 36/40 |
| Extraction / healthy matrix / scoring (R1 primary) | complete: 34 keys / 9 tasks; R1 −1.71e-4 [−2.5e-4, −9.8e-5], negative on 34/34, resolved (prereg §7) |
