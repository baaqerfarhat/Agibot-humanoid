# Preregistration: latency and recovery time (re4 Part E, 2026-09-11)

Definitions fixed before the data; no directional prediction is registered, because this part
measures rather than tests. Instrumented on every Part C and D run via `--timing` (one JSON line
per control step; `openpi/adaptive_law.py`), summarised by `openpi/re4_record.py timing`.

- **Per-step wall clock** (`time.perf_counter`): `policy_ms` on replan steps only (the policy call
  that returns a chunk); `adapter_ms` = correction computation + FIR prediction + estimator update;
  `env_ms` = simulator step; `s2c_ms` = observation acquired to the next command issued;
  `loop_ms` = step start to adapter end. Reported: median, p95, p99, max per arm.
- **Deadline misses** at the 20 Hz interface: `loop_ms` > 50 ms, counted separately on replan and
  non-replan steps. This is a simulated robot on a Turing GPU, so policy and simulator times are
  this machine's, not a real-time controller's; the adapter's own compute is the number the
  paper's claim needs.
- **Recovery time.** Executed-action error on the corrected channels e_k = ‖(f_true + c)[mask]‖.
  Threshold τ = 0.2 · ‖f_true[mask]‖ (20 % of the fault on those channels), sustained for W = 20
  consecutive steps (1.0 s at 20 Hz). Recovery time = the first step k with e_j < τ for all j in
  [k, k + W). Reported in control time (k / 20 Hz) and in wall-clock seconds since the episode's
  first policy step. Episodes that end first are censored and counted, never dropped.
