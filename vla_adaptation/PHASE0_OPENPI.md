# Phase 0 on the GPU machine — openpi (π0/π0-FAST) + LIBERO

Vehicle decision (19 Aug 2026): **openpi over GR00T N1** — open torch weights, MuJoCo-based
benchmarks (no Isaac Sim), a built-in websocket policy server for remote inference, action
chunks (native fit for the prediction-vs-realisation signal of
`docs/PLAN_CROSS_EMBODIMENT.md` §1.6), and a continuous flow-matching action head (clean
final-projection edit site). GR00T needs a 16 GB+ floor AND Isaac-leaning eval; openpi's
server runs on any ≥16 GB card and the sim client can live on the laptop.

Each step has a gate; failing it stops the phase rather than triggering tuning.

## 0.1 Stand up openpi and reproduce a published number  *(gate: reproduces)*

```bash
git clone https://github.com/Physical-Intelligence/openpi && cd openpi
# follow README install (uv). Download the LIBERO fine-tuned checkpoint they publish.
# serve the policy (their websocket server), then run their LIBERO eval client.
```
Reproduce the repo's reported LIBERO success rates for π0-FAST (their eval script, their
seeds). Do not proceed on a partial reproduction — an environment mismatch here poisons
every later number (this cost the walker project weeks twice; see docs/LESSONS §4z).

## 0.2 Confirm the edit sites  *(gate: both exist and are reachable)*

- **Server-side:** locate the action head's final projection (weights + bias) in the torch
  model; confirm a bias/diagonal edit can be applied to the loaded model between episodes.
- **Client-side fallback (works even with zero model access):** an additive offset /
  diagonal rescale applied to the action CHUNK as it arrives — this is exactly the
  output-bias (b6) / output-rescale (w6) class from the walker, and the walker showed the
  client-side residual route and the direct-edit route differ ONLY through the envelope.
  Decide and RECORD which route Phase 1 uses; do not mix them within one comparison.

## 0.3 Fault suite  *(client-side wrappers around the LIBERO env; no model surgery)*

| side | faults |
|---|---|
| perception | camera brightness/bias/blur; proprioception offset |
| actuation | per-joint action gain (multiplicative); action offset; action delay |
| dynamics | object mass/friction scale; table height |

Implement as env wrappers so every fault is exactly repeatable and per-seed.

## 0.4 Headroom + reachability gates  *(before ANY adaptation — the framework's order)*

For each fault × 2–3 severities × ~20 episodes: frozen success rate vs nominal.
Keep cells with a ≥30-percentage-point drop that are not floor-dead (>5% success — the
walker's tq04 lesson: a dead robot gives no search signal). For each kept cell, if an
analytic or oracle repair is computable (e.g. inverse gain for actuation faults), replay it
through the CHOSEN route from 0.2 to measure the ceiling — the envelope-selects-the-class
result says this number, not the fault label, predicts searchability.

## 0.5 First adaptation run  *(pre-register first — copy the template discipline)*

`core/episodic_search.py` over the 0.2 edit site on the best-gated cell:
- `seeds_per_gen >= 2` (non-negotiable; see lesson 1 in the module docstring),
- fresh episode seeds only, held-out seeds disjoint,
- arms: search / refit-off control / frozen; ACE arm for the record,
- primary endpoint and thresholds written down BEFORE the run, in a
  `prereg_records/PREREG_OPENPI_<cell>.md` following the walker templates.

## Kill criterion (from PLAN_CROSS_EMBODIMENT Phase 0)

No runnable policy-server + benchmark pair within a week → the direction is not testable
with available resources; record that and stop rather than descending into harness surgery.
