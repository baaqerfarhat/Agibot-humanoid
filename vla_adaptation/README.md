# VLA_Adaptation — online layer adaptation of frozen policies by realised-metric search

**Branch purpose.** Port the adaptation method validated on this repo's X2 walking policy to a
frozen VLA (openpi π0 / π0-FAST on LIBERO), on a machine with a ≥16 GB GPU. The method is
**gradient-free** — it needs only forward passes / a served policy — so it applies to models we
did not train and cannot backprop through.

## The method, in five lines

1. **Edit site:** a small (≤ ~20-dim) parameter vector on ONE layer — output-layer bias
   (= constant action offset), a diagonal output rescale, or the first layer's bias.
2. **Objective:** the *realised task metric* (distance, success rate) over whole episodes.
   Never a surrogate — every instantaneous surrogate tested was at chance w.r.t. the true
   fix (`docs/WHEN_ADAPTATION_WORKS.md`, condition D; mechanism in `docs/ACE_CONNECTION.md`).
3. **Search:** across-episode CEM, fresh seeds only (no replay), **≥ 2 episodes per
   candidate** — single-episode scoring is variance-seeking and measurably fails
   (`prereg_records/PREREG_ONLINE_TQ05.md` → `_V2.md`).
4. **Controls:** the same loop with the refit disabled (selection-without-learning), plus
   norm-matched random edits; settled estimate = elite-mean of the final generation for every
   arm.
5. **Gates before any search:** headroom (does the fault cost anything?) and reachability
   (is a repair inside the class *and* the deployment envelope?). The envelope can select
   the function class before the search does — measured: a multiplicative fault's inverse
   recovers 5.8% as a clipped action residual vs **100.0%** as a direct layer rescale.

## Validated results (X2 walker, all pre-registered; records in `prereg_records/`)

| result | numbers |
|---|---|
| **Online** adaptation of the output-layer bias, torque fault | **+1.38 m (~55% of headroom) held-out, 120 fresh-seed episodes, no replay**; beats matched selection control +1.33 m; deployed reward better 6/6 generations |
| Same protocol, single-episode scoring | **fails** (+0.11 m); ‖settled‖ walks 0.104→0.199 past the fix scale 0.141–0.149 — the variance-seeking mechanism |
| Test-time search, sensor-bias fault | 80.9% (output bias) / 68.3% (input bias) of headroom, held-out |
| Which layer? | tractability, not correspondence: output bias won on every fault; an input-side search matched the analytic inverse's recovery while orthogonal to it (cos +0.04) |
| Envelope vs class | every state-dependent/large inverse needs 3–4× the deployed residual cap; lifting the contract (direct layer edit) takes a 5.8%-ceiling fault to 100.0% |

## Layout

```
vla_adaptation/
├── core/episodic_search.py     # model-agnostic search core + statistical self-test (numpy only)
├── docs/                       # the six-conditions framework, mechanism, positioning, 25 rules
├── prereg_records/             # pre-registrations WITH results — the worked methodology examples
├── walker_reference/           # the validated X2 scripts (need the x2_ttcl lab workspace to run)
│   └── results/                # raw result JSONs backing the table above
└── PHASE0_OPENPI.md            # what to do on the GPU machine, step by step
```

`walker_reference/` scripts import the private `x2_ttcl` testbed (mjlab-based) and run on the
lab machine; they are included as the *reference implementation* the openpi port copies.
`core/episodic_search.py` has no dependencies beyond numpy and is the piece to build on.

## GPU machine (≥16 GB VRAM)

Follow `PHASE0_OPENPI.md`: stand up openpi, serve π0-FAST over its websocket policy server,
reproduce the published LIBERO numbers (gate 0), then the fault suite + headroom gates, then
`core/episodic_search.py` over the action-head edit.

## What runs on the 8 GB laptop

- `python core/episodic_search.py` — the self-test (2 s).
- LIBERO environment + fault-injection development with a scripted policy (MuJoCo, no VLA).
- The openpi *client* side, against a policy server on the GPU machine.
- All X2 walker studies (CPU, via the lab workspace).

Provenance: distilled 2026-08-17…19 from the walker study series; a parameter-route search
verdict (3-class comparison under the lifted contract) may land in a follow-up commit.
