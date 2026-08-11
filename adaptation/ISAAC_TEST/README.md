# Testing the box-pickup policy in IsaacSim

This folder exists because **IsaacSim will not run on the laptop the rest of this work was done
on**, and the test it would perform is worth doing somewhere that it does.

## What question this answers

The capture-point layer-adaptation result (`../RESULTS_CAPTURE_2026-08-07/`) is confirmed in
MuJoCo: frozen 150.3 → adapted 180.2 control steps on held-out seeds, 25/31, p = 0.0009, beating
both displacement-matched random nulls. Two things MuJoCo cannot settle:

1. **Does the frozen policy complete the motion in the simulator it was TRAINED in?** It never
   does in MuJoCo (0/32 seeds). PhysX contact solving on the robot's 22 foot spheres, and the
   presence of the box, are the last two differences from the training plant. If the frozen
   policy succeeds under IsaacSim, the MuJoCo failure is a modelling artefact and this changes
   how every result here should be read.
2. **Does the capture-point law transfer to PhysX?**

## Why it could not be run here

IsaacSim's own compatibility checker, on the machine in question:

```
|-- Driver version [supported]
|-- GPU 0 [supported]      NVIDIA GeForce RTX 4070 Laptop GPU
|-- GPU 0: VRAM [not enough]           <- 8188 MiB total
System checking result: FAILED
```

The **driver is fine**; the GPU is below IsaacSim 5.1's VRAM threshold. Consistent with that, a
bare `SimulationApp` with no scene at all dies with an access violation in
`rtx.scenedb.plugin.dll`, headless and windowed, with caches cleared and the GPU pinned. The
full GUI reaches "app ready" and then exits. IsaacSim 4.5.0 on the same machine fails
differently (`WinError 1114` loading its bundled `torch\lib\c10.dll`). In WSL the NVIDIA Vulkan
ICD fails `CreateInstance` — driver userspace 595.84 in the rootfs against 596.47 on Windows —
so Vulkan falls back to `llvmpipe` and there is no GPU renderer at all.

**So: run this on a machine with more than 8 GB of VRAM and a working IsaacSim.**

## Route A — the holosoma harness (preferred, needs the training stack)

`box_eval_isaac.py` runs the **trained checkpoint under its own saved config**. That matters:
the config already pins IsaacSim, so the **box is physically present** and the reference motion
is the trained `_w_obj` clip. Neither is true of the MuJoCo testbed.

```bash
python box_eval_isaac.py <checkpoint.pt> out.npz 734          # full 734-frame motion
python box_eval_isaac.py <checkpoint.pt> out.npz 734 --dump-obs   # + per-term obs trace
```

Only three things are changed, none of which touch a frozen policy's actions: demo mode (start
at t=0, no init noise, no early `bad_tracking` reset), reward terms cleared (never evaluated
without a learning update), and observation noise off. Each is justified inline in the file.

`--dump-obs` writes a per-term observation trace. Diffing it against the MuJoCo trace localises
any remaining discrepancy to a specific observation term instead of leaving it to inspection.

Checkpoints on the original machine are under
`SimuAgibot/holosoma/logs/WholeBodyTracking/*/`.

## Route B — standalone, no holosoma

If the training stack is not installed, `isaac_capture_demo.py` loads the USD directly and
reimplements the deployment loop. **It has never been executed** — IsaacSim would not start
here — so treat it as scaffolding, not working code. Every part that is likely to need fixing is
marked `# VERIFY`.

The robot asset is `x2_31dof_w_object_halfspherehand` (~129 MB, includes the box); it is not
committed here. On the original machine it is at
`SimuAgibot/holosoma/src/holosoma/holosoma/data/robots/converted_rank0/`, copied to
`C:\SimuAgibot\x2_usd\`.

## The one thing to check first, in any environment

**Actuator order versus policy joint order.** In the MuJoCo MJCF these differ on 16 of 31
positions, and indices 0–14 (legs and waist) coincide by accident, so the failure looks like
"topples after two seconds" rather than anything obviously wrong. It cost this project a great
deal. Resolve the mapping **by name**, and assert it:

```python
assert [name_of(actuator[i]) for i in range(nu)] == policy_meta["joint_names"]
```

See `../ACC_ADAPTATION_PACKAGE/ERRATA.md` for that and the other two plant defects (zero joint
armature; the deployment safety loop mistaken for the training loop), plus the list of things
already audited clean so they need not be re-checked.

## What to report back

| | frozen | capture-point adapted |
|---|---|---|
| survival (steps before pelvis < 0.35 m) | | |
| completed the 734-frame motion? | | |
| mean leg tracking error (deg) | | |

MuJoCo reference, held-out seeds 800–831: frozen **150.3** steps / 16.41°, adapted **180.2** /
18.08°, **0/32 completed** in both. Tracking getting *worse* while survival improves is expected
and is the mechanism, not a defect — see the results README.
