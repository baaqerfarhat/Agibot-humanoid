"""Backend check for cached LIBERO resets; no policy server is contacted."""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from libero_reset import physics_arrays, physics_fingerprint, reset_libero


def differences(reference, current):
    return {key: float(np.max(np.abs(current[key] - value))) if value.size else 0.
            for key, value in reference.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", type=int, default=4)
    parser.add_argument("--init", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    import main as lm
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[args.suite]()
    state = suite.get_task_init_states(args.task)[args.init]
    env, _ = lm._get_libero_env(suite.get_task(args.task), 32, 7)
    result = dict(args={k: str(v) if isinstance(v, pathlib.Path) else v
                        for k, v in vars(args).items()}, modes={})
    dummy = np.array([0., 0., 0., 0., 0., 0., -1.])
    try:
        for mode in ("legacy", "scenario_seeded"):
            reference_reset = reference_warmup = None
            rows = []
            for repeat in range(args.repeats):
                env.sim.data.qfrc_applied[:] = 0.
                env.sim.data.xfrc_applied[:] = 0.
                if mode == "legacy":
                    env.reset()
                    env.set_init_state(state)
                    provenance = None
                else:
                    _, provenance = reset_libero(env, state, suite=args.suite,
                                                  task=args.task, init=args.init)
                initial = physics_arrays(env)
                if reference_reset is None:
                    reference_reset = initial
                for _ in range(10):
                    env.step(dummy)
                warmup = physics_arrays(env)
                if reference_warmup is None:
                    reference_warmup = warmup
                rows.append(dict(repeat=repeat, reset_differences=differences(reference_reset, initial),
                                 warmup_differences=differences(reference_warmup, warmup),
                                 warmup_fingerprint=physics_fingerprint(env), provenance=provenance))
                # The same cached environment has moved and global sampling has advanced.
                for _ in range(5):
                    env.step(np.array([.2, -.1, .1, .1, 0., -.1, -1.]))
                np.random.random(31 + repeat)
            result["modes"][mode] = rows
        seeded = result["modes"]["scenario_seeded"]
        result["seeded_exact"] = all(v == 0. for row in seeded
            for stage in ("reset_differences", "warmup_differences") for v in row[stage].values())
        result["legacy_maxima"] = {stage: {k: max(r[stage][k] for r in result["modes"]["legacy"])
                                             for k in result["modes"]["legacy"][0][stage]}
                                   for stage in ("reset_differences", "warmup_differences")}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(json.dumps({"seeded_exact": result["seeded_exact"],
                          "legacy_nonzero_maxima": {stage: {k: v for k, v in values.items() if v}
                              for stage, values in result["legacy_maxima"].items()}}), flush=True)
        if not result["seeded_exact"]:
            raise RuntimeError("scenario seeded reset failed exact full-physics repetition")
    finally:
        env.close()


if __name__ == "__main__":
    main()
