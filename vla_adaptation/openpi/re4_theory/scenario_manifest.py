#!/usr/bin/env python3
"""Scenario manifest for a locked confirmation partition (unified plan, shared protocol items 1 and 3).

Enumerates every (suite, task, init) already used by a stored LIBERO result or calibration log,
then picks, per task, the requested number of UNUSED stored initial states by a fixed rule
(the largest unused indices below 50, descending) without looking at any outcome. If a task has
too few unused states the manifest says so and reuses the largest used ones with fresh sampler
seeds, flagged per scenario (`reused_state: true`). Sampler seeds are assigned per (task, init,
replicate) from a declared base. Output: JSON with the fields a paired key needs (robot, suite,
task, init, sampler_seed, horizon, fault profile placeholder) plus the exclusion ledger.
"""
import argparse, collections, glob, hashlib, json, pathlib
HERE = pathlib.Path(__file__).resolve().parents[2]
N_INIT = 50


def used_states(root):
    used = collections.defaultdict(set); sources = collections.defaultdict(set)
    for f in glob.glob(str(root / "results/**/*.json"), recursive=True):
        try:
            r = json.loads(pathlib.Path(f).read_text())
        except Exception:
            continue
        if isinstance(r, dict) and "arms" in r:
            a = r.get("args") or {}; suite = a.get("suite")
            if not suite and "/suites/" in f:
                suite = pathlib.Path(f).stem.split("_rotonly")[0].split("_adaptive")[0].split("_transonly")[0].split("_nofault")[0]
            if not suite or not str(suite).startswith("libero"):
                continue
            for arm in r["arms"].values():
                for ep in (arm.get("per_ep") or []):
                    if isinstance(ep, dict) and "task" in ep and "init" in ep:
                        used[suite].add((int(ep["task"]), int(ep["init"]))); sources[suite].add(str(pathlib.Path(f).relative_to(root)))
        elif isinstance(r, dict) and r.get("calib_episodes"):
            suite = r.get("suite") or "libero_spatial"   # calibration logs are spatial unless they say otherwise
            for t, i in r["calib_episodes"]:
                used[suite].add((int(t), int(i))); sources[suite].add(str(pathlib.Path(f).relative_to(root)))
    # the shipped calibration (bare-list log) probed init 45 on tasks 0-2 of libero_spatial
    used["libero_spatial"] |= {(t, 45) for t in range(3)}; sources["libero_spatial"].add("results/phase05/error_signal_so3.json (init base 45, 3 episodes)")
    return used, sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True); ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--inits-per-task", type=int, default=2); ap.add_argument("--replicates", type=int, default=3,
                    help="sampler seeds per (task, init)")
    ap.add_argument("--seed-base", type=int, default=1000); ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--fault", default="uniform +0.05 on six action channels")
    ap.add_argument("--out", type=pathlib.Path, required=True); a = ap.parse_args()
    lo, hi = (int(x) for x in a.tasks.split("-")); tasks = list(range(lo, hi + 1))
    used, sources = used_states(HERE)
    su = used.get(a.suite, set()); scenarios = []; ledger = {}
    for t in tasks:
        taken = {i for tt, i in su if tt == t}; free = [i for i in range(N_INIT - 1, -1, -1) if i not in taken]
        chosen = free[:a.inits_per_task]; reused = []
        if len(chosen) < a.inits_per_task:
            reused = sorted(taken, reverse=True)[:a.inits_per_task - len(chosen)]
        ledger[str(t)] = dict(used_states=sorted(taken), chosen_unused=chosen, reused_states=reused)
        for k, init in enumerate(chosen + reused):
            for rep in range(a.replicates):
                seed = a.seed_base + 100 * t + 10 * k + rep
                scenarios.append(dict(robot="Panda (LIBERO, OSC_POSE)", suite=a.suite, task=t, init=int(init),
                                      sampler_seed=int(seed), replicate=rep, reused_state=init in reused,
                                      horizon=a.horizon, fault=a.fault))
    out = dict(rule="per task, the largest unused stored initial states below 50, descending; outcomes never inspected; "
                    "sampler seed = seed_base + 100*task + 10*state_rank + replicate",
               suite=a.suite, n_scenarios=len(scenarios), inits_per_task=a.inits_per_task, replicates=a.replicates,
               seed_base=a.seed_base, exclusion_sources=sorted(sources.get(a.suite, [])), ledger=ledger, scenarios=scenarios)
    js = json.dumps(out, indent=1); out["manifest_sha256"] = hashlib.sha256(js.encode()).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1))
    print(f"{a.suite}: {len(scenarios)} scenarios; per task chosen/reused:",
          {t: (v["chosen_unused"], v["reused_states"]) for t, v in ledger.items()})


if __name__ == "__main__":
    main()
