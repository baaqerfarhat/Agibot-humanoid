"""re4 theory plan, Part 8.1: task-clustered bootstrap and cluster sign-flip test for the headline."""
import json, pathlib, sys
import numpy as np

B, SEED = 20000, 0
SETS = {"shipped": {"libero_spatial": "results/suites/libero_spatial_rotonly_paired.json", "libero_object": "results/suites/libero_object_rotonly_paired.json",
                    "libero_goal": "results/suites/libero_goal_rotonly_n40.json", "libero_10": "results/suites/libero_10_rotonly_n40.json"},
        "heldout": {s: f"results/heldout/{s}_rotonly_heldout.json" for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10")}}


def clusters(path, suite):
    d = json.loads(pathlib.Path(path).read_text())["arms"]
    fr = {(e["task"], e["init"]): e["ok"] for e in d["frozen_faulted"]["per_ep"]}; ad = {(e["task"], e["init"]): e["ok"] for e in d["adaptive"]["per_ep"]}
    out = {}
    for k in fr:
        out.setdefault((suite, k[0]), []).append(int(ad[k]) - int(fr[k]))
    return out


def stats(cl, rng):
    keys = list(cl); sums = np.array([sum(cl[k]) for k in keys], float); n = len(keys)
    tot = sums.sum()
    boot = np.array([sums[rng.integers(0, n, n)].sum() for _ in range(B)])
    flips = np.array([(sums * rng.choice([-1, 1], n)).sum() for _ in range(B)])
    p = float((np.abs(flips) >= abs(tot)).mean())
    diffs = np.concatenate([np.array(v, float) for v in cl.values()])
    iid = np.array([diffs[rng.integers(0, diffs.size, diffs.size)].sum() for _ in range(B)])
    return dict(clusters=n, pairs=int(diffs.size), effect=int(tot), ci95_cluster=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                ci95_iid=[float(np.percentile(iid, 2.5)), float(np.percentile(iid, 97.5))], signflip_p=p)


def main():
    rng = np.random.default_rng(SEED); out = {}
    for cal, files in SETS.items():
        allc = {}
        for suite, path in files.items():
            cl = clusters(path, suite); allc.update(cl); out[f"{cal}:{suite}"] = stats(cl, rng)
        out[f"{cal}:pooled"] = stats(allc, rng)
    for k, v in out.items():
        w = (v["ci95_cluster"][1] - v["ci95_cluster"][0]) / max(v["ci95_iid"][1] - v["ci95_iid"][0], 1e-9)
        print(f"{k:24s} clusters {v['clusters']:2d} pairs {v['pairs']:3d} effect {v['effect']:+3d}  cluster CI {v['ci95_cluster'][0]:+6.1f},{v['ci95_cluster'][1]:+6.1f}  iid CI {v['ci95_iid'][0]:+6.1f},{v['ci95_iid'][1]:+6.1f}  width ratio {w:.2f}  sign-flip p {v['signflip_p']:.4f}")
    p = pathlib.Path("results/re4_theory/8_statistics/clustered_bootstrap.json"); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(B=B, seed=SEED, results=out), indent=1)); print("wrote", p)


if __name__ == "__main__":
    main()
