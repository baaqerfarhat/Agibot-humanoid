"""Can a computed edit on ONE layer actually repair the fault? The reachability ceiling.

This is the gate the framework puts before any search: if a hand-computed repair cannot
recover the headroom, no search over the same class can either, and 400 episodes of CEM
would be spent against a ceiling that is not there.

The repair is NOT a scalar, which is the part that is easy to get wrong. pi0.5 emits
NORMALISED actions and `Unnormalize` runs afterwards, so an edit at action_out_proj/bias
lives in normalised units while the fault is applied in env units. With quantile norm,
env = (norm+1)/2 * (q99-q01) + q01, so the per-dim scale is (q99-q01)/2 -- and that scale
differs 7x across the arm channels. A +0.05 env offset is 3.0% of the action range on
translation and 19.5% on drx. Cancelling it needs

    beta_i = (0.05 / scale_i) / 0.34          (0.34 = the trained flow's measured
                                               attenuation of a bias edit, §0)

which spans 0.157 .. 1.149 -- 29x to 193x the bias vector's own RMS of 0.0059.

Sweeping a scale k on that vector measures the ceiling and checks the 0.34 attenuation in
closed loop, where §0 only measured it open-loop on a synthetic observation.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

import ace_screen
from ace_screen import Runner

FAULT_ENV = 0.05
ATTEN = 0.34
NORM_STATS = ("/home/mtaheri/.cache/openpi/openpi-assets/checkpoints/pi05_libero/"
              "assets/physical-intelligence/libero/norm_stats.json")


def oracle_beta(n_dims: int = 7) -> np.ndarray:
    d = json.load(open(NORM_STATS))
    n = d.get("norm_stats", d)["actions"]
    q01 = np.array(n["q01"][:n_dims], float)
    q99 = np.array(n["q99"][:n_dims], float)
    scale = (q99 - q01) / 2.0
    return (FAULT_ENV / scale) / ATTEN


def episodes_for(n: int):
    """Initial states 8-9, disjoint from gate (0-1), screen (2-5) and baseline (6-7)."""
    plan = [(t, 8) for t in range(10)] + [(t, 9) for t in range(10)]
    return plan[:n]


class OracleRunner(Runner):
    def set_bias(self, add, tag):
        self.a.ack.unlink(missing_ok=True)
        self.a.control.write_text(json.dumps(dict(bias_add=list(map(float, add)), draw=tag)))
        t0 = time.time()
        while True:
            self.probe()
            if self.a.ack.exists():
                ack = json.loads(self.a.ack.read_text())
                if ack.get("site") == "bias_add":
                    break
            if time.time() - t0 > 300:
                raise RuntimeError("server never acked the bias edit")
        if not ack.get("ok"):
            raise RuntimeError(f"bias edit not live: {ack}")
        return ack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--control", type=pathlib.Path, required=True)
    p.add_argument("--ack", type=pathlib.Path, required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--replan-steps", type=int, default=5)
    a = p.parse_args()

    beta = oracle_beta()
    print("oracle beta (arm dims + grip):", np.round(beta, 3))
    r = OracleRunner(a)
    eps = episodes_for(a.n)

    # (label, scale on beta, apply the env fault?)
    conds = [("k=0.0 faulted (floor)", 0.0, True),
             ("k=0.5", 0.5, True),
             ("k=1.0 oracle", 1.0, True),
             ("k=1.5", 1.5, True),
             ("k=-1.0 wrong sign", -1.0, True),
             ("k=1.0 NO fault", 1.0, False)]

    out = []
    if a.out.exists():
        out = json.loads(a.out.read_text())["conds"]
        print(f"resuming with {len(out)} conditions done")
    done = {c["label"] for c in out}

    for label, k, faulted in conds:
        if label in done:
            continue
        add = np.zeros(7)
        add[:6] = k * beta[:6]          # arm dims only; the gripper channel is untouched
        ack = r.set_bias(add, label)
        ace_screen.SEVERITY = FAULT_ENV if faulted else 0.0
        t0 = time.time()
        succ = sum(r.episode(t, i) for t, i in eps)
        rec = dict(label=label, k=k, faulted=faulted, n=len(eps), successes=succ,
                   success_rate=succ / len(eps), bias_l2=ack.get("applied_l2"),
                   wall_s=round(time.time() - t0, 1))
        out.append(rec)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(beta=list(beta), conds=out), indent=1))
        print(f"{label:<24} {succ}/{len(eps)} = {100*rec['success_rate']:>5.1f}%  ({rec['wall_s']}s)")
    print("=== ORACLE SWEEP DONE")


if __name__ == "__main__":
    main()
