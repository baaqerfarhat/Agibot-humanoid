#!/usr/bin/env python3
"""Stage 2 telemetry summary (PREREG_FYA_RECOVERY_DEADLINE_V1.md section 6: estimate transients, cap activity,
episode lengths and runtime), per arm. Reads the adaptive_law.py telemetry (.jsonl or .jsonl.gz) of each arm.

Per adaptive arm: final r_y estimate (median, IQR over the 20 keys), steps after enablement until the r_y estimate
first exceeds half the fault, fraction of enabled steps with |f_hat_ry| at the cap (.05), and the r_y estimate
during the delayed window (must be zero for the delayed arms). Per arm: episode length (policy steps) median and
mean, and wall time per episode from the log timestamps if present.
"""
import argparse, gzip, json, pathlib
import numpy as np

ARMS = ["healthy_off", "healthy_nt", "healthy_innovation", "fault_off", "fault_nt", "fault_innovation", "delay_nt", "delay_innovation"]


def read(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("results/frozen_yet_adaptive_deadline_v1/reacting_policy"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/frozen_yet_adaptive_deadline_v1/analysis/stage2_telemetry.json")); a = ap.parse_args()
    out = {}
    for arm in ARMS:
        p = next((a.root / f"{arm}_telemetry.jsonl{s}" for s in ("", ".gz") if (a.root / f"{arm}_telemetry.jsonl{s}").exists()), None)
        if p is None:
            continue
        eps = {}
        for r in read(p):
            if r.get("type") != "step" or r.get("phase") != "rollout":
                continue
            eps.setdefault(r["episode"], []).append((r["t"] - 10, float(np.asarray(r["f_hat"])[4]), bool(r.get("live")), float(np.asarray(r["f_true"])[4])))
        adapt_from = 40 if arm.startswith("delay") else 30
        lengths = [len(v) for v in eps.values()]; finals = [v[-1][1] for v in eps.values()]
        rise = []; at_cap = []; pre = []
        for v in eps.values():
            f = [x for x in v if x[0] >= adapt_from]; pre += [x[1] for x in v if x[0] < adapt_from]
            if f:
                at_cap.append(float(np.mean([abs(x[1]) >= 0.05 - 1e-9 for x in f])))
                k = next((i for i, x in enumerate(f) if x[1] > 0.025), None); rise.append(k)
        out[arm] = dict(n_episodes=len(eps), episode_policy_steps=dict(median=float(np.median(lengths)), mean=float(np.mean(lengths)), min=int(min(lengths)), max=int(max(lengths))),
                        final_fhat_ry=dict(median=float(np.median(finals)), q25=float(np.percentile(finals, 25)), q75=float(np.percentile(finals, 75))),
                        steps_to_half_fault_after_enable=dict(median=(float(np.median([x for x in rise if x is not None])) if any(x is not None for x in rise) else None), never=int(sum(x is None for x in rise))),
                        fraction_enabled_steps_at_cap=float(np.mean(at_cap)) if at_cap else None,
                        max_abs_fhat_ry_before_enable=float(max(abs(x) for x in pre)) if pre else 0.0, adapt_from_policy_step=adapt_from)
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
