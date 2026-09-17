#!/usr/bin/env python3
"""Extraction for the crossed command-stream replay (papers/frozen_yet_adaptive_closed_loop/EXPERIMENT_PLAN.md §3;
prereg_records/PREREG_FYA_CROSSED_REPLAY_V1.md). Read-only on the archived Stage 2 telemetry.

For every Stage 2 key (task, init, sampler seed) and the three arms healthy_off / fault_off / fault_nt, extract the
seven-dimensional raw policy actions of the warm-up, the thirty-step common prefix and the post-prefix continuation,
the recorded physical states (position, quaternion, joint positions), the recorded corrections and estimates
(fault_nt), and the deployed configuration from the fault_nt telemetry header (W, M, mask, constants). Eligibility:
every arm has >= prefix + window policy steps and the three arms' prefixes agree exactly on raw actions, positions and
joints (the strict common-reference rule). Writes extracted_streams/<task>_<init>_<seed>.json, source_keys.csv and
extraction_audit.json with the byte hashes of every input. Nothing is simulated or modified.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, pathlib
import numpy as np

ARMS = ("healthy_off", "fault_off", "fault_nt")
OPTIONAL_ARMS = ("delay_nt", "fault_innovation", "healthy_nt")   # optional matrices (delayed NT, innovation, healthy control)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def read_arm(path):
    header = None; eps = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("type") != "step":
                if header is None and "config" in r:
                    header = r
                continue
            eps.setdefault((r["task"], r["init"]), []).append(r)
    return header, eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=pathlib.Path, default=pathlib.Path("results/frozen_yet_adaptive_deadline_v1/reacting_policy"))
    ap.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("results/frozen_yet_adaptive_deadline_v1/stage2_manifest.json"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/fya_crossed_replay_v1"))
    ap.add_argument("--prefix", type=int, default=30); ap.add_argument("--window", type=int, default=50); a = ap.parse_args()
    man = json.loads(a.manifest.read_text()); keys = [(s["task"], s["init"], s["sampler_seed"]) for s in man["scenarios"]]
    data = {}; hashes = {str(a.manifest): sha(a.manifest)}; headers = {}
    for arm in ARMS + OPTIONAL_ARMS:
        p = a.src / f"{arm}_telemetry.jsonl.gz"
        if not p.exists():
            continue
        hashes[str(p)] = sha(p); headers[arm], data[arm] = read_arm(p)
    cfg_hdr = headers["fault_nt"]; cfg = dict(W=cfg_hdr["config"]["W"], M=cfg_hdr["config"]["M"], mask=cfg_hdr["config"]["mask"], fault_vector=cfg_hdr["config"]["fault_vector"],
                                            args={k: cfg_hdr["args"][k] for k in ("gamma", "dead", "norm_r", "clip", "corr_dims", "norm_channels", "law", "onset", "adapt_from", "fault_vec", "scenario_reset", "deadzone_mode", "replan_steps")},
                                            constants=cfg_hdr["constants"], source="fault_nt telemetry header (deployed arrays)", runner_source_sha256=hashlib.sha256(cfg_hdr["runner_source"].encode()).hexdigest())
    (a.out / "extracted_streams").mkdir(parents=True, exist_ok=True)
    (a.out / "configuration.json").write_text(json.dumps(cfg, indent=1))
    rows = []
    for (t, i, seed) in keys:
        rec = {}; ok = True; reasons = []; opt_ok = {}
        for arm in ARMS + tuple(x for x in OPTIONAL_ARMS if x in data):
            steps = data[arm].get((t, i))
            if not steps:
                if arm in ARMS:
                    ok = False; reasons.append(f"{arm}: missing")
                continue
            warm = [s for s in steps if s["phase"] == "warmup"]; roll = [s for s in steps if s["phase"] == "rollout"]
            roll.sort(key=lambda s: s["t"])
            assert [s["t"] for s in roll] == list(range(10, 10 + len(roll))), (arm, t, i, "non-contiguous rollout steps")
            rec[arm] = dict(n_policy_steps=len(roll), warmup_commands=[s["command"] for s in warm],
                            raw_action=[s["raw_action"] for s in roll], correction=[s["correction"] for s in roll], nominal_command=[s["nominal_command"] for s in roll],
                            command=[s["command"] for s in roll], position=[s["position"] for s in roll], quaternion=[s["quaternion"] for s in roll],
                            joint_position=[s["joint_position"] for s in roll], joint_before=[s.get("joint_before") for s in roll],
                            f_hat_before=[s["f_hat_before"] for s in roll], f_hat_after=[s["f_hat"] for s in roll], f_true=[s["f_true"] for s in roll],
                            live=[s["live"] for s in roll], done=[s["done"] for s in roll], measured=[s["measured"] for s in roll], r=[s["r"] for s in roll],
                            success_recorded=bool(any(s["done"] for s in roll)))
            if len(roll) < a.prefix + a.window:
                if arm in ARMS:
                    ok = False; reasons.append(f"{arm}: {len(roll)} < {a.prefix + a.window} policy steps")
                else:
                    opt_ok[arm] = False
        if all(arm in rec for arm in ARMS):
            ref = rec["fault_off"]
            for arm in ("healthy_off", "fault_nt") + tuple(x for x in OPTIONAL_ARMS if x in rec):
                for field, tol in (("raw_action", 0.0), ("position", 0.0), ("joint_position", 0.0)):
                    n = min(a.prefix, rec[arm]["n_policy_steps"], ref["n_policy_steps"])
                    gapv = float(np.max(np.abs(np.array(rec[arm][field][:n]) - np.array(ref[field][:n])))) if n else float("inf")
                    rec[arm][f"prefix_{field}_max_gap_vs_fault_off"] = gapv
                    if gapv > tol:
                        if arm in ARMS:
                            ok = False; reasons.append(f"{arm}: prefix {field} gap {gapv:.3e} vs fault_off")
                        else:
                            opt_ok[arm] = False
            for arm in OPTIONAL_ARMS:
                if arm in rec:
                    opt_ok.setdefault(arm, rec[arm]["n_policy_steps"] >= a.prefix + a.window)
                    if arm == "delay_nt" and (np.any(np.abs(np.array(rec[arm]["correction"][:a.prefix + 10])) > 0) or np.any(np.abs(np.array(rec[arm]["f_hat_before"][:a.prefix + 10])) > 0)):
                        opt_ok[arm] = False
            # corrections in the prefix must be zero and the pre-update estimate at the first window step must be zero (fault_nt)
            if np.any(np.abs(np.array(rec["fault_nt"]["correction"][:a.prefix])) > 0) or np.any(np.abs(np.array(rec["fault_nt"]["f_hat_before"][a.prefix])) > 0):
                ok = False; reasons.append("fault_nt: correction or estimate nonzero inside the prefix")
            for arm in ARMS:
                if arm != "healthy_off" and not all(rec[arm]["live"][a.prefix:a.prefix + a.window]):
                    ok = False; reasons.append(f"{arm}: fault not live throughout the window")
        key_id = f"{t}_{i}_{seed}"
        out = dict(key=dict(task=t, init=i, sampler_seed=seed), eligible=ok, reasons=reasons, optional_eligible=opt_ok, prefix=a.prefix, window=a.window, arms=rec)
        (a.out / "extracted_streams" / f"{key_id}.json").write_text(json.dumps(out))
        rows.append(dict(task=t, init=i, sampler_seed=seed, eligible=ok, reasons="; ".join(reasons), **{f"len_{arm}": rec[arm]["n_policy_steps"] if arm in rec else None for arm in ARMS},
                         **{f"eligible_{arm}": (ok and opt_ok.get(arm, False)) for arm in OPTIONAL_ARMS},
                         stream_sha256_M0=hashlib.sha256(json.dumps(rec["fault_off"]["raw_action"]).encode()).hexdigest() if "fault_off" in rec else None,
                         stream_sha256_M1=hashlib.sha256(json.dumps(rec["fault_nt"]["raw_action"]).encode()).hexdigest() if "fault_nt" in rec else None,
                         stream_sha256_ref=hashlib.sha256(json.dumps(rec["healthy_off"]["raw_action"]).encode()).hexdigest() if "healthy_off" in rec else None))
    with open(a.out / "source_keys.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    audit = dict(inputs_sha256=hashes, extractor_sha256=sha(__file__), n_keys=len(rows), n_eligible=int(sum(r["eligible"] for r in rows)),
                 eligible_keys=[[r["task"], r["init"], r["sampler_seed"]] for r in rows if r["eligible"]], ineligible=[{k: r[k] for k in ("task", "init", "sampler_seed", "reasons")} for r in rows if not r["eligible"]],
                 warmup_command=data["fault_off"][keys[0][:2]][0]["command"], prefix=a.prefix, window=a.window,
                 configuration_sha256=sha(a.out / "configuration.json"), note="read-only extraction; prefix agreement exact (tolerance 0) on raw actions, positions and joints across the three arms")
    (a.out / "extraction_audit.json").write_text(json.dumps(audit, indent=1)); print(json.dumps({k: v for k, v in audit.items() if k != "inputs_sha256"}, indent=1))


if __name__ == "__main__":
    main()
