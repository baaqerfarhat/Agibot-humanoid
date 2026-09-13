"""re4 evidence: the global logging contract (docs/RE4_EVIDENCE_PLAN.md, section 0).

The three templates the plan names (reproduction/audit/run_configuration_template.json,
episode_record_template.csv, timing_and_error_budget_schema.md) are not on this machine, so the
required fields are taken from the plan's own list. No field is left null: a field that does
not apply says why.

  record       finished runner result -> <root>/<part>/<run_id>/run_configuration.json + episodes.csv
  calibration  healthy log + openloop  -> <root>/calibration/<id>.json  (FIR taps H_l, bias c, M, M^-1, SHAs)
  timing       a --timing JSONL        -> latency percentiles, deadline misses, recovery time (E prereg)
"""
from __future__ import annotations
import argparse, csv, datetime, hashlib, json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adaptive_law as AL            # pure helpers only; simulator imports stay inside run()
from mcnemar import mcnemar_exact

SOURCES = ["adaptive_law.py", "so3.py", "joint_fault.py", "libero_reset.py", "gate_faults.py",
           "paired_probe.py", "weighted_dob.py", "mcnemar.py", "error_signal.py", "openloop_id.py",
           "re4_record.py", "aloha_adapt.py", "gr1_adapt.py"]
# Per-runner interface facts, read from the runners (aloha_adapt.py: NJ=14, HORIZON=10, K_FIR=6,
# DT=0.02; gr1_adapt.py: NJ=29, K_FIR=6, 20 Hz, executes --horizon of each 16-step chunk).
RUNNERS = dict(
    libero=dict(script="openpi/adaptive_law.py", rate_hz=20.0, K=None, nj=6,
                units="LIBERO OSC_POSE normalised action units (policy output space)"),
    aloha=dict(script="openpi/aloha_adapt.py", rate_hz=50.0, K=6, nj=14,
               units="absolute joint targets: rad for the 12 arm joints, normalised gripper for joints 6 and 13"),
    gr1=dict(script="openpi/gr1_adapt.py", rate_hz=20.0, K=6, nj=29,
             units="absolute joint targets in rad (29: two 7-joint arms, two 6-joint hands, 3 waist)"))


def runner_of(a):
    if a.get("corr_joints") is not None or a.get("fault_vec", "") and len(str(a.get("fault_vec")).split(",")) in (14, 29):
        return "gr1" if str(a.get("task", "")).startswith("gr1") or "arm:" in str(a.get("fault_vec", "")) else "aloha"
    return "libero"
RATE_HZ = 20.0                       # LIBERO control rate
NAMES = ["x", "y", "z", "rx", "ry", "rz"]


def sha(path):
    p = pathlib.Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else f"missing: {path}"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def gate_convention(a):
    if a.get("static_corr"):
        return "not applicable: fixed correction, no estimator, no deadzone"
    # the same rule holds in all three runners: legacy+zero deadzone zeroes the observation
    # (leakage), legacy+hold skips the update, innovation zeroes its step (hold)
    if a.get("baseline") not in (None, "none"):
        return f"baseline estimator '{a.get('baseline')}': see estimator_step in openpi/adaptive_law.py"
    if a.get("law") == "innov":
        return "full hold (chi=0): below the deadzone the innovation step is zero and f_hat is held"
    if a.get("deadzone_mode") == "hold":
        return "full hold (chi=0): below the deadzone the update is skipped"
    return ("zero-observation leakage (chi=1, s=0): below the deadzone the observation is zeroed, "
            "so f_hat decays by (1-gamma) per step")


def fault_of(a):
    if a.get("joint_fault"):
        return "joint-level (below the controller)", str(a["joint_fault"])
    if a.get("fault_vec"):
        return "additive action offset, per-channel vector", str(a["fault_vec"])
    sev = float(a.get("sev") or 0.0)
    if sev > 0:
        return "additive action offset, uniform on all six channels", f"{sev}"
    return "none (healthy)", "0"


def channels(a):
    if a.get("corr_joints"):
        cj = str(a["corr_joints"])
        if cj.startswith("arm:"):
            return {"left": list(range(0, 7)), "right": list(range(7, 14))}.get(cj.split(":")[1], cj)
        return [int(x) for x in cj.split(",")]
    if a.get("corr_dims"):
        return [int(x) for x in str(a["corr_dims"]).split(",")]
    if a.get("gate_stats"):
        return "healthy-phantom gate, re-evaluated every step (see gate_stats)"
    return list(range(6))


def paired(arms):
    fr = arms.get("frozen_faulted")
    out = {}
    if not fr or not fr.get("per_ep"):
        return out
    kf = {(e["task"], e["init"]): e["ok"] for e in fr["per_ep"]}
    for name, arm in arms.items():
        if name == "frozen_faulted" or not arm.get("per_ep"):
            continue
        ka = {(e["task"], e["init"]): e["ok"] for e in arm["per_ep"]}
        keys = sorted(set(kf) & set(ka))
        b = sum(ka[k] and not kf[k] for k in keys); c = sum(kf[k] and not ka[k] for k in keys)
        out[f"{name}_vs_frozen_faulted"] = dict(pairs=len(keys), frozen=sum(kf[k] for k in keys),
                                               arm=sum(ka[k] for k in keys), fixed=b, broken=c,
                                               exact_mcnemar_p=mcnemar_exact(b, c))
    return out


def cmd_record(ns):
    res = json.loads(ns.result.read_text())
    a = res.get("args") or {}
    if not a:
        sys.exit(f"{ns.result}: no stored 'args'; this tool needs a result from the current runner")
    out = ns.root / ns.part / ns.run_id
    out.mkdir(parents=True, exist_ok=True)
    fam, mag = fault_of(a)
    rn = runner_of(a); RN = RUNNERS[rn]
    K = int(a["fir_k"]) if a.get("fir_k") is not None else (RN["K"] if RN["K"] is not None else AL.K_FIR)
    rate = RN["rate_hz"]
    cfg = dict(
        schema_version="re4-v1", run_id=ns.run_id, part=ns.part, cohort=ns.cohort or a.get("suite"),
        created_utc=now(), result_file=str(ns.result), result_sha256=sha(ns.result),
        runner=RN["script"], interface=rn,
        code_path=("external action subtraction: the frozen policy's action a is sent as a + c with "
                   "c = -f_hat on the corrected channels; the policy network is not edited "
                   "(the native action_out_proj/bias edit exists only in the ACE experiments)"),
        gate_convention=gate_convention(a),
        clipping_order=(("f_hat is updated, then projected onto the box [-clip, clip]^n; the correction is "
                         "-f_hat times the channel mask; clipping precedes masking"
                         + ("; the gripper is never corrected" if rn == "libero" else ""))
                        if not a.get("static_corr") else
                        ("fixed correction times the channel mask; no estimate, no projection"
                         + ("" if rn == "libero" else "; the joint-space runners negate --static-corr internally "
                            "(they are given the positive fault)"))),
        projection_box=float(a.get("clip")) if a.get("clip") is not None else "not recorded",
        units_per_channel=({n: ("LIBERO OSC_POSE normalised action units (policy output space); measured "
                                f"motion is divided by OUT={AL.OUT} to the same units") for n in NAMES}
                           if rn == "libero" else RN["units"]),
        update_law=dict(law=a.get("law", "legacy"), baseline=a.get("baseline", "none"),
                        gamma=a.get("gamma"), deadzone=a.get("dead"), normaliser_rho=a.get("norm_r"),
                        normaliser_channels=a.get("norm_channels", "all"),
                        deadzone_mode=a.get("deadzone_mode", "zero"),
                        bias_subtracted=a.get("bias") or "none: no bias vector subtracted",
                        static_correction=a.get("static_corr") or "none: estimator-driven correction",
                        estimate_only=bool(a.get("estimate_only")),
                        initial_estimate=a.get("f_init") or "zero at every episode start",
                        freeze_after=a.get("freeze_after") if a.get("freeze_after") is not None else "never (updates throughout)",
                        identify_episodes=a.get("identify_episodes") if a.get("identify_episodes") is not None else "none"),
        corrected_channels=channels(a), fir_order_K=K,
        snapshot_application_rule=(f"the policy returns a chunk; "
                                   f"{a.get('replan_steps', a.get('horizon', 10 if rn == 'aloha' else 5))} actions are executed "
                                   "per chunk; the correction is recomputed and applied at every control step "
                                   f"(tau_k = k, {rate:g} Hz)"),
        calibration=dict(healthy_log=a.get("log"), healthy_log_sha256=sha(a["log"]) if a.get("log") else "none",
                         openloop=a.get("openloop"), openloop_sha256=sha(a["openloop"]) if a.get("openloop") else "none",
                         calib_episodes=a.get("calib_episodes") or "all episodes of the healthy log"),
        reset_protocol=(("libero-reset-v1 (libero_reset.reset_libero: forces cleared, cached env seeded per "
                         "scenario, state fingerprint)" if a.get("scenario_reset") else
                         "env.reset() + set_init_state(init) (historical protocol)") if rn == "libero" else
                        ("gym_aloha env.reset(seed=seed+episode)" if rn == "aloha" else
                         "robocasa env rng reseeded per reset: env.unwrapped.env.rng = default_rng(seed+episode) (gr1_adapt.GR1.reset)")),
        policy_rng_pinned=bool(a.get("pin_rng")),
        policy_seed=("server key 0 on every policy call (pin_rng)" if a.get("pin_rng") else "not applicable: pin_rng=False, policy sampling unpinned"),
        fault=dict(family=fam, magnitude=mag, profile=a.get("profile", "step"), onset=a.get("onset", 0)),
        episodes_per_arm=a.get("episodes"), suite=a.get("suite"), eval_init_base=a.get("eval_init", 45),
        source_sha256={f: sha(HERE / f) for f in SOURCES},
        timing_file=str(ns.timing) if ns.timing else "not recorded for this run",
        arms={k: dict(successes=v.get("successes"), n=v.get("n")) for k, v in res["arms"].items()},
        paired=paired(res["arms"]),
        command=" ".join(["python openpi/adaptive_law.py"] +
                         [f"--{k.replace('_', '-')}" + ("" if v is True else f" {v}")
                          for k, v in sorted(a.items()) if v not in (None, False, "")]),
    )
    def nonull(x, path="cfg"):
        if x is None:
            return f"not recorded by the runner ({path})"
        if isinstance(x, dict):
            return {k: nonull(v, f"{path}.{k}") for k, v in x.items()}
        if isinstance(x, list):
            return [nonull(v, f"{path}[{i}]") for i, v in enumerate(x)]
        return x
    cfg = nonull(cfg)
    (out / "run_configuration.json").write_text(json.dumps(cfg, indent=1))
    with open(out / "episodes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "init", "scenario_seed", "policy_seed", "arm", "outcome",
                    "fault_family", "fault_magnitude", "fall_flag", "violation_flag"])
        base = int(a.get("seed") or 0)
        for arm, v in res["arms"].items():
            for e in v.get("per_ep") or []:
                scen = ((f"libero-reset-v1 ({a.get('suite')}, task {e['task']}, init {e['init']})" if a.get("scenario_reset")
                         else "init state only") if rn == "libero" else base + int(e["init"]))
                w.writerow([e["task"], e["init"], scen, ("pinned:key0" if a.get("pin_rng") else "unpinned"), arm, int(bool(e["ok"])),
                            fam, mag, "not applicable (fixed-base arm)", "not applicable (no limit monitor)"])
    print(f"wrote {out}/run_configuration.json and episodes.csv")
    for k, v in cfg["paired"].items():
        print(f"  {k}: {v['frozen']}/{v['pairs']} -> {v['arm']}/{v['pairs']}  {v['fixed']} fixed / "
              f"{v['broken']} broken  p={v['exact_mcnemar_p']:.3g}")


def cmd_calibration(ns):
    if ns.fir_k is not None:
        AL.K_FIR = int(ns.fir_k)
    W = AL.fit_plant(ns.log)
    K = AL.K_FIR
    M = np.array(json.loads(pathlib.Path(ns.openloop).read_text())["M"], float)
    out = ns.root / "calibration"; out.mkdir(parents=True, exist_ok=True)
    doc = dict(schema_version="re4-v1", calibration_id=ns.id, created_utc=now(), interface="LIBERO OSC_POSE, 6 channels",
               fir_order_K=K, H_l=W[:, :K + 1].tolist(), c=W[:, -1].tolist(),
               dc_gain=W[:, :K + 1].sum(axis=1).tolist(), S_M=M.tolist(), M_inv=np.linalg.pinv(M).tolist(),
               cond_M=float(np.linalg.cond(M)), b_cal=ns.bias or "none: the headline law subtracts no bias",
               healthy_log=str(ns.log), healthy_log_sha256=sha(ns.log),
               openloop=str(ns.openloop), openloop_sha256=sha(ns.openloop))
    path = out / f"{ns.id}.json"; path.write_text(json.dumps(doc, indent=1))
    print(f"wrote {path}  (K={K}, cond M={doc['cond_M']:.2f}, DC gain {np.round(doc['dc_gain'], 3)})")


def pct(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return "no samples"
    return dict(n=int(x.size), median=float(np.median(x)), p95=float(np.percentile(x, 95)),
                p99=float(np.percentile(x, 99)), max=float(x.max()))


def cmd_timing(ns):
    rows = [json.loads(l) for l in open(ns.timing) if l.strip()]
    res = json.loads(ns.result.read_text()) if ns.result else {}
    a = res.get("args") or {}
    ch = channels(a) if isinstance(channels(a), list) else list(range(6))
    rate = RUNNERS[runner_of(a)]["rate_hz"] if a else RATE_HZ      # ALOHA runs at 50 Hz, LIBERO and GR1 at 20
    deadline = 1e3 / rate
    out = dict(schema_version="re4-v1", timing_file=str(ns.timing), steps=len(rows), rate_hz=rate,
               deadline_ms=deadline, note=("wall-clock of a simulated robot: policy_ms and env_ms are this "
                                           "machine's GPU/CPU times, not a real-time controller's"))
    for arm in sorted({r["arm"] for r in rows}):
        R = [r for r in rows if r["arm"] == arm]
        rep = [r for r in R if r["replan"]]; non = [r for r in R if not r["replan"]]
        blk = dict(policy_ms_replan_steps=pct([r["policy_ms"] for r in rep]),
                   adapter_ms=pct([r["adapter_ms"] for r in R]), env_ms=pct([r["env_ms"] for r in R]),
                   sensor_to_command_ms=pct([r["s2c_ms"] for r in R]), loop_ms=pct([r["loop_ms"] for r in R]),
                   deadline_misses=dict(all=int(sum(r["loop_ms"] > deadline for r in R)),
                                        replan_steps=int(sum(r["loop_ms"] > deadline for r in rep)),
                                        non_replan_steps=int(sum(r["loop_ms"] > deadline for r in non)),
                                        of_steps=len(R)))
        # recovery (E prereg): e_k = ||(f_true + c)[ch]||, tau = frac * ||f_true[ch]||, sustained W steps
        rec, cens, none = [], 0, 0
        for ep in sorted({r["episode"] for r in R}):
            E = sorted([r for r in R if r["episode"] == ep], key=lambda r: r["t"])
            ft = np.array([r["f_true"] for r in E])[:, ch]; cc = np.array([r["correction"] for r in E])[:, ch]
            fn = np.linalg.norm(ft, axis=1)
            if not np.any(fn > 0):
                none += 1; continue
            err = np.linalg.norm(ft + cc, axis=1); tau = ns.frac * fn
            ok = err < tau; k = next((i for i in range(len(ok) - ns.window + 1) if ok[i:i + ns.window].all()), None)
            if k is None:
                cens += 1; continue
            rec.append(dict(episode=ep, step=int(E[k]["t"]), control_s=E[k]["t"] / rate,
                            wall_s=float(E[k]["wall"] - E[0]["wall"])))
        blk["recovery"] = dict(threshold=f"{ns.frac:g} x ||f_true|| on channels {ch}", window_steps=ns.window,
                               window_s=ns.window / rate,
                               recovered=len(rec), censored=cens, no_fault_episodes=none,
                               control_s=pct([x["control_s"] for x in rec]), wall_s=pct([x["wall_s"] for x in rec]),
                               per_episode=rec)
        out[arm] = blk
    dst = ns.out or ns.timing.with_name("timing_summary.json")
    dst.write_text(json.dumps(out, indent=1))
    print(f"wrote {dst}")
    for arm in [k for k in out if isinstance(out[k], dict) and "adapter_ms" in out[k]]:
        b = out[arm]; am = b["adapter_ms"]; lp = b["policy_ms_replan_steps"]; rc = b["recovery"]
        print(f"  [{arm}] adapter median {am['median']:.3f} ms p99 {am['p99']:.3f} max {am['max']:.3f}; "
              f"policy median {lp['median'] if isinstance(lp, dict) else lp}; "
              f"recovered {rc['recovered']}, censored {rc['censored']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record"); r.add_argument("result", type=pathlib.Path)
    r.add_argument("--part", required=True); r.add_argument("--run-id", required=True)
    r.add_argument("--cohort"); r.add_argument("--timing", type=pathlib.Path)
    r.add_argument("--root", type=pathlib.Path, default=pathlib.Path("results/re4_evidence"))
    c = sub.add_parser("calibration"); c.add_argument("--log", required=True); c.add_argument("--openloop", required=True)
    c.add_argument("--id", required=True); c.add_argument("--bias"); c.add_argument("--fir-k", type=int)
    c.add_argument("--root", type=pathlib.Path, default=pathlib.Path("results/re4_evidence"))
    t = sub.add_parser("timing"); t.add_argument("timing", type=pathlib.Path)
    t.add_argument("--result", type=pathlib.Path); t.add_argument("--out", type=pathlib.Path)
    t.add_argument("--frac", type=float, default=0.2); t.add_argument("--window", type=int, default=20)
    ns = ap.parse_args()
    dict(record=cmd_record, calibration=cmd_calibration, timing=cmd_timing)[ns.cmd](ns)


if __name__ == "__main__":
    main()
