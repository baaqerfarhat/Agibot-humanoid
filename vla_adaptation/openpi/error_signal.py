"""Is the fault OBSERVABLE from the robot's own motion? -- the error an adaptive law needs.

Searching over task success ignores everything we know about the plant. LIBERO's arm runs an
OSC_POSE controller at 20 Hz with `control_delta=True` and
`output_max = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]`, so a commanded action unit is exactly 0.05 m
of end-effector translation (0.5 rad of rotation), and `robot0_eef_pos` is in the observation
every step. The commanded and the achieved motion are therefore both measurable, and their
difference is a direct estimate of an additive action fault:

    e_t = dx_t / 0.05  -  a_cmd,t          ->   f      (for a pure additive fault)

That is the error signal an adaptive law would drive to zero, and it needs no notion of task
success at all. What is NOT obvious is whether it survives real OSC tracking error, contact,
and joint limits -- which is what this measures, by logging commanded vs achieved motion over
episodes with the fault on and off.
"""
from __future__ import annotations

import argparse, collections, datetime, hashlib, json, os, pathlib, tempfile
import numpy as np

from so3 import rot_delta

OUT_MAX = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
FAULT, DIMS = "offset", 6


def log_episode(pr, tid, init, sev, max_steps=None):
    import main as lm
    from openpi_client import image_tools
    from gate_faults import apply_action_fault
    import paired_probe as _pp
    max_steps = max_steps or _pp.MAXS
    env, desc, inits = pr.env_for(tid)
    from libero_reset import reset_libero
    obs, _ = reset_libero(env, inits[init], suite=pr.suite_name, task=tid, init=init)
    plan, t, rec = collections.deque(), 0, []
    while t < max_steps + 10:
        if t < 10:
            obs, _, done, _ = env.step(lm.LIBERO_DUMMY_ACTION); t += 1; continue
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
        wr = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
        if not plan:
            el = {"observation/image": img, "observation/wrist_image": wr,
                  "observation/state": np.concatenate((obs["robot0_eef_pos"],
                      lm._quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                  "prompt": str(desc)}
            plan.extend(pr.client.infer(el)["actions"][: pr.a.replan_steps])
        a_cmd = np.asarray(plan.popleft(), float)
        a_exec = apply_action_fault(a_cmd, FAULT, sev, DIMS)
        x0 = np.array(obs["robot0_eef_pos"], float)
        q0 = np.array(obs["robot0_eef_quat"], float)
        obs, _, done, _ = env.step(a_exec.tolist())
        x1 = np.array(obs["robot0_eef_pos"], float)
        q1 = np.array(obs["robot0_eef_quat"], float)
        rec.append(dict(a_cmd=a_cmd[:6].tolist(), a_exec=a_exec[:6].tolist(),
                        command=a_cmd.tolist(), executed=a_exec.tolist(),
                        dx=(x1 - x0).tolist(), dr=rot_delta(q0, q1).tolist()))
        if done:
            break
        t += 1
    return rec, bool(done)


def analyse(recs, sev, label):
    if not recs or any(not episode for episode in recs):
        raise ValueError("calibration episodes must contain at least one policy action")
    A = np.array([r["a_cmd"] for e in recs for r in e])
    E = np.array([r["a_exec"] for e in recs for r in e])
    D = np.array([r["dx"] + r["dr"] for e in recs for r in e])
    ach = D / OUT_MAX                       # achieved motion, in action units
    err = ach - A                           # what an adaptive law would see
    true = E - A                            # the fault actually injected
    print(f"\n=== {label}  ({len(recs)} episodes, {len(A)} steps)")
    print(f"{'dim':<5} {'true fault':>11} {'mean e_t':>10} {'sd e_t':>9} {'SNR':>7}")
    for i, nm in enumerate(["dx", "dy", "dz", "drx", "dry", "drz"]):
        m, s = err[:, i].mean(), err[:, i].std()
        print(f"{nm:<5} {true[:, i].mean():>11.4f} {m:>10.4f} {s:>9.4f} "
              f"{abs(m)/max(s,1e-9):>7.2f}")
    return dict(label=label, sev=sev, n_steps=int(len(A)),
                true=true.mean(0).tolist(), mean_e=err.mean(0).tolist(),
                sd_e=err.std(0).tolist(),
                # raw pairs, so the command->motion map can be IDENTIFIED offline rather
                # than assumed from output_max (which is the target scaling, not the
                # realised one: nominal e_t is -0.175 on dx, so the arm achieves only a
                # fraction of the commanded delta within one 50 ms control step)
                raw_a=[r["a_cmd"] for e in recs for r in e],
                raw_exec=[r["a_exec"] for e in recs for r in e],
                raw_cmd=[r.get("command", r["a_cmd"]) for e in recs for r in e],
                raw_executed=[r.get("executed", r["a_exec"]) for e in recs for r in e],
                raw_d=[r["dx"] + r["dr"] for e in recs for r in e],
                ep_len=[len(e) for e in recs])


def atomic_json(path, payload):
    """Save a complete artifact snapshot, including new parent directories."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=1, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def calibration_records(payload):
    """Read historical bare lists and the provenance-carrying records schema."""
    if isinstance(payload, dict):
        if "status" in payload and payload["status"] != "complete":
            raise ValueError("calibration artifact is not complete")
        records = payload.get("records")
    else:
        records = payload
    if not isinstance(records, list) or not records or any(not isinstance(r, dict) for r in records):
        raise ValueError("calibration requires a nonempty list of records")
    return records


def source_hashes(paths):
    return {str(pathlib.Path(path).resolve()): hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
            for path in paths}


def collect(a, pr, episode_logger=None):
    """Collect complete conditions without changing the records container's type."""
    reset_protocol = "libero-reset-v1" if episode_logger is None else "injected_episode_logger"
    episode_logger = episode_logger or log_episode
    count = pr.suite.n_tasks
    if count < 1 or a.episodes < 1 or a.init_base < 0:
        raise ValueError("suite, episode count, and initial-state index must be valid")
    scenarios = [[int(k % count), int(a.init_base + k // count)] for k in range(a.episodes)]
    for task, init in scenarios:
        _, _, inits = pr.env_for(task)
        if init >= len(inits):
            raise ValueError(f"task {task} has {len(inits)} initial states; requested {init}")
    folder = pathlib.Path(__file__).resolve().parent
    payload = dict(schema_version=2, status="planned", records=[], calib_episodes=scenarios,
                   init_base=int(a.init_base), suite=a.suite, reset_protocol=reset_protocol,
                   args={key: str(value) if isinstance(value, pathlib.Path) else value
                         for key, value in vars(a).items()},
                   created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   source_hashes=source_hashes([folder / name for name in
                       ("error_signal.py", "paired_probe.py", "gate_faults.py", "so3.py", "libero_reset.py")]),
                   setup_probe=dict(task=0, init=8, scored=False), policy_rng_pinned=False)
    atomic_json(a.out, payload)
    try:
        payload["policy_control_ack"] = pr.control(dict(site=None, pin_rng=False))
        payload["status"] = "running"
        conditions = [(0.0, "NOMINAL (no fault)")]
        if not a.healthy_only:
            conditions.append((0.05, "FAULTED (+0.05 on arm dims)"))
        for sev, label in conditions:
            recs, outcomes = [], []
            for k, (task, init) in enumerate(scenarios):
                rec, success = episode_logger(pr, task, init, sev, max_steps=a.max_steps)
                recs.append(rec)
                outcomes.append(dict(task=task, init=init, ok=bool(success), steps=len(rec)))
                print(f"  {label}: episode {k} -> {len(rec)} steps, success={success}")
            record = analyse(recs, sev, label)
            record.update(episode_keys=scenarios, per_ep=outcomes)
            payload["records"].append(record)
            atomic_json(a.out, payload)
        payload["status"] = "complete"
        atomic_json(a.out, payload)
    except BaseException as exc:
        payload.update(status="failed", error=dict(type=type(exc).__name__, message=str(exc)))
        atomic_json(a.out, payload)
        raise
    return payload


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--control", type=pathlib.Path, required=True)
    p.add_argument("--ack", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--host", default="0.0.0.0"); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--healthy-only", action="store_true",
                   help="collect only the nominal episodes needed for FIR identification")
    p.add_argument("--max-steps", type=int, help="policy action cap; default is the suite cap")
    p.add_argument("--init-base", type=int, default=45,
                   help="first initial state; keep DISJOINT from evaluation states")
    a = p.parse_args(argv)
    if a.episodes < 1 or a.init_base < 0 or a.replan_steps < 1 or (a.max_steps is not None and a.max_steps < 1):
        p.error("episode count, initial state, and step counts must be valid")
    if a.out.exists():
        p.error("--out already exists; use a fresh path to preserve calibration provenance")
    from paired_probe import Probe, SUITE_MAX
    if a.max_steps is None:
        a.max_steps = SUITE_MAX.get(a.suite, 220)
    pr = Probe(a)
    try:
        out = collect(a, pr)
    finally:
        pr.close()
    if a.healthy_only:
        return out
    n, f = calibration_records(out)
    print("\n=== IDENTIFIABILITY: faulted minus nominal, per dim")
    print(f"{'dim':<5} {'injected':>10} {'recovered':>10} {'ratio':>8}")
    for i, nm in enumerate(["dx", "dy", "dz", "drx", "dry", "drz"]):
        rec = f["mean_e"][i] - n["mean_e"][i]
        print(f"{nm:<5} {0.05:>10.4f} {rec:>10.4f} {rec/0.05:>8.2f}")


if __name__ == "__main__":
    main()
