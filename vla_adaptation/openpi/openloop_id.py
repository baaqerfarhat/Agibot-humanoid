"""Open-loop replay: is the fault identifiable when NO policy reacts to it?

The closed-loop residual failed to recover the fault even with a plant model explaining 98%
of translation motion. Two explanations compete: the plant model is inadequate, or the
faulted data is off-distribution because the policy compensates and drives the arm through
different states.

Replay separates them. A recorded command sequence is played back open loop -- no policy, no
feedback -- with and without the fault added. The commands are then IDENTICAL by construction,
so any difference in the achieved motion is the fault propagating through the plant, and
d(motion)/d(fault) is measurable directly. That derivative is the map an adaptive law needs.

No inference is involved, so this runs on CPU in seconds.
"""
from __future__ import annotations

import argparse, datetime, hashlib, json, pathlib
import numpy as np
from so3 import rot_delta
from error_signal import atomic_json, calibration_records, source_hashes

OUT = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])


def replay(env, inits, init_idx, cmds, f, *, suite=None, task=0):
    import main as lm
    if suite is None:
        env.reset()
        obs = env.set_init_state(inits[init_idx])
    else:
        from libero_reset import reset_libero
        obs, _ = reset_libero(env, inits[init_idx], suite=suite, task=task, init=init_idx)
    for _ in range(10):
        obs, *_ = env.step(lm.LIBERO_DUMMY_ACTION)
    D, X = [], []
    for a in cmds:
        # New logs preserve the policy's gripper command. Historical six-channel logs
        # cannot recover it; their explicitly recorded fallback is held open.
        a = np.asarray(a, float).copy()
        if a.shape == (6,):
            a = np.concatenate([a, [-1.0]])
        if a.shape != (7,):
            raise ValueError("replay commands must contain six arm channels or all seven channels")
        a[:6] += f
        x0 = np.array(obs["robot0_eef_pos"], float)
        q0 = np.array(obs["robot0_eef_quat"], float)
        obs, _, done, _ = env.step(a.tolist())
        x1 = np.array(obs["robot0_eef_pos"], float)
        q1 = np.array(obs["robot0_eef_quat"], float)
        D.append(np.concatenate([x1 - x0, rot_delta(q0, q1)]))
        X.append(np.concatenate([x1, lm._quat2axisangle(q1)]))
        if done:
            break
    return np.array(D), np.array(X)


def nominal_commands(payload, episode=0, steps=80):
    """Select one nominal episode; never join commands across reset boundaries."""
    records = calibration_records(payload)
    record_index = next((i for i, record in enumerate(records) if record.get("sev") == 0), None)
    if record_index is None:
        raise ValueError("source log contains no nominal (sev=0) record")
    record = records[record_index]
    lengths = record.get("ep_len")
    if (not isinstance(lengths, list) or not lengths
            or any(type(length) is not int or length < 1 for length in lengths)):
        raise ValueError("source log needs positive ep_len entries to preserve episode boundaries")
    if not 0 <= episode < len(lengths) or steps < 1:
        raise ValueError("source episode or replay step count is invalid")
    command_field = "raw_cmd" if "raw_cmd" in record else "raw_a"
    commands = np.asarray(record.get(command_field), dtype=float)
    if (commands.ndim != 2 or commands.shape[1] not in (6, 7)
            or commands.shape[0] != sum(lengths) or not np.isfinite(commands).all()):
        raise ValueError("recorded commands must be finite, have six/seven channels, and match ep_len")
    raw_arm = np.asarray(record.get("raw_a"), dtype=float)
    if raw_arm.shape != (len(commands), 6) or not np.array_equal(commands[:, :6], raw_arm):
        raise ValueError("full command stream disagrees with recorded arm commands")
    begin = sum(lengths[:episode])
    end = begin + min(steps, lengths[episode])
    keys = record.get("episode_keys")
    if keys is None and isinstance(payload, dict):
        keys = payload.get("calib_episodes")
    key = None
    if keys is not None:
        if (not isinstance(keys, list) or len(keys) != len(lengths)
                or any(not isinstance(k, (list, tuple)) or len(k) != 2
                       or any(type(v) is not int or v < 0 for v in k) for k in keys)):
            raise ValueError("source episode keys do not match episode boundaries")
        key = list(keys[episode])
    selected = np.asarray(commands[begin:end], dtype="<f8").copy()
    provenance = dict(record_index=record_index, episode_index=episode, scenario=key,
                      command_field=command_field, source_episode_steps=lengths[episode],
                      requested_steps=steps, selected_steps=len(selected),
                      flat_command_range=[begin, end],
                      gripper="recorded" if selected.shape[1] == 7 else "legacy_missing_held_open_minus_one",
                      command_sha256=hashlib.sha256(selected.tobytes()).hexdigest())
    return selected, provenance


def identify(env, inits, init_idx, cmds, probe, replay_fn=None):
    """Measure sensitivity, aligning both signs to one common episode prefix."""
    replay_fn = replay_fn or replay
    if not np.isfinite(probe) or probe <= 0 or not 0 <= init_idx < len(inits):
        raise ValueError("probe magnitude must be finite/positive and initial state must exist")
    def rollout(fault):
        motion, _ = replay_fn(env, inits, init_idx, cmds, fault)
        motion = np.asarray(motion, dtype=float)
        if motion.ndim != 2 or motion.shape[1] != 6 or not len(motion) or not np.isfinite(motion).all():
            raise ValueError("sensitivity replay returned no finite six-dimensional motion")
        return motion
    base = rollout(np.zeros(6))
    rows = []
    for magnitude in (0.01, 0.02, 0.05, -0.05):
        motion = rollout(np.full(6, magnitude))
        count = min(len(motion), len(base))
        delta = (motion[:count] - base[:count]).mean(0) / OUT
        rows.append(dict(f=magnitude, d_motion=delta.tolist(), sens=(delta/magnitude).tolist(),
                         replay_steps=len(motion), common_steps=count))
    matrix = np.zeros((6, 6))
    columns = []
    for axis in range(6):
        fault = np.zeros(6)
        fault[axis] = probe
        plus, minus = rollout(fault), rollout(-fault)
        count = min(len(base), len(plus), len(minus))
        matrix[:, axis] = (plus[:count] - minus[:count]).mean(0) / OUT / (2 * probe)
        columns.append(dict(axis=axis, plus_steps=len(plus), minus_steps=len(minus), common_steps=count))
    return dict(rows=rows, M=matrix.tolist(), baseline_steps=len(base), columns=columns,
                sensitivity_method="central difference on common baseline/plus/minus prefix per input axis")


def main(argv=None, environment_factory=None, replay_fn=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--source-episode", type=int, default=0,
                    help="nominal recorded episode index; replay never crosses its reset boundary")
    ap.add_argument("--probe-task", type=int,
                    help="task for sensitivity probes; default source task if recorded, otherwise task 0")
    ap.add_argument("--probe", type=float, default=0.02,
                    help="central-difference magnitude for M. The default 0.02 sits in "
                         "the linear region; the experiments inject 0.05, where "
                         "translation does not respond linearly. Setting this to the "
                         "operating point measures M where it is actually used.")
    ap.add_argument("--probe-init", type=int, default=45,
                    help="initial state to replay for the sensitivity probes. Default 45 "
                         "reproduces every stored M, but 45 is ALSO adaptive_law.py's "
                         "default --eval-init, so the default overlaps evaluation.")
    a = ap.parse_args(argv)
    if a.out.exists():
        ap.error("--out already exists; use a fresh path to preserve calibration provenance")
    if a.steps < 1 or a.source_episode < 0 or a.probe_init < 0 or not np.isfinite(a.probe) or a.probe <= 0:
        ap.error("steps, episode/initial-state indices, and probe magnitude must be valid")

    d = json.loads(a.log.read_text())
    if isinstance(d, dict) and d.get("suite") is not None and d["suite"] != a.suite:
        raise ValueError("source calibration suite differs from requested sensitivity suite")
    cmds, command_source = nominal_commands(d, a.source_episode, a.steps)
    if a.probe_task is None:
        a.probe_task = command_source["scenario"][0] if command_source["scenario"] is not None else 0
    if a.probe_task < 0:
        ap.error("--probe-task must be nonnegative")
    if environment_factory is None:
        import main as lm
        from libero.libero import benchmark
        def environment_factory(suite_name, task_id):
            suite = benchmark.get_benchmark_dict()[suite_name]()
            if task_id >= suite.n_tasks:
                raise ValueError("probe task is outside the selected suite")
            task = suite.get_task(task_id)
            env, _ = lm._get_libero_env(task, lm.LIBERO_ENV_RESOLUTION, 7)
            return env, suite.get_task_init_states(task_id)
    env, inits = environment_factory(a.suite, a.probe_task)
    reset_protocol = "injected_replay_callable"
    if replay_fn is None:
        from functools import partial
        replay_fn = partial(replay, suite=a.suite, task=a.probe_task)
        reset_protocol = "libero-reset-v1"
    try:
        result = identify(env, inits, a.probe_init, cmds, a.probe, replay_fn)
    finally:
        env.close()
    M = np.asarray(result["M"])
    print(f"replayed {result['baseline_steps']} baseline steps; sensitivity matrix M[out, in]:")
    hdr = ["dx", "dy", "dz", "drx", "dry", "drz"]
    print("        " + " ".join(f"{h:>8}" for h in hdr) + "   <- fault applied to")
    for i in range(6):
        print(f"{hdr[i]:>6}  " + " ".join(f"{M[i, j]:>8.3f}" for j in range(6)))
    off = np.abs(M - np.diag(np.diag(M))).sum() / max(np.abs(M).sum(), 1e-9)
    print(f"\noff-diagonal share of |M| = {off:.2f}   (0 = decoupled, per-axis gains suffice)")
    print(f"diagonal: {np.round(np.diag(M), 3)}")
    print(f"condition number of M = {np.linalg.cond(M):.1f}   (large = ill-posed to invert)")
    folder = pathlib.Path(__file__).resolve().parent
    result.update(schema_version=2, status="complete", probe=a.probe, reset_protocol=reset_protocol,
                  probe_init=int(a.probe_init), probe_task=int(a.probe_task), suite=a.suite,
                  probe_episodes=[[int(a.probe_task), int(a.probe_init)]],
                  command_source=command_source,
                  source_calib_episodes=d.get("calib_episodes") if isinstance(d, dict) else None,
                  args={key: str(value) if isinstance(value, pathlib.Path) else value
                        for key, value in vars(a).items()},
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_hashes=source_hashes([a.log, folder/"openloop_id.py", folder/"error_signal.py", folder/"so3.py", folder/"libero_reset.py"]))
    atomic_json(a.out, result)
    return result


if __name__ == "__main__":
    main()
