"""re4 theory plan, T.0 + Part 1: same-command replay from saved and perturbed simulator states on
the LIBERO Panda (prereg_records/PREREG_RE4T_1_2_CONTRACTION.md). Simulator only; no policy server.

For each recorded command sequence: reset the scenario (libero_reset), replay commands to a save
step, snapshot the MuJoCo state, run the baseline continuation for H steps, then for each
perturbation restore the snapshot, perturb (arm joint positions, or the first replayed command),
replay the same H commands and record the per-step gap in four coordinate sets. A zero
perturbation is replayed first as the determinism check.
Run in the LIBERO client venv with MUJOCO_GL=egl and PYTHONPATH including openpi/ and the
openpi examples/libero + third_party/libero dirs.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
SAVE_STEPS, H, MAGS, NDIR, DU = (10, 25, 40, 55), 15, (0.005, 0.01, 0.02), 6, 0.02
ARM = 7


def contact_flag(sim):
    """True if a non-finger robot geom touches anything, or any robot geom touches a non-arena geom."""
    m = sim.model
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        n1, n2 = (m.geom_id2name(c.geom1) or ""), (m.geom_id2name(c.geom2) or "")
        r1, r2 = n1.startswith(("robot0", "gripper0")), n2.startswith(("robot0", "gripper0"))
        if not (r1 or r2):
            continue
        other = n2 if r1 else n1; robot = n1 if r1 else n2
        arena = other.startswith(("table", "floor", "wall", "bin", "shelf", "cabinet")) or other == ""
        if "finger" not in robot and "pad" not in robot:
            return True
        if not arena:
            return True
    return False


def ee(sim, site):
    return np.array(sim.data.site_xpos[site], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(HERE.parent / "results/heldout/error_signal_init25.json"))
    ap.add_argument("--episodes", default="0,1,2"); ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--out", type=pathlib.Path, default=HERE.parent / "results/re4_theory/1_contraction/panda_replay.json")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=70)
    a = ap.parse_args()
    from libero.libero import benchmark
    import main as libero_main
    from libero_reset import reset_libero
    d = json.loads(pathlib.Path(a.log).read_text()); rec = d["records"][0]
    cmds_all = np.asarray(rec["raw_cmd"], float); lens = rec["ep_len"]; keys = rec["episode_keys"]
    starts = np.cumsum([0] + list(lens[:-1]))
    suite = benchmark.get_benchmark_dict()[a.suite]()
    rng = np.random.default_rng(a.seed)
    dirs = rng.normal(size=(NDIR, ARM)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    results, envs = [], {}
    t0 = time.time()
    for ei in [int(x) for x in a.episodes.split(",")]:
        tid, init = int(keys[ei][0]), int(keys[ei][1]); n = min(int(lens[ei]), a.steps)
        cmds = cmds_all[starts[ei]:starts[ei] + n]
        if tid not in envs:
            task = suite.get_task(tid)
            env, desc = libero_main._get_libero_env(task, libero_main.LIBERO_ENV_RESOLUTION, 7)
            envs[tid] = (env, suite.get_task_init_states(tid))
        env, inits = envs[tid]
        obs, _ = reset_libero(env, inits[init], suite=a.suite, task=tid, init=init)
        for _ in range(10):
            obs, _, _, _ = env.step(libero_main.LIBERO_DUMMY_ACTION)
        sim = env.sim; robot = env.robots[0]
        jidx = np.array(robot._ref_joint_pos_indexes, int); vidx = np.array(robot._ref_joint_vel_indexes, int)
        site = sim.model.site_name2id(robot.controller.eef_name) if hasattr(robot.controller, "eef_name") else sim.model.site_name2id("gripper0_grip_site")

        core = env.env if hasattr(env, "env") else env      # robosuite MujocoEnv: clears its terminated flag

        def run(cmd_seq):
            traj = []
            for c in cmd_seq:
                core.done = False                                # a restored state is never 'terminated'
                obs2, _, done, _ = env.step(list(c))
                traj.append(dict(q=sim.data.qpos[jidx].copy(), v=sim.data.qvel[vidx].copy(), ee=ee(sim, site), contact=contact_flag(sim)))
                # do not stop on 'done': the same H commands must run in every replay
            return traj

        k = 0
        for ks in SAVE_STEPS:
            if ks + H > n - 1:                                   # never replay into the success step
                break
            run(cmds[k:ks]); k = ks
            snap = sim.get_state(); q_at = sim.data.qpos[jidx].copy()
            # MjSimState holds only time/qpos/qvel; contact-solver warm starts and actuator states
            # must be restored too, or a restored state with contacts replays differently.
            extra = dict(warm=sim.data.qacc_warmstart.copy(), act=sim.data.act.copy(), ctrl=sim.data.ctrl.copy(),
                         grip=np.array(robot.gripper.current_action, float).copy())   # accumulated gripper target
            seg = cmds[ks:ks + H]
            base = run(seg)
            per = []
            def restore():
                sim.set_state(snap)
                sim.data.qacc_warmstart[:] = extra["warm"]; sim.data.act[:] = extra["act"]; sim.data.ctrl[:] = extra["ctrl"]
                robot.gripper.current_action = extra["grip"].copy()
                sim.forward()

            def replay(kind, mag, vec, dcmd=None):
                restore()
                if kind == "joint":
                    sim.data.qpos[jidx] = q_at + mag * vec; sim.forward()
                seg2 = seg.copy()
                if dcmd is not None:
                    seg2[0, :6] = seg2[0, :6] + dcmd
                tr = run(seg2)
                L = min(len(tr), len(base))
                gq = [float(np.linalg.norm(tr[i]["q"] - base[i]["q"])) for i in range(L)]
                gqv = [float(np.sqrt(np.sum((tr[i]["q"] - base[i]["q"]) ** 2) + np.sum(((tr[i]["v"] - base[i]["v"]) / 20.0) ** 2))) for i in range(L)]
                gee = [(tr[i]["ee"] - base[i]["ee"]).tolist() for i in range(L)]
                per.append(dict(kind=kind, mag=mag, dir=(vec.tolist() if vec is not None else None), dcmd=(None if dcmd is None else dcmd.tolist()),
                                gap_q=gq, gap_qv=gqv, gap_ee=gee, contact=[bool(b["contact"]) for b in base[:L]]))
            replay("none", 0.0, None)
            for mag in MAGS:
                for v in dirs:
                    replay("joint", mag, v)
            for axis in range(6):
                for s in (1.0, -1.0):
                    dc = np.zeros(6); dc[axis] = s * DU; replay("cmd", DU, None, dc)
            # restore the baseline end state so the next save step continues from it
            restore(); run(seg); k = ks + H
            results.append(dict(episode=ei, task=tid, init=init, save_step=ks, horizon=H, perturbations=per,
                                baseline_contact=[bool(b["contact"]) for b in base]))
            print(f"episode {ei} (task {tid}, init {init}) save step {ks}: {len(per)} replays, baseline contact steps {sum(b['contact'] for b in base)}/{len(base)}, determinism gap {max(per[0]['gap_q']):.2e}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(log=a.log, suite=a.suite, save_steps=SAVE_STEPS, horizon=H, mags=MAGS, ndir=NDIR, du=DU,
                                     seed=a.seed, seconds=time.time() - t0, results=results)))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
