"""Local healthy LIBERO joint-state sensitivity under identical recorded commands.

No policy queries. Each central-difference branch starts from the same complete
physics/controller/gripper snapshot. Native relative OSC then recomputes its goals
from each branch's pose; the first-step goal differences are recorded explicitly.
Singular gains measure local finite-horizon expansion in the declared coordinates.
Eigenvalue radii are frozen-map diagnostics, not contraction certificates.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from error_signal import atomic_json
from libero_local_probe import Snapshot, advance, parity
from openloop_id import nominal_commands


def diagnostics(matrix):
    matrix = np.asarray(matrix, float)
    singular = np.linalg.svd(matrix, compute_uv=False)
    return dict(singular_values=singular.tolist(), sigma_max=float(singular[0]),
                eigenvalue_radius=float(np.max(np.abs(np.linalg.eigvals(matrix)))),
                eigenvalue_interpretation="frozen endpoint Jacobian diagnostic; not a contraction certificate")


def finite_difference(evaluate, dimension, epsilon, horizons, tolerance):
    """Differentiate one restored-trajectory evaluator; test repeated-center agreement."""
    zero = np.zeros(dimension)
    center = evaluate(zero)
    plus, minus = [], []
    for axis in range(dimension):
        delta = np.zeros(dimension)
        delta[axis] = epsilon
        plus.append(evaluate(delta))
        minus.append(evaluate(-delta))
    repeat = evaluate(zero)
    rows = []
    for horizon in horizons:
        index = horizon-1
        before = np.asarray(center["states"])[index]
        after = np.asarray(repeat["states"])[index]
        error = float(np.max(np.abs(before-after)))
        if error > tolerance:
            raise RuntimeError(f"restored center replay differs by {error} at horizon {horizon}")
        jacobian = np.column_stack([(np.asarray(p["states"])[index]-np.asarray(m["states"])[index])/(2*epsilon)
                                     for p, m in zip(plus, minus)])
        qmatrix = jacobian[:7, :7]
        goals = np.column_stack([(np.asarray(p["first_goal"])-np.asarray(m["first_goal"]))/ (2*epsilon)
                                 for p, m in zip(plus, minus)])
        rows.append(dict(horizon=horizon, epsilon=epsilon, repeat_max_abs=error,
            state_jacobian=jacobian.tolist(), q_to_q_jacobian=qmatrix.tolist(),
            q_to_q=diagnostics(qmatrix), arm_state=diagnostics(jacobian),
            first_goal_jacobian=goals.tolist(), first_goal_position_sigma_max=float(np.linalg.svd(goals[:3], compute_uv=False)[0]),
            nominal_endpoint=before.tolist(), nominal_first_goal=center["first_goal"]))
    return rows


def state_coordinates(rs, qaddr, dofs, velocity_scale):
    return np.r_[rs.sim.data.qpos[qaddr], velocity_scale*rs.sim.data.qvel[dofs]]


def goal_coordinates(rs):
    controller = rs.robots[0].controller
    return np.r_[np.asarray(controller.goal_pos).reshape(-1), np.asarray(controller.goal_ori).reshape(-1)]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--init", type=int, default=27)
    p.add_argument("--source-episode", type=int, default=0)
    p.add_argument("--checkpoints", default="0,20,60")
    p.add_argument("--horizons", default="1,5,20")
    p.add_argument("--epsilons", default="0.0001,0.00005")
    p.add_argument("--velocity-scale", type=float, default=.05,
                   help="seconds: arm-state coordinates are [q, velocity_scale*qdot]")
    p.add_argument("--tolerance", type=float, default=1e-8)
    p.add_argument("--contraction-margin", type=float, default=.001)
    p.add_argument("--selftest", action="store_true")
    return p


def selftest():
    import unittest
    class Tests(unittest.TestCase):
        def test_known_map_and_nonnormal_radius_is_not_contraction(self):
            matrix = np.eye(14)*.5
            matrix[0, 1] = 2.
            def evaluate(delta):
                return dict(states=[(np.linalg.matrix_power(matrix, h)@delta).tolist() for h in range(1, 6)],
                            first_goal=(np.zeros(12)+delta[0]).tolist())
            rows = finite_difference(evaluate, 14, 1e-4, [1, 5], 1e-12)
            np.testing.assert_allclose(rows[0]["state_jacobian"], matrix)
            self.assertLess(rows[0]["arm_state"]["eigenvalue_radius"], 1.)
            self.assertGreater(rows[0]["arm_state"]["sigma_max"], 1.)
            np.testing.assert_allclose(rows[1]["state_jacobian"], np.linalg.matrix_power(matrix, 5))
        def test_replay_drift_is_rejected(self):
            count = 0
            def evaluate(delta):
                nonlocal count
                count += 1
                return dict(states=[np.asarray(delta)+count*.01], first_goal=np.zeros(12).tolist())
            with self.assertRaisesRegex(RuntimeError, "center replay differs"):
                finite_difference(evaluate, 14, 1e-4, [1], 1e-8)
    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful()


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.selftest:
        return 0 if selftest() else 1
    checkpoints = sorted(set(map(int, args.checkpoints.split(","))))
    horizons = sorted(set(map(int, args.horizons.split(","))))
    epsilons = list(map(float, args.epsilons.split(",")))
    if (not checkpoints or min(checkpoints) < 0 or not horizons or min(horizons) < 1
            or len(epsilons) < 2 or len(set(epsilons)) != len(epsilons)
            or any(not np.isfinite(eps) or eps <= 0 for eps in epsilons)
            or not np.isfinite(args.velocity_scale) or args.velocity_scale <= 0):
        p.error("checkpoints, horizons, distinct positive epsilons, and velocity scale must be valid")
    if args.out.exists():
        p.error("output already exists; use a fresh path")
    raw = args.log.read_bytes()
    source = json.loads(raw)
    if isinstance(source, dict) and source.get("suite", args.suite) != args.suite:
        p.error("command donor suite differs from replay suite")
    commands, donor = nominal_commands(source, args.source_episode, max(checkpoints)+max(horizons))
    if len(commands) < max(checkpoints)+max(horizons):
        p.error("selected donor episode is too short; reset boundaries cannot be concatenated")
    if commands.shape[1] == 6:
        commands = np.c_[commands, -np.ones(len(commands))]
    folder = Path(__file__).resolve().parent
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in
              (Path(__file__), folder/"libero_local_probe.py", folder/"libero_reset.py", folder/"openloop_id.py")}
    hashes[str(args.log.resolve())] = hashlib.sha256(raw).hexdigest()
    result = dict(schema_version=1, status="planned", stage="local_healthy_model_qualification",
                  args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), source_hashes=hashes,
                  command_source=donor, rows=[], checkpoints=[],
                  protocol=dict(healthy=True, policy_queries=0, dimension=14,
                    arm_state="[seven arm joint positions, velocity_scale * seven joint velocities]",
                    q_metric="Euclidean radians", arm_state_metric="Euclidean in declared scaled coordinates",
                    controller="identical initial controller snapshot; native relative OSC recomputes goals per branch",
                    perturbation="central +/- state coordinates with unchanged initial controller/gripper/other-body state",
                    criterion="all sampled q-to-q singular gains < 1-margin at both epsilons, with derivative consistency",
                    scope="local finite horizons on one command replay; excludes policy response and unperturbed hidden state"))
    atomic_json(args.out, result)
    env = None
    try:
        import main as lm
        from libero.libero import benchmark
        from libero_reset import reset_libero, physics_fingerprint
        suite = benchmark.get_benchmark_dict()[args.suite]()
        if not 0 <= args.task < suite.n_tasks:
            raise ValueError("requested task does not exist")
        inits = suite.get_task_init_states(args.task)
        if not 0 <= args.init < len(inits):
            raise ValueError("requested initial state does not exist")
        env, _ = lm._get_libero_env(suite.get_task(args.task), 32, 7)
        _, reset = reset_libero(env, inits[args.init], suite=args.suite, task=args.task, init=args.init)
        rs = env.env
        for name, observable in rs._observables.items():
            if any(token in name for token in ("image", "depth", "segmentation")):
                observable.set_enabled(False)
        names = rs.robots[0].robot_model.joints
        qaddr = np.array([rs.sim.model.get_joint_qpos_addr(name) for name in names])
        dofs = np.array([rs.sim.model.get_joint_qvel_addr(name) for name in names])
        if len(qaddr) != 7:
            raise ValueError("expected seven Panda arm joints")
        result.update(status="running", reset=reset, joint_names=names,
                      control_dt=rs.control_timestep, physics_dt=rs.model_timestep,
                      controller_type=type(rs.robots[0].controller).__name__)
        result["warmup_parity"] = parity(rs, np.asarray(lm.LIBERO_DUMMY_ACTION), int(dofs[0]), 0., args.tolerance)
        for _ in range(10):
            advance(rs, np.asarray(lm.LIBERO_DUMMY_ACTION), int(dofs[0]), 0.)
        for checkpoint in range(max(checkpoints)+1):
            if checkpoint in checkpoints:
                saved = Snapshot(rs)
                initial_goal = goal_coordinates(rs).copy()
                step_parity = parity(rs, commands[checkpoint], int(dofs[0]), 0., args.tolerance)
                saved.restore()
                result["checkpoints"].append(dict(step=checkpoint, snapshot_id=saved.identity,
                    full_physics=physics_fingerprint(env), initial_state=state_coordinates(rs, qaddr, dofs, args.velocity_scale).tolist(),
                    initial_controller_goal=initial_goal.tolist(), full_step_parity=step_parity))
                def evaluate(delta):
                    saved.restore()
                    if not np.array_equal(goal_coordinates(rs), initial_goal):
                        raise RuntimeError("controller goal snapshot did not restore identically")
                    rs.sim.data.qpos[qaddr] += delta[:7]
                    rs.sim.data.qvel[dofs] += delta[7:]/args.velocity_scale
                    rs.sim.forward()
                    states, first_goal = [], None
                    for offset in range(max(horizons)):
                        advance(rs, commands[checkpoint+offset], int(dofs[0]), 0.)
                        states.append(state_coordinates(rs, qaddr, dofs, args.velocity_scale).tolist())
                        if first_goal is None:
                            first_goal = goal_coordinates(rs).tolist()
                    return dict(states=states, first_goal=first_goal)
                for epsilon in epsilons:
                    rows = finite_difference(evaluate, 14, epsilon, horizons, args.tolerance)
                    for row in rows:
                        row["checkpoint"] = checkpoint
                        result["rows"].append(row)
                        print(f"checkpoint={checkpoint} h={row['horizon']} eps={epsilon} "
                              f"sigma_q={row['q_to_q']['sigma_max']:.6f} "
                              f"sigma_arm={row['arm_state']['sigma_max']:.6f} "
                              f"rho_arm={row['arm_state']['eigenvalue_radius']:.6f}", flush=True)
                    atomic_json(args.out, result)
                saved.restore()
            if checkpoint < max(checkpoints):
                advance(rs, commands[checkpoint], int(dofs[0]), 0.)
        consistency = []
        for checkpoint in checkpoints:
            for horizon in horizons:
                rows = [row for row in result["rows"] if row["checkpoint"] == checkpoint and row["horizon"] == horizon]
                base = np.asarray(rows[0]["q_to_q_jacobian"])
                difference = max(float(np.linalg.norm(np.asarray(row["q_to_q_jacobian"])-base)/max(np.linalg.norm(base), 1e-12))
                                 for row in rows[1:])
                consistency.append(dict(checkpoint=checkpoint, horizon=horizon, relative_q_jacobian_difference=difference))
        result["derivative_consistency"] = consistency
        result["qualification"] = dict(
            sampled_q_contraction_criterion_met=all(row["q_to_q"]["sigma_max"] < 1-args.contraction_margin for row in result["rows"]),
            derivative_consistent=all(row["relative_q_jacobian_difference"] < .05 for row in consistency),
            max_q_singular_gain=max(row["q_to_q"]["sigma_max"] for row in result["rows"]),
            max_arm_singular_gain=max(row["arm_state"]["sigma_max"] for row in result["rows"]),
            global_contraction_established=False, composite_enabled_by_this_probe=False,
            interpretation="Failure excludes this sampled q-reference contraction premise; passage alone would still require model/controller qualification.")
        result["status"] = "complete"
        atomic_json(args.out, result)
    except BaseException as error:
        result.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        atomic_json(args.out, result)
        raise
    finally:
        if env is not None:
            env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
