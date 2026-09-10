"""Physical ALOHA cell orchestration tests with a synthetic MuJoCo-like plant."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import aloha_joint_fault
import run_aloha_joint_followup as runner


class FakeEnvironment:
    def __init__(self, mismatch=False):
        names = [f"vx300s_{side}/{name}" for side in ("left", "right")
                 for name in aloha_joint_fault.JOINT_NAMES]
        model = SimpleNamespace(name2id=lambda name, kind: names.index(name),
            actuator_trnid=np.c_[np.arange(12), np.zeros(12)],
            actuator_gainprm=np.c_[runner.EXPECTED_GAINS, np.zeros((12, 9))],
            actuator_gear=np.c_[np.ones(12), np.zeros((12, 5))],
            jnt_dofadr=np.arange(12), jnt_qposadr=np.arange(12),
            actuator_forcelimited=np.ones(12), actuator_forcerange=np.tile([-100., 100.], (12, 1)),
            actuator_ctrllimited=np.ones(12), actuator_ctrlrange=np.tile([-1., 1.], (12, 1)),
            jnt_limited=np.ones(12), jnt_range=np.tile([-2., 2.], (12, 1)))
        data = SimpleNamespace(qpos=np.zeros(24), qvel=np.zeros(23), qfrc_applied=np.zeros(23),
                               actuator_force=np.zeros(12), ctrl=np.zeros(12))
        self.unwrapped = self
        self._env = SimpleNamespace(physics=SimpleNamespace(model=model, data=data))
        self.forces = []
        self.reset_count = 0
        self.mismatch = mismatch

    def reset(self, seed):
        self.reset_count += 1
        data = self._env.physics.data
        data.qpos[:] = 0
        data.qvel[:] = 0
        data.qpos[20] = seed/10000.
        if self.mismatch and self.reset_count == 2:
            data.qpos[20] += .001  # cube mismatch, outside the robot joint slice
        return {"agent_pos": np.zeros(14)}, {}

    def step(self, command):
        data = self._env.physics.data
        self.forces.append(data.qfrc_applied.copy())
        data.ctrl[:] = np.clip(np.asarray(command)[runner.CORRECTION_COORDINATES], -1., 1.)
        data.actuator_force[:] = 0
        data.actuator_force[0] = 100.
        return {"agent_pos": np.zeros(14)}, 0, False, False, {}


class Tests(unittest.TestCase):
    def setup_cell(self, directory, healthy=False, mismatch=False):
        cell = ["--healthy"] if healthy else ["--joint", "7"]
        args = runner.parser().parse_args([*cell, "--out-dir", directory, "--episodes", "2"])
        args.arm_names = args.arms.split(",")
        args.telemetry = args.out_dir/"telemetry.jsonl"
        env = FakeEnvironment(mismatch)
        aloha = SimpleNamespace(env=env, seed=args.seed)
        aloha.reset = lambda episode: aloha.env.reset(seed=aloha.seed+episode)[0]
        observer = dict(W=np.zeros((14, 8)).tolist(), M=np.eye(14).tolist(),
                        Q=(np.eye(14)*.001).tolist(), R=(np.eye(14)*.01).tolist(),
                        bias=None, dt=.02, gamma=.08)
        artifact = dict(observer=observer, qualification=dict(allowed=True))
        calls = []
        def episode(client, ep, **kwargs):
            calls.append(kwargs)
            client.reset(ep)
            self.assertEqual(kwargs["corr"], runner.CORRECTION_COORDINATES)
            np.testing.assert_array_equal(kwargs["fvec"], np.zeros(14))
            self.assertNotIn("joint_fault", kwargs)
            self.assertNotIn("reference_model", kwargs)
            trace = []
            for t in range(2):
                command = np.zeros(14)
                command[0] = 1.5
                client.env.step(command)
                trace.append(dict(f_hat=[.08]*14, f_true=[0.]*14))
            return kwargs["adapt"], np.full(14, .08), trace
        law = SimpleNamespace(NJ=14, K_FIR=6, DT=.02, HORIZON=50, MAX_STEPS=987,
                              TASK="gym_aloha/AlohaTransferCube-v0", episode=episode,
                              write_telemetry=lambda stream, row: stream.write(
                                  json.dumps(row, default=runner.json_default)+"\n"))
        return args, aloha, env, law, artifact, calls

    def test_faulted_cell_shared_control_mapping_torque_and_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            args, aloha, env, law, artifact, calls = self.setup_cell(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                result = runner.execute(args, aloha, law, artifact, {})
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(calls), 8)
            self.assertEqual(result["physical_fault"]["command_coordinate"], 8)
            self.assertEqual(result["physical_fault"]["torque_nm"], 32.)
            for force in env.forces:
                expected = np.zeros(23)
                expected[7] = 32.
                np.testing.assert_array_equal(force, expected)
            self.assertIs(aloha.env, env)
            np.testing.assert_array_equal(env._env.physics.data.qfrc_applied, np.zeros(23))
            self.assertEqual(law.MAX_STEPS, 987)
            self.assertEqual(result["pairing"]["checked_states"], 8)
            self.assertEqual(result["arms"]["off"]["n"], 2)
            self.assertEqual(result["schedule"][1]["arms"], runner.arm_order(args.arm_names, 1))
            views = [json.loads((Path(directory)/f"{name}.json").read_text()) for name in args.arm_names[1:]]
            self.assertEqual(len({view["shared_control_id"] for view in views}), 1)
            self.assertEqual(views[0]["arms"]["frozen_faulted"], views[-1]["arms"]["frozen_faulted"])
            diagnostics = result["arms"]["off"]["diagnostics"][0]
            self.assertEqual(diagnostics["force_limit"][0], 2)
            self.assertEqual(diagnostics["requested_outside_control_range"][0], 2)
            self.assertEqual(diagnostics["estimate_clip_steps"], [2]*14)
            self.assertEqual(result["robot_joints"][6]["command_coordinate"], 7)

    def test_healthy_cell_never_injects_torque(self):
        with tempfile.TemporaryDirectory() as directory:
            args, aloha, env, law, artifact, _ = self.setup_cell(directory, healthy=True)
            with contextlib.redirect_stdout(io.StringIO()):
                result = runner.execute(args, aloha, law, artifact, {})
            self.assertIsNone(result["physical_fault"])
            self.assertIsNone(result["args"]["joint_fault"])
            for force in env.forces:
                np.testing.assert_array_equal(force, np.zeros(23))

    def test_cube_pairing_mismatch_aborts_and_restores_torque(self):
        with tempfile.TemporaryDirectory() as directory:
            args, aloha, env, law, artifact, _ = self.setup_cell(directory, mismatch=True)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "pairing failed"):
                runner.execute(args, aloha, law, artifact, {})
            result = json.loads((Path(directory)/"study.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["pairing"]["valid_so_far"])
            self.assertEqual(sum(arm["n"] for arm in result["arms"].values()), 1)
            self.assertIs(aloha.env, env)
            np.testing.assert_array_equal(env._env.physics.data.qfrc_applied, np.zeros(23))
            self.assertEqual(law.MAX_STEPS, 987)


if __name__ == "__main__":
    unittest.main(verbosity=2)
