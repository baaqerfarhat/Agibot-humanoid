"""Exercise adaptive_law.run with real JointFault cleanup and a minimal cached plant.

Only simulator/client imports are replaced. The production rollout, warmup, fault
application, estimator path, exits, and JointFault.restore execute unchanged.
"""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import adaptive_law


MODEL_FIELDS = ("dof_frictionloss", "dof_damping", "actuator_gainprm", "jnt_range")
FAULTS = ("torque:3:5", "friction:3:2", "damping:3:1.5", "gain:3:0.8", "lock:3:0.05")


class SyntheticEnvironment:
    """Persistent model parameters; reset changes state but deliberately not the model."""

    def __init__(self, mode="timeout"):
        self.mode = mode
        self.reset_count = 0
        self.steps = 0
        self.history = []
        names = [f"joint{i}" for i in range(7)]
        actuator_names = [f"actuator{i}" for i in range(7)]
        # Nontrivial addresses ensure the real fault wrapper resolves model coordinates.
        dofs, qpos, actuators = [2, 5, 1, 7, 3, 8, 4], [1, 3, 4, 6, 8, 9, 10], [1, 4, 6, 0, 7, 3, 2]
        model = SimpleNamespace(
            get_joint_qvel_addr=lambda name: dofs[names.index(name)],
            get_joint_qpos_addr=lambda name: qpos[names.index(name)],
            joint_name2id=names.index,
            actuator_name2id=lambda name: actuators[actuator_names.index(name)],
            dof_frictionloss=np.arange(11, dtype=float)/10+.1,
            dof_damping=np.arange(11, dtype=float)/20+.2,
            actuator_gainprm=np.arange(90, dtype=float).reshape(9, 10)/100+.9,
            jnt_range=np.c_[np.arange(9)/10.-2., np.arange(9)/10.+2.])
        data = SimpleNamespace(qpos=np.arange(12, dtype=float)/10., qfrc_applied=np.zeros(11))
        self.sim = SimpleNamespace(model=model, data=data)
        self.robots = [SimpleNamespace(robot_model=SimpleNamespace(joints=names, actuators=actuator_names),
                                       _joint_positions=np.zeros(7))]
        self.baseline = {name: np.asarray(getattr(model, name)).copy() for name in MODEL_FIELDS}

    def observation(self):
        return dict(agentview_image=np.zeros((4, 4, 3), dtype=np.uint8),
                    robot0_eye_in_hand_image=np.zeros((4, 4, 3), dtype=np.uint8),
                    robot0_eef_pos=np.array([.01*self.steps, 0., 0.]),
                    robot0_eef_quat=np.array([0., 0., 0., 1.]),
                    robot0_gripper_qpos=np.zeros(2))

    def reset(self):
        self.reset_count += 1
        self.steps = 0
        return self.observation()

    def set_init_state(self, initial):
        self.sim.data.qpos[:] = initial
        return self.observation()

    def step(self, action):
        self.steps += 1
        state = {name: np.asarray(getattr(self.sim.model, name)).copy() for name in MODEL_FIELDS}
        state["qfrc_applied"] = self.sim.data.qfrc_applied.copy()
        state["step"] = self.steps
        self.history.append(state)
        if self.mode == "warmup_error" and self.steps == 1:
            raise RuntimeError("synthetic warmup failure")
        if self.mode == "step_error" and self.steps == adaptive_law.WARMUP_STEPS+1:
            raise RuntimeError("synthetic physics step failure")
        done = self.mode == "done" and self.steps >= adaptive_law.WARMUP_STEPS+1
        return self.observation(), 0., done, {}


def fake_imports():
    lm = ModuleType("main")
    lm.LIBERO_DUMMY_ACTION = [0.]*6+[-1.]
    lm._quat2axisangle = lambda quaternion: np.zeros(3)
    image_tools = SimpleNamespace(convert_to_uint8=lambda image: image,
                                  resize_with_pad=lambda image, *shape: image)
    client = ModuleType("openpi_client")
    client.image_tools = image_tools
    faults = ModuleType("gate_faults")
    def action_fault(*args, **kwargs):
        raise AssertionError("these tests explicitly use the zero fvec path")
    faults.apply_action_fault = action_fault
    probe = ModuleType("paired_probe")
    probe.MAXS = 2
    return {"main": lm, "openpi_client": client, "gate_faults": faults, "paired_probe": probe}


class LifecycleTests(unittest.TestCase):
    def run_episode(self, env, fault, observer=None):
        def infer(observation):
            if env.mode == "inference_error":
                raise RuntimeError("synthetic policy inference failure")
            return dict(actions=np.tile([.1, 0., 0., 0., 0., 0., -1.], (2, 1)))
        initial = np.arange(12, dtype=float)/10.
        probe = SimpleNamespace(env_for=lambda task: (env, "synthetic task", [initial]),
                                client=SimpleNamespace(infer=infer), a=SimpleNamespace(replan_steps=2))
        with mock.patch.dict(sys.modules, fake_imports()):
            return adaptive_law.run(probe, 0, 0, 0., np.eye(6),
                np.zeros((6, adaptive_law.K_FIR+2)), .08, True, max_steps=2,
                fvec=np.zeros(6), joint_fault=fault, step_observer=observer)

    def assert_restored(self, env):
        for name, baseline in env.baseline.items():
            np.testing.assert_array_equal(getattr(env.sim.model, name), baseline,
                                          err_msg=f"cached model retained a change to {name}")
        np.testing.assert_array_equal(env.sim.data.qfrc_applied, np.zeros(11),
                                      err_msg="injected torque leaked beyond the episode")

    def assert_fault_observed(self, env, spec):
        kind = spec.split(":")[0]
        observed = env.history[0]
        if kind == "torque":
            for state in env.history:
                expected = np.zeros(11)
                if state["step"] > adaptive_law.WARMUP_STEPS:
                    expected[7] = 5.
                np.testing.assert_array_equal(state["qfrc_applied"], expected)
            return
        expected = {name: value.copy() for name, value in env.baseline.items()}
        if kind == "friction":
            expected["dof_frictionloss"][7] += 2.
        elif kind == "damping":
            expected["dof_damping"][7] += 1.5
        elif kind == "gain":
            expected["actuator_gainprm"][0, 0] = .8
        elif kind == "lock":
            expected["jnt_range"][3] = [.55, .65]
        for name in MODEL_FIELDS:
            np.testing.assert_allclose(observed[name], expected[name], rtol=0, atol=1e-14)
        self.assertTrue(any(not np.array_equal(observed[name], env.baseline[name]) for name in MODEL_FIELDS))

    def test_done_timeout_and_errors_restore_every_fault_kind(self):
        for spec in FAULTS:
            for mode in ("done", "timeout", "step_error", "inference_error", "warmup_error"):
                with self.subTest(fault=spec, exit=mode):
                    env = SyntheticEnvironment(mode)
                    if mode.endswith("error"):
                        with self.assertRaisesRegex(RuntimeError, "synthetic"):
                            self.run_episode(env, spec)
                    else:
                        success, _, trajectory = self.run_episode(env, spec)
                        self.assertEqual(success, mode == "done")
                        self.assertEqual(len(trajectory), 1 if mode == "done" else 2)
                    self.assert_fault_observed(env, spec)
                    self.assert_restored(env)

    def test_sequential_friction_runs_on_cached_model_do_not_accumulate(self):
        env = SyntheticEnvironment("timeout")
        self.run_episode(env, "friction:3:2")
        self.assert_restored(env)
        first_count = len(env.history)
        self.run_episode(env, "friction:3:2")
        self.assert_restored(env)
        self.assertEqual(env.reset_count, 2)
        for state in env.history:
            self.assertEqual(state["dof_frictionloss"][7], env.baseline["dof_frictionloss"][7]+2.)
        np.testing.assert_array_equal(env.history[0]["dof_frictionloss"],
                                      env.history[first_count]["dof_frictionloss"])

    def test_mixed_faults_then_unfaulted_episode_on_same_model(self):
        env = SyntheticEnvironment("timeout")
        for spec in FAULTS:
            self.run_episode(env, spec)
            self.assert_restored(env)
        boundary = len(env.history)
        self.run_episode(env, None)
        for state in env.history[boundary:]:
            for name in MODEL_FIELDS:
                np.testing.assert_array_equal(state[name], env.baseline[name])
            np.testing.assert_array_equal(state["qfrc_applied"], np.zeros(11))

    def test_after_step_observer_exception_restores_active_torque(self):
        env = SyntheticEnvironment("timeout")
        def observer(phase, values):
            if phase == "after":
                self.assertEqual(env.sim.data.qfrc_applied[7], 5.)
                raise RuntimeError("synthetic telemetry observer failure")
        with self.assertRaisesRegex(RuntimeError, "observer failure"):
            self.run_episode(env, "torque:3:5", observer)
        self.assert_restored(env)

    def test_keyboard_interrupt_restores_model(self):
        env = SyntheticEnvironment("timeout")
        def interrupt(phase, values):
            raise KeyboardInterrupt("synthetic cancellation")
        with self.assertRaises(KeyboardInterrupt):
            self.run_episode(env, "gain:3:0.8", interrupt)
        self.assert_restored(env)

    def test_preexisting_forces_survive_faults_and_torque_is_additive(self):
        baseline = np.linspace(-.4, .6, 11)
        for spec in FAULTS:
            for mode in ("done", "timeout", "step_error"):
                with self.subTest(fault=spec, exit=mode):
                    env = SyntheticEnvironment(mode)
                    env.sim.data.qfrc_applied[:] = baseline
                    if mode == "step_error":
                        with self.assertRaisesRegex(RuntimeError, "physics step failure"):
                            self.run_episode(env, spec)
                    else:
                        self.run_episode(env, spec)
                    for state in env.history:
                        expected = baseline.copy()
                        if spec.startswith("torque:") and state["step"] > adaptive_law.WARMUP_STEPS:
                            expected[7] += 5.
                        np.testing.assert_array_equal(state["qfrc_applied"], expected)
                    np.testing.assert_array_equal(env.sim.data.qfrc_applied, baseline)
                    for name, original in env.baseline.items():
                        np.testing.assert_array_equal(getattr(env.sim.model, name), original)

if __name__ == "__main__":
    unittest.main(verbosity=2)
