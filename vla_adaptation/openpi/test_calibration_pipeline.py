"""Calibration regressions with mocked collection/replay; no simulator or policy server."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import adaptive_law
import error_signal
import openloop_id


class CalibrationTests(unittest.TestCase):
    def args(self, folder, **updates):
        values = dict(out=Path(folder)/"nested"/"fir.json", suite="libero_spatial",
                      episodes=3, init_base=26, healthy_only=False, max_steps=12,
                      control=Path(folder)/"control.json", ack=Path(folder)/"ack.json",
                      host="localhost", port=8000, replan_steps=5)
        values.update(updates)
        return SimpleNamespace(**values)

    def collect(self, args, logger=None):
        calls = []
        probe = SimpleNamespace(suite=SimpleNamespace(n_tasks=2),
                                env_for=lambda _: (None, "mock task", range(50)),
                                control=lambda request: dict(request, acknowledged=True))
        def episode(pr, task, init, severity, max_steps=None):
            calls.append((task, init, severity, max_steps))
            rng = np.random.default_rng(task*100+init)
            actions = rng.uniform(-.5, .5, (10+task, 6))
            records = []
            for index, action in enumerate(actions):
                executed = action+severity
                motion = .7*executed*error_signal.OUT_MAX
                gripper = 1.0 if index % 2 else -1.0
                records.append(dict(a_cmd=action.tolist(), a_exec=executed.tolist(),
                                    command=[*action, gripper], executed=[*executed, gripper],
                                    dx=motion[:3].tolist(), dr=motion[3:].tolist()))
            return records, task == 0
        with contextlib.redirect_stdout(io.StringIO()):
            artifact = error_signal.collect(args, probe, logger or episode)
        return artifact, calls

    def test_two_condition_collection_roundtrip_fir_and_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.args(folder)
            artifact, calls = self.collect(args)
            self.assertEqual(artifact, json.loads(args.out.read_text()))
            self.assertEqual(artifact["status"], "complete")
            self.assertEqual(len(artifact["records"]), 2)  # regression: list became dict after first condition
            self.assertEqual(artifact["calib_episodes"], [[0, 26], [1, 26], [0, 27]])
            self.assertEqual([(task, init) for task, init, _, _ in calls[:3]],
                             [(0, 26), (1, 26), (0, 27)])
            self.assertEqual(len(calls), 6)
            self.assertTrue(all(count == 12 for _, _, _, count in calls))
            self.assertEqual(artifact["records"][0]["ep_len"], [10, 11, 10])
            self.assertTrue(artifact["source_hashes"])
            self.assertEqual(artifact["policy_control_ack"]["pin_rng"], False)
            legacy_path = Path(folder)/"legacy.json"
            legacy_path.write_text(json.dumps(artifact["records"]))
            with contextlib.redirect_stdout(io.StringIO()):
                wrapped_fit = adaptive_law.fit_plant(args.out)
                self.assertEqual(adaptive_law.CALIB_EPISODES, artifact["calib_episodes"])
                legacy_fit = adaptive_law.fit_plant(legacy_path)
            np.testing.assert_allclose(wrapped_fit, legacy_fit)
            for episode, expected_len, expected_key in [(0, 10, [0, 26]), (1, 11, [1, 26])]:
                commands, source = openloop_id.nominal_commands(artifact, episode, 80)
                self.assertEqual(commands.shape, (expected_len, 7))
                self.assertEqual(source["scenario"], expected_key)
                self.assertEqual(source["gripper"], "recorded")
                self.assertEqual(commands[1, 6], 1.0)
            legacy = copy.deepcopy(artifact["records"])
            legacy[0].pop("raw_cmd")
            commands, source = openloop_id.nominal_commands(legacy, 0, 80)
            self.assertEqual(commands.shape, (10, 6))
            self.assertEqual(source["gripper"], "legacy_missing_held_open_minus_one")

    def test_healthy_only_and_invalid_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            artifact, calls = self.collect(self.args(folder, healthy_only=True))
            self.assertEqual(len(artifact["records"]), 1)
            self.assertTrue(all(severity == 0 for _, _, severity, _ in calls))
            for status in ("running", "failed"):
                invalid = copy.deepcopy(artifact)
                invalid["status"] = status
                with self.assertRaisesRegex(ValueError, "not complete"):
                    openloop_id.nominal_commands(invalid)
            malformed = copy.deepcopy(artifact)
            malformed["records"][0]["ep_len"][0] += 1
            with self.assertRaisesRegex(ValueError, "match ep_len"):
                openloop_id.nominal_commands(malformed)
            malformed = copy.deepcopy(artifact)
            malformed["records"][0]["raw_cmd"][0][0] += 1
            with self.assertRaisesRegex(ValueError, "disagrees"):
                openloop_id.nominal_commands(malformed)

    def test_failed_collection_retains_failed_status(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.args(folder)
            def fail(*args, **kwargs):
                raise RuntimeError("mock policy unavailable")
            with self.assertRaisesRegex(RuntimeError, "mock policy unavailable"):
                self.collect(args, fail)
            artifact = json.loads(args.out.read_text())
            self.assertEqual(artifact["status"], "failed")
            self.assertEqual(artifact["error"]["type"], "RuntimeError")

    def test_sensitivity_roundtrip_uses_one_source_episode_and_common_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.args(folder, healthy_only=True)
            artifact, _ = self.collect(args)
            target = Path(folder)/"other"/"M.json"
            expected = np.eye(6)*.8
            expected[2, 0] = .15
            calls, closed = [], []
            def environment(suite, task):
                self.assertEqual((suite, task), ("libero_spatial", 1))
                return SimpleNamespace(close=lambda: closed.append(True)), range(50)
            def replay(env, inits, initial, commands, fault):
                self.assertEqual(initial, 25)
                self.assertEqual(commands.shape, (11, 7))
                calls.append(fault.copy())
                # A time-varying nominal trajectory plus asymmetric termination:
                # unequal prefixes would confound its drift with sensitivity.
                length = 3 if fault[0] > 0 else 7
                base = np.arange(length)[:, None]*np.arange(1, 7)[None, :]*.01
                motion = base + openloop_id.OUT*(expected@fault)
                return motion, np.zeros((length, 6))
            with contextlib.redirect_stdout(io.StringIO()):
                result = openloop_id.main(["--log", str(args.out), "--out", str(target),
                    "--source-episode", "1", "--steps", "80", "--probe-init", "25"],
                    environment_factory=environment, replay_fn=replay)
            self.assertEqual(result, json.loads(target.read_text()))
            np.testing.assert_allclose(result["M"], expected, atol=1e-12)
            self.assertEqual(len(calls), 17)
            self.assertEqual(closed, [True])
            self.assertEqual(result["columns"][0]["common_steps"], 3)
            self.assertEqual(result["probe_episodes"], [[1, 25]])
            self.assertEqual(result["command_source"]["scenario"], [1, 26])
            self.assertEqual(result["source_calib_episodes"], artifact["calib_episodes"])
            self.assertIn(str(args.out.resolve()), result["source_hashes"])

    def test_replay_preserves_gripper_and_discloses_legacy_fallback(self):
        calls = []
        observation = dict(robot0_eef_pos=np.zeros(3), robot0_eef_quat=np.array([0., 0., 0., 1.]))
        env = SimpleNamespace(reset=lambda: None, set_init_state=lambda _: observation,
                              step=lambda action: (calls.append(action) or observation, 0, False, {}))
        fake_main = SimpleNamespace(LIBERO_DUMMY_ACTION=[0.]*7, _quat2axisangle=lambda _: np.zeros(3))
        with mock.patch.dict(sys.modules, {"main": fake_main}):
            openloop_id.replay(env, [None], 0, [[0., 0., 0., 0., 0., 0., 1.]], np.full(6, .02))
            self.assertEqual(calls[-1], [.02]*6+[1.])
            openloop_id.replay(env, [None], 0, [[0., 0., 0., 0., 0., 0.]], np.full(6, .02))
            self.assertEqual(calls[-1], [.02]*6+[-1.])


if __name__ == "__main__":
    unittest.main(verbosity=2)
