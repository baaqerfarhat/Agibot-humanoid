"""Simulator-free checks of the opt-in pose tracking term and the replay mode of adaptive_law.py.

Only the external packages are stubbed (LIBERO, openpi_client, examples/libero/main.py, tqdm);
adaptive_law, paired_probe, error_signal, libero_reset, gate_faults, so3 and joint_fault run
unchanged against a deterministic synthetic plant: first-order increments per Cartesian channel
(normalised units), an integrating pose and an xyzw quaternion.

Run:  python -B -m unittest openpi/test_pose_tracking.py -v
Byte identity against another runner version (e.g. a clean HEAD copy):
      ADAPTIVE_LAW_REFERENCE=/path/to/adaptive_law.py python -B -m unittest openpi/test_pose_tracking.py -v
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import pathlib
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adaptive_law  # noqa: E402
import so3  # noqa: E402

OUT = adaptive_law.OUT
POLES = np.array([0.5, 0.6, 0.7, 0.83, 0.93, 0.8])      # increment poles (rx, ry as measured on the Panda)
GAINS = np.array([0.30, 0.27, 0.25, 0.25, 0.28, 0.24])   # DC gains, normalised motion per command unit
BASE = np.array([0.10, -0.05, 0.90])
M_PROBED = np.diag(GAINS) + 0.004 * np.array([[0, 1, -1, 0, 1, 0], [1, 0, 0, -1, 0, 0], [0, 1, 0, 0, 0, -1],
                                              [0, 0, 1, 0, -1, 0], [1, 0, 0, 1, 0, 0], [0, -1, 0, 0, 1, 0]])
SHIPPED = ["--gamma", "0.08", "--dead", "0.008", "--norm-r", "0.15", "--clip", "0.15", "--corr-dims", "3,4,5"]


def goal(task):
    return BASE + np.array([0.08 * np.cos(task), 0.08 * np.sin(task), 0.05])


def goal_rotation(task):
    return np.array([0.1 * np.sin(task), -0.1, 0.05 * np.cos(task)])


def quat_exp(w):
    angle = float(np.linalg.norm(w))
    if angle == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.r_[w / angle * np.sin(angle / 2), np.cos(angle / 2)]


class SyntheticPanda:
    """Deterministic stand-in for a LIBERO env; done when the goal is reached (if a tolerance is set)."""

    def __init__(self, task=0, tolerance=0.01):
        self.task, self.tolerance = int(task), tolerance
        self.sim = SimpleNamespace(model=SimpleNamespace(), data=SimpleNamespace(
            qpos=np.zeros(9), qvel=np.zeros(9), qfrc_applied=np.zeros(9), xfrc_applied=np.zeros((3, 6))))
        self.robots = [SimpleNamespace(_joint_positions=np.zeros(7), _joint_velocities=np.zeros(7),
                                       torques=np.zeros(7),
                                       controller=SimpleNamespace(goal_pos=np.zeros(3), goal_ori=np.eye(3)))]
        self.seeds, self.closed = [], False
        self._restart(np.zeros(3))

    def _restart(self, state):
        self.velocity = np.zeros(6)
        self.position = BASE + np.asarray(state, float)[:3]
        self.quaternion = np.array([0.0, 0.0, 0.0, 1.0])

    def seed(self, value):
        self.seeds.append(value)

    def reset(self):
        self._restart(np.zeros(3))
        return self.observation()

    def set_init_state(self, state):
        self._restart(state)
        return self.observation()

    def observation(self):
        return dict(agentview_image=np.zeros((4, 4, 3), np.uint8),
                    robot0_eye_in_hand_image=np.zeros((4, 4, 3), np.uint8),
                    robot0_eef_pos=self.position.copy(), robot0_eef_quat=self.quaternion.copy(),
                    robot0_gripper_qpos=np.zeros(2))

    def step(self, action):
        action = np.asarray(action, float)
        self.velocity = POLES * self.velocity + (1.0 - POLES) * GAINS * action[:6]
        self.position = self.position + self.velocity[:3] * OUT[:3]
        quaternion = so3.qmul(quat_exp(self.velocity[3:] * OUT[3:]), self.quaternion)
        self.quaternion = quaternion / np.linalg.norm(quaternion)
        robot = self.robots[0]
        robot._joint_positions = np.r_[self.position, self.quaternion]
        robot._joint_velocities = np.r_[self.velocity, 0.0]
        done = self.tolerance is not None and np.linalg.norm(self.position - goal(self.task)) < self.tolerance
        return self.observation(), 0.0, bool(done), {}

    def close(self):
        self.closed = True


class FakeSuite:
    n_tasks = 10

    def get_task(self, task):
        return SimpleNamespace(task_id=int(task))

    def get_task_init_states(self, task):
        return [np.array([0.004 * i, -0.003 * task, 0.002 * i]) for i in range(50)]


class FakePolicy:
    """Deterministic closed-loop policy; also plays the server's side of the control/ack handshake."""
    control = ack = None
    constructed = 0

    def __init__(self, host=None, port=None):
        type(self).constructed += 1

    def infer(self, observation):
        if FakePolicy.control is not None and FakePolicy.control.exists() and not FakePolicy.ack.exists():
            FakePolicy.ack.write_text(json.dumps(dict(json.loads(FakePolicy.control.read_text()), ok=True)))
        state = np.asarray(observation["observation/state"], float)
        task = int(str(observation["prompt"]).split()[-1])
        command = np.r_[np.clip(0.08 * (goal(task) - state[:3]) / OUT[:3], -1, 1),
                        np.clip(0.1 * (goal_rotation(task) - state[3:6]) / OUT[3:], -1, 1)]
        return {"actions": np.array([np.r_[command * 0.97 ** k, -1.0] for k in range(10)])}


class RefusingPolicy:
    def __init__(self, *args, **kwargs):
        raise AssertionError("replay mode must not construct a policy client")


def stub_modules(policy=FakePolicy, tolerance=0.01):
    main = ModuleType("main")
    main.LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    main.LIBERO_ENV_RESOLUTION = 256
    main._quat2axisangle = so3.axis_angle
    main._get_libero_env = lambda task, resolution, seed: (SyntheticPanda(task.task_id, tolerance),
                                                           f"synthetic task {task.task_id}")
    image_tools = ModuleType("openpi_client.image_tools")
    image_tools.convert_to_uint8 = lambda image: image
    image_tools.resize_with_pad = lambda image, height, width: image
    websocket = ModuleType("openpi_client.websocket_client_policy")
    websocket.WebsocketClientPolicy = policy
    client = ModuleType("openpi_client")
    client.image_tools, client.websocket_client_policy = image_tools, websocket
    benchmark = ModuleType("libero.libero.benchmark")
    benchmark.get_benchmark_dict = lambda: {"libero_spatial": FakeSuite, "libero_10": FakeSuite}
    libero, libero_libero = ModuleType("libero"), ModuleType("libero.libero")
    libero.libero, libero_libero.benchmark = libero_libero, benchmark
    tqdm = ModuleType("tqdm")
    tqdm.tqdm = lambda iterable, **kwargs: iterable
    return {"main": main, "openpi_client": client, "openpi_client.image_tools": image_tools,
            "openpi_client.websocket_client_policy": websocket, "libero": libero,
            "libero.libero": libero_libero, "libero.libero.benchmark": benchmark, "tqdm": tqdm}


def dc_consistent_fir(scale=None):
    """Per-axis FIR: the synthetic plant's truncated impulse response, taps summing to its DC gain
    (times `scale` per channel), zero intercept."""
    taps = np.array([(1 - POLES) * POLES ** j for j in range(adaptive_law.K_FIR + 1)]).T
    taps = taps / taps.sum(axis=1, keepdims=True) * GAINS[:, None]
    if scale is not None:
        taps = taps * np.asarray(scale, float)[:, None]
    return np.c_[taps, np.zeros(6)]


def run_rollout(policy_command, steps, *, track=None, W=None, fault=None, gamma=0.0, clip=1.0,
                corr_dims=None, adapt=True, telemetry=None, norm_r=0.15, dead=0.0, replan=5,
                M_inv=None, **kwargs):
    """One adaptive_law.run episode on the synthetic plant with a fixed or callable policy."""
    env = SyntheticPanda(0, tolerance=None)
    calls = []

    def infer(observation):
        calls.append(len(calls))
        command = policy_command(observation) if callable(policy_command) else np.asarray(policy_command, float)
        return {"actions": np.tile(command, (replan, 1))}

    probe = SimpleNamespace(env_for=lambda task: (env, "synthetic task 0", FakeSuite().get_task_init_states(0)),
                            client=SimpleNamespace(infer=infer), a=SimpleNamespace(replan_steps=replan),
                            suite_name="libero_spatial")
    W = np.zeros((6, adaptive_law.K_FIR + 2)) if W is None else W
    M_inv = np.linalg.inv(np.diag(GAINS)) if M_inv is None else M_inv
    with mock.patch.dict(sys.modules, stub_modules()):
        return adaptive_law.run(probe, 0, 0, 0.0, M_inv, W, gamma, adapt,
                                max_steps=steps, fvec=np.zeros(6) if fault is None else np.asarray(fault, float),
                                dead=dead, norm_r=norm_r, clip=clip, corr_dims=corr_dims,
                                telemetry=telemetry, track=track, **kwargs)


def rollout_records(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines() if json.loads(line).get("phase") == "rollout"]


class Workspace:
    """A temporary directory holding a synthetic healthy log (written by error_signal.collect through
    paired_probe.Probe) and an open-loop artifact, plus a main() driver."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.log, self.openloop = self.dir / "healthy.json", self.dir / "openloop.json"
        self.openloop.write_text(json.dumps(dict(M=M_PROBED.tolist())))
        FakePolicy.control, FakePolicy.ack = self.dir / "ctl.json", self.dir / "ack.json"
        args = SimpleNamespace(control=FakePolicy.control, ack=FakePolicy.ack, out=self.log, host="0.0.0.0",
                               port=8000, replan_steps=5, suite="libero_spatial", episodes=3,
                               healthy_only=True, max_steps=220, init_base=25)
        with mock.patch.dict(sys.modules, stub_modules()), contextlib.redirect_stdout(io.StringIO()):
            import error_signal
            import paired_probe
            error_signal.collect(args, paired_probe.Probe(args))
        self.record = json.loads(self.log.read_text())["records"][0]

    def close(self):
        FakePolicy.control = FakePolicy.ack = None
        self.tmp.cleanup()

    def argv(self, *extra, telemetry=True, policy=True):
        argv = ["--log", str(self.log), "--openloop", str(self.openloop), "--out", str(self.dir / "result.json")]
        if policy:
            argv += ["--control", str(FakePolicy.control), "--ack", str(FakePolicy.ack)]
        if telemetry:
            argv += ["--telemetry", str(self.dir / "telemetry.jsonl")]
        return argv + [str(x) for x in extra]

    def main(self, argv, module=adaptive_law, policy=FakePolicy):
        """Run module.main(argv); returns (stdout, result text, telemetry lines) and removes the outputs."""
        stdout = io.StringIO()
        with mock.patch.dict(sys.modules, stub_modules(policy)), \
                mock.patch.object(sys, "argv", ["adaptive_law.py", *argv]), contextlib.redirect_stdout(stdout):
            module.main()
        result = (self.dir / "result.json").read_text()
        telemetry_path = self.dir / "telemetry.jsonl"
        telemetry = telemetry_path.read_text().splitlines() if telemetry_path.exists() else []
        for path in (self.dir / "result.json", telemetry_path):
            path.unlink(missing_ok=True)
        return stdout.getvalue(), result, telemetry


class RecordingEstimator:
    """Wraps estimator_step and records (f_hat, residual, output) for every call, as bytes."""

    def __init__(self):
        self.real, self.calls = adaptive_law.estimator_step, []

    def __call__(self, f_hat, res, M_inv, **kwargs):
        out, diag = self.real(f_hat, res, M_inv, **kwargs)
        self.calls.append((np.asarray(f_hat).tobytes(), np.asarray(res).tobytes(), out.tobytes()))
        return out, diag


class ExplodingTracker:
    def __init__(self, *args, **kwargs):
        raise AssertionError("tracking code entered with kappa 0")


class TrackingOffTests(unittest.TestCase):
    """(a) kappa 0 never enters the tracking code; estimator_step sees and returns the same numbers."""

    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()

    @classmethod
    def tearDownClass(cls):
        cls.ws.close()

    def test_kappa_zero_main_is_identical_to_no_flags_and_never_builds_a_tracker(self):
        base = self.ws.argv("--sev", 0.05, "--episodes", 2, "--scenario-reset", *SHIPPED)
        runs = {}
        for name, extra in (("none", []), ("kappa0", ["--track-kappa", "0", "--track-ref", "fitted",
                                                      "--track-dims", "0,1", "--track-anchor", "3"])):
            recorder = RecordingEstimator()
            with mock.patch.object(adaptive_law, "estimator_step", recorder), \
                    mock.patch.object(adaptive_law, "PoseTracker", ExplodingTracker):
                runs[name] = self.ws.main(base + extra), recorder.calls
        (stdout_none, result_none, tele_none), calls_none = runs["none"]
        (stdout_zero, result_zero, tele_zero), calls_zero = runs["kappa0"]
        self.assertGreater(len(calls_none), 100)
        self.assertEqual(calls_none, calls_zero)                   # inputs and outputs, bit for bit
        self.assertEqual(stdout_none, stdout_zero)
        result_none, result_zero = json.loads(result_none), json.loads(result_zero)
        self.assertTrue(set(adaptive_law.OPT_IN_DEFAULTS).isdisjoint(result_none["args"]))
        self.assertEqual(result_zero["args"]["track_kappa"], 0.0)
        self.assertEqual(result_zero["args"]["track_dims"], "0,1")   # recorded as given, not resolved
        for result in (result_none, result_zero):
            result.pop("args")
            self.assertNotIn("tracking", result)
        self.assertEqual(result_none, result_zero)
        self.assertEqual(tele_none[1:], tele_zero[1:])               # every step record
        self.assertFalse(any(key.startswith("track_") for line in tele_zero[1:] for key in json.loads(line)))
        self.assertNotIn("tracking", json.loads(tele_zero[0])["config"])

    def test_run_without_track_or_with_kappa_zero_is_the_default_path(self):
        command = np.r_[0.2, -0.1, 0.1, 0.2, -0.2, 0.1, -1.0]
        outputs = []
        for track in (None, dict(W_ref=dc_consistent_fir(), kappa=0.0, anchor=5, dims=range(6))):
            stream = io.StringIO()
            with mock.patch.object(adaptive_law, "PoseTracker", ExplodingTracker):
                result = run_rollout(command, 40, track=track, fault=np.full(6, 0.05), gamma=0.08,
                                     W=dc_consistent_fir(), telemetry=stream)
            outputs.append((result[1].tobytes(), json.dumps(result[2]), rollout_records(stream)))
        self.assertEqual(outputs[0], outputs[1])
        self.assertFalse(any(key.startswith("track_") for key in outputs[0][2][0]))


class FixedPointTests(unittest.TestCase):
    """(b) Integrating pose, first-order increments, constant offset f, constant nominal command u.

    Stated limit (tracking alone, all six dims tracked and corrected): f_hat -> f + (I - G^-1 G_ref) u.
    Per axis with a reference DC gain s times the true one: f_hat -> f + (1 - s) u. A DC-consistent
    reference (s = 1) therefore drives the correction to the true offset; s != 1 converges elsewhere.
    With the legacy base law also running (gain gamma, attenuation A = 1/(1 + |G f|^2 / rho^2) and an
    exact plant), the limit is the gain-weighted mean (gamma A f + kappa (f + (1 - s) u)) / (gamma + kappa).
    """
    command = np.r_[0.2, -0.15, 0.1, 0.3, -0.25, 0.2, -1.0]
    fault = np.array([0.03, -0.02, 0.04, 0.05, -0.04, 0.03])

    def converge(self, scale=None, gamma=0.0, W=None, steps=1500):
        track = dict(W_ref=dc_consistent_fir(scale), kappa=0.05, anchor=5, dims=range(6))
        _, f_hat, trajectory = run_rollout(self.command, steps, track=track, fault=self.fault, gamma=gamma, W=W)
        return f_hat, np.array([step["f_hat"] for step in trajectory])

    def test_dc_consistent_reference_drives_the_correction_to_the_true_offset(self):
        f_hat, history = self.converge()
        np.testing.assert_allclose(f_hat, self.fault, rtol=0, atol=1e-10)
        # it got there by moving: the estimate starts at zero
        self.assertGreater(np.abs(history[0] - self.fault).min(), 0.01)

    def test_wrong_dc_gain_converges_to_the_stated_limit(self):
        scale = np.array([1.0, 0.6, 1.0, 1.0, 0.37, 1.25])       # 0.37: the Panda's fitted/probed r_y ratio
        f_hat, _ = self.converge(scale)
        limit = self.fault + (1.0 - scale) * self.command[:6]
        np.testing.assert_allclose(f_hat, limit, rtol=0, atol=1e-10)
        self.assertGreater(np.abs(f_hat - self.fault)[[1, 4, 5]].min(), 0.02)
        np.testing.assert_allclose(f_hat[[0, 2, 3]], self.fault[[0, 2, 3]], rtol=0, atol=1e-10)

    def test_with_the_base_law_the_limit_is_the_gain_weighted_mean(self):
        gamma, kappa, rho, scale = 0.08, 0.05, 0.15, np.array([1.0, 1.0, 1.0, 1.0, 0.37, 1.0])
        f_hat, _ = self.converge(scale, gamma=gamma, W=dc_consistent_fir(), steps=2500)
        attenuation = 1.0 / (1.0 + (np.linalg.norm(GAINS * self.fault) / rho) ** 2)
        tracking_limit = self.fault + (1.0 - scale) * self.command[:6]
        expected = (gamma * attenuation * self.fault + kappa * tracking_limit) / (gamma + kappa)
        np.testing.assert_allclose(f_hat, expected, rtol=0, atol=1e-10)


class AnchorTests(unittest.TestCase):
    """(c) The anchor resets e_p; the reference uses the runner's own FIR arithmetic."""

    def feed(self, anchor, steps=13, seed=3):
        rng = np.random.default_rng(seed)
        W_ref, M_inv = rng.normal(size=(6, adaptive_law.K_FIR + 2)) * 0.1, rng.normal(size=(6, 6))
        tracker = adaptive_law.PoseTracker(W_ref, M_inv, kappa=0.1, anchor=anchor, dims=range(6), clip=0.15)
        u = rng.normal(size=(steps, 7)) * 0.3
        positions = BASE + np.cumsum(rng.normal(size=(steps + 1, 3)) * 0.003, axis=0)
        measured = rng.normal(size=(steps, 6)) * 0.2
        outputs = [tracker.observe(k, u[k], positions[k], positions[k + 1], measured[k]) for k in range(steps)]
        # independent recomputation: histories of the nominal command, windows from the anchors
        padded = np.r_[np.zeros((adaptive_law.K_FIR, 6)), u[:, :6]]
        predicted = []
        for k in range(steps):
            H = padded[k:k + adaptive_law.K_FIR + 1][::-1]
            predicted.append(np.array([W_ref[i, :-1] @ H[:, i] + W_ref[i, -1] for i in range(6)]))
        for k, (error, z, n) in enumerate(outputs):
            start = k - k % anchor if anchor else 0
            self.assertEqual(n, k - start + 1)
            expected = np.r_[(positions[k + 1] - positions[start]) / OUT[:3], measured[start:k + 1, 3:].sum(axis=0)] \
                - np.sum(predicted[start:k + 1], axis=0)
            np.testing.assert_allclose(error, expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(z, M_inv @ (expected / n), rtol=1e-11, atol=1e-11)
        return outputs, positions, measured, predicted

    def test_anchor_resets_the_pose_error(self):
        outputs, positions, measured, predicted = self.feed(anchor=4)
        self.assertEqual([n for _, _, n in outputs], [1, 2, 3, 4] * 3 + [1])
        for k in (0, 4, 8, 12):                     # at an anchor the error is that single step's
            single = np.r_[(positions[k + 1] - positions[k]) / OUT[:3], measured[k, 3:]] - predicted[k]
            np.testing.assert_allclose(outputs[k][0], single, rtol=1e-12, atol=1e-13)

    def test_anchor_zero_accumulates_from_the_first_step(self):
        outputs, *_ = self.feed(anchor=0)
        self.assertEqual([n for _, _, n in outputs], list(range(1, 14)))

    def test_run_anchors_on_the_replan_steps_and_matches_the_residual_exactly(self):
        # Frozen arm, no fault: the command sent is the nominal one, so with W_ref = W and a one-step
        # window the pose error must equal the runner's own FIR residual r bit for bit.
        rng = np.random.default_rng(11)
        W = dc_consistent_fir() + np.c_[np.zeros((6, adaptive_law.K_FIR + 1)), rng.normal(size=6) * 1e-3]

        def policy(observation):
            state = np.asarray(observation["observation/state"], float)
            return np.r_[0.3 * np.sin(10 * state[:3]), 0.2 * np.cos(5 * state[:3]), -1.0]

        for anchor, expected_n in ((1, None), (5, [1, 2, 3, 4, 5])):
            stream = io.StringIO()
            track = dict(W_ref=W, kappa=0.5, anchor=anchor, dims=range(6))
            run_rollout(policy, 23, track=track, W=W, adapt=False, telemetry=stream)
            records = rollout_records(stream)
            self.assertEqual(len(records), 23)
            n = [r["track_n"] for r in records]
            self.assertEqual(n, [1] * 23 if expected_n is None else (expected_n * 5)[:23])
            for record in records:
                self.assertEqual(record["f_hat"], [0.0] * 6)          # frozen: measured, never applied
                if record["track_n"] == 1:
                    self.assertEqual(record["track_e"], record["r"])
                    expected_z = np.linalg.inv(np.diag(GAINS)) @ np.asarray(record["r"])
                    self.assertEqual(record["track_z"], expected_z.tolist())


class ProjectionTests(unittest.TestCase):
    """(d) Only tracked dims move; they are projected onto the box; untracked dims are the base law's."""

    def test_correct_keeps_untracked_dims_bit_for_bit_and_clips_tracked_dims(self):
        tracker = adaptive_law.PoseTracker(dc_consistent_fir(), np.eye(6), kappa=2.0, anchor=5, dims=[3, 4], clip=0.15)
        f_hat = np.array([-0.0, 0.3, -0.2, 0.1, -0.1, 0.149])
        out = tracker.correct(f_hat, np.array([9.0, 9.0, 9.0, 0.02, -1.0, 9.0]))
        for d in (0, 1, 2, 5):
            self.assertEqual(out[d:d + 1].tobytes(), f_hat[d:d + 1].tobytes())   # incl. -0.0 and out-of-box
        self.assertEqual(out[3], 0.1 + 2.0 * 0.02)
        self.assertEqual(out[4], -0.15)
        self.assertIsNot(out, f_hat)

    def test_run_adds_tracking_after_the_base_update_on_tracked_dims_only(self):
        recorder = RecordingEstimator()
        stream = io.StringIO()
        track = dict(W_ref=dc_consistent_fir(), kappa=3.0, anchor=5, dims=[3, 4])
        with mock.patch.object(adaptive_law, "estimator_step", recorder):
            run_rollout(np.r_[0.2, -0.1, 0.1, 0.2, -0.2, 0.1, -1.0], 60, track=track, W=dc_consistent_fir(),
                        fault=np.full(6, 0.05), gamma=0.08, clip=0.15, corr_dims=[3, 4, 5], telemetry=stream)
        records = rollout_records(stream)
        self.assertEqual(len(records), len(recorder.calls))
        railed = 0
        for record, (_, _, base_bytes) in zip(records, recorder.calls):
            base = np.frombuffer(base_bytes)
            f_hat = np.asarray(record["f_hat"])
            np.testing.assert_array_equal(f_hat[[0, 1, 2, 5]], base[[0, 1, 2, 5]])
            expected = np.clip(base[[3, 4]] + 3.0 * np.asarray(record["track_z"])[[3, 4]], -0.15, 0.15)
            np.testing.assert_array_equal(f_hat[[3, 4]], expected)
            self.assertTrue(np.all(np.abs(f_hat) <= 0.15))
            railed += int(np.any(np.abs(f_hat[[3, 4]]) == 0.15))
        self.assertGreater(railed, 0)


class PositionModeTests(unittest.TestCase):
    """--track-mode position: z_T = M_inv e_p, the retained pose offset, with a leaky reference.

    Rate mode feeds back e_p/n, the mean pose rate since the anchor, so with --track-anchor 0 it drives
    the mean rate to zero and leaves the retained offset in place. Position mode feeds the offset back.
    The leak makes e_p a leaky sum: a constant per-step model error d settles at d/leak instead of
    integrating without bound.
    """
    command = np.r_[0.2, -0.15, 0.1, 0.3, -0.25, 0.2, -1.0]
    fault = np.full(6, 0.05)

    def tracker(self, **kwargs):
        options = dict(W_ref=dc_consistent_fir(), M_inv=np.eye(6), kappa=0.1, anchor=0, dims=range(6),
                       clip=0.15, mode="position")
        options.update(kwargs)
        return adaptive_law.PoseTracker(options.pop("W_ref"), options.pop("M_inv"), **options)

    def test_position_mode_is_the_undivided_error_and_the_leak_follows_its_recursion(self):
        rng = np.random.default_rng(5)
        for leak in (0.0, 0.05, 0.3):
            M_inv = rng.normal(size=(6, 6))
            tracker = self.tracker(M_inv=M_inv, leak=leak, anchor=4, W_ref=rng.normal(size=(6, adaptive_law.K_FIR + 2)) * 0.1)
            u = rng.normal(size=(12, 7)) * 0.3
            positions = BASE + np.cumsum(rng.normal(size=(13, 3)) * 0.003, axis=0)
            measured = rng.normal(size=(12, 6)) * 0.2
            previous = None
            for k in range(12):
                error, z, n = tracker.observe(k, u[k], positions[k], positions[k + 1], measured[k])
                np.testing.assert_allclose(z, M_inv @ error, rtol=0, atol=1e-14)   # no division by n
                self.assertEqual(n, k % 4 + 1)
                if previous is not None and k % 4:
                    # e_p(k) = (1 - leak) e_p(k-1) + (pose increment - predicted increment)
                    step = np.r_[(positions[k + 1] - positions[k]) / OUT[:3], measured[k, 3:]]
                    np.testing.assert_allclose(error - (1 - leak) * previous, step - self.predicted(tracker, u, k),
                                               rtol=0, atol=1e-12)
                previous = error

    @staticmethod
    def predicted(tracker, u, k):
        padded = np.r_[np.zeros((adaptive_law.K_FIR, 6)), u[:, :6]]
        return adaptive_law.fir_increment(tracker.W_ref, padded[k:k + adaptive_law.K_FIR + 1][::-1])

    def test_leak_bounds_drift_from_a_constant_model_error(self):
        # A zero reference plant and a constant per-step motion make every step's model error exactly d.
        d = np.array([0.02, -0.01, 0.03, -0.04, 0.05, 0.01])
        for leak in (0.0, 0.01, 0.05, 0.25):
            tracker = self.tracker(W_ref=np.zeros((6, adaptive_law.K_FIR + 2)), M_inv=np.eye(6), leak=leak)
            positions = BASE + np.outer(np.arange(301), d[:3] * OUT[:3])
            errors = [tracker.observe(k, np.zeros(7), positions[k], positions[k + 1], d)[0] for k in range(300)]
            errors = np.array(errors)
            if leak:
                expected = np.outer(1.0 - (1.0 - leak) ** np.arange(1, 301), d / leak)
                np.testing.assert_allclose(errors, expected, rtol=0, atol=1e-11)
                self.assertTrue(np.all(np.abs(errors) <= np.abs(d / leak) + 1e-12))    # bounded by d/leak
                remaining = np.abs(d / leak) * (1.0 - leak) ** 300                     # and converging to it
                self.assertTrue(np.all(np.abs(errors[-1] - d / leak) <= remaining + 1e-11))
            else:
                np.testing.assert_allclose(errors, np.outer(np.arange(1, 301), d), rtol=0, atol=1e-11)

    def converge(self, track, steps=2500, **kwargs):
        stream = io.StringIO()
        options = dict(gamma=0.08, norm_r=0.03, W=dc_consistent_fir(), fault=self.fault, telemetry=stream)
        options.update(kwargs)
        _, f_hat, _ = run_rollout(self.command, steps, track=track, **options)
        return f_hat, np.array([r["track_e"] for r in rollout_records(stream)])

    def test_position_removes_the_velocity_error_that_rate_mode_leaves(self):
        # Legacy base law: it settles at A f, below the fault. Then (PoseTracker docstring) position mode
        # holds the balance offset (gamma/kappa)(1 - A) G f with zero pose rate, while rate mode settles
        # at the gain-weighted mean, whose residual velocity error makes the offset grow without bound.
        G, gamma = np.diag(GAINS), 0.08
        attenuation = 1.0 / (1.0 + (np.linalg.norm(GAINS * self.fault) / 0.03) ** 2)
        self.assertLess(attenuation, 0.5)
        balances = {}
        for kappa in (0.002, 0.005):
            f_hat, errors = self.converge(dict(W_ref=dc_consistent_fir(), kappa=kappa, anchor=0,
                                               dims=range(6), mode="position"))
            np.testing.assert_allclose(f_hat, self.fault, rtol=0, atol=1e-9)          # no velocity error left
            balance = (gamma / kappa) * (1 - attenuation) * (G @ self.fault)
            np.testing.assert_allclose(errors[-1], balance, rtol=0, atol=1e-6)
            np.testing.assert_allclose(errors[-1], errors[-1000], rtol=0, atol=1e-6)  # stationary, not drifting
            balances[kappa] = np.linalg.norm(errors[-1])
        self.assertAlmostEqual(balances[0.005], 0.4 * balances[0.002], places=6)      # toward zero as kappa grows

        kappa = 0.05
        f_hat, errors = self.converge(dict(W_ref=dc_consistent_fir(), kappa=kappa, anchor=0, dims=range(6)))
        expected = (gamma * attenuation + kappa) / (gamma + kappa) * self.fault
        # anchor 0 divides by n, so rate mode approaches its limit slowly: 7e-6 after 2500 steps
        np.testing.assert_allclose(f_hat, expected, rtol=0, atol=1e-4)
        self.assertGreater(np.abs(self.fault - f_hat).min(), 0.01)                    # still below the fault
        growth = errors[-1] - errors[-1001]                  # the offset grows at the residual velocity error
        np.testing.assert_allclose(growth, 1000 * (G @ (self.fault - f_hat)), rtol=1e-3, atol=0)
        self.assertGreater(np.linalg.norm(errors[-1]), 20.0)

    def test_the_leak_makes_the_dc_gain_kappa_over_leak(self):
        # Docstring claim: against a base pulling toward f_base = A f at rate gamma, a leaking position
        # term settles at (gamma f_base + (kappa/leak) f_track) / (gamma + kappa/leak), f_track = f here.
        gamma, leak = 0.08, 0.02
        attenuation = 1.0 / (1.0 + (np.linalg.norm(GAINS * self.fault) / 0.03) ** 2)
        for kappa in (0.002, 0.005):
            f_hat, _ = self.converge(dict(W_ref=dc_consistent_fir(), kappa=kappa, anchor=0, dims=range(6),
                                          mode="position", leak=leak))
            rate = kappa / leak
            expected = (gamma * attenuation + rate) / (gamma + rate) * self.fault
            np.testing.assert_allclose(f_hat, expected, rtol=0, atol=1e-6)
            self.assertLess(np.abs(f_hat - self.fault).max(), (1 - attenuation) * 0.05)   # closer than the base

    def test_position_drives_the_retained_offset_to_zero_on_an_unbiased_base(self):
        base = dict(law="innov", M=np.diag(GAINS), dead=0.0, norm_r=0.15)
        _, position = self.converge(dict(W_ref=dc_consistent_fir(), kappa=0.002, anchor=0, dims=range(6),
                                         mode="position"), **base)
        _, rate = self.converge(dict(W_ref=dc_consistent_fir(), kappa=0.05, anchor=0, dims=range(6)), **base)
        self.assertLess(np.linalg.norm(position[-1]), 1e-9)
        self.assertGreater(np.linalg.norm(rate[-1]), 1e-2)

    def test_leak_bounds_an_uncorrected_channel_in_closed_loop(self):
        # Translation is neither corrected nor tracked, so its pose error is a constant per-step model
        # error for the tracker; M_inv couples it into the tracked dims' z.
        drift = GAINS[:3] * self.fault[:3]
        for leak in (0.0, 0.02, 0.05):
            track = dict(W_ref=dc_consistent_fir(), kappa=0.002, anchor=0, dims=[3, 5], mode="position", leak=leak)
            _, errors = self.converge(track, steps=2000, corr_dims=[3, 4, 5], M_inv=np.linalg.inv(M_PROBED),
                                      norm_r=0.15, dead=0.0, clip=0.15)
            if leak:
                np.testing.assert_allclose(errors[-1, :3], drift / leak, rtol=0, atol=1e-6)
                self.assertTrue(np.all(np.abs(errors[:, :3]) <= drift / leak + 1e-6))
            else:
                np.testing.assert_allclose(errors[-1, :3] - errors[-1001, :3], 1000 * drift, rtol=1e-6, atol=0)
                self.assertGreater(np.abs(errors[-1, :3]).min(), 20.0)


class TrackModeArgumentTests(unittest.TestCase):
    """The mode flags are opt-in: rate mode records and prints exactly what it did before they existed."""

    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()

    @classmethod
    def tearDownClass(cls):
        cls.ws.close()

    def test_explicit_rate_mode_is_the_recorded_default(self):
        base = self.ws.argv("--sev", 0.05, "--episodes", 1, "--track-kappa", 0.05, *SHIPPED)
        implicit = self.ws.main(base)
        explicit = self.ws.main(base + ["--track-mode", "rate", "--track-leak", "0"])
        self.assertEqual(implicit[:2], explicit[:2])       # stdout and result JSON, including args
        self.assertEqual(implicit[2][1:], explicit[2][1:])  # every telemetry step record
        headers = [json.loads(run[2][0]) for run in (implicit, explicit)]
        for header in headers:
            header.pop("argv")                             # the two command lines do differ
        self.assertEqual(*headers)
        arguments = json.loads(implicit[1])["args"]
        self.assertTrue({"track_mode", "track_leak"}.isdisjoint(arguments))
        self.assertEqual(arguments["track_kappa"], 0.05)

    def test_position_mode_is_recorded_everywhere(self):
        import re4_record
        stdout, result, telemetry = self.ws.main(self.ws.argv(
            "--sev", 0.05, "--episodes", 1, "--track-kappa", 0.002, "--track-mode", "position",
            "--track-leak", 0.02, "--track-anchor", 0, "--track-dims", "3,5", *SHIPPED))
        self.assertIn("position mode: z_T = M_inv e_p", stdout)
        result = json.loads(result)
        self.assertEqual((result["args"]["track_mode"], result["args"]["track_leak"]), ("position", 0.02))
        self.assertEqual((result["tracking"]["mode"], result["tracking"]["leak"]), ("position", 0.02))
        header = json.loads(telemetry[0])
        self.assertEqual(header["config"]["tracking"]["mode"], "position")
        self.assertIn("position mode", header["fields"]["track_z"])
        rollout = [json.loads(line) for line in telemetry[1:]]
        rollout = [r for r in rollout if r.get("phase") == "rollout"]
        self.assertTrue(rollout and all("track_e" in r for r in rollout))
        # z_T is the undivided error mapped through M_inv, on a run of the real runner
        M_inv = np.linalg.inv(np.array(header["config"]["M"]))
        for record in rollout[:20]:
            np.testing.assert_allclose(record["track_z"], M_inv @ np.asarray(record["track_e"]), rtol=1e-12, atol=1e-14)
        path = self.ws.dir / "position.json"
        path.write_text(json.dumps(result))
        with contextlib.redirect_stdout(io.StringIO()):
            re4_record.cmd_record(SimpleNamespace(result=path, root=self.ws.dir / "records", part="p",
                                                  run_id="position", cohort=None, timing=None))
        config = json.loads((self.ws.dir / "records" / "p" / "position" / "run_configuration.json").read_text())
        self.assertEqual((config["track_mode"], config["track_leak"]), ("position", 0.02))
        self.assertEqual(config["track_dims"], [3, 5])

    def test_argument_errors(self):
        for extra in (["--track-leak", "0.1"],                      # leak needs position mode
                      ["--track-leak", "-0.1", "--track-mode", "position"],
                      ["--track-leak", "1.5", "--track-mode", "position"],
                      ["--track-leak", "nan", "--track-mode", "position"],
                      ["--track-mode", "sideways"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                self.ws.main(self.ws.argv("--track-kappa", 0.01, *extra))


INNOV_GRID = HERE.parent / "results" / "tracking_pretest" / "innov_grid"
NAMES = {0: "x", 1: "y", 2: "z", 3: "r_x", 4: "r_y", 5: "r_z"}


def translation_forcing_shares(path):
    """Offline, on logged telemetry (recorded with the full observation): split each tracked dim's
    forcing z_T[d] = M_inv[d] @ e_p into its translation-sourced, tracked-rotation and untracked-rotation
    parts, and recompute the tracked-subspace observation solve(M[R,R], e_p[R]), M = inv(M_inv), on the
    same e_p. The forcing kappa z_T[d] is what tracking adds to f_hat[d] each step, so its sum over an
    episode is the integrated forcing; kappa cancels in every share. Position-mode telemetry only."""
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    header = rows[0]
    tracking = header["config"]["tracking"]
    if tracking.get("mode", "rate") != "position":
        raise ValueError("the share decomposition is written for position-mode telemetry (v = e_p)")
    M_inv = np.array(header["config"]["M_inv"])
    M = np.linalg.inv(M_inv)
    R = list(tracking["dims"])
    if min(R) < 3:
        raise ValueError("the split assumes rotation-only tracked dims (translation is the untracked source)")
    T, U = [0, 1, 2], [d for d in range(3, 6) if d not in R]
    steps = [r for r in rows[1:] if r.get("type") == "step" and r.get("phase") == "rollout"]
    E = np.array([r["track_e"] for r in steps])
    logged = np.array([r["track_z"] for r in steps])
    episodes = np.array([r["episode"] for r in steps])
    tracked = np.zeros_like(E)
    tracked[:, R] = np.linalg.solve(M[np.ix_(R, R)], E[:, R].T).T
    without_translation = E.copy()
    without_translation[:, T] = 0.0
    tracked_without = np.zeros_like(E)
    tracked_without[:, R] = np.linalg.solve(M[np.ix_(R, R)], without_translation[:, R].T).T
    out = dict(steps=len(steps), episodes=int(len(set(episodes.tolist()))), kappa=tracking["kappa"],
               dims=R, logged_vs_recomputed=float(np.abs(logged - E @ M_inv.T).max()),
               tracked_independent_of_translation=bool(np.array_equal(tracked, tracked_without)))
    for d in R:
        parts = dict(translation=E[:, T] @ M_inv[d, T], tracked_rotation=E[:, R] @ M_inv[d, R],
                     untracked_rotation=E[:, U] @ M_inv[d, U])
        total = sum(parts.values())
        per_episode = [parts["translation"][episodes == e].sum() / total[episodes == e].sum()
                       for e in sorted(set(episodes.tolist()))]
        magnitude = np.abs(parts["translation"]).sum() / sum(np.abs(p).sum() for p in parts.values())
        out[NAMES[d]] = dict(
            full=dict(integrated={k: float(v.sum()) for k, v in parts.items()}, total=float(total.sum()),
                      translation_share=float(parts["translation"].sum() / total.sum()),
                      untracked_rotation_share=float(parts["untracked_rotation"].sum() / total.sum()),
                      translation_share_per_episode=[float(s) for s in per_episode],
                      translation_magnitude_share=float(magnitude)),
            tracked=dict(total=float(tracked[:, d].sum()), translation_share=0.0,
                         ratio_to_full=float(tracked[:, d].sum() / total.sum())))
    residual_fault = np.r_[np.full(3, 0.05), np.zeros(len(U))]       # uncorrected translation, r_y corrected
    bias = np.linalg.solve(M[np.ix_(R, R)], M[np.ix_(R, T + U)] @ residual_fault) / 0.05
    out["dc_bias_of_tracked"] = {NAMES[d]: float(b) for d, b in zip(R, bias)}
    return out


class TrackedObservationTests(unittest.TestCase):
    """--track-obs tracked: z_T[R] = solve(M[R,R], v[R]), z_T[not R] = 0, M = inv(M_inv)."""

    def feed(self, obs, mode, untracked_scale, M_inv, dims=(3, 5), steps=40, seed=9):
        rng = np.random.default_rng(seed)
        W_ref = rng.normal(size=(6, adaptive_law.K_FIR + 2)) * 0.1
        tracker = adaptive_law.PoseTracker(W_ref, M_inv, kappa=0.1, anchor=0, dims=dims, clip=0.15,
                                           mode=mode, leak=0.0, obs=obs)
        u = rng.normal(size=(steps, 7)) * 0.3
        motion = rng.normal(size=(steps, 6)) * 0.01
        untracked = [d for d in range(6) if d not in dims]
        motion[:, untracked] += untracked_scale * 0.02 * np.arange(1, steps + 1)[:, None]   # growing error
        positions = BASE + np.r_[np.zeros((1, 3)), np.cumsum(motion[:, :3] * OUT[:3], axis=0)]
        return [tracker.observe(k, u[k], positions[k], positions[k + 1], motion[k]) for k in range(steps)]

    def test_tracked_observation_ignores_the_untracked_channels_exactly(self):
        rng = np.random.default_rng(4)
        M = np.eye(6) * 0.25 + rng.normal(size=(6, 6)) * 0.03             # coupled probed sensitivity
        M_inv = np.linalg.inv(M)
        for mode in ("rate", "position"):
            for dims in ((3, 5), (0, 2, 4)):
                quiet = self.feed("tracked", mode, 0.0, M_inv, dims)
                loud = self.feed("tracked", mode, 1.0, M_inv, dims)
                full_quiet = self.feed("full", mode, 0.0, M_inv, dims)
                full_loud = self.feed("full", mode, 1.0, M_inv, dims)
                untracked = [d for d in range(6) if d not in dims]
                for (e_q, z_q, n), (e_l, z_l, _), (_, zf_q, _), (_, zf_l, _) in zip(quiet, loud, full_quiet, full_loud):
                    v = e_q if mode == "position" else e_q / n
                    expected = np.zeros(6)
                    expected[list(dims)] = np.linalg.solve(np.linalg.inv(M_inv)[np.ix_(dims, dims)], v[list(dims)])
                    np.testing.assert_array_equal(z_q, expected)
                    np.testing.assert_array_equal(z_q[untracked], 0.0)
                    self.assertEqual(z_q.tobytes(), z_l.tobytes())    # the growing error does not enter at all
                    dv = (e_l - e_q) if mode == "position" else (e_l - e_q) / n
                    np.testing.assert_allclose(zf_l - zf_q, M_inv[:, untracked] @ dv[untracked], rtol=0, atol=1e-12)
                self.assertGreater(np.abs(zf_l - zf_q)[list(dims)].max(), 0.1)  # full does feel it

    def test_tracked_trades_the_coupling_for_the_verifiers_dc_bias(self):
        # One step of DC pose error e = M f from the probed M actually used on the robot: full recovers
        # f on R, tracked attributes the physical coupling of the uncorrected translation fault to R.
        M = np.array(json.loads((HERE.parent / "results" / "phase05" / "openloop_so3.json").read_text())["M"])
        M_inv = np.linalg.pinv(M)
        f = np.r_[0.05, 0.05, 0.05, 0.05, 0.0, 0.05]          # r_y's residual 0: corrected by the base
        e = M @ f
        results = {}
        for obs in ("full", "tracked"):
            tracker = adaptive_law.PoseTracker(np.zeros((6, adaptive_law.K_FIR + 2)), M_inv, kappa=0.1, anchor=0,
                                               dims=[3, 5], clip=0.15, mode="position", obs=obs)
            results[obs] = tracker.observe(0, np.zeros(7), BASE, BASE + e[:3] * OUT[:3], e)[1]
        np.testing.assert_allclose(results["full"][[3, 5]], f[[3, 5]], rtol=0, atol=1e-13)
        bias = results["tracked"][[3, 5]] / 0.05 - 1.0
        np.testing.assert_allclose(bias, [-0.019625, 0.008626], rtol=0, atol=1e-6)   # verifier: -1.9 %, +0.8 %

    def test_closed_loop_tracked_removes_the_coupling_and_full_shows_it(self):
        # The true plant is diagonal; the probed M (M_PROBED) couples translation into r_x and r_z.
        # Translation is uncorrected and untracked, so with no leak its pose error grows without bound.
        # The base law is off (gamma 0) so that only the tracking term moves f_hat; its position loop is
        # then undamped and f_hat oscillates, which is why whole trajectories are compared, not limits.
        M_inv = np.linalg.inv(M_PROBED)
        runs = {}
        for obs in ("full", "tracked"):
            for name, fault in (("with translation fault", np.full(6, 0.05)),
                                ("rotation fault only", np.r_[0.0, 0.0, 0.0, 0.05, 0.05, 0.05])):
                stream = io.StringIO()
                track = dict(W_ref=dc_consistent_fir(), kappa=0.002, anchor=0, dims=[3, 5], mode="position", obs=obs)
                _, _, trajectory = run_rollout(PositionModeTests.command, 600, track=track, fault=fault, gamma=0.0,
                                               W=dc_consistent_fir(), corr_dims=[3, 4, 5], M_inv=M_inv, clip=0.15,
                                               telemetry=stream)
                records = rollout_records(stream)
                runs[obs, name] = (np.array([s["f_hat"] for s in trajectory]),
                                   np.array([r["track_e"] for r in records]), np.array([r["track_z"] for r in records]))
        tracked = [runs["tracked", name][0] for name in ("with translation fault", "rotation fault only")]
        self.assertEqual(tracked[0].tobytes(), tracked[1].tobytes())         # removed exactly
        full = [runs["full", name][0] for name in ("with translation fault", "rotation fault only")]
        self.assertGreater(np.abs(full[0][:, [3, 5]] - full[1][:, [3, 5]]).max(), 1e-3)
        f_hat, errors, z = runs["full", "with translation fault"]
        coupling = errors[:, :3] @ M_inv[[3, 5]][:, :3].T                      # M_inv[R, xyz] e_xyz
        np.testing.assert_allclose(z[:, [3, 5]] - errors[:, 3:] @ M_inv[[3, 5]][:, 3:].T, coupling, rtol=0, atol=1e-12)
        self.assertGreater(np.linalg.norm(coupling[-1]), 5 * np.linalg.norm(coupling[99]))   # and it grows
        _, errors, z = runs["tracked", "with translation fault"]
        M_RR = M_PROBED[np.ix_([3, 5], [3, 5])]
        np.testing.assert_allclose(z[:, [3, 5]], np.linalg.solve(M_RR, errors[:, [3, 5]].T).T, rtol=0, atol=1e-12)

    @unittest.skipUnless((INNOV_GRID / "CP_k0.001" / "telemetry.jsonl.gz").exists(), "committed telemetry absent")
    def test_offline_translation_share_on_the_committed_robot_telemetry(self):
        shares = translation_forcing_shares(INNOV_GRID / "CP_k0.001" / "telemetry.jsonl.gz")
        self.assertEqual((shares["steps"], shares["episodes"], shares["dims"]), (1006, 10, [3, 5]))
        self.assertLess(shares["logged_vs_recomputed"], 1e-12)       # the full observation, recomputed
        self.assertTrue(shares["tracked_independent_of_translation"])  # tracked: exactly no translation share
        self.assertAlmostEqual(shares["r_x"]["full"]["translation_share"], 0.0985, delta=5e-4)
        self.assertAlmostEqual(shares["r_z"]["full"]["translation_share"], -0.0241, delta=5e-4)
        self.assertAlmostEqual(shares["r_x"]["tracked"]["ratio_to_full"], 0.886, delta=5e-4)
        self.assertAlmostEqual(shares["r_z"]["tracked"]["ratio_to_full"], 1.044, delta=5e-4)
        np.testing.assert_allclose(list(shares["dc_bias_of_tracked"].values()), [-0.019625, 0.008626], atol=1e-6)


class TrackObsArgumentTests(unittest.TestCase):
    """--track-obs is opt-in: the full observation records and prints exactly what it did before."""

    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()

    @classmethod
    def tearDownClass(cls):
        cls.ws.close()

    POSITION = ("--sev", 0.05, "--episodes", 1, "--track-kappa", 0.002, "--track-mode", "position",
                "--track-leak", 0.02, "--track-anchor", 0, "--track-dims", "3,5", *SHIPPED)

    def test_explicit_full_observation_is_the_recorded_default(self):
        for extra in (self.POSITION, ("--sev", 0.05, "--episodes", 1, "--track-kappa", 0.05, *SHIPPED)):
            with self.subTest(mode="position" if "position" in extra else "rate"):
                implicit = self.ws.main(self.ws.argv(*extra))
                explicit = self.ws.main(self.ws.argv(*extra, "--track-obs", "full"))
                self.assertEqual(implicit[:2], explicit[:2])        # stdout and result JSON, args included
                self.assertEqual(implicit[2][1:], explicit[2][1:])  # every telemetry step record
                headers = [json.loads(run[2][0]) for run in (implicit, explicit)]
                for header in headers:
                    header.pop("argv")
                self.assertEqual(*headers)
                self.assertNotIn("track_obs", json.loads(implicit[1])["args"])

    def test_tracked_observation_is_recorded_everywhere(self):
        import re4_record
        stdout, result, telemetry = self.ws.main(self.ws.argv(*self.POSITION, "--track-obs", "tracked"))
        self.assertIn("tracked-subspace observation", stdout)
        result = json.loads(result)
        self.assertEqual(result["args"]["track_obs"], "tracked")
        self.assertEqual(result["tracking"]["obs"], "tracked")
        header = json.loads(telemetry[0])
        self.assertEqual(header["config"]["tracking"]["obs"], "tracked")
        self.assertIn("solve(M[R,R]", header["fields"]["track_z"])
        M = np.linalg.inv(np.array(header["config"]["M_inv"]))
        for line in telemetry[1:40]:
            record = json.loads(line)
            if record.get("phase") != "rollout":
                continue
            expected = np.zeros(6)
            expected[[3, 5]] = np.linalg.solve(M[np.ix_([3, 5], [3, 5])], np.asarray(record["track_e"])[[3, 5]])
            np.testing.assert_allclose(record["track_z"], expected, rtol=1e-12, atol=1e-15)
        path = self.ws.dir / "tracked.json"
        path.write_text(json.dumps(result))
        with contextlib.redirect_stdout(io.StringIO()):
            re4_record.cmd_record(SimpleNamespace(result=path, root=self.ws.dir / "records", part="p",
                                                  run_id="tracked", cohort=None, timing=None))
        config = json.loads((self.ws.dir / "records" / "p" / "tracked" / "run_configuration.json").read_text())
        self.assertEqual((config["track_obs"], config["track_mode"], config["track_dims"]), ("tracked", "position", [3, 5]))

    def test_argument_errors(self):
        for extra in (["--track-obs", "tracked"],                     # tracking is off
                      ["--track-kappa", "0.01", "--track-obs", "sideways"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                self.ws.main(self.ws.argv(*extra))


class ReplayTests(unittest.TestCase):
    """Replay: recorded commands, recorded scenario and length, no policy client, no control file."""

    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()

    @classmethod
    def tearDownClass(cls):
        cls.ws.close()

    def replay(self, *extra, episodes="0,2"):
        before = sorted(p.name for p in self.ws.dir.iterdir())
        stdout, result, telemetry = self.ws.main(self.ws.argv("--replay-log", self.ws.log, "--replay-episodes",
                                                              episodes, *extra, policy=False), policy=RefusingPolicy)
        self.assertEqual(sorted(p.name for p in self.ws.dir.iterdir()), before)   # no control/ack/lock files
        return stdout, json.loads(result), [json.loads(line) for line in telemetry]

    def test_healthy_replay_reproduces_the_recorded_commands_and_motion(self):
        record = self.ws.record
        starts = np.cumsum([0] + record["ep_len"][:-1])
        _, result, telemetry = self.replay("--sev", 0, "--arms", "frozen")
        self.assertEqual(result["args"]["episodes"], 2)
        self.assertTrue(result["args"]["scenario_reset"])
        self.assertEqual(result["replay"]["episodes"], [0, 2])
        self.assertEqual([tuple(s) for s in result["replay"]["scenarios"]], [(0, 25), (2, 25)])
        per_ep = result["arms"]["frozen_faulted"]["per_ep"]
        for index, episode in enumerate((0, 2)):
            steps = [r for r in telemetry if r.get("type") == "step" and r["episode"] == index]
            rollout = [r for r in steps if r["phase"] == "rollout"]
            length = record["ep_len"][episode]
            self.assertEqual(len(rollout), length)
            self.assertEqual(per_ep[index]["steps"], length)
            self.assertEqual(per_ep[index]["ok"], record["per_ep"][episode]["ok"])
            rows = slice(starts[episode], starts[episode] + length)
            self.assertEqual([r["raw_action"] for r in rollout], record["raw_cmd"][rows])
            self.assertEqual([r["command"] for r in rollout], record["raw_cmd"][rows])
            positions = [steps[len(steps) - len(rollout) - 1]["position"]] + [r["position"] for r in rollout]
            quaternions = [steps[len(steps) - len(rollout) - 1]["quaternion"]] + [r["quaternion"] for r in rollout]
            motion = [list(np.subtract(positions[k + 1], positions[k])) +
                      so3.rot_delta(quaternions[k], quaternions[k + 1]).tolist() for k in range(length)]
            self.assertEqual(motion, record["raw_d"][rows])          # bit for bit: same scenario, same physics

    def test_faulted_adaptive_replay_uses_the_runner_paths(self):
        stdout, result, telemetry = self.replay("--sev", 0.05, "--arms", "adaptive", "--dc-constrain", "corrected",
                                                "--track-kappa", 0.05, *SHIPPED, episodes="1")
        self.assertIn("POSE TRACKING: kappa 0.05 on dims [3, 4, 5]", stdout)
        self.assertEqual(result["tracking"]["dims"], [3, 4, 5])
        self.assertEqual(result["args"]["track_anchor"], 5)
        rollout = [r for r in telemetry if r.get("phase") == "rollout"]
        self.assertTrue(rollout and all("track_e" in r and "track_z" in r and "track_n" in r for r in rollout))
        record, start = self.ws.record, self.ws.record["ep_len"][0]
        for k, r in enumerate(rollout):
            np.testing.assert_array_equal(r["raw_action"], record["raw_cmd"][start + k])
            expected = np.asarray(r["raw_action"]) + np.asarray(r["correction"]) + np.r_[np.full(6, 0.05), 0.0]
            np.testing.assert_array_equal(r["command"], expected)
            self.assertEqual(r["correction"][:3], [0.0, 0.0, 0.0])
        header = json.loads(self.ws.main(self.ws.argv("--replay-log", self.ws.log, "--replay-episodes", 1,
                                                      "--sev", 0.05, "--arms", "adaptive", "--track-kappa", 0.05,
                                                      *SHIPPED, policy=False), policy=RefusingPolicy)[2][0])
        self.assertEqual(header["config"]["replay"]["episodes"], [1])
        self.assertEqual(header["config"]["tracking"]["reference"], "dc")

    def test_recorder_fields(self):
        import re4_record
        runs = {"tracked": ("--sev", 0.05, "--arms", "adaptive", "--track-kappa", 0.05, "--track-anchor", 0,
                            "--track-ref", "fitted", *SHIPPED),
                "default": ("--sev", 0.05, "--arms", "adaptive", *SHIPPED)}
        for name, extra in runs.items():
            argv = (self.ws.argv("--replay-log", self.ws.log, "--replay-episodes", 1, *extra, policy=False)
                    if name == "tracked" else self.ws.argv("--episodes", 1, *extra))
            _, result, _ = self.ws.main(argv, policy=RefusingPolicy if name == "tracked" else FakePolicy)
            path = self.ws.dir / f"{name}.json"
            path.write_text(result)
            with contextlib.redirect_stdout(io.StringIO()):
                re4_record.cmd_record(SimpleNamespace(result=path, root=self.ws.dir / "records", part="p",
                                                      run_id=name, cohort=None, timing=None))
            config = json.loads((self.ws.dir / "records" / "p" / name / "run_configuration.json").read_text())
            if name == "tracked":
                self.assertEqual((config["track_kappa"], config["track_ref"], config["track_anchor"],
                                  config["track_dims"], config["replay_episodes"]), (0.05, "fitted", 0, [3, 4, 5], [1]))
                self.assertEqual(config["replay_log"]["path"], str(self.ws.log))
                self.assertEqual(config["replay_log"]["sha256"], re4_record.sha(self.ws.log))
                self.assertIn("--replay-log", config["command"])
                self.assertTrue(config["reset_protocol"].startswith("libero-reset-v1"))
            else:
                self.assertEqual(config["track_kappa"], 0.0)
                self.assertTrue(config["track_ref"].startswith("not applicable"))
                self.assertEqual(config["replay_log"], "none: live policy server")
                self.assertNotIn("--track-", config["command"])
                self.assertNotIn("--replay-", config["command"])

    def test_argument_errors(self):
        cases = [["--replay-episodes", "0"],
                 ["--replay-log", self.ws.log, "--manifest", self.ws.log],
                 ["--replay-log", self.ws.log, "--pin-rng"],
                 ["--replay-log", self.ws.log, "--replay-episodes", "7"],
                 ["--replay-log", self.ws.log, "--suite", "libero_10"],
                 ["--replay-log", self.ws.openloop],
                 ["--track-kappa", "-0.1"], ["--track-kappa", "nan"], ["--track-dims", "2,6"],
                 ["--track-dims", "x"], ["--track-anchor", "-1"],
                 ["--track-kappa", "0.1", "--mimo"], ["--track-kappa", "0.1", "--ar", "1"]]
        for extra in cases:
            with self.subTest(extra=extra), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.ws.main(self.ws.argv(*extra, policy="--replay-log" not in extra), policy=RefusingPolicy)


REFERENCE = os.environ.get("ADAPTIVE_LAW_REFERENCE")


@unittest.skipUnless(REFERENCE, "set ADAPTIVE_LAW_REFERENCE to another adaptive_law.py (e.g. a clean HEAD copy)")
class ReferenceByteIdentityTests(unittest.TestCase):
    """Default flags: stdout, result JSON and every telemetry record byte-identical to the reference
    runner's; the header differs at most in its runner_source snapshot."""

    CONFIGS = (
        ["--sev", 0.05, "--episodes", 3, "--scenario-reset", *SHIPPED],
        ["--sev", 0.05, "--episodes", 3, "--scenario-reset", "--dc-constrain", "corrected", *SHIPPED],
        ["--sev", 0.05, "--episodes", 2, "--baseline", "kalman", "--gamma", 0.05],
        ["--episodes", 2, "--law", "innov", "--profile", "ramp", "--prof-p", 20, "--onset", 5,
         "--fault-vec", "0.02,-0.03,0.01,0.04,-0.02,0.03", "--arms", "adaptive"],
        ["--sev", 0, "--episodes", 2, "--estimate-only", "--corr-dims", "3,4,5", "--task-stride", 3],
    )

    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()
        spec = importlib.util.spec_from_file_location("adaptive_law_reference", REFERENCE)
        cls.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.reference)

    @classmethod
    def tearDownClass(cls):
        cls.ws.close()

    # Runs the reference also supports once it has the tracking term and replay mode (b01492a onwards):
    # rate mode must stay byte-identical too, not only the defaults.
    TRACKING_CONFIGS = (
        ["--sev", 0.05, "--episodes", 2, "--scenario-reset", "--dc-constrain", "corrected",
         "--track-kappa", 0.05, "--track-ref", "dc", *SHIPPED],
        ["--sev", 0.05, "--episodes", 2, "--track-kappa", 0.02, "--track-anchor", 3, "--track-dims", "3,4,5",
         "--track-ref", "fitted", "--arms", "adaptive", *SHIPPED],
        ["--sev", 0.05, "--episodes", 2, "--track-kappa", 0.01, "--track-anchor", 0, "--arms", "adaptive"],
    )

    def compare(self, argv, policy=FakePolicy):
        ref_stdout, ref_result, ref_tele = self.ws.main(argv, module=self.reference, policy=policy)
        new_stdout, new_result, new_tele = self.ws.main(argv, policy=policy)
        self.assertEqual(new_stdout, ref_stdout)
        self.assertEqual(new_result, ref_result)
        self.assertEqual(len(new_tele), len(ref_tele))
        self.assertGreater(len(new_tele), 50)
        ref_header, new_header = json.loads(ref_tele[0]), json.loads(new_tele[0])
        ref_header.pop("runner_source"), new_header.pop("runner_source")
        self.assertEqual(new_header, ref_header)
        self.assertEqual(new_tele[1:], ref_tele[1:])

    def test_default_runs_match_the_reference_byte_for_byte(self):
        for config in self.CONFIGS:
            with self.subTest(config=config):
                self.compare(self.ws.argv(*config))

    def test_rate_mode_and_replay_match_the_reference_byte_for_byte(self):
        if not hasattr(self.reference, "PoseTracker"):
            self.skipTest("the reference predates the tracking term and replay mode")
        for config in self.TRACKING_CONFIGS:
            with self.subTest(config=config):
                self.compare(self.ws.argv(*config))
        replay = ["--replay-log", self.ws.log, "--replay-episodes", "0,2", "--sev", 0.05, "--arms", "adaptive",
                  "--dc-constrain", "corrected", "--track-kappa", 0.05, "--track-anchor", 0, *SHIPPED]
        with self.subTest(config="replay"):
            self.compare(self.ws.argv(*replay, policy=False), policy=RefusingPolicy)

    # Runs the reference also supports once it has position mode (433e5c3 onwards): the full observation
    # must stay byte-identical in position mode too.
    POSITION_CONFIGS = (
        ["--sev", 0.05, "--episodes", 2, "--law", "innov", "--dc-constrain", "corrected", "--track-kappa", 0.002,
         "--track-mode", "position", "--track-anchor", 0, "--track-dims", "3,5", "--track-leak", 0.02, *SHIPPED],
        ["--sev", 0.05, "--episodes", 2, "--track-kappa", 0.001, "--track-mode", "position", "--track-anchor", 5,
         "--arms", "adaptive", *SHIPPED],
        ["--sev", 0, "--episodes", 2, "--law", "innov", "--track-kappa", 0.005, "--track-mode", "position",
         "--track-anchor", 0, "--track-dims", "3,4,5", "--track-leak", 0.05, "--track-ref", "fitted", *SHIPPED],
    )

    def test_position_mode_matches_the_reference_byte_for_byte(self):
        if not hasattr(self.reference, "TRACK_MODE_DEFAULTS"):
            self.skipTest("the reference predates position mode")
        for config in self.POSITION_CONFIGS:
            with self.subTest(config=config):
                self.compare(self.ws.argv(*config))
        replay = ["--replay-log", self.ws.log, "--replay-episodes", "1,2", "--sev", 0.05, "--arms", "adaptive",
                  "--law", "innov", "--dc-constrain", "corrected", "--track-kappa", 0.001, "--track-mode", "position",
                  "--track-anchor", 0, "--track-dims", "3,5", "--track-leak", 0.02, *SHIPPED]
        with self.subTest(config="replay, position"):
            self.compare(self.ws.argv(*replay, policy=False), policy=RefusingPolicy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
