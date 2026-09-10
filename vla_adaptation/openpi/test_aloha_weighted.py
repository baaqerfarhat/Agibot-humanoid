"""Check the weighted observer through ALOHA's actual fourteen-coordinate episode."""
import io
import json
import unittest

import numpy as np

import aloha_adapt as law


class Tests(unittest.TestCase):
    def test_physical_offset_is_estimated_without_correcting_grippers(self):
        class Servo:
            seed = 1
            def __init__(self):
                self.env = self.client = self
                self.commands = []
            def reset(self, episode):
                self.t = 0
                return {"agent_pos": np.zeros(14)}
            def policy_obs(self, observation):
                return observation
            def infer(self, observation):
                return {"actions": np.full((law.HORIZON, 14), .3)}
            def step(self, command):
                self.commands.append(command.copy())
                self.t += 1
                fault = np.zeros(14); fault[0] = .02; fault[13] = .4
                return {"agent_pos": command + fault}, 0, self.t == 100, False, {}
        servo = Servo()
        W = np.zeros((14, law.K_FIR+2)); W[:, 0] = 1.
        log = io.StringIO()
        _, estimate, _ = law.episode(servo, 0, W=W, M_inv=np.eye(14), M=np.eye(14),
            baseline="weighted_dob", kf_r=np.eye(14)*1e-6, adapt=True,
            corr=list(range(6))+list(range(7, 13)), gamma=.08, clip=.08, telemetry=log)
        rows = [json.loads(line) for line in log.getvalue().splitlines()]
        self.assertGreater(estimate[0], .0199)
        np.testing.assert_array_equal(estimate[[6, 13]], [0., 0.])
        np.testing.assert_array_equal(servo.commands[0], np.full(14, .3))
        self.assertLess(abs(servo.commands[-1][0]+.02-.3), .0001)
        np.testing.assert_array_equal(np.asarray(servo.commands)[:, [6, 13]], np.full((100, 2), .3))
        self.assertTrue(all(row['optimizer']['status'] == 'optimal' for row in rows))
        self.assertTrue(all(row['optimizer']['objective'] <= row['optimizer']['zero_objective']+1e-10
                            for row in rows))


if __name__ == '__main__':
    unittest.main()
