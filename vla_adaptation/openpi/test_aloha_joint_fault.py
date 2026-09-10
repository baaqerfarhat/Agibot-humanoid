import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from aloha_joint_fault import joint_metadata, torque_fault


class TorqueTests(unittest.TestCase):
    def test_named_joint_mapping_excludes_grippers(self):
        names = []
        def name2id(name, kind):
            names.append(name)
            return len(names) - 1
        model = SimpleNamespace(name2id=name2id, jnt_dofadr=np.arange(12)+2,
            jnt_qposadr=np.arange(12)+3, actuator_trnid=np.c_[np.arange(12), np.zeros(12)],
            actuator_gainprm=np.ones((12, 10))*20, actuator_gear=np.ones((12, 6)),
            actuator_forcelimited=np.zeros(12), actuator_forcerange=np.ones((12, 2)))
        metadata = [joint_metadata(SimpleNamespace(model=model), j) for j in range(12)]
        self.assertEqual([r['command_coordinate'] for r in metadata], list(range(6))+list(range(7, 13)))
        self.assertEqual(names[6], 'vx300s_right/waist')
        self.assertEqual(metadata[-1]['dof'], 13)

    def test_torque_is_below_command_and_does_not_accumulate_or_leak(self):
        force = np.array([3., 4., 5.])
        received = []
        physics = SimpleNamespace(data=SimpleNamespace(qfrc_applied=force))
        env = SimpleNamespace(unwrapped=SimpleNamespace(_env=SimpleNamespace(physics=physics)),
            step=lambda command: received.append((command.copy(), force.copy())),
            reset=lambda **kw: force.__setitem__(slice(None), [3., 4., 5.]))
        client = SimpleNamespace(env=env)
        with patch('aloha_joint_fault.joint_metadata', return_value={'dof': 1}):
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with torque_fault(client, 0, -.2) as wrapper:
                    command = np.array([.1, .2])
                    client.env.reset(seed=1)
                    client.env.step(command); client.env.step(command)
                    self.assertEqual(wrapper.steps, 2)
                    np.testing.assert_array_equal(received[0][0], command)
                    np.testing.assert_array_equal(received[0][1], [3., 3.8, 5.])
                    np.testing.assert_array_equal(received[0][1], received[1][1])
                    raise RuntimeError('fixture')
        self.assertIs(client.env, env)
        np.testing.assert_array_equal(force, [3., 4., 5.])


if __name__ == '__main__':
    unittest.main()
