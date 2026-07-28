#!/usr/bin/env bash
echo "=== python / numpy ==="
python3 --version 2>&1
python3 -c "import numpy; print('numpy', numpy.__version__)" 2>&1
echo "=== ruckig (needed by robot_states_control import) ==="
python3 -c "import ruckig; print('ruckig ok')" 2>&1
echo "=== ROS msgs available? ==="
python3 -c "import rclpy; from aimdk_msgs.msg import JointCommand; print('rclpy + aimdk_msgs ok')" 2>&1
echo "=== git available ==="
git --version 2>&1
echo "=== internet to github (5s) ==="
timeout 5 curl -sI https://github.com >/dev/null 2>&1 && echo REACHABLE || echo UNREACHABLE
echo "=== existing box policy on robot? ==="
ls -la ~/box_pickup/policy/ 2>/dev/null || echo "NO ~/box_pickup"
ls ~/agibot_control_functions/deploy_x2_box_pickup.py 2>/dev/null || echo "NO deploy_x2_box_pickup.py on robot"
