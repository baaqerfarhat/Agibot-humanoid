#!/usr/bin/env bash
echo "===== aima help ====="
aima --help 2>&1 | head -n 40
echo ""
echo "===== aima em help ====="
aima em --help 2>&1 | head -n 40
echo ""
echo "===== running apps / states (try common subcommands) ====="
aima em list-app 2>&1 | head -n 60
aima em status 2>&1 | head -n 60
echo ""
echo "===== other compute node reachable? ====="
ping -c 2 10.0.1.40 2>&1 | tail -n 4
echo ""
echo "===== publishers on the chest IMU topic (who should publish it?) ====="
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/aimdk_ws/install/setup.bash 2>/dev/null
ros2 topic info /aima/hal/imu/chest/state 2>&1
echo "--- torso for comparison ---"
ros2 topic info /aima/hal/imu/torso/state 2>&1
