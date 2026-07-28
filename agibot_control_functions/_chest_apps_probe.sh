#!/usr/bin/env bash
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/aimdk_ws/install/setup.bash 2>/dev/null

echo "===== chest IMU: longer echo (8s) ====="
timeout 8 ros2 topic echo --once /aima/hal/imu/chest/state 2>&1 | head -n 12 || echo "  chest STILL SILENT after 8s"

echo ""
echo "===== chest IMU rate (5s window) ====="
timeout 6 ros2 topic hz /aima/hal/imu/chest/state 2>&1 | head -n 4 || echo "  no rate (dead)"

echo ""
echo "===== torso IMU rate (5s window, for comparison) ====="
timeout 6 ros2 topic hz /aima/hal/imu/torso/state 2>&1 | head -n 4 || echo "  no rate"

echo ""
echo "===== em apps installed ====="
aima em list-apps 2>&1 | head -n 80

echo ""
echo "===== em doctor (health + running apps) ====="
aima em doctor 2>&1 | head -n 80
