#!/usr/bin/env bash
# Quick connectivity + ROS 2 environment check for the AgiBot X2 board.
source /opt/ros/humble/setup.bash
source ~/aimdk_ws/install/setup.bash

echo "===== ENV ====="
echo "ROS_DISTRO=${ROS_DISTRO:-?}"
echo "RMW=${RMW_IMPLEMENTATION:-default}"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"

echo "===== PYTHON DEPS ====="
python3 - <<'PY'
mods = ["rclpy", "aimdk_msgs", "sensor_msgs", "numpy"]
for m in mods:
    try:
        mod = __import__(m)
        v = getattr(mod, "__version__", "ok")
        print(f"  {m}: {v}")
    except Exception as e:
        print(f"  {m}: MISSING ({e})")
PY

echo "===== AIMA HAL TOPICS ====="
ros2 topic list 2>/dev/null | grep -E 'aima/hal/(imu|joint)' | sort

echo "===== IMU SAMPLE (torso) ====="
timeout 5 ros2 topic echo --once /aima/hal/imu/torso/state 2>/dev/null | head -n 20 || echo "  (no IMU data within 5s)"

echo "DONE"
