#!/usr/bin/env bash
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/aimdk_ws/install/setup.bash 2>/dev/null
export RCUTILS_LOGGING_SEVERITY=FATAL
export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/software/entry/cfg/ros_dds_configuration.xml

echo "===== chest IMU rate over 5s (stderr hidden) ====="
timeout 6 ros2 topic hz /aima/hal/imu/chest/state 2>/dev/null | head -n 5
echo "--- (blank above = chest publishes NO data) ---"

echo ""
echo "===== torso IMU rate over 5s (reference) ====="
timeout 6 ros2 topic hz /aima/hal/imu/torso/state 2>/dev/null | head -n 5
