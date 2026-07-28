#!/usr/bin/env bash
# Launcher for the camera stream server on the robot.
# Kept as a separate script so `pkill -f camera_stream_server.py` does not match
# the SSH/launch command line itself.
pkill -f camera_stream_server.py 2>/dev/null
sleep 1
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/aimdk_ws/install/setup.bash 2>/dev/null
export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/software/entry/cfg/ros_dds_configuration.xml
export RCUTILS_LOGGING_SEVERITY=FATAL
exec python3 -u /tmp/camera_stream_server.py
