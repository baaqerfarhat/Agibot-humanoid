#!/usr/bin/env bash
echo "=== deploy processes ==="
pgrep -af "deploy_x2" || echo "  ALL CLEAR - no deploy processes"
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/aimdk_ws/install/setup.bash 2>/dev/null
export RCUTILS_LOGGING_SEVERITY=FATAL
echo "=== command-topic publishers (want 0 = nothing driving joints) ==="
for t in /aima/hal/joint/leg/command /aima/hal/joint/arm/command /aima/hal/joint/waist/command /aima/hal/joint/head/command ; do
  n=$(ros2 topic info "$t" 2>/dev/null | awk -F': ' '/Publisher count/{print $2}')
  echo "  $t publishers=$n"
done
