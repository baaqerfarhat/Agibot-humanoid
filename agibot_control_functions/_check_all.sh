#!/usr/bin/env bash
# One-shot pre-flight for the X2 box-pickup deploy. Read-only: moves nothing.
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/aimdk_ws/install/setup.bash 2>/dev/null
export RCUTILS_LOGGING_SEVERITY=FATAL

echo "==================== 1) no stale deploy processes ===================="
pgrep -af "deploy_x2" || echo "  ALL CLEAR - nothing running"

echo ""
echo "==================== 2) required STATE topics live? ===================="
for t in \
  /aima/hal/imu/torso/state \
  /aima/hal/joint/head/state \
  /aima/hal/joint/waist/state \
  /aima/hal/joint/arm/state \
  /aima/hal/joint/leg/state ; do
  if timeout 3 ros2 topic echo --once "$t" >/dev/null 2>&1; then
    echo "  LIVE   $t"
  else
    echo "  SILENT $t   <-- PROBLEM"
  fi
done
echo "  (chest IMU is known-dead and intentionally NOT required)"

echo ""
echo "==================== 3) is mc released? command publishers (want 0) ===================="
for t in /aima/hal/joint/leg/command /aima/hal/joint/arm/command ; do
  n=$(ros2 topic info "$t" 2>/dev/null | awk -F': ' '/Publisher count/{print $2}')
  echo "  $t publishers=$n"
done
echo "  (0 = mc stopped on .40, joints free. >0 = mc still running -> stop it on .40)"

echo ""
echo "==================== 4) new policy loads + is the right model ===================="
python3 ~/agibot_control_functions/_verify_policy.py ~/box_pickup/policy/x2_box_policy.npz 2>&1
