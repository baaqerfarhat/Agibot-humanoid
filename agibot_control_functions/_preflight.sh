#!/usr/bin/env bash
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/aimdk_ws/install/setup.bash 2>/dev/null
export RCUTILS_LOGGING_SEVERITY=FATAL

echo "===== 1) required STATE topics live? (torso IMU + 4 joint groups) ====="
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

echo ""
echo "===== 2) is any controller still PUBLISHING joint commands? (want 0) ====="
for t in \
  /aima/hal/joint/leg/command \
  /aima/hal/joint/arm/command \
  /aima/hal/joint/waist/command \
  /aima/hal/joint/head/command ; do
  n=$(ros2 topic info "$t" 2>/dev/null | awk -F': ' '/Publisher count/{print $2}')
  echo "  $t  publishers=$n"
done
echo "  (0 publishers = mc released the joints, good. >0 = something still commanding.)"
