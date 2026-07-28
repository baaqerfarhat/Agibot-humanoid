#!/usr/bin/env bash
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/aimdk_ws/install/setup.bash 2>/dev/null

echo "===== which required topics EXIST in the graph? ====="
ros2 topic list 2>/dev/null | grep -E '/aima/hal/(imu/(chest|torso)|joint/(head|waist|arm|leg))/(state|command)' | sort

echo ""
echo "===== does each STATE topic actually PUBLISH data? (2s each) ====="
for t in \
  /aima/hal/imu/chest/state \
  /aima/hal/imu/torso/state \
  /aima/hal/joint/head/state \
  /aima/hal/joint/waist/state \
  /aima/hal/joint/arm/state \
  /aima/hal/joint/leg/state ; do
  if timeout 2 ros2 topic echo --once "$t" >/dev/null 2>&1; then
    echo "  LIVE   $t"
  else
    echo "  SILENT $t"
  fi
done
