#!/usr/bin/env bash
for pid in 2022 3389 2710; do
  echo "=== PID $pid ($(cat /proc/$pid/comm 2>/dev/null)) ==="
  echo "1" | sudo -S cat /proc/$pid/environ 2>/dev/null | tr '\0' '\n' \
    | grep -iE 'ROS_DOMAIN|RMW_IMPL|FASTRTPS|FASTDDS|ROS_LOCALHOST|CYCLONEDDS|ROS_DISCOVERY'
done
