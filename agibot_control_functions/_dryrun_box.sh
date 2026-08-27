#!/usr/bin/env bash
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/aimdk_ws/install/setup.bash 2>/dev/null
export RCUTILS_LOGGING_SEVERITY=FATAL
cd ~/agibot_control_functions || exit 1

echo "===== DRY RUN (no --engage: computes + prints, NEVER publishes) ====="
# feed the 'Press Enter' prompt; SIGINT after ~16s (dry run loops through the motion)
printf '\n' | timeout -s INT 20 python3 deploy_x2_box_pickup.py \
    --policy policies/x2_box_policy_walk_feasible_v17_iter49000.npz \
    --ramp-seconds 2 --settle-seconds 1
echo "===== dry run ended (exit $?) ====="
