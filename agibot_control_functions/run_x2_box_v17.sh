#!/usr/bin/env bash
# On-robot runner for box-walk v17 (iter 49000).
#
# Default is DRY-RUN (computes + logs, does not publish).
# First motion trials: robot SUSPENDED, NO BOX.
# Stop MC first:  aima em stop-app mc   (on 10.0.1.40)
# When done:      aima em start-app mc
#
# Usage (on the robot, as user `run`):
#   cd ~/agibot_control_functions
#   ./run_x2_box_v17.sh                 # dry-run
#   ./run_x2_box_v17.sh --engage        # publish (after the safety ladder)
# Extra args are forwarded to deploy_x2_box_pickup.py
set -euo pipefail

source /opt/ros/humble/setup.bash
source ~/aimdk_ws/install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/agibot/software/entry/cfg/ros_dds_configuration.xml}"
export RCUTILS_LOGGING_SEVERITY="${RCUTILS_LOGGING_SEVERITY:-WARN}"

cd ~/agibot_control_functions || {
  echo "missing ~/agibot_control_functions — run push_x2_box_v17_to_robot.sh from the laptop first"
  exit 1
}

POLICY="${POLICY:-policies/x2_box_policy_walk_feasible_v17_iter49000.npz}"
if [[ ! -f "$POLICY" ]]; then
  echo "missing policy: $POLICY"
  exit 1
fi

echo "===== v17 box-walk  policy=$POLICY  args: $* ====="
exec python3 deploy_x2_box_pickup.py --policy "$POLICY" "$@"
