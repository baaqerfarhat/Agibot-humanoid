#!/usr/bin/env bash
# On-robot runner for the mjlab 40% in-place squat (iter 16499).
#
# Default is DRY-RUN (computes + logs, does not publish).
# First motion trials: robot SUSPENDED.
# Stop MC first:  aima em stop-app mc   (on 10.0.1.40)
# When done:      aima em start-app mc
#
# Usage (on the robot, as user `run`):
#   cd ~/agibot_control_functions
#   ./run_x2_squat.sh                 # dry-run
#   ./run_x2_squat.sh --engage        # publish (after the safety ladder)
# Extra args are forwarded to deploy_x2_squat.py
set -eo pipefail

set +u
source /opt/ros/humble/setup.bash
source ~/aimdk_ws/install/setup.bash
set -u

export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/agibot/software/entry/cfg/ros_dds_configuration.xml}"
export RCUTILS_LOGGING_SEVERITY="${RCUTILS_LOGGING_SEVERITY:-WARN}"

cd ~/agibot_control_functions || {
  echo "missing ~/agibot_control_functions — run push_x2_squat_to_robot.sh from the laptop first"
  exit 1
}

POLICY="${POLICY:-policies/x2_squat_policy_40pct_iter16499.npz}"
if [[ ! -f "$POLICY" ]]; then
  echo "missing policy: $POLICY"
  exit 1
fi

echo "===== squat 40%  policy=$POLICY  args: $* ====="
exec python3 deploy_x2_squat.py --policy "$POLICY" "$@"
