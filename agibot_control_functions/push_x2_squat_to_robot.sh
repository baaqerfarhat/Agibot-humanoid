#!/usr/bin/env bash
# Copy the 40% squat policy + deploy scripts onto the robot.
# Run this from a machine on the robot Ethernet (laptop 10.0.1.50, robot 10.0.1.41).
# This training box (fremont) cannot reach 10.0.1.41.
#
#   ping -c 1 10.0.1.41
#   ./agibot_control_functions/push_x2_squat_to_robot.sh
#
# Then SSH in and dry-run:
#   ssh -i ~/.ssh/agibot_ed25519 run@10.0.1.41
#   cd ~/agibot_control_functions && ./run_x2_squat.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${ROBOT_HOST:-run@10.0.1.41}"
KEY="${AGIBOT_SSH_KEY:-$HOME/.ssh/agibot_ed25519}"
POLICY_NAME="x2_squat_policy_40pct_iter16499.npz"
POLICY_SRC="$ROOT/agibot_control_functions/policies/$POLICY_NAME"

SSH=(ssh)
SCP=(scp)
if [[ -f "$KEY" ]]; then
  SSH+=(-i "$KEY")
  SCP+=(-i "$KEY")
fi
SSH+=(-o ConnectTimeout=5 "$HOST")
SCP+=(-o ConnectTimeout=5)

if [[ ! -f "$POLICY_SRC" ]]; then
  echo "missing $POLICY_SRC"
  exit 1
fi

echo "===== ping $HOST ====="
if ! ping -c 1 -W 2 "${HOST#*@}" >/dev/null; then
  echo "robot not reachable. Plug the laptop Ethernet (10.0.1.50) into the robot LAN and retry."
  exit 1
fi

echo "===== mkdir on robot ====="
"${SSH[@]}" "mkdir -p agibot_control_functions/policies"

echo "===== copy policy + scripts ====="
"${SCP[@]}" "$POLICY_SRC" "$HOST:agibot_control_functions/policies/$POLICY_NAME"
for f in \
  deploy_x2_squat.py \
  export_squat_policy_npz.py \
  base_frame.py \
  robot_states_control.py \
  run_logger.py \
  run_x2_squat.sh
do
  "${SCP[@]}" "$ROOT/agibot_control_functions/$f" "$HOST:agibot_control_functions/$f"
done

"${SSH[@]}" "chmod +x agibot_control_functions/run_x2_squat.sh"

echo "===== remote ls ====="
"${SSH[@]}" "ls -lh agibot_control_functions/policies/$POLICY_NAME agibot_control_functions/deploy_x2_squat.py agibot_control_functions/run_x2_squat.sh"

echo
echo "On the robot:"
echo "  ssh ${KEY:+-i $KEY }$HOST"
echo "  aima em stop-app mc          # on 10.0.1.40, before any --engage"
echo "  cd ~/agibot_control_functions && ./run_x2_squat.sh"
echo "  # first --engage: SUSPENDED"
echo "  ./run_x2_squat.sh --engage"
