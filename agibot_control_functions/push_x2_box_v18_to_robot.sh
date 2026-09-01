#!/usr/bin/env bash
# Copy v18 iter 81499 box-walk policy + deploy scripts onto the robot.
# Run this from a machine on the robot Ethernet (laptop 10.0.1.50, robot 10.0.1.41).
# This training box (fremont) cannot reach 10.0.1.41.
#
#   ping -c 1 10.0.1.41
#   ./agibot_control_functions/push_x2_box_v18_to_robot.sh
#
# Then SSH in and dry-run:
#   ssh -i ~/.ssh/agibot_ed25519 run@10.0.1.41
#   cd ~/agibot_control_functions && ./run_x2_box_v18.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${ROBOT_HOST:-run@10.0.1.41}"
KEY="${AGIBOT_SSH_KEY:-$HOME/.ssh/agibot_ed25519}"
POLICY_NAME="x2_box_policy_ankle_scale_v18_iter81499.npz"
POLICY_SRC="$ROOT/box_pickup/policy/$POLICY_NAME"

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
"${SSH[@]}" "mkdir -p agibot_control_functions/policies box_pickup/policy"

echo "===== copy policy + scripts ====="
"${SCP[@]}" "$POLICY_SRC" "$HOST:agibot_control_functions/policies/$POLICY_NAME"
"${SCP[@]}" "$POLICY_SRC" "$HOST:box_pickup/policy/$POLICY_NAME"
for f in \
  deploy_x2_box_pickup.py \
  base_frame.py \
  robot_states_control.py \
  run_logger.py \
  run_x2_box_v18.sh \
  _dryrun_box.sh
do
  "${SCP[@]}" "$ROOT/agibot_control_functions/$f" "$HOST:agibot_control_functions/$f"
done

"${SSH[@]}" "chmod +x agibot_control_functions/run_x2_box_v18.sh agibot_control_functions/_dryrun_box.sh"

echo "===== remote ls ====="
"${SSH[@]}" "ls -lh agibot_control_functions/policies/$POLICY_NAME agibot_control_functions/deploy_x2_box_pickup.py agibot_control_functions/run_x2_box_v18.sh"

echo
echo "On the robot:"
echo "  ssh ${KEY:+-i $KEY }$HOST"
echo "  aima em stop-app mc          # on 10.0.1.40, before any --engage"
echo "  cd ~/agibot_control_functions && ./run_x2_box_v18.sh"
echo "  # first --engage: SUSPENDED, NO BOX"
echo "  ./run_x2_box_v18.sh --engage"
