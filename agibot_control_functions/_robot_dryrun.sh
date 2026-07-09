#!/usr/bin/env bash
# Run the trained X2 policy in DRY-RUN (computes + prints, never publishes).
source /opt/ros/humble/setup.bash
source ~/aimdk_ws/install/setup.bash

echo "===== dependency: ruckig (needed by robot_states_control import) ====="
python3 - <<'PY'
try:
    import ruckig
    print("  ruckig:", getattr(ruckig, "__version__", "ok"))
except Exception as e:
    print("  ruckig MISSING:", e)
    print("  -> install with:  pip3 install --user ruckig")
PY

cd ~/agibot_control_functions || { echo "deploy dir missing"; exit 1; }

echo "===== DRY RUN (no --engage, robot will NOT move) ====="
# feed a newline to satisfy the 'Press Enter to START' prompt;
# stop cleanly with SIGINT after a few seconds (dry-run loops forever otherwise).
printf '\n' | timeout -s INT 22 python3 deploy_x2_walk.py \
    --policy policies/x2_policy_original.npz \
    --ramp-seconds 2 --settle-seconds 1 --cmd-ramp-seconds 2 --run-seconds 6
echo "===== dry run finished (exit $?) ====="
