#!/usr/bin/env bash
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/aimdk_ws/install/setup.bash 2>/dev/null
export RCUTILS_LOGGING_SEVERITY=FATAL

echo "==================== policies load + metadata ===================="
python3 - <<'PY'
import json, numpy as np
for tag, p in [("BOX ", "/home/run/box_pickup/policy/x2_box_policy.npz"),
               ("WALK", "/home/run/agibot_control_functions/policies/x2_walk_carry.npz")]:
    d = np.load(p, allow_pickle=True)
    m = json.loads(str(d["meta_json"]))
    print(f"{tag} {p.split('/')[-1]}")
    print(f"     obs_dim={m['obs_dim']} action_dim={m['action_dim']}")
    print(f"     obs_names={m['observation_names']}")
    if 'run_path' in m: print(f"     run_path={m['run_path']}")
    if 'hold_frame_range' in m: print(f"     hold_frame_range={m['hold_frame_range']}  (hybrid uses the middle)")
    if 'motion_frames' in m: print(f"     motion_frames={m['motion_frames']} @ {m.get('motion_fps')}Hz")
PY

echo ""
echo "==================== state topics + mc ===================="
for t in /aima/hal/imu/torso/state /aima/hal/joint/leg/state ; do
  timeout 3 ros2 topic echo --once "$t" >/dev/null 2>&1 && echo "  LIVE   $t" || echo "  SILENT $t"
done
n=$(ros2 topic info /aima/hal/joint/leg/command 2>/dev/null | awk -F': ' '/Publisher count/{print $2}')
echo "  leg/command publishers=$n   (0 = mc stopped on .40, >0 = stop mc on .40)"
