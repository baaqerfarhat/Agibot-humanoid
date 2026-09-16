#!/usr/bin/env bash
# Physical pre-test of the pose-referenced tracking term (IMP1), with NO policy server: recorded healthy
# commands are replayed open loop through adaptive_law.py --replay-log, i.e. the same runner code path
# (faults, adapter, masks, projection, telemetry) the later policy-in-the-loop experiment will use.
#
# Runs on lambda and writes only under $ROOT (/data/fxxie/tracking_pretest). Stage from the workstation:
#   rsync -a --delete --exclude __pycache__ openpi/ lambda:/data/fxxie/tracking_pretest/code/openpi/
#   rsync -a scripts/tracking_pretest.sh lambda:/data/fxxie/tracking_pretest/code/scripts/
#   rsync -a results/phase05/error_signal_so3.json results/phase05/openloop_so3.json \
#            results/heldout/error_signal_init25.json lambda:/data/fxxie/tracking_pretest/data/
# then, on lambda (KAPPA is required; EPISODES are indices into the replay log):
#   RUN=smoke EPISODES=0 KAPPA=0.05 bash /data/fxxie/tracking_pretest/code/scripts/tracking_pretest.sh
#   RUN=smoke bash .../tracking_pretest.sh check          # re-run only the checks on an existing run
#
# Arms, all replaying the same episodes; fault = the headline uniform command offset $SEV on all six
# arm dims, step profile from the first step:
#   A0a A0b  no fault, no correction: the reference trajectory, twice (determinism)
#   A1       fault, no correction
#   A2       fault + the shipped law (gamma 0.08, dead 0.008, rho 0.15, clip 0.15, rotation corrected)
#   A3       A2 + --dc-constrain corrected
#   A4       A3 + tracking, --track-ref dc (reference pinned on the tracked = corrected dims)
#   A5       A4 with --track-dims 0,1,2,3,4,5 and --dc-constrain all (the correction mask stays 3,4,5)
#   A6       A3 + tracking, --track-ref fitted
#   H3 H4 H5 A3 A4 A5 without the fault
# Each arm writes $ROOT/$RUN/<arm>/{command.txt,run.log,result.json,telemetry.jsonl} and a re4 record
# under $ROOT/$RUN/records/. `check` compares A0a/A0b EEF poses at every step (<= 1e-12), requires
# track_e telemetry in A4-A6 and H4-H5, and writes $ROOT/$RUN/summary.json.
set -uo pipefail

ROOT=${ROOT:-/data/fxxie/tracking_pretest}
CODE=${CODE:-$ROOT/code/openpi}
DATA=${DATA:-$ROOT/data}
OPENPI=${OPENPI:-/data/fxxie/vla/openpi}
PY=${PY:-/data/fxxie/vla/envs/libero/bin/python}
RUN=${RUN:?set RUN, e.g. RUN=smoke}
SUITE=${SUITE:-libero_spatial}
REPLAY_LOG=${REPLAY_LOG:-$DATA/error_signal_init25.json}
CAL_LOG=${CAL_LOG:-$DATA/error_signal_so3.json}
OPENLOOP=${OPENLOOP:-$DATA/openloop_so3.json}
SEV=${SEV:-0.05}
JOBS=${JOBS:-1}
ARMS=${ARMS:-"A0a A0b A1 A2 A3 A4 A5 A6 H3 H4 H5"}
OUT=$ROOT/$RUN
case "$OUT" in "$ROOT"/?*) ;; *) echo "refusing: $OUT is not inside $ROOT"; exit 1 ;; esac
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
export PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:$CODE

check() {
  "$PY" - "$OUT" "$REPLAY_LOG" <<'PYCHECK'
import json, pathlib, sys
import numpy as np
from so3 import rot_delta              # $CODE is on PYTHONPATH

out, replay_log = pathlib.Path(sys.argv[1]), sys.argv[2]
TRACKED = ("A4", "A5", "A6", "H4", "H5")


def load(arm):
    path = out / arm / "telemetry.jsonl"
    if not path.exists():
        return None, []
    rows = [json.loads(line) for line in path.open()]
    return rows[0], [r for r in rows[1:] if r.get("type") == "step"]


def angle(q0, q1):
    return float(np.degrees(np.linalg.norm(rot_delta(q0, q1))))


ok, summary = True, {}
_, a = load("A0a")
_, b = load("A0b")
if a and b:
    same = [(r["episode"], r["phase"], r["t"]) for r in a] == [(r["episode"], r["phase"], r["t"]) for r in b]
    dp = float(np.max(np.abs(np.array([r["position"] for r in a]) - np.array([r["position"] for r in b])))) if same else float("inf")
    dq = float(np.max(np.abs(np.array([r["quaternion"] for r in a]) - np.array([r["quaternion"] for r in b])))) if same else float("inf")
    passed = same and max(dp, dq) <= 1e-12
    ok &= passed
    summary["determinism"] = dict(step_records=len(a), same_steps=same, max_abs_position=dp, max_abs_quaternion=dq,
                                  tolerance=1e-12, passed=passed)
    print(f"A0 determinism: {len(a)} step records each (warmup + rollout), same steps {same}; "
          f"max|d position| = {dp:.3e} m, max|d quaternion| = {dq:.3e} -> {'PASS' if passed else 'FAIL'} (<= 1e-12)")
    # Replay fidelity (information only): A0a's motion against the motion recorded with the policy in the loop.
    log = json.loads(open(replay_log).read())["records"][0]
    starts = np.cumsum([0] + log["ep_len"][:-1])
    result = json.loads((out / "A0a" / "result.json").read_text())
    for index, episode in enumerate(result["replay"]["episodes"]):
        steps = [r for r in a if r["episode"] == index]
        roll = [r for r in steps if r["phase"] == "rollout"]
        prev = steps[len(steps) - len(roll) - 1]
        pos = np.array([prev["position"]] + [r["position"] for r in roll])
        quat = [prev["quaternion"]] + [r["quaternion"] for r in roll]
        motion = np.c_[np.diff(pos, axis=0), [rot_delta(quat[k], quat[k + 1]) for k in range(len(roll))]]
        recorded = np.array(log["raw_d"][starts[episode]:starts[episode] + len(roll)])
        gap = np.abs(motion - recorded)
        summary.setdefault("replay_fidelity", []).append(dict(
            episode=episode, steps=len(roll), recorded_steps=log["ep_len"][episode],
            max_translation_increment_gap_m=float(gap[:, :3].max()), max_rotation_increment_gap_rad=float(gap[:, 3:].max())))
        print(f"  replay fidelity, episode {episode}: {len(roll)}/{log['ep_len'][episode]} steps; per-step motion vs the "
              f"recording: max gap {gap[:, :3].max():.2e} m, {gap[:, 3:].max():.2e} rad (recorded on another host)")
elif not ((out / "A0a").exists() or (out / "A0b").exists()):
    summary["determinism"] = dict(skipped=True, reason="A0a/A0b not in this run's ARMS")
    print("A0 determinism: skipped (A0a/A0b not in this run's ARMS)")
else:
    ok = False
    print("A0 determinism: missing A0a or A0b telemetry -> FAIL")

print(f"{'arm':<4} {'ep':>3} {'ok':>3} {'steps':>9} {'f_hat rx ry rz (final)':>26} {'max|dx| mm':>10} "
      f"{'end|dx| mm':>10} {'end rot deg':>11} {'track_e':>8}")
for arm in ("A0a", "A0b", "A1", "A2", "A3", "A4", "A5", "A6", "H3", "H4", "H5"):
    header, rows = load(arm)
    result_path = out / arm / "result.json"
    if header is None or not result_path.exists():
        print(f"{arm:<4} missing")
        ok = False
        continue
    result = json.loads(result_path.read_text())
    per_ep = next(iter(result["arms"].values()))["per_ep"]
    summary[arm] = []
    for index, episode_result in enumerate(per_ep):
        roll = [r for r in rows if r["episode"] == index and r["phase"] == "rollout"]
        ref = [r for r in a if r["episode"] == index and r["phase"] == "rollout"] if a else []
        n = min(len(roll), len(ref))
        dx = [1e3 * float(np.linalg.norm(np.subtract(roll[k]["position"], ref[k]["position"]))) for k in range(n)]
        rot = angle(ref[n - 1]["quaternion"], roll[n - 1]["quaternion"]) if n else float("nan")
        tracked = sum("track_e" in r for r in roll)
        if arm in TRACKED and (not roll or tracked != len(roll)):
            ok = False
        f_hat = roll[-1]["f_hat"] if roll else [float("nan")] * 6
        entry = dict(episode=episode_result.get("replay_episode"), ok=episode_result["ok"], steps=len(roll),
                     recorded_steps=episode_result.get("recorded_steps"), final_f_hat=f_hat,
                     max_position_deviation_mm=max(dx) if dx else None, final_position_deviation_mm=dx[-1] if dx else None,
                     final_rotation_deviation_deg=rot, track_e_records=tracked)
        if tracked:
            entry["final_track_e"] = roll[-1]["track_e"]
        summary[arm].append(entry)
        fh = " ".join(f"{v:+.4f}" for v in f_hat[3:6])
        print(f"{arm:<4} {str(entry['episode']):>3} {int(entry['ok']):>3} {len(roll):>4}/{str(entry['recorded_steps']):<4} "
              f"{fh:>26} {max(dx) if dx else float('nan'):>10.2f} {dx[-1] if dx else float('nan'):>10.2f} "
              f"{rot:>11.3f} {tracked:>4}/{len(roll):<3}")
summary["passed"] = bool(ok)
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(f"checks {'PASSED' if ok else 'FAILED'}; wrote {out / 'summary.json'}")
sys.exit(0 if ok else 1)
PYCHECK
}

if [ "${1:-}" = check ]; then check; exit $?; fi

KAPPA=${KAPPA:?set KAPPA, the tracking gain (e.g. KAPPA=0.05)}
EPISODES=${EPISODES:?set EPISODES, comma-separated episode indices of the replay log (e.g. EPISODES=0)}
SHIPPED="--gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.15 --corr-dims 3,4,5"
FAULT="--sev $SEV --profile step"
declare -A FLAGS=(
  [A0a]="--sev 0 --arms frozen"
  [A0b]="--sev 0 --arms frozen"
  [A1]="$FAULT --arms frozen"
  [A2]="$FAULT --arms adaptive $SHIPPED"
  [A3]="$FAULT --arms adaptive $SHIPPED --dc-constrain corrected"
  [A4]="$FAULT --arms adaptive $SHIPPED --dc-constrain corrected --track-kappa $KAPPA --track-ref dc"
  [A5]="$FAULT --arms adaptive $SHIPPED --dc-constrain all --track-kappa $KAPPA --track-ref dc --track-dims 0,1,2,3,4,5"
  [A6]="$FAULT --arms adaptive $SHIPPED --dc-constrain corrected --track-kappa $KAPPA --track-ref fitted"
  [H3]="--sev 0 --arms adaptive $SHIPPED --dc-constrain corrected"
  [H4]="--sev 0 --arms adaptive $SHIPPED --dc-constrain corrected --track-kappa $KAPPA --track-ref dc"
  [H5]="--sev 0 --arms adaptive $SHIPPED --dc-constrain all --track-kappa $KAPPA --track-ref dc --track-dims 0,1,2,3,4,5"
)
for name in $ARMS; do [ -n "${FLAGS[$name]:-}" ] || { echo "unknown arm $name"; exit 1; }; done
for f in "$CODE/adaptive_law.py" "$REPLAY_LOG" "$CAL_LOG" "$OPENLOOP" "$PY"; do
  [ -e "$f" ] || { echo "missing $f"; exit 1; }
done
mkdir -p "$OUT"
{ echo "run $RUN  $(date -Is)  host $(hostname)"
  echo "KAPPA=$KAPPA EPISODES=$EPISODES SEV=$SEV SUITE=$SUITE JOBS=$JOBS ARMS=$ARMS MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
  sha256sum "$CODE/adaptive_law.py" "$CODE/re4_record.py" "$CODE/paired_probe.py" "$CODE/libero_reset.py" \
            "$CODE/gate_faults.py" "$CODE/so3.py" "$REPLAY_LOG" "$CAL_LOG" "$OPENLOOP" "$0"
} > "$OUT/manifest.txt"

arm() {
  local name=$1 d=$OUT/$1
  mkdir -p "$d"
  local cmd="$PY -u $CODE/adaptive_law.py --suite $SUITE --replay-log $REPLAY_LOG --replay-episodes $EPISODES"
  cmd="$cmd --log $CAL_LOG --openloop $OPENLOOP ${FLAGS[$name]} --telemetry $d/telemetry.jsonl --out $d/result.json"
  echo "$cmd" > "$d/command.txt"
  (cd "$d" && $cmd > "$d/run.log" 2>&1)
  local rc=$?
  if [ $rc -eq 0 ] && [ -s "$d/result.json" ]; then
    (cd "$d" && "$PY" "$CODE/re4_record.py" record "$d/result.json" --part "$RUN" --run-id "$name" \
       --root "$OUT/records" > "$d/record.log" 2>&1) || echo "RECORD FAILED $name"
  fi
  echo "$name exit $rc  $(grep -E ': [0-9]+/[0-9]+ = ' "$d/run.log" | tr '\n' ' ')"
  return $rc
}

fail=0 running=0
for name in $ARMS; do
  arm "$name" &
  running=$((running + 1))
  if [ "$running" -ge "$JOBS" ]; then wait -n || fail=1; running=$((running - 1)); fi
done
while [ "$running" -gt 0 ]; do wait -n || fail=1; running=$((running - 1)); done
[ $fail -eq 0 ] || echo "AT LEAST ONE ARM FAILED (see $OUT/<arm>/run.log)"
check; rc=$?
[ $fail -eq 0 ] && [ $rc -eq 0 ] && echo "PRETEST $RUN DONE" || { echo "PRETEST $RUN FAILED"; exit 1; }
