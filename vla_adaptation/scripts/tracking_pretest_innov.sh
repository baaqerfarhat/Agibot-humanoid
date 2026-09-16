#!/usr/bin/env bash
# Pre-test of the tracking term on an INNOVATION-law base, with NO policy server: recorded healthy
# commands replayed open loop through adaptive_law.py --replay-log, the same runner code path the
# policy-in-the-loop experiment will use. Companion to scripts/tracking_pretest.sh (legacy-law arms).
#
# Why this base: on the legacy (attenuated) law, rate-mode tracking with --track-ref dc on top of
# --dc-constrain is an unattenuated, deadzone-free integrator of the runner's own residual, so part of
# its benefit is attenuation removal that the innovation law already gives for free. Base C is the
# innovation law, which isolates what tracking adds. And with --track-anchor 0, rate mode feeds back
# e_p/n, the episode-mean pose RATE; position mode feeds back the retained pose offset e_p itself.
#
# Runs on lambda and writes only under $ROOT (/data/fxxie/tracking_pretest). Stage from the workstation:
#   rsync -a --delete --exclude __pycache__ openpi/ lambda:/data/fxxie/tracking_pretest/code/openpi/
#   rsync -a scripts/tracking_pretest_innov.sh lambda:/data/fxxie/tracking_pretest/code/scripts/
#   rsync -a results/phase05/error_signal_so3.json results/phase05/openloop_so3.json \
#            results/heldout/error_signal_init25.json lambda:/data/fxxie/tracking_pretest/data/
# then, on lambda (KAPPA_RATE required; EPISODES are indices into the replay log):
#   RUN=innov_smoke EPISODES=0 KAPPA_RATE=0.05 bash /data/fxxie/tracking_pretest/code/scripts/tracking_pretest_innov.sh
#   RUN=innov_smoke bash .../tracking_pretest_innov.sh check      # re-run only the checks
#
# Arms, all replaying the same episodes. Base C = --law innov --dc-constrain corrected with the shipped
# constants (gamma 0.08, dead 0.008, rho 0.15, clip 0.15, rotation corrected); fault = the headline
# uniform command offset $SEV, step profile:
#   A0a A0b   no fault, no correction: the reference trajectory, twice (determinism)
#   C         fault + base;                      HC        healthy base
#   CR        C + rate mode, anchor 5, dims 3,4,5, --track-ref dc, kappa $KAPPA_RATE
#   CP_k<k>   C + position mode, anchor 0, dims 3,5, --track-ref dc, --track-leak $LEAK, for k in $KAPPAS_POS
#   CPT_k<k>  CP_k<k> + --track-obs tracked: z_T[R] = solve(M[R,R], e_p[R]) on R = {r_x, r_z}, so the
#             uncorrected translation pose error no longer forces r_x/r_z through M_inv's off-diagonals
#             (IMP3; defined here, not yet run)
#   HCP_k<k>  healthy versions of CP_k<k>
#   CP0       the position arm at $KAPPA_CP0 with --track-leak 0 (stability check)
# Each arm writes $ROOT/$RUN/<arm>/{command.txt,run.log,result.json,telemetry.jsonl} plus a re4 record
# under $ROOT/$RUN/records/. `check` compares A0a/A0b EEF poses at every step (<= 1e-12), requires
# track_e in every rollout record of every tracking arm, and writes $ROOT/$RUN/summary.json.
set -uo pipefail

ROOT=${ROOT:-/data/fxxie/tracking_pretest}
CODE=${CODE:-$ROOT/code/openpi}
DATA=${DATA:-$ROOT/data}
OPENPI=${OPENPI:-/data/fxxie/vla/openpi}
PY=${PY:-/data/fxxie/vla/envs/libero/bin/python}
RUN=${RUN:?set RUN, e.g. RUN=innov_smoke}
SUITE=${SUITE:-libero_spatial}
REPLAY_LOG=${REPLAY_LOG:-$DATA/error_signal_init25.json}
CAL_LOG=${CAL_LOG:-$DATA/error_signal_so3.json}
OPENLOOP=${OPENLOOP:-$DATA/openloop_so3.json}
SEV=${SEV:-0.05}
LEAK=${LEAK:-0.02}
KAPPAS_POS=${KAPPAS_POS:-"0.001 0.002 0.005"}
KAPPA_CP0=${KAPPA_CP0:-0.002}
JOBS=${JOBS:-1}
OUT=$ROOT/$RUN
case "$OUT" in "$ROOT"/?*) ;; *) echo "refusing: $OUT is not inside $ROOT"; exit 1 ;; esac
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
export PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:$CODE

check() {
  "$PY" - "$OUT" "$REPLAY_LOG" <<'PYCHECK'
import json, pathlib, sys
import numpy as np
from so3 import rot_delta                  # $CODE is on PYTHONPATH

out, replay_log = pathlib.Path(sys.argv[1]), sys.argv[2]


def load(arm):
    path = out / arm / "telemetry.jsonl"
    if not path.exists():
        return None, []
    rows = [json.loads(line) for line in path.open()]
    return rows[0], [r for r in rows[1:] if r.get("type") == "step"]


arms = sorted(p.name for p in out.iterdir() if (p / "telemetry.jsonl").exists())
order = {name: i for i, name in enumerate(["A0a", "A0b", "C", "HC", "CR"])}
arms.sort(key=lambda name: (order.get(name, len(order)), name))
ok, summary = True, {}

header_a, a = load("A0a")
header_b, b = load("A0b")
if a and b:
    same = [(r["episode"], r["phase"], r["t"]) for r in a] == [(r["episode"], r["phase"], r["t"]) for r in b]
    dp = float(np.max(np.abs(np.array([r["position"] for r in a]) - np.array([r["position"] for r in b])))) if same else float("inf")
    dq = float(np.max(np.abs(np.array([r["quaternion"] for r in a]) - np.array([r["quaternion"] for r in b])))) if same else float("inf")
    passed = same and max(dp, dq) <= 1e-12
    ok &= passed
    summary["determinism"] = dict(step_records=len(a), same_steps=same, max_abs_position=dp,
                                  max_abs_quaternion=dq, tolerance=1e-12, passed=passed)
    print(f"A0 determinism: {len(a)} step records each, same steps {same}; max|d position| = {dp:.3e} m, "
          f"max|d quaternion| = {dq:.3e} -> {'PASS' if passed else 'FAIL'} (<= 1e-12)")
elif not ((out / "A0a").exists() or (out / "A0b").exists()):
    summary["determinism"] = dict(skipped=True, reason="A0a/A0b not in this run's ARMS")
    print("A0 determinism: skipped (A0a/A0b not in this run's ARMS)")
else:
    ok = False
    print("A0 determinism: missing A0a or A0b telemetry -> FAIL")

print(f"{'arm':<10} {'ep':>3} {'ok':>3} {'steps':>9} {'f_hat rx ry rz':>24} {'swing(50)':>9} {'rail':>5} "
      f"{'end|dx|mm':>9} {'end rot':>8} {'|e_p| tracked':>13} {'|e_p| trans':>11} {'track_e':>8}")
for arm in arms:
    header, rows = load(arm)
    result_path = out / arm / "result.json"
    if header is None or not result_path.exists():
        print(f"{arm:<10} missing")
        ok = False
        continue
    tracking = header.get("config", {}).get("tracking")
    tracked = tracking["dims"] if tracking else []
    result = json.loads(result_path.read_text())
    summary[arm] = []
    for index, episode_result in enumerate(next(iter(result["arms"].values()))["per_ep"]):
        roll = [r for r in rows if r["episode"] == index and r["phase"] == "rollout"]
        ref = [r for r in a if r["episode"] == index and r["phase"] == "rollout"] if a else []
        n = min(len(roll), len(ref))
        dx = 1e3 * float(np.linalg.norm(np.subtract(roll[n - 1]["position"], ref[n - 1]["position"]))) if n else float("nan")
        rot = float(np.degrees(np.linalg.norm(rot_delta(ref[n - 1]["quaternion"], roll[n - 1]["quaternion"])))) if n else float("nan")
        f_hat = np.array([r["f_hat"] for r in roll]) if roll else np.full((1, 6), np.nan)
        swing = float(np.max(f_hat[-50:].max(axis=0) - f_hat[-50:].min(axis=0))) if roll else float("nan")
        clip = float(result["args"].get("clip") or 0.15)
        railed = float(np.mean(np.abs(f_hat) >= clip - 1e-12)) if roll else float("nan")
        with_e = sum("track_e" in r for r in roll)
        if tracking and (not roll or with_e != len(roll)):
            ok = False
        entry = dict(episode=episode_result.get("replay_episode"), ok=episode_result["ok"], steps=len(roll),
                     recorded_steps=episode_result.get("recorded_steps"), final_f_hat=f_hat[-1].tolist(),
                     f_hat_swing_last50=swing, railed_fraction=railed,
                     final_position_deviation_mm=dx, final_rotation_deviation_deg=rot, track_e_records=with_e)
        if with_e:
            errors = np.array([r["track_e"] for r in roll])
            entry.update(final_track_e=errors[-1].tolist(),
                         max_abs_track_e=float(np.abs(errors).max()),
                         final_track_e_tracked=float(np.linalg.norm(errors[-1][tracked])) if tracked else None,
                         final_track_e_translation=float(np.linalg.norm(errors[-1][:3])),
                         tracking=dict(mode=tracking.get("mode", "rate"), leak=tracking.get("leak", 0.0),
                                       obs=tracking.get("obs", "full"),
                                       kappa=tracking["kappa"], anchor=tracking["anchor"], dims=tracked))
        summary[arm].append(entry)
        fh = " ".join(f"{v:+.4f}" for v in entry["final_f_hat"][3:6])
        et = f"{entry['final_track_e_tracked']:.3f}" if with_e and tracked else "-"
        ep = f"{entry['final_track_e_translation']:.3f}" if with_e else "-"
        print(f"{arm:<10} {str(entry['episode']):>3} {int(entry['ok']):>3} {len(roll):>4}/{str(entry['recorded_steps']):<4} "
              f"{fh:>24} {swing:>9.4f} {railed:>5.2f} {dx:>9.2f} {rot:>8.3f} {et:>13} {ep:>11} {with_e:>4}/{len(roll):<3}")
summary["passed"] = bool(ok)
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print("columns: swing(50) = max f_hat range over the last 50 steps (oscillation indicator; on an episode "
      "shorter than that it still contains the startup transient); rail = fraction of |f_hat| at the clip; "
      "end|dx|/end rot = deviation from the A0a reference trajectory; |e_p| = tracking pose error, on the "
      "tracked dims and on the (uncorrected, untracked) translation dims")
print(f"checks {'PASSED' if ok else 'FAILED'}; wrote {out / 'summary.json'}")
sys.exit(0 if ok else 1)
PYCHECK
}

if [ "${1:-}" = check ]; then check; exit $?; fi

KAPPA_RATE=${KAPPA_RATE:?set KAPPA_RATE, the rate-mode tracking gain (e.g. KAPPA_RATE=0.05)}
EPISODES=${EPISODES:?set EPISODES, comma-separated episode indices of the replay log (e.g. EPISODES=0)}
SHIPPED="--gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.15 --corr-dims 3,4,5"
BASE_C="--law innov --dc-constrain corrected $SHIPPED"
FAULT="--sev $SEV --profile step"
POSITION="--track-mode position --track-anchor 0 --track-dims 3,5 --track-ref dc"
declare -A FLAGS=(
  [A0a]="--sev 0 --arms frozen"
  [A0b]="--sev 0 --arms frozen"
  [C]="$FAULT --arms adaptive $BASE_C"
  [HC]="--sev 0 --arms adaptive $BASE_C"
  [CR]="$FAULT --arms adaptive $BASE_C --track-kappa $KAPPA_RATE --track-mode rate --track-anchor 5 --track-dims 3,4,5 --track-ref dc"
  [CP0]="$FAULT --arms adaptive $BASE_C --track-kappa $KAPPA_CP0 $POSITION --track-leak 0"
)
ORDER="A0a A0b C HC CR"
for k in $KAPPAS_POS; do
  FLAGS[CP_k$k]="$FAULT --arms adaptive $BASE_C --track-kappa $k $POSITION --track-leak $LEAK"
  FLAGS[CPT_k$k]="${FLAGS[CP_k$k]} --track-obs tracked"
  FLAGS[HCP_k$k]="--sev 0 --arms adaptive $BASE_C --track-kappa $k $POSITION --track-leak $LEAK"
  ORDER="$ORDER CP_k$k CPT_k$k HCP_k$k"
done
ORDER="$ORDER CP0"
ARMS=${ARMS:-$ORDER}
for name in $ARMS; do [ -n "${FLAGS[$name]:-}" ] || { echo "unknown arm $name"; exit 1; }; done
for f in "$CODE/adaptive_law.py" "$REPLAY_LOG" "$CAL_LOG" "$OPENLOOP" "$PY"; do
  [ -e "$f" ] || { echo "missing $f"; exit 1; }
done
mkdir -p "$OUT"
{ echo "run $RUN  $(date -Is)  host $(hostname)"
  echo "KAPPA_RATE=$KAPPA_RATE KAPPAS_POS=$KAPPAS_POS KAPPA_CP0=$KAPPA_CP0 LEAK=$LEAK EPISODES=$EPISODES SEV=$SEV"
  echo "SUITE=$SUITE JOBS=$JOBS ARMS=$ARMS MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
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
