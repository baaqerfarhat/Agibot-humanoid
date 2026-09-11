#!/usr/bin/env bash
# Error-geometry study (prereg_records/PREREG_ERROR_GEOMETRY.md).
#
# Each cell injects a pure translation command error via --fault-vec and runs the FROZEN policy.
# --estimate-only withholds the correction (apply_corr=not estimate_only, adaptive_law.py:1104),
# so BOTH arms execute a+f and each cell yields 2*EPISODES independent frozen episodes.
#
# Fails loudly. The earlier sweep in this project reported "done" through 18 consecutive failures
# because it never checked that the output file existed; this checks the file AND its contents,
# and prints the actual error.
set -u
ROOT=/data/fxxie/vla
PY=$ROOT/envs/libero/bin/python
REPO=$ROOT/repo
OUT=${1:?usage: run_geometry.sh <outdir> <spec> [<spec>...]   spec=name:fx,fy,fz}
shift
mkdir -p "$OUT"
EPISODES=${EPISODES:-20}
fail=0
for spec in "$@"; do
  name=${spec%%:*}; vec=${spec#*:}
  o="$OUT/$name.json"; lg="$OUT/$name.log"
  if [ -s "$o" ] && "$PY" -c "import json,sys; d=json.load(open('$o')); sys.exit(0 if len(d['arms'])==2 and all(v['n']==$EPISODES for v in d['arms'].values()) else 1)" 2>/dev/null; then
    echo "skip $name (complete)"; continue
  fi
  echo "run  $name  fault-vec=$vec,0,0,0"
  "$PY" "$REPO/openpi/adaptive_law.py" \
    --port 8000 --suite libero_spatial --episodes "$EPISODES" --eval-init 45 \
    --fault-vec "$vec,0,0,0" --estimate-only \
    --corr-dims 0,1,2 --dead 0.008 --norm-r 0.15 --clip 0.30 --gamma 0.08 \
    --log  "$REPO/results/phase05/error_signal_so3.json" \
    --openloop "$REPO/results/phase05/openloop_so3.json" \
    --control /tmp/ctl.json --ack /tmp/ack.json \
    --telemetry "$OUT/${name}_tel.jsonl" --out "$o" > "$lg" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$o" ]; then
    echo "FAIL $name rc=$rc -- last 25 lines:"; tail -25 "$lg"; fail=1; continue
  fi
  if ! "$PY" -c "import json,sys; d=json.load(open('$o')); a=d['arms']; assert len(a)==2, a.keys(); assert all(v['n']==$EPISODES for v in a.values()), {k:v['n'] for k,v in a.items()}" 2>/dev/null; then
    echo "FAIL $name incomplete arms -- last 25 lines:"; tail -25 "$lg"; fail=1; continue
  fi
  "$PY" -c "
import json; a=json.load(open('$o'))['arms']
print('  ok $name  ' + '  '.join('%s %d/%d'%(k,v['successes'],v['n']) for k,v in a.items()))"
done
[ $fail -eq 0 ] && echo "GEOMETRY_BATCH_DONE" || { echo "GEOMETRY_BATCH_FAILED"; exit 1; }
