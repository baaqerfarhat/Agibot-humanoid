#!/usr/bin/env bash
echo "=== newest 3 log dirs ==="
ls -dt /agibot/data/log/log_* 2>/dev/null | head -3
NEW=$(ls -dt /agibot/data/log/log_* 2>/dev/null | head -1)
echo "newest=$NEW"
echo "=== orbbec_camera log dir ==="
ls -la "$NEW/orbbec_camera" 2>/dev/null | head -20
echo "=== tail of each orbbec log file ==="
find "$NEW/orbbec_camera" -type f 2>/dev/null | head -6 | while read f; do
  echo "--- $f ---"
  tail -30 "$f" 2>/dev/null
done
echo "=== grep color/stream/error/frame across newest orbbec log ==="
grep -riE "color|stream start|frame|error|fail|busy|device" "$NEW/orbbec_camera" 2>/dev/null | tail -30
