#!/usr/bin/env python3
"""Crash-safe per-run data logger for the X2 deploy scripts.

Records, at every control tick, the measured joint encoder state (position +
velocity), the commanded joint target, and the base-IMU state (angular
velocity + orientation + roll). Data is streamed straight to a CSV file and
flushed on every row, so a Ctrl-C or an unexpected exception keeps everything
logged up to that instant -- nothing is buffered only in memory.

A `<name>.meta.json` sidecar records which policy/script/args produced the run
so the logs are self-describing for offline policy-training analysis.

Usage (from a deploy script):

    from run_logger import RunLogger
    logger = RunLogger(joint_names, base_imu="torso",
                       run_name="box_pickup_x2_box_policy",
                       meta={"script": "deploy_x2_box_pickup.py", ...},
                       log_dir="run_logs", enabled=True)
    ...
    logger.log(t_s, phase, frame, imus, jmap, target_by_name)
    ...
    logger.close()          # ALWAYS call in a finally: block
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from datetime import datetime


def _roll_of(q) -> float:
    x, y, z, w = q
    return math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


class RunLogger:
    def __init__(self, joint_names, base_imu, run_name, meta=None,
                 log_dir="run_logs", enabled=True, extra_columns=None):
        """extra_columns: names appended AFTER the per-joint columns, so that
        positional parsers of older logs keep working. Values come from the
        `extra` dict passed to log()."""
        self.enabled = bool(enabled)
        self.joint_names = list(joint_names)
        self.base_imu = base_imu
        self.extra_columns = list(extra_columns or [])
        self.path = None
        self._f = None
        self._w = None
        self._n = 0
        if not self.enabled:
            return

        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{stamp}_{run_name}"
        self.path = os.path.join(log_dir, base + ".csv")
        # newline="" so csv does not emit blank rows on Windows-mounted FSes.
        self._f = open(self.path, "w", newline="")
        self._w = csv.writer(self._f)

        header = ["t_s", "wall_time", "phase", "frame", "roll",
                  "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z",
                  "base_quat_x", "base_quat_y", "base_quat_z", "base_quat_w"]
        for n in self.joint_names:
            header += [f"{n}__pos_meas", f"{n}__vel_meas", f"{n}__tgt"]
        header += self.extra_columns
        self._w.writerow(header)
        self._f.flush()

        info = {"run_name": run_name, "base_imu": base_imu, "created": stamp,
                "joint_names": self.joint_names, "csv": os.path.basename(self.path),
                "columns_per_joint": ["pos_meas", "vel_meas", "tgt"],
                "extra_columns": self.extra_columns}
        if meta:
            info.update(meta)
        try:
            with open(os.path.join(log_dir, base + ".meta.json"), "w") as mf:
                json.dump(info, mf, indent=2, default=str)
        except Exception:
            pass
        print(f"[log] recording joint/IMU data -> {self.path}")

    def log(self, t_s, phase, frame, imus, jmap, target_by_name, extra=None) -> None:
        """Append one tick. Never raises: logging must not crash a live run."""
        if not self.enabled or self._w is None:
            return
        try:
            imu = imus[self.base_imu]
            av = [float(v) for v in imu.ang_vel]
            q = [float(v) for v in imu.quat]
            roll = _roll_of(q)
        except Exception:
            av = [float("nan")] * 3
            q = [float("nan")] * 4
            roll = float("nan")

        row = [f"{float(t_s):.4f}", f"{time.time():.4f}", phase, int(frame),
               f"{roll:.6f}", av[0], av[1], av[2], q[0], q[1], q[2], q[3]]
        for n in self.joint_names:
            jr = jmap.get(n) if hasattr(jmap, "get") else None
            if jr is None:
                row += [float("nan"), float("nan")]
            else:
                row += [float(jr.position), float(jr.velocity)]
            tgt = target_by_name.get(n) if hasattr(target_by_name, "get") else None
            row.append(float(tgt) if tgt is not None else float("nan"))
        for c in self.extra_columns:
            v = (extra or {}).get(c)
            row.append(float(v) if v is not None else float("nan"))

        try:
            self._w.writerow(row)
            self._n += 1
            # flush every tick so a hard interrupt loses nothing.
            self._f.flush()
        except Exception:
            pass

    def close(self) -> None:
        if self._f is None:
            return
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass
        print(f"[log] saved {self._n} rows -> {self.path}")
        self._f = None
        self._w = None
