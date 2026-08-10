"""Path resolution for the adaptation scripts, so they run on any machine.

Every location can be overridden by an environment variable; the defaults
reproduce the layout `box_pickup/setup_holosoma_x2.sh` creates (holosoma cloned
as a sibling of this repo).

    HOLOSOMA_ROOT   holosoma checkout            default: <repo>/../holosoma
    X2_CKPT         v31 torch checkpoint         default: see resolve_ckpt()
    X2_MOTION       reference motion clip        default: box_speed100.npz
    X2_POLICY_NPZ   exported numpy policy        default: <repo>/box_pickup/policy/...
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOLOSOMA_ROOT = Path(os.environ.get("HOLOSOMA_ROOT", REPO_ROOT.parent / "holosoma"))
HOLOSOMA_PKG = HOLOSOMA_ROOT / "src" / "holosoma"

MOTION = Path(os.environ.get(
    "X2_MOTION",
    HOLOSOMA_PKG / "holosoma/data/motions/x2_31dof/whole_body_tracking"
                   "/box_multispeed/box_speed100.npz",
))
POLICY_NPZ = Path(os.environ.get(
    "X2_POLICY_NPZ", REPO_ROOT / "box_pickup/policy/x2_box_policy_v31.npz"
))

# The v31 checkpoint is gitignored (*.pt) but ships inside the tracked
# x2_box_policy_share.tar.gz, so the extracted copy is the portable fallback.
_CKPT_CANDIDATES = [
    HOLOSOMA_ROOT / "logs/WholeBodyTracking"
                    "/20260730_215012-x2_box_v31_flatfoot-locomotion/model_202500.pt",
    REPO_ROOT / "x2_box_policy_share/checkpoint/model_202500.pt",
]


def resolve_ckpt() -> Path:
    override = os.environ.get("X2_CKPT")
    if override:
        return Path(override)
    for c in _CKPT_CANDIDATES:
        if c.exists():
            return c
    raise SystemExit(
        "v31 checkpoint not found. Extract it with\n"
        f"    tar -xzf {REPO_ROOT / 'x2_box_policy_share.tar.gz'} -C {REPO_ROOT}\n"
        "or set X2_CKPT to your own checkpoint."
    )


def enter_holosoma() -> None:
    """chdir into holosoma and put it on sys.path (its configs use relative paths)."""
    import sys

    if not HOLOSOMA_PKG.exists():
        raise SystemExit(
            f"holosoma not found at {HOLOSOMA_ROOT}.\n"
            "Run box_pickup/setup_holosoma_x2.sh, or set HOLOSOMA_ROOT."
        )
    os.chdir(HOLOSOMA_ROOT)
    sys.path.insert(0, str(HOLOSOMA_PKG))
