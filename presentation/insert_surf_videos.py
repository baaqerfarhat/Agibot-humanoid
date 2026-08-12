#!/usr/bin/env python3
"""Build a copy of the SURF deck with the real videos embedded in the slots.

    python3 presentation/insert_surf_videos.py

Reads the same nine rollouts the placeholders are posters of, splits the
frozen-versus-adapted fault comparison into its two halves, and rebuilds the
deck with each clip embedded in place of its still.

    presentation/SURF_Final_Baaqer_Farhat.pptx              placeholders, ~4 MB
    presentation/SURF_Final_Baaqer_Farhat_with_videos.pptx  embedded,     ~120 MB

Keep both. The placeholder deck is the one to edit; regenerate the video deck
from it whenever the slides change.

PowerPoint plays embedded video on Windows and macOS. Keep the videos in
box_pickup/videos/ so the poster frames can be regenerated.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VIDEOS = ROOT / "box_pickup" / "videos"
CLIPS = HERE / "surf_assets" / "clips"

SIDE_BY_SIDE = VIDEOS / "isaac_fault_knee03_frozen_vs_waistadapt.mp4"

# the fault comparison ships as one 1280x720 side-by-side render; the deck wants
# the halves separately, cropped below the caption band baked into the frame
HALVES = {
    "fault_knee03_frozen.mp4": "crop=640:632:0:88",
    "fault_knee03_waistadapt.mp4": "crop=640:632:640:88",
}

NEEDED = [
    "x2_box_v8_iter8000_full.mp4",
    "x2_box_v10_iter28000_final.mp4",
    "x2_box_v33_waist_track_iter253000.mp4",
    "x2_crawl_slope_v2_palmflat_iter14500.mp4",
    "x2_crawl_slope_v3_tracking_iter49999.mp4",
    "x2_crawl_slope_v5_track_xyz_iter86000.mp4",
    "x2_box_v31_flatfoot_iter202500.mp4",
]


def main() -> int:
    missing = [n for n in NEEDED + [SIDE_BY_SIDE.name] if not (VIDEOS / n).exists()]
    if missing:
        print("missing source videos in box_pickup/videos:", file=sys.stderr)
        for m in missing:
            print("  ", m, file=sys.stderr)
        return 1

    if not shutil.which("ffmpeg"):
        print("ffmpeg is required to split the fault comparison", file=sys.stderr)
        return 1

    CLIPS.mkdir(parents=True, exist_ok=True)
    for name, vf in HALVES.items():
        out = CLIPS / name
        if out.exists():
            print(f"have  {out.name}")
            continue
        print(f"split {out.name}")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(SIDE_BY_SIDE),
             "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
             "-pix_fmt", "yuv420p", "-an", str(out)],
            check=True,
        )

    env = dict(os.environ, SURF_EMBED_VIDEOS="1")
    subprocess.run([sys.executable, str(HERE / "build_surf_slides.py")],
                   check=True, env=env)

    out = HERE / "SURF_Final_Baaqer_Farhat_with_videos.pptx"
    print(f"\n{out.relative_to(ROOT)}  ({out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
