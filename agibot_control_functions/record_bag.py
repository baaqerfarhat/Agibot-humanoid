#!/usr/bin/env python3
"""Record a 'bag' of camera frames by pulling the robot's MJPEG stream.

Saves each JPEG frame to an output folder plus a manifest.json with timestamps.
Run on the laptop.

  python record_bag.py --seconds 15
  python record_bag.py --seconds 15 --outdir captures/bag_walk1
"""
import argparse
import json
import os
import time
import urllib.request


def record(url, seconds, outdir):
    os.makedirs(outdir, exist_ok=True)
    print(f"Connecting to {url} ...")
    stream = urllib.request.urlopen(url, timeout=10)
    buf = b""
    frames = []
    idx = 0
    t0 = time.time()
    print(f"Recording {seconds}s ...")
    while time.time() - t0 < seconds:
        chunk = stream.read(65536)
        if not chunk:
            break
        buf += chunk
        while True:
            a = buf.find(b"\xff\xd8")            # JPEG start
            b = buf.find(b"\xff\xd9", a + 2)      # JPEG end
            if a != -1 and b != -1:
                jpg = buf[a:b + 2]
                buf = buf[b + 2:]
                idx += 1
                ts = round(time.time() - t0, 3)
                fn = f"frame_{idx:05d}.jpg"
                with open(os.path.join(outdir, fn), "wb") as f:
                    f.write(jpg)
                frames.append({"index": idx, "file": fn, "t": ts,
                               "bytes": len(jpg)})
            else:
                break
    dur = time.time() - t0
    fps = len(frames) / dur if dur else 0
    manifest = {
        "source": url,
        "seconds": round(dur, 2),
        "frame_count": len(frames),
        "fps": round(fps, 2),
        "frames": frames,
    }
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done: {len(frames)} frames in {dur:.1f}s ({fps:.1f} fps) -> {outdir}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://10.0.1.41:8099/stream.mjpg")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = args.outdir or os.path.join(
        "Agibot-humanoid/agibot_control_functions/captures",
        "bag_" + time.strftime("%Y%m%d_%H%M%S"))
    record(args.url, args.seconds, outdir)
    print(outdir)


if __name__ == "__main__":
    main()
