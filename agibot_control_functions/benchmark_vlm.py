#!/usr/bin/env python3
"""Benchmark VLM models on the robot's live camera feed (run on the laptop).

Measures end-to-end latency, throughput (Hz), and Ollama's internal timing
(model load, prompt eval, generation tokens/sec) for one or more models.

Example:
  python benchmark_vlm.py --models qwen2.5vl:7b qwen2.5vl:3b moondream --frames 6
"""
import argparse
import base64
import statistics
import time

import cv2
import numpy as np
import requests

DEF_PROMPT = ("You are a robot's vision system. List the main objects and people "
              "you see and roughly where they are (left/center/right). Then one "
              "short summary sentence. Be concise.")


def grab_frames(robot_url, n, fallback):
    frames = []
    sess = requests.Session()
    for i in range(n):
        try:
            r = sess.get(robot_url + "/frame.jpg", timeout=5)
            if r.status_code == 200:
                frames.append(r.content)
        except Exception:
            pass
        time.sleep(0.3)
    if not frames and fallback:
        with open(fallback, "rb") as f:
            frames = [f.read()] * n
    return frames


def call(ollama, model, b64, prompt, num_predict):
    body = {
        "model": model, "prompt": prompt, "images": [b64],
        "stream": False, "keep_alive": "30m",
        "options": {"temperature": 0.2},
    }
    if num_predict > 0:
        body["options"]["num_predict"] = num_predict
    t0 = time.time()
    r = requests.post(ollama + "/api/generate", json=body, timeout=600)
    wall = time.time() - t0
    d = r.json()
    ns = 1e9
    return {
        "wall": wall,
        "total": d.get("total_duration", 0) / ns,
        "load": d.get("load_duration", 0) / ns,
        "p_tokens": d.get("prompt_eval_count", 0),
        "p_eval": d.get("prompt_eval_duration", 0) / ns,
        "g_tokens": d.get("eval_count", 0),
        "g_eval": d.get("eval_duration", 0) / ns,
        "resp": d.get("response", "").strip(),
    }


def bench_model(ollama, model, frames, prompt, num_predict):
    print(f"\n=== {model} ===")
    b64s = [base64.b64encode(f).decode() for f in frames]
    print("  warmup (loads model into VRAM)...", flush=True)
    w = call(ollama, model, b64s[0], prompt, num_predict)
    print(f"  warmup wall={w['wall']:.1f}s  (load={w['load']:.1f}s)")
    lat, tps = [], []
    for i, b in enumerate(b64s):
        m = call(ollama, model, b, prompt, num_predict)
        gtps = m["g_tokens"] / m["g_eval"] if m["g_eval"] > 0 else 0
        lat.append(m["wall"])
        tps.append(gtps)
        print(f"  frame {i+1}/{len(b64s)}: {m['wall']:.2f}s  "
              f"gen={m['g_tokens']}tok @ {gtps:.1f} tok/s")
    mean = statistics.mean(lat)
    return {
        "model": model,
        "mean": mean,
        "median": statistics.median(lat),
        "min": min(lat),
        "max": max(lat),
        "hz": 1.0 / mean if mean else 0,
        "tps": statistics.mean(tps) if tps else 0,
        "sample": w["resp"][:200],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["qwen2.5vl:7b", "qwen2.5vl:3b", "moondream"])
    ap.add_argument("--robot-url", default="http://10.0.1.41:8099")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--prompt", default=DEF_PROMPT)
    ap.add_argument("--num-predict", type=int, default=0,
                    help="cap generated tokens for consistent timing (0=uncapped)")
    ap.add_argument("--fallback", default="Agibot-humanoid/agibot_control_functions/captures/laptop_pull.jpg")
    args = ap.parse_args()

    print(f"Grabbing {args.frames} live frames from {args.robot_url} ...")
    frames = grab_frames(args.robot_url, args.frames, args.fallback)
    print(f"Got {len(frames)} frames.")
    if not frames:
        print("No frames available; aborting.")
        return

    results = []
    for m in args.models:
        try:
            results.append(bench_model(args.ollama, m, frames, args.prompt,
                                       args.num_predict))
        except Exception as e:
            print(f"  ERROR benchmarking {m}: {e}")

    print("\n\n================  SUMMARY  ================")
    print(f"{'model':<16} {'mean s':>7} {'med s':>7} {'min s':>7} "
          f"{'max s':>7} {'Hz':>6} {'tok/s':>7}")
    print("-" * 62)
    for r in results:
        print(f"{r['model']:<16} {r['mean']:>7.2f} {r['median']:>7.2f} "
              f"{r['min']:>7.2f} {r['max']:>7.2f} {r['hz']:>6.2f} {r['tps']:>7.1f}")
    print("\nSample outputs:")
    for r in results:
        print(f"  [{r['model']}] {r['sample']}")


if __name__ == "__main__":
    main()
