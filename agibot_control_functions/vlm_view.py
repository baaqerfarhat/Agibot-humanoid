#!/usr/bin/env python3
"""Live camera + VLM viewer that runs ON THE LAPTOP.

Pulls the robot's camera stream over the wired link and runs a local Ollama
VLM (Qwen2.5-VL) on it, overlaying the analysis on a live window.

Usage:
  python vlm_view.py
  python vlm_view.py --model moondream --interval 0
  python vlm_view.py --robot-url http://10.0.1.41:8099
"""
import argparse
import base64
import threading
import time
from collections import deque

import cv2
import numpy as np
import requests

latest_frame = None          # decoded BGR frame (np.ndarray)
frame_lock = threading.Lock()
caption = "starting..."
caption_lock = threading.Lock()
vlm_stat = {"latency": 0.0, "count": 0, "state": "idle"}
stop = threading.Event()


def fetcher(url, fps):
    global latest_frame
    period = 1.0 / max(1, fps)
    sess = requests.Session()
    while not stop.is_set():
        t0 = time.time()
        try:
            r = sess.get(url + "/frame.jpg", timeout=4)
            if r.status_code == 200:
                arr = np.frombuffer(r.content, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    with frame_lock:
                        latest_frame = img
        except Exception:
            pass
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


def analyzer(ollama, model, prompt, interval):
    global caption
    sess = requests.Session()
    while not stop.is_set():
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.2)
            continue
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        b64 = base64.b64encode(jpg.tobytes()).decode()
        vlm_stat["state"] = "analyzing"
        t0 = time.time()
        try:
            r = sess.post(ollama + "/api/generate", json={
                "model": model, "prompt": prompt, "images": [b64],
                "stream": False, "keep_alive": "30m",
                "options": {"temperature": 0.2},
            }, timeout=300)
            resp = r.json().get("response", "").strip()
        except Exception as e:
            resp = f"[VLM error: {e}]"
        dt = time.time() - t0
        vlm_stat["latency"] = dt
        vlm_stat["count"] += 1
        vlm_stat["state"] = "idle"
        with caption_lock:
            caption = resp
        print(f"\n[{time.strftime('%H:%M:%S')}] ({dt:.1f}s) {resp}", flush=True)
        if interval > 0:
            time.sleep(interval)


def wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_overlay(img):
    h, w = img.shape[:2]
    with caption_lock:
        cap = caption
    lines = []
    for para in cap.split("\n"):
        lines.extend(wrap(para, max(20, w // 11)) or [""])
    bar_h = 26 * len(lines) + 40
    y0 = h - bar_h
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    status = (f"model VLM | {vlm_stat['state']} | "
              f"{vlm_stat['latency']:.1f}s | n={vlm_stat['count']}")
    cv2.putText(img, status, (10, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)
    y = y0 + 46
    for ln in lines:
        cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-url", default="http://10.0.1.41:8099")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen2.5vl:7b")
    ap.add_argument("--fps", type=int, default=15, help="display poll rate")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="seconds between VLM calls (0 = back-to-back)")
    ap.add_argument("--prompt", default=(
        "You are a robot's vision system. List the main objects and people you "
        "see and roughly where they are (left/center/right). Then one short "
        "summary sentence. Be concise."))
    args = ap.parse_args()

    print(f"Robot stream: {args.robot_url}  |  VLM: {args.model} @ {args.ollama}")
    print("Keys:  q or ESC = quit\n")

    threading.Thread(target=fetcher, args=(args.robot_url, args.fps),
                     daemon=True).start()
    threading.Thread(target=analyzer,
                     args=(args.ollama, args.model, args.prompt, args.interval),
                     daemon=True).start()

    win = "AgiBot X2 - live VLM (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1024, 576)
    times = deque(maxlen=30)
    try:
        while True:
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()
            if frame is None:
                blank = np.zeros((360, 640, 3), np.uint8)
                cv2.putText(blank, "waiting for robot stream...", (30, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.imshow(win, blank)
            else:
                times.append(time.time())
                disp = draw_overlay(frame)
                cv2.imshow(win, disp)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        stop.set()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
