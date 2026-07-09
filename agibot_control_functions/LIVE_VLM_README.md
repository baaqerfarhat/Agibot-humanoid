# Live Camera Feed + VLM Analysis (AgiBot X2)

Stream the robot's head camera over the wired Ethernet link to your laptop and
run a local Vision-Language Model (Qwen2.5-VL via Ollama) on the live feed.

The VLM runs **on the laptop** (the robot has no reliable internet). The robot
only streams JPEG frames.

```
Robot Orbbec Gemini 335 ──ROS2──► camera_stream_server.py  (robot, port 8099)
                                          │  HTTP / MJPEG over Ethernet
                                          ▼
Laptop  vlm_view.py ──► Ollama Qwen2.5-VL (127.0.0.1:11434) ──► live window + text
```

---

## 0. Prerequisites

**Network (wired):**
- Laptop Ethernet is `10.0.1.50`, robot is `10.0.1.41` (same cable/subnet).
- Quick check: `ping 10.0.1.41` should reply.

**SSH key (laptop):** `%USERPROFILE%\.ssh\agibot_ed25519`
```powershell
ssh -i "$env:USERPROFILE\.ssh\agibot_ed25519" run@10.0.1.41   # password: 1 (if asked)
```

**Laptop software (already installed):**
- Ollama running at `http://127.0.0.1:11434` with model `qwen2.5vl:7b`
  (fast fallback: `moondream`).
- Python 3.11 with `opencv-python`, `requests`, `numpy`.
- Check Ollama + model: `curl http://127.0.0.1:11434/api/tags`

**Files** (in `agibot_control_functions/`):
- `camera_stream_server.py` — runs on the robot.
- `vlm_view.py` — runs on the laptop.

---

## 1. Start the camera stream ON THE ROBOT

Copy the server to the robot (only needed if it changed) and start it detached
so it keeps running after you disconnect:

```powershell
# from the repo root on the laptop (PowerShell)
$key = "$env:USERPROFILE\.ssh\agibot_ed25519"
scp -i $key ".\Agibot-humanoid\agibot_control_functions\camera_stream_server.py" run@10.0.1.41:/tmp/camera_stream_server.py

ssh -n -i $key run@10.0.1.41 "source /opt/ros/humble/setup.bash 2>/dev/null; source ~/aimdk_ws/install/setup.bash 2>/dev/null; export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/software/entry/cfg/ros_dds_configuration.xml; export RCUTILS_LOGGING_SEVERITY=FATAL; pkill -f camera_stream_server.py; sleep 1; setsid bash -c 'python3 /tmp/camera_stream_server.py > /tmp/cam_server.log 2>&1' < /dev/null & echo LAUNCHED"
```

> The `FASTRTPS_DEFAULT_PROFILES_FILE` line is **required** — the robot routes
> camera data over a custom FastDDS profile; without it you get zero frames.

**Confirm it is streaming** (should show a growing frame count and small `age_s`):
```powershell
curl http://10.0.1.41:8099/status
# {"frames": 389, "age_s": 0.03}
```

### Server endpoints (port 8099)
| URL | What it gives |
|-----|---------------|
| `http://10.0.1.41:8099/frame.jpg`   | latest single JPEG |
| `http://10.0.1.41:8099/stream.mjpg` | continuous MJPEG stream |
| `http://10.0.1.41:8099/status`      | JSON: frame count + age |
| `http://10.0.1.41:8099/`            | simple browser viewer page |

---

## 2. View the live feed (no VLM)

Fastest way — just open in a browser on the laptop:

```
http://10.0.1.41:8099/            (full-page live view)
http://10.0.1.41:8099/stream.mjpg (raw MJPEG stream)
```

Or grab a single snapshot:
```powershell
curl -o snap.jpg http://10.0.1.41:8099/frame.jpg
```

---

## 3. Run the VLM on the live feed

From the laptop (repo root):

```powershell
python .\Agibot-humanoid\agibot_control_functions\vlm_view.py
```

A window titled **"AgiBot X2 - live VLM"** opens showing the live camera with the
VLM's analysis overlaid at the bottom. It re-analyzes the newest frame
continuously.

### Options
```powershell
# faster but lower quality model
python .\Agibot-humanoid\agibot_control_functions\vlm_view.py --model moondream

# ask it to look for something specific
python .\Agibot-humanoid\agibot_control_functions\vlm_view.py --prompt "find all people and any safety hazards; give their location"

# space out VLM calls (e.g. one every 3 s) to reduce GPU load
python .\Agibot-humanoid\agibot_control_functions\vlm_view.py --interval 3

# point at a different robot / ollama host
python .\Agibot-humanoid\agibot_control_functions\vlm_view.py --robot-url http://10.0.1.41:8099 --ollama http://127.0.0.1:11434
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--model`     | `qwen2.5vl:7b` | Ollama vision model tag |
| `--robot-url` | `http://10.0.1.41:8099` | camera server |
| `--ollama`    | `http://127.0.0.1:11434` | local Ollama API |
| `--fps`       | `15` | display refresh (frame poll) rate |
| `--interval`  | `0` | seconds between VLM calls (0 = back-to-back) |
| `--prompt`    | objects+people | what the VLM should report |

### Controls
- Press **`q`** or **`ESC`** in the window to quit.

---

## 4. Check the live results

Two places show results simultaneously:

1. **On the video window** — the latest analysis is overlaid at the bottom, with a
   status line: `state | latency | n=<count>`.
2. **In the terminal** — every analysis is printed with a timestamp and latency:
   ```
   [14:50:44] (4.0s) Main objects: yellow toolboxes, rolling chairs, desks,
   computer monitors. People: one person in red shirt at desk.
   ```

**Expected speed:** warm `qwen2.5vl:7b` ≈ **~4 s per frame (~0.25 Hz)** on this
laptop. The first call after startup takes ~50 s because it loads the 6 GB model
once. Use `--model moondream` for faster (lower-quality) updates.

---

## 5. Stop / restart

**Stop the laptop viewer:** press `q` in the window (or close it).

**Stop the robot server:**
```powershell
ssh -n -i "$env:USERPROFILE\.ssh\agibot_ed25519" run@10.0.1.41 "pkill -f camera_stream_server.py"
```

**Restart after a robot reboot:** repeat Step 1.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `/status` shows `age_s` large or curl fails | Camera server not running → redo Step 1. Check `/tmp/cam_server.log` on the robot. |
| Zero frames / server log has no frames | Missing `FASTRTPS_DEFAULT_PROFILES_FILE` env when launching the server. |
| `ping 10.0.1.41` fails | Reseat the Ethernet cable; confirm laptop Ethernet is `10.0.1.50`. |
| Viewer window: "waiting for robot stream..." | Server down or wrong `--robot-url`. Verify `curl http://10.0.1.41:8099/status`. |
| First VLM result takes ~50 s | Normal cold model load; subsequent calls ~4 s. |
| VLM error in overlay/terminal | Ollama not running, or model not pulled. `curl http://127.0.0.1:11434/api/tags`. |
| Image upside down | The server already rotates 180°. If a future camera is upright, set `ROTATE_180 = False` in `camera_stream_server.py`. |

## Notes
- Camera source topic: `/aima/hal/sensor/rgbd_head_front/rgb_image` (Orbbec
  Gemini 335, 1280x720 `rgb8`, ~10 fps), subscribed with BEST_EFFORT QoS.
- A generative VLM cannot hit 5-10 Hz on this hardware. For true real-time
  (5-10 Hz) open-vocabulary object detection, use a detector such as
  YOLO-World or NanoOWL instead of a generative VLM.
