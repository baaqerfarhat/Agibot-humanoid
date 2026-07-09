import base64, sys, time, json
import requests

img_path = sys.argv[1] if len(sys.argv) > 1 else "captures/laptop_pull.jpg"
model = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5vl:7b"

with open(img_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

prompt = ("You are a robot's vision system. List the main objects and people "
          "you see and roughly where they are (left/center/right). Then one short "
          "summary sentence. Be concise.")

t0 = time.time()
r = requests.post("http://127.0.0.1:11434/api/generate", json={
    "model": model, "prompt": prompt, "images": [b64],
    "stream": False, "options": {"temperature": 0.2},
}, timeout=300)
dt = time.time() - t0
data = r.json()
print(f"=== model={model}  latency={dt:.1f}s ===")
print(data.get("response", data))
