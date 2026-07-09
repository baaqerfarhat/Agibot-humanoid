import base64, sys, time, requests

model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5vl:7b"
img = sys.argv[2] if len(sys.argv) > 2 else "Agibot-humanoid/agibot_control_functions/captures/laptop_pull.jpg"
with open(img, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

r = requests.post("http://127.0.0.1:11434/api/generate", json={
    "model": model, "prompt": "Describe the scene in one sentence.",
    "images": [b64], "stream": False, "keep_alive": "30m",
}, timeout=600).json()
ns = 1e9
print(f"model={model}")
print(f"  total       {r.get('total_duration',0)/ns:6.2f}s")
print(f"  load        {r.get('load_duration',0)/ns:6.2f}s")
print(f"  prompt_eval {r.get('prompt_eval_duration',0)/ns:6.2f}s "
      f"({r.get('prompt_eval_count',0)} tokens)  <- image+prompt encoding")
print(f"  generate    {r.get('eval_duration',0)/ns:6.2f}s "
      f"({r.get('eval_count',0)} tokens)")
