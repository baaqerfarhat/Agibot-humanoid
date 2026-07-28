#!/usr/bin/env python3
"""Run a recorded frame-bag through multiple Ollama VLMs and build a comparison.

Processes each model to completion before switching (and unloads the others
first) so the 8 GB laptop GPU never thrashes. Produces:
  - results.json  : full structured data
  - results.csv   : flat table (frame x model)
  - report.html   : side-by-side visual comparison (thumbnails + outputs)

  python analyze_bag.py --bag Agibot-humanoid/agibot_control_functions/captures/bag_XXXX
"""
import argparse
import base64
import csv
import html
import json
import os
import statistics
import subprocess
import time

import requests

OLLAMA = "http://127.0.0.1:11434"
OLLAMA_BIN = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
DEFAULT_MODELS = ["qwen2.5vl:3b", "qwen2.5vl:7b", "moondream"]
PROMPT = ("You are a robot's vision system. List the main objects and people you "
          "see and roughly where they are (left/center/right). Then one short "
          "summary sentence. Be concise.")


def stop_all(models):
    for m in models:
        try:
            subprocess.run([OLLAMA_BIN, "stop", m], capture_output=True, timeout=30)
        except Exception:
            pass
    time.sleep(1)


def call(model, b64):
    t0 = time.time()
    r = requests.post(OLLAMA + "/api/generate", json={
        "model": model, "prompt": PROMPT, "images": [b64],
        "stream": False, "keep_alive": "30m",
        "options": {"temperature": 0.2},
    }, timeout=600)
    wall = time.time() - t0
    d = r.json()
    ns = 1e9
    return {
        "response": d.get("response", "").strip(),
        "wall_s": round(wall, 3),
        "gen_tokens": d.get("eval_count", 0),
        "gen_tps": round(d.get("eval_count", 0) / (d.get("eval_duration", 1) / ns), 1)
        if d.get("eval_duration") else 0,
    }


def pick_frames(manifest, num):
    frames = manifest["frames"]
    if num <= 0 or num >= len(frames):
        return frames
    step = len(frames) / num
    return [frames[int(i * step)] for i in range(num)]


def build_html(bag, models, sampled, results, summary, path, b64s=None):
    def esc(s):
        return html.escape(s).replace("\n", "<br>")
    rows = []
    for fr in sampled:
        idx = fr["index"]
        # embed thumbnails as data URIs so the report is a single portable file
        src = (f"data:image/jpeg;base64,{b64s[idx]}" if b64s and idx in b64s
               else fr["file"])
        cells = [f"<td class='fr'><img src='{src}'><div class='cap'>"
                 f"frame {idx} @ {fr['t']}s</div></td>"]
        for m in models:
            res = results[idx][m]
            cells.append(
                f"<td><div class='lat'>{res['wall_s']}s | {res['gen_tps']} tok/s</div>"
                f"<div class='txt'>{esc(res['response'])}</div></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    head = "<th>frame</th>" + "".join(f"<th>{html.escape(m)}</th>" for m in models)
    sumrows = []
    for m in models:
        s = summary[m]
        sumrows.append(
            f"<tr><td>{html.escape(m)}</td><td>{s['mean_s']}</td><td>{s['median_s']}</td>"
            f"<td>{s['min_s']}</td><td>{s['max_s']}</td><td>{s['hz']}</td>"
            f"<td>{s['tps']}</td></tr>")
    doc = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>VLM comparison - {html.escape(os.path.basename(bag))}</title>
<style>
 body{{font-family:Segoe UI,Arial,sans-serif;margin:20px;background:#0f1115;color:#e6e6e6}}
 h1,h2{{font-weight:600}}
 table{{border-collapse:collapse;width:100%;margin-bottom:24px}}
 th,td{{border:1px solid #2a2f3a;padding:8px;vertical-align:top;text-align:left}}
 th{{background:#1b2130;position:sticky;top:0}}
 td.fr{{width:280px}} img{{width:260px;border-radius:6px;display:block}}
 .cap{{font-size:12px;color:#9aa4b2;margin-top:4px}}
 .lat{{font-size:12px;color:#37d39b;margin-bottom:6px;font-weight:600}}
 .txt{{font-size:13px;line-height:1.4}}
 .sum td,.sum th{{text-align:center}}
</style></head><body>
<h1>VLM comparison &mdash; {html.escape(os.path.basename(bag))}</h1>
<h2>Summary (per-frame timing)</h2>
<table class='sum'><tr><th>model</th><th>mean s</th><th>median s</th><th>min s</th>
<th>max s</th><th>Hz</th><th>gen tok/s</th></tr>
{''.join(sumrows)}</table>
<h2>Frame-by-frame outputs</h2>
<table><tr>{head}</tr>
{''.join(rows)}</table>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


def find_browser():
    candidates = [
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def make_pdf(html_path, pdf_path):
    browser = find_browser()
    if not browser:
        print("  (PDF skipped: no Edge/Chrome found)")
        return False
    uri = "file:///" + os.path.abspath(html_path).replace("\\", "/")
    cmd = [browser, "--headless=old", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={os.path.abspath(pdf_path)}", uri]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
        if os.path.isfile(pdf_path):
            return True
    except Exception as e:
        print(f"  (PDF failed: {e})")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--num-frames", type=int, default=12,
                    help="evenly-spaced frames to analyze (0 = all)")
    ap.add_argument("--no-pdf", action="store_true",
                    help="skip exporting report.pdf")
    args = ap.parse_args()

    with open(os.path.join(args.bag, "manifest.json")) as f:
        manifest = json.load(f)
    sampled = pick_frames(manifest, args.num_frames)
    print(f"Bag: {args.bag}  ({manifest['frame_count']} frames @ "
          f"{manifest['fps']} fps).  Analyzing {len(sampled)} sampled frames "
          f"with {len(args.models)} models.")

    # preload jpeg bytes as base64
    b64s = {}
    for fr in sampled:
        with open(os.path.join(args.bag, fr["file"]), "rb") as fh:
            b64s[fr["index"]] = base64.b64encode(fh.read()).decode()

    results = {fr["index"]: {} for fr in sampled}
    summary = {}
    for model in args.models:
        print(f"\n=== {model} ===  (unloading others first)")
        stop_all([m for m in args.models if m != model])
        # warmup / load
        _ = call(model, b64s[sampled[0]["index"]])
        lat, tps = [], []
        for fr in sampled:
            res = call(model, b64s[fr["index"]])
            results[fr["index"]][model] = res
            lat.append(res["wall_s"])
            tps.append(res["gen_tps"])
            print(f"  frame {fr['index']:>4}: {res['wall_s']:.2f}s  "
                  f"{res['gen_tps']} tok/s  | {res['response'][:70]}")
        mean = statistics.mean(lat)
        summary[model] = {
            "mean_s": round(mean, 2),
            "median_s": round(statistics.median(lat), 2),
            "min_s": round(min(lat), 2),
            "max_s": round(max(lat), 2),
            "hz": round(1.0 / mean, 2) if mean else 0,
            "tps": round(statistics.mean(tps), 1) if tps else 0,
        }

    # write outputs
    out = {
        "bag": args.bag, "prompt": PROMPT, "models": args.models,
        "manifest_fps": manifest["fps"], "summary": summary,
        "frames": [
            {"index": fr["index"], "t": fr["t"], "file": fr["file"],
             "outputs": results[fr["index"]]}
            for fr in sampled],
    }
    jpath = os.path.join(args.bag, "results.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    cpath = os.path.join(args.bag, "results.csv")
    with open(cpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s"] + sum(
            [[f"{m} latency_s", f"{m} tok/s", f"{m} output"] for m in args.models], []))
        for fr in sampled:
            row = [fr["index"], fr["t"]]
            for m in args.models:
                r = results[fr["index"]][m]
                row += [r["wall_s"], r["gen_tps"], r["response"].replace("\n", " ")]
            w.writerow(row)

    hpath = os.path.join(args.bag, "report.html")
    build_html(args.bag, args.models, sampled, results, summary, hpath, b64s)

    ppath = os.path.join(args.bag, "report.pdf")
    pdf_ok = False
    if not args.no_pdf:
        pdf_ok = make_pdf(hpath, ppath)

    print("\n================ SUMMARY ================")
    print(f"{'model':<16}{'mean s':>8}{'median':>8}{'min':>7}{'max':>7}{'Hz':>7}{'tok/s':>8}")
    for m in args.models:
        s = summary[m]
        print(f"{m:<16}{s['mean_s']:>8}{s['median_s']:>8}{s['min_s']:>7}"
              f"{s['max_s']:>7}{s['hz']:>7}{s['tps']:>8}")
    print(f"\nSaved:\n  {jpath}\n  {cpath}\n  {hpath}")
    if pdf_ok:
        print(f"  {ppath}")


if __name__ == "__main__":
    main()
