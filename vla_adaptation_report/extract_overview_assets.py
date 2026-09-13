#!/usr/bin/env python3
"""Extract authentic, source-bound robot scenes for the overview figure.

Only full-panel annotation-bar crops are used; there is no scene retouching.
The selected pairs have matched displayed steps, not matched policy randomness.
Off outcomes are eventual timeouts, not failures declared at the shown step.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/overview_assets"
VIDEOS = {
    "panda": {
        "path": "results/phase05/adaptive_vs_frozen.mp4",
        "sha256": "86b7533a406302ee98d64412990fffdbeedaf334755317eb91651acd20c6bdf9",
        "origin_commit": "a04c285fcf8a8a878890ad21bb698586f0089698",
        "renderer": "openpi/compare_video.py",
        "renderer_sha256": "5e2747c8ee671ba2ea446dda9ef715189e20e0eaf252c72a828659c79508d545",
        "fps": 20, "geometry": [768, 464], "frames": 960,
        "robot": "Panda", "policy": "pi0.5", "suite": "libero_spatial",
        "task": "pick up the black bowl next to the ramekin and place it on the plate",
        "fault": "command offset +0.05 on all six motion-action channels",
        "method": "original continuous calibrated adaptation; rotation-only correction",
        "timing": "Images precede env.step; the reported result belongs to the following transition.",
        "selection": "Four repaired clips retained from six candidates; two off-success skips (record section 28.3).",
        "pairing": "Same task and stored initial state by recording code; policy samples not pinned; no archived full-state hash.",
        "header_geometry": "74px annotation bar in 458px original canvas, resized to 464px; crop starts at 75px.",
    },
    "aloha": {
        "path": "results/phase05/adaptive_vs_frozen_aloha.mp4",
        "sha256": "47154db5ba21397394e2215c32febb966d35e1b5c75bf84bb2c8e1f299e90c97",
        "origin_commit": "f7174bbd038c3107f072592b092e62ea0bf7027d",
        "renderer": "openpi/aloha_video.py",
        "renderer_sha256": "f510f0de4d37480168433ef4057fbbf2ce12e67878b63c9f16adb9e9e99fec4d",
        "fps": 25, "geometry": [1280, 560], "frames": 1300,
        "robot": "ALOHA", "policy": "pi0", "suite": "gym_aloha transfer_cube",
        "task": "Transfer cube",
        "fault": "joint-target command offset  +0.02 rad on left arm joints 0–5",
        "method": "identify in one episode, then hold correction fixed during task",
        "timing": "Renderer stores the post-env.step scene and that transition's success result.",
        "selection": "Four historical off-failure/on-success clips; candidate count not archived.",
        "pairing": "Same episode seed by recording code; policy samples not pinned; no archived full-state hash.",
        "header_geometry": "74px annotation bar above 480px scene, encoded canvas resized from 554 to 560px; crop starts at 75px.",
        "misleading_original_label": "The generic header says ADAPTIVE (online), but detailed overlay and origin commit establish identify-then-hold.",
    },

}

# Two Panda panels share a displayed step; ALOHA is a separate held-success scene.
# The Panda terminal full frame is retained only as outcome evidence.
ASSETS = {
    "panda_off_matched": ("panda", 152, "off", (0, 75, 384, 389), 162, False, 229),
    "panda_on_success": ("panda", 152, "on", (384, 75, 384, 389), 162, True, 162),
    "aloha_on_success": ("aloha", 272, "on", (640, 75, 640, 485), 272, True, 272),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_bytes(commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{commit}:{relative}"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {"schema_version": 1, "sources": {}, "assets": {}, "decoded_evidence_frames": [], "commands": [],
               "scope": "Selected historical simulation illustrations; no new benchmark outcomes or estimator ranking.",
               "correction_on_failure_available_for_current_protocol": False,
               "excluded_failure_video": {
                   "path": "results/phase05/adaptive_vs_frozen_offset010.mp4",
                   "sha256": "69d3f81a3e597ea5fdf8b8551c200fbbb2d5a719a3fe37d68ade3db79e09bf99",
                   "observed_failure_frames": [459, 699],
                   "reason": "On-failure annotations are visible, but exact generating source/CLI is absent; 432px encoded height mismatches 458px canvas of committed renderer. Historical renderer used all-six correction, axis-angle differences and recording-camera policy input, unlike current rotation-only SO(3) protocol. Not used as Figure 1 algorithm evidence.",
               },
               "limitations": [
                   "No exact render CLI, selected task/init identifier or per-video episode JSON was saved.",
                   "No selected clip is a deterministic counterfactual with identical policy samples.",
                   "No image depicts physical joint 5 harm, torque faults, composite adaptation or the weighted allocator.",
                   "All four corrected ALOHA clips report SUCCESS. An initial reduced-preview misreading was corrected against full-resolution frames before producing assets.",
               ]}
    for name, source in VIDEOS.items():
        path = ROOT / source["path"]
        if digest(path.read_bytes()) != source["sha256"]:
            raise ValueError(f"Video changed: {name}")
        if digest(git_bytes(source["origin_commit"], source["path"])) != source["sha256"]:
            raise ValueError(f"Video does not match archived origin: {name}")
        if digest((ROOT / source["renderer"]).read_bytes()) != source["renderer_sha256"]:
            raise ValueError(f"Renderer changed: {name}")
        if digest(git_bytes(source["origin_commit"], source["renderer"])) != source["renderer_sha256"]:
            raise ValueError(f"Renderer does not match archived origin: {name}")
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_frames",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames:frame=best_effort_timestamp_time",
            "-of", "json", str(path)]))
        stream = probe["streams"][0]
        if ([stream["width"], stream["height"]] != source["geometry"]
                or stream["r_frame_rate"] != f'{source["fps"]}/1'
                or int(stream["nb_frames"]) != source["frames"]):
            raise ValueError(f"Unexpected frame geometry: {name}")
        receipt["sources"][name] = {**source, "stream": stream,
                                     "origin_video_and_renderer_bytes_verified": True}
        for frame in sorted({v[1] for v in ASSETS.values() if v[0] == name}
                            | ({219} if name == "panda" else set())):
            timestamp = float(probe["frames"][frame]["best_effort_timestamp_time"])
            if abs(timestamp - frame / source["fps"]) > 1e-9:
                raise ValueError(f"Unexpected timestamp: {name}:{frame}")
            full = OUT / f"{name}_frame_{frame:04d}_annotated.png"
            argv = ["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                    f"select=eq(n\\,{frame}),format=rgb24", "-frames:v", "1", "-y", str(full)]
            subprocess.run(argv, check=True)
            receipt["commands"].append(argv)
            receipt["decoded_evidence_frames"].append({
                "video_key": name, "frame_index_zero_based": frame,
                "video_presentation_seconds": timestamp,
                "path": str(full.relative_to(ROOT)), "sha256": digest(full.read_bytes()),
            })
    for name, (video, frame, arm, box, step, success, last_step) in ASSETS.items():
        source = VIDEOS[video]
        full = OUT / f"{video}_frame_{frame:04d}_annotated.png"
        x, y, width, height = box
        destination = OUT / f"{name}.png"
        argv = ["ffmpeg", "-v", "error", "-i", str(ROOT / source["path"]), "-vf",
                f"select=eq(n\\,{frame}),format=rgb24,crop={width}:{height}:{x}:{y}:exact=1",
                "-frames:v", "1", "-y", str(destination)]
        subprocess.run(argv, check=True)
        receipt["commands"].append(argv)
        raw = Image.open(full).convert("RGB")
        crop = Image.open(destination).convert("RGB")
        if crop.tobytes() != raw.crop((x, y, x + width, y + height)).tobytes():
            raise ValueError(f"Crop altered scene pixels: {name}")
        receipt["assets"][name] = {
            "path": str(destination.relative_to(ROOT)), "sha256": digest(destination.read_bytes()),
            "video_key": video, "video_sha256": source["sha256"], "arm": arm,
            "clip_index_zero_based": 0, "frame_index_zero_based": frame,
            "video_presentation_seconds": frame / source["fps"], "displayed_step": step,
            "source_frame": str(full.relative_to(ROOT)), "source_frame_sha256": digest(full.read_bytes()),
            "crop_xywh": box, "native_pixels_preserved": True, "panel_held_at_frame": False,
            "eventual_success": success, "eventual_outcome": "success" if success else "timeout",
            "outcome_reported_at_displayed_step": last_step,
            "outcome_already_reported_at_shown_frame": step == last_step,
            "outcome_source": "Original burned-in annotations, read at full resolution; off terminal images retained separately.",
            "recommended_label": ("Online: success" if video == "panda" else "Held: success") if success
                                 else ("Off: timeout" if step == last_step else "Off: eventual timeout"),
        }
    receipt["script"] = {"path": str(pathlib.Path(__file__).resolve().relative_to(ROOT)),
                         "sha256": digest(pathlib.Path(__file__).read_bytes())}
    receipt["ffmpeg_version"] = subprocess.check_output(["ffmpeg", "-version"]).decode().splitlines()[0]
    (OUT / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (OUT / "README.md").write_text("""# Native overview scenes

Figure 1 uses `panda_off_matched.png`, `panda_on_success.png`, and
`aloha_on_success.png`. Scene pixels are exact rectangular crops of decoded
original frames. Only the known annotation bar is removed; within-scene
black backgrounds remain intact. Add titles outside the scene pixels.

| Scene | Exact frame | Video time | Displayed step | Observed outcome |
| --- | ---: | ---: | ---: | --- |
| Panda, off | 152 | 7.60 s | 162 | Eventual timeout at step 229 |
| Panda, online correction | 152 | 7.60 s | 162 | Success reported on following transition |
| ALOHA, held correction | 272 | 10.88 s | 272 | Success reported on the shown transition |

The two Panda panels are the first spatial-video clip. A +0.05 command
offset affects all six motion-action coordinates; the original continuously
updated method corrects rotation only. Images precede `env.step`, so the
success annotation belongs to the transition after the shown image. Label
the off image **eventual timeout**, not failure already declared at step 162.
Annotated frame 219 (video 10.95 s, step 229) is retained as timeout evidence.

The ALOHA image is the first transfer-cube-video clip, right panel. The
command fault is +0.02 rad on left-arm joint targets 0–5. Its correction is
identified in one episode, then **held** during the task. The ALOHA scene
follows `env.step`. The generic original header says “online,” but the
detailed overlay, constant estimates and origin commit establish held
correction; do not reproduce that generic header as the method label.

**Selection and limits.** These are selected historical command-offset
illustrations, not new benchmark samples. They do not depict torque faults,
joint 5 harm, composite adaptation or weighted allocation. Policy sampling
was not pinned. Exact CLI, init identifiers, full-state hashes and per-video
episode JSON are absent. No outcome count should be inferred from the
images. All four corrected ALOHA clips succeed; an initial small
contact-sheet misreading was corrected against full-resolution frames before
these final assets were made. A separate quantitative joint-5 panel, if
included in Figure 1, must have its own data provenance and is not depicted
by any of these scenes.

The old `adaptive_vs_frozen_offset010.mp4` does contain correction-on failure
annotations, but its exact source/CLI is unavailable. Its geometry differs
from the renderer in its first commit, which used an older camera/residual
protocol and corrected all six dimensions. It is excluded rather than being
attributed to the current evaluated method. GR1 scenes were also inspected
but are not included: that recording redraws scenes at every reset and does
not provide an identical-scene pair.

`provenance.json` records every crop/source-frame/video SHA256, displayed
step, presentation timestamp, method, eventual outcome and interpretation
limit. The extractor verifies that the videos and renderer files match their
original Git commits, checks actual presentation timestamps, and asserts
pixel identity for every crop. Rebuild with
`python paper/extract_overview_assets.py`; no simulator is invoked.
""")
    print(json.dumps({"assets": list(receipt["assets"]), "provenance": str(OUT / "provenance.json")}, indent=2))


if __name__ == "__main__":
    main()
