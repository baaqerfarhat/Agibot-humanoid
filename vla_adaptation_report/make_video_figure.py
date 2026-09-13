#!/usr/bin/env python3
"""Extract a source-bound, selected illustration from an existing paired video.

No simulator, policy, image generation, retouching, or outcome recomputation is
used. ffmpeg decodes exact frame indices; the only crop removes the two known
annotation bars and separates the two original panels. Original annotated
frames are retained beside the publication figure for provenance review.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
ASSETS = PAPER / "video_assets"
VIDEO = ROOT / "results/phase05/adaptive_vs_frozen.mp4"
VIDEO_SHA256 = "86b7533a406302ee98d64412990fffdbeedaf334755317eb91651acd20c6bdf9"
RENDER_COMMIT = "a04c285fcf8a8a878890ad21bb698586f0089698"
RENDERER_SHA256 = "5e2747c8ee671ba2ea446dda9ef715189e20e0eaf252c72a828659c79508d545"
SELECTED_FRAME = 152
EVIDENCE_FRAMES = (0, 151, SELECTED_FRAME, 219)
CROPS = {"off": (0, 75, 384, 389), "on": (384, 75, 384, 389)}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str]) -> bytes:
    return subprocess.check_output(argv)


def main() -> None:
    if sha256(VIDEO) != VIDEO_SHA256:
        raise ValueError("The video changed; inspect and re-select before regenerating.")
    renderer = ROOT / "openpi/compare_video.py"
    if sha256(renderer) != RENDERER_SHA256:
        raise ValueError("The renderer source changed; review timing/crop provenance.")
    # Verify the selected video is exactly the documented corrected re-render.
    historical_video = run(["git", "-C", str(ROOT), "show",
                            f"{RENDER_COMMIT}:results/phase05/adaptive_vs_frozen.mp4"])
    if hashlib.sha256(historical_video).hexdigest() != VIDEO_SHA256:
        raise ValueError("Selected video does not match the documented re-render commit.")
    probe = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                           "-show_streams", "-show_frames", "-show_entries",
                           "stream=width,height,r_frame_rate,nb_frames,duration:"
                           "frame=best_effort_timestamp_time", "-of", "json", str(VIDEO)]))
    stream = probe["streams"][0]
    if (stream["width"], stream["height"], stream["r_frame_rate"],
            int(stream["nb_frames"])) != (768, 464, "20/1", 960):
        raise ValueError("Unexpected encoded geometry or frame numbering.")
    timestamps = [float(f["best_effort_timestamp_time"]) for f in probe["frames"]]
    if any(abs(timestamps[n] - n / 20) > 1e-9 for n in EVIDENCE_FRAMES):
        raise ValueError("Unexpected presentation timestamps.")

    ASSETS.mkdir(parents=True, exist_ok=True)
    commands = []
    generated = []
    for n in EVIDENCE_FRAMES:
        destination = ASSETS / f"spatial_clip1_frame_{n:04d}_annotated.png"
        argv = ["ffmpeg", "-v", "error", "-i", str(VIDEO), "-vf",
                f"select=eq(n\\,{n}),format=rgb24", "-frames:v", "1", "-y", str(destination)]
        subprocess.run(argv, check=True)
        commands.append(argv)
        generated.append(destination)

    scenes = {}
    annotated = Image.open(ASSETS / f"spatial_clip1_frame_{SELECTED_FRAME:04d}_annotated.png")
    for arm, (x, y, width, height) in CROPS.items():
        destination = ASSETS / f"spatial_clip1_frame_{SELECTED_FRAME:04d}_{arm}.png"
        argv = ["ffmpeg", "-v", "error", "-i", str(VIDEO), "-vf",
                f"select=eq(n\\,{SELECTED_FRAME}),format=rgb24,"
                f"crop={width}:{height}:{x}:{y}:exact=1", "-frames:v", "1", "-y", str(destination)]
        subprocess.run(argv, check=True)
        commands.append(argv)
        scenes[arm] = Image.open(destination).convert("RGB")
        # Exact pixel check: the published scene is only a rectangular source crop.
        if scenes[arm].tobytes() != annotated.crop((x, y, x + width, y + height)).tobytes():
            raise ValueError("Scene pixels differ from their exact source-frame crop.")
        generated.append(destination)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(6.6, 3.58), facecolor="white")
    fig.text(.5, .975, "Pick up the black bowl next to the ramekin and place it on the plate",
             ha="center", va="top", fontsize=10, weight="bold")
    fig.text(.5, .919, r"$\pi_{0.5}$ / Panda / LIBERO-Spatial  |  same recorded step 162 (video 7.60 s)",
             ha="center", va="top", fontsize=9, color="#404040")
    labels = {"off": ("Correction off", "Eventual timeout at step 229", "#9d292b"),
              "on": ("Original online correction", "Success reported at step 162", "#237448")}
    for arm, left in (("off", .01), ("on", .51)):
        heading, outcome, color = labels[arm]
        fig.text(left + .24, .852, heading, ha="center", va="top",
                 weight="bold", fontsize=10, color=color)
        ax = fig.add_axes([left, .122, .48, .698])
        ax.imshow(scenes[arm], interpolation="none")
        ax.set_axis_off()
        fig.text(left + .24, .105, outcome, ha="center", va="top",
                 fontsize=9, color=color)
    fig.text(.5, .035, "Command offset +0.05 on all six action channels; rotation-only correction",
             ha="center", va="bottom", fontsize=8.5, color="#404040")
    pdf = PAPER / "fig_video_evidence.pdf"
    preview = ASSETS / "fig_video_evidence.png"
    fig.savefig(pdf, dpi=220, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(preview, dpi=180)
    plt.close(fig)
    generated.extend((pdf, preview))

    sources = [VIDEO, renderer, ROOT / "openpi/gate_faults.py",
               ROOT / "docs/ADAPTIVE_CONTROL_VLA.md", pathlib.Path(__file__).resolve()]
    record = {
        "schema_version": 1,
        "artifact_kind": "selected_existing_video_illustration_not_benchmark",
        "source_files": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "video_origin_commit": RENDER_COMMIT,
        "video_origin_verified_byte_identical": True,
        "video_stream": stream,
        "episode": {
            "clip_index_zero_based": 0, "clip_start_frame": 0,
            "task_text": "pick up the black bowl next to the ramekin and place it on the plate",
            "suite": "libero_spatial", "policy": "pi0.5", "robot": "Panda",
            "fault": {"kind": "command_offset", "value": .05,
                      "channels": [0, 1, 2, 3, 4, 5], "units": "environment action coordinates"},
            "correction": {"family": "original_online_calibrated_adaptation",
                           "channels": [3, 4, 5], "continuous_updates": True},
            "task_id": None, "initial_state_id": None,
            "full_render_cli": None, "per_video_episode_json_available": False,
            "reported_outcomes": {"off": {"success": False, "displayed_last_step": 229},
                                  "on": {"success": True, "displayed_last_step": 162}},
            "outcome_evidence": "Burned-in original annotations at frames 152 and 219; renderer attaches env.step done to raw pre-transition frame.",
        },
        "figure_frame": {
            "zero_based_index": SELECTED_FRAME, "presentation_time_seconds": timestamps[SELECTED_FRAME],
            "displayed_step_off": 162, "displayed_step_on": 162,
            "either_panel_held_at_selected_frame": False,
            "on_first_success_annotation": True,
            "image_timing": "Both raw scene images precede the env.step transition numbered 162; success is reported from that following transition.",
        },
        "evidence_frames": [{"index_zero_based": n, "presentation_time_seconds": timestamps[n],
                             "path": f"paper/video_assets/spatial_clip1_frame_{n:04d}_annotated.png"}
                            for n in EVIDENCE_FRAMES],
        "image_operations": {
            "decode": "ffmpeg exact zero-based frame select, RGB24, lossless PNG",
            "crop_xywh": CROPS,
            "crop_reason": "Separate original 384-pixel panels and remove only encoded 75-pixel annotation bar; retain full 389-pixel-high scene.",
            "encoded_geometry_note": "compare_video creates 768x458 with a 74-pixel bar; original imageio encoding resized height to 464. Header boundary is at approximately 74*464/458=74.969 pixels.",
            "scene_pixel_identity_checked": True,
            "retouching_or_generated_scene_content": False,
            "figure_annotations": "New text outside scene; no arrows, object relocation, selective scene cropping or recoloring.",
        },
        "selection_and_interpretation": {
            "original_selection": "docs/ADAPTIVE_CONTROL_VLA.md section 28.3: 4 selected repaired clips from 6 spatial candidates; 2 off-success skips; 0 corrected-failure skips.",
            "figure_selection": "First spatial clip selected for recognizable bowl-to-plate goal; frame 152 is first corrected-success annotation and last matched live step.",
            "pairing": "Renderer reuses task and stored initial state; no archived full-state hash or task/init sidecar is available for this video.",
            "policy_randomness": "Renderer explicitly requests pin_rng=False; two outcomes are not a deterministic counterfactual with identical policy samples.",
            "statistics": "No success rate or estimator ranking may be inferred from selected clips. Stored benchmark outcome JSON is not treated as the outcome record for this re-render.",
            "not_shown": ["composite adaptation", "weighted allocator", "physical torque fault"],
            "camera": "Current video byte-matches corrected re-render; documented native policy camera and separately widened recording image.",
            "historical_limits": "Exact render CLI, selected initial-state ID, and per-rollout source/full-state receipts were not saved; no modern provenance guarantee is inferred.",
        },
        "commands": commands,
        "ffmpeg_version": run(["ffmpeg", "-version"]).decode().splitlines()[0],
        "output_files": {str(p.relative_to(ROOT)): sha256(p) for p in generated},
    }
    (ASSETS / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    (ASSETS / "README.md").write_text("""# Existing-video illustration

`../fig_video_evidence.pdf` uses the **first clip** in
`results/phase05/adaptive_vs_frozen.mp4`: π0.5 / Panda / LIBERO-Spatial,
“pick up the black bowl next to the ramekin and place it on the plate.”
Both arms receive a +0.05 **command offset on all six action channels**;
the original continuously updated calibrated method corrects rotation only.
This is neither composite adaptation nor the weighted allocator, and it is
not a physical joint-torque experiment.

The figure uses exact zero-based **frame 152, presentation time 7.60 s**, with
both original panels displaying **step 162**. Neither panel is held at this
frame. The corrected arm first reports success here; the uncorrected arm
eventually reports timeout at step 229 (archived frame 219, 10.95 s). The renderer
records each raw image **before** `env.step` and adds that transition's
outcome afterward: “success reported at step 162” describes this timing.
Frames 0, 151, 152, 219 are retained with original embedded annotations.

Only the known annotation bar is cropped away, and the original left/right
panels are separated. There is no selective crop of the scene, retouching,
object relocation, or generated image content. The program verifies exact
pixel equality of each scene crop against its archived decoded full frame.
`provenance.json` records source/output SHA256s, exact extraction commands,
timestamps, crop rectangles, tool version, settings and interpretation limits.

The MP4 is byte-identical to commit
`a04c285fcf8a8a878890ad21bb698586f0089698` (2026-09-05), which replaced the
older camera-confounded renders. `docs/ADAPTIVE_CONTROL_VLA.md` §28.3 records
the corrected native policy camera, SO(3) residual, and outcome selection:
four spatial repair clips retained from six candidates, with two skipped
because the uncorrected arm succeeded. The current `compare_video.py` is
also byte-identical to its source at that commit. Its source and the embedded
annotations establish the method, action-fault convention, and timing.

**Limits.** This is an outcome-selected illustration, not a new benchmark,
rate estimate, or deterministic counterfactual. Policy randomness was not
pinned. Pairing on the task and stored initial state follows the recording
code; the historical clip has no saved full-state hashes, exact CLI, episode
JSON or initial-state identifier. Existing benchmark JSON is therefore not
substituted for this re-render's outcome record. Unknown settings remain
unspecified rather than being inferred from CLI defaults.

Rebuild from the repository root with `python paper/make_video_figure.py`.
It requires ffmpeg/ffprobe, Pillow, Matplotlib, and the original Git object;
it refuses changed video or renderer bytes. It does not invoke a simulator.
""")
    print(json.dumps({"figure": str(pdf), "preview": str(preview),
                      "provenance": str(ASSETS / "provenance.json")}, indent=2))


if __name__ == "__main__":
    main()
