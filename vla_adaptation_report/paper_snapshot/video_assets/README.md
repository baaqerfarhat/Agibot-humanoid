# Existing-video illustration

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
