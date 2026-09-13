# Native overview scenes

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
images. All four corrected ALOHA clips succeed, as verified against their
full-resolution annotations. The separate quantitative joint-5 panel has its
own data provenance and is not depicted by any of these scenes.

The old `adaptive_vs_frozen_offset010.mp4` does contain correction-on failure
annotations, but its exact source/CLI is unavailable. Its geometry differs
from the renderer in its first commit, which used an older camera/residual
protocol and corrected all six dimensions. It is excluded rather than being
attributed to the current evaluated method. The original GR1 recording was also
inspected and excluded because its panels showed different initial scenes.
Main commits `c367fb2` and `96555f3` subsequently replace that video with a render
using a reseeded scene generator and a smaller encoding. The replacement is
available in `results/phase05/adaptive_vs_frozen_gr1.mp4` but is not used in the
current overview. Neither version changes the historical unpaired GR1 benchmark
cohorts or supplies new aggregate task-success results.

`provenance.json` records every crop/source-frame/video SHA256, displayed
step, presentation timestamp, method, eventual outcome and interpretation
limit. The extractor verifies that the videos and renderer files match their
original Git commits, checks actual presentation timestamps, and asserts
pixel identity for every crop. Rebuild with
`python paper/extract_overview_assets.py`; no simulator is invoked.
