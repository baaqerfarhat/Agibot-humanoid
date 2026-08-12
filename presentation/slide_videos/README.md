# Slide videos

The six clips that belong in the video slots on slides 7 and 11 of
`SURF_Final_Baaqer_Farhat.pptx`, named by the slot they go in.

Each slot in the deck already shows a still frame from its own clip, with the
source filename printed under it in grey. Match that filename to the file here.

## Slide 7, Crawling

| Slot | Caption on the slide | File |
|------|----------------------|------|
| VIDEO 1 | Early · v2, iter 14500 | `crawling/video1_x2_crawl_slope_v2_palmflat_iter14500.mp4` |
| VIDEO 2 | Intermediate · v3, iter 49999 | `crawling/video2_x2_crawl_slope_v3_tracking_iter49999.mp4` |
| VIDEO 3 | Current · v5, iter 86000 | `crawling/video3_x2_crawl_slope_v5_track_xyz_iter86000.mp4` |

## Slide 11, Same policy, same fault, one difference

| Slot | Caption on the slide | File |
|------|----------------------|------|
| VIDEO 1 | Nominal, frozen | `adaptation/video1_x2_box_v31_flatfoot_iter202500.mp4` |
| VIDEO 2 | Knee fault, frozen | `adaptation/video2_fault_knee03_frozen.mp4` |
| VIDEO 3 | Knee fault, waist adaptation | `adaptation/video3_fault_knee03_waistadapt.mp4` |

Videos 2 and 3 are the left and right halves of
`box_pickup/videos/isaac_fault_knee03_frozen_vs_waistadapt.mp4`, which renders
both runs side by side in one frame. They are cropped apart so each slot plays
one run, and the caption band baked into the source frame is cropped off.

## Adding one by hand in PowerPoint

1. Click the still frame in the slot and delete it. Leave the coloured header
   bar, the title and the grey caption underneath.
2. Insert, then Video, then This Device, and pick the file.
3. Drag it into the empty space. Hold Shift while resizing so the aspect ratio
   holds.
4. With the video selected, open Playback and set Start to `Automatically`, and
   tick `Loop until Stopped` so it runs while you talk over it.

Do this on the copy you present from, not on a rebuild of the deck, because
`build_surf_slides.py` regenerates the slide from scratch and would discard it.

## Doing all nine at once instead

`insert_surf_videos.py` rebuilds the whole deck with every clip already
embedded and set to play, which is faster and less error prone than nine manual
inserts:

    python3 presentation/insert_surf_videos.py

That writes `SURF_Final_Baaqer_Farhat_with_videos.pptx`, about 95 MB. It is
gitignored, so it will not appear in a fresh clone until you run the script.
Present from that file and keep `SURF_Final_Baaqer_Farhat.pptx` as the one you
edit.

The three box pickup clips on slide 4 are not copied here. The script reads
them, and every other source clip, straight out of `box_pickup/videos/`.
