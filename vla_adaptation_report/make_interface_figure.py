"""Render Figure 1: symbolic repair loop plus source-verified historical evidence.

All icons, lines and plots are vector primitives. The three robot images are
unaltered, provenance-checked native frame crops; see extract_overview_assets.py.
The joint-5 bars are recomputed from episode records, not illustrative numbers.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"
PROVENANCE = OUT / "overview_assets/provenance.json"
CELL = ROOT / "results/joint_map/cell_torque_5.json"
ASSETS = ("panda_off_matched", "panda_on_success", "aloha_on_success")
INK, GRAY, PALE = "#23384A", "#667580", "#F2F5F7"
TEAL, TEAL_BG = "#237E7B", "#EDF7F4"
PURPLE, PURPLE_BG = "#77588D", "#F5F0F8"
RED, RED_BG = "#AE4E3C", "#FAEFEA"
LINE = "#CDD6DC"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                     "mathtext.fontset": "dejavusans", "pdf.fonttype": 42})

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

provenance = json.loads(PROVENANCE.read_text())
images, source_hashes = {}, {str(PROVENANCE.relative_to(ROOT)): sha(PROVENANCE),
                            str(CELL.relative_to(ROOT)): sha(CELL),
                            str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__)}
for name in ASSETS:
    record = provenance["assets"][name]
    path = ROOT / record["path"]
    if sha(path) != record["sha256"] or not record["native_pixels_preserved"]:
        raise ValueError("Scene/provenance mismatch: " + name)
    source_hashes[record["path"]] = record["sha256"]
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError("Expected native RGB scene pixels: " + name)
        images[name] = np.asarray(image).copy()
if (provenance["assets"]["panda_off_matched"]["frame_index_zero_based"] != 152
        or provenance["assets"]["panda_on_success"]["frame_index_zero_based"] != 152
        or provenance["assets"]["panda_off_matched"]["eventual_success"]
        or not provenance["assets"]["panda_on_success"]["eventual_success"]
        or not provenance["assets"]["aloha_on_success"]["eventual_success"]
        or "hold" not in provenance["sources"]["aloha"]["method"]):
    raise ValueError("The selected methods/outcomes changed; review figure labels.")

cell = json.loads(CELL.read_text())
if (cell["joint_fault"] != "torque:5:5.0" or cell["args"]["corr_dims"] != "0,1,2"
        or cell["args"]["static_corr"] is not None or cell["args"]["law"] != "legacy"):
    raise ValueError("Joint-5 figure requires the historical continuous translation-only cell.")
rates, counts = [], []
for name in ("frozen_faulted", "adaptive"):
    arm = cell["arms"][name]
    episodes = arm["per_ep"]
    if len({(e["task"], e["init"]) for e in episodes}) != len(episodes):
        raise ValueError("Duplicate episode key in joint-5 source")
    count, n = sum(e["ok"] for e in episodes), len(episodes)
    if count != arm["successes"] or n != arm["n"]:
        raise ValueError("Joint-5 totals disagree with episode outcomes")
    counts.append((count, n))
    rates.append(count/n)
if counts != [(12, 20), (3, 20)]:
    raise ValueError("Historical counts changed; review the manuscript/caption together.")

fig, ax = plt.subplots(figsize=(5.5, 4.10))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set(xlim=(0, 11), ylim=(0, 8.2))
ax.axis("off")


def text(x, y, value, size=8, color=INK, ha="center", weight="normal", va="center"):
    return ax.text(x, y, value, fontsize=size, color=color, ha=ha, va=va,
                   fontweight=weight, linespacing=1.10, zorder=8)


def box(x, y, w, h, fill="white", edge=LINE, radius=.10, lw=.9, zorder=2):
    item = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=fill, edgecolor=edge, linewidth=lw, zorder=zorder)
    ax.add_patch(item)
    return item


def arrow(points, color=INK, lw=1.0, head=True, zorder=4):
    path = MplPath(points, [MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>" if head else "-",
                                mutation_scale=8, linewidth=lw, color=color,
                                capstyle="round", joinstyle="round", zorder=zorder))


def circle(x, y, value, color=TEAL, radius=.16, size=10, fill="white"):
    ax.add_patch(Circle((x, y), radius, facecolor=fill, edgecolor=color, lw=1, zorder=6))
    text(x, y, value, size=size, color=color)


def title(letter, caption, y):
    box(.15, y-.12, .27, .25, fill=INK, edge=INK, radius=.035)
    text(.285, y, letter, size=7, color="white", weight="bold")
    text(.57, y, caption, size=8.2, weight="bold", ha="left")


# A. The main method uses symbols, with labels only where an icon is ambiguous.
title("A", "Frozen policy, adaptive interface", 7.97)
box(.10, 6.35, 10.78, 1.06, fill=PALE, edge=PALE, lw=0, zorder=0)
box(.10, 4.65, 10.78, .94, fill=TEAL_BG, edge=TEAL_BG, lw=0, zorder=0)

# Images and language token glyphs, feeding a locked policy chip.
box(.38, 7.55, .39, .26, edge=GRAY, radius=.025, lw=.7)
ax.plot([.42, .51, .59, .66, .73], [7.58, 7.70, 7.62, 7.68, 7.58],
        color=GRAY, lw=.6)
ax.add_patch(Circle((.67, 7.74), .025, color=GRAY))
for x, w in ((.94, .26), (1.25, .18), (.94, .49)):
    y = 7.74 if x != .94 or w == .26 else 7.61
    box(x, y, w, .055, fill=GRAY, edge=GRAY, radius=.015, lw=0)
arrow([(.93, 7.49), (.93, 7.25)], lw=.8)
box(.32, 6.44, 1.22, .81, fill=INK, edge=INK)
text(.86, 6.87, r"$\pi_\theta$", size=21, color="white")
# Native lock icon, not a learned or changing policy weight.
ax.add_patch(Arc((1.33, 7.13), .15, .17, theta1=0, theta2=180, color="white", lw=.9, zorder=7))
box(1.25, 6.98, .16, .14, fill="white", edge="white", radius=.02, lw=.5)
text(.93, 6.24, "frozen", size=6.7, color=GRAY)
arrow([(1.54, 6.84), (2.42, 6.84)])
text(1.97, 7.08, r"$a_t$", size=10)

# Negative bounded correction, then a known-command tap BEFORE additive fault.
circle(2.60, 6.84, r"$\Sigma$", color=INK, radius=.17)
text(2.34, 7.02, "+", size=9)
text(2.82, 6.59, "−", size=10, color=TEAL)
arrow([(2.79, 6.84), (4.32, 6.84)], head=False)
text(3.30, 7.08, r"$u_t$", size=10)
ax.add_patch(Circle((3.82, 6.84), .035, color=TEAL, zorder=7))
circle(4.50, 6.84, "+", color=RED, radius=.135, size=11, fill=RED_BG)
text(4.50, 7.56, r"$f_t$", size=12, color=RED)
arrow([(4.50, 7.36), (4.50, 7.00)], color=RED)
arrow([(4.65, 6.84), (5.18, 6.84)])

# Low-level servo glyph (no assumed PID/transfer function).
cx, cy = 5.50, 6.84
ax.add_patch(Circle((cx, cy), .215, facecolor="white", edgecolor=INK, lw=1))
for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
    direction = np.array([np.cos(angle), np.sin(angle)])
    p0, p1 = np.array([cx, cy])+.22*direction, np.array([cx, cy])+.29*direction
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=INK, lw=1.6)
ax.add_patch(Circle((cx, cy), .07, facecolor=PALE, edgecolor=INK, lw=.9))
text(5.50, 6.25, "servo", size=6.7, color=GRAY)
arrow([(5.81, 6.84), (6.49, 6.84)])

# Symbolic robot linkage. Joint disturbance enters after the low-level servo.
joints = np.array([[7.30, 6.46], [6.95, 6.95], [7.60, 7.17], [7.93, 6.85]])
ax.plot(joints[:, 0], joints[:, 1], color=INK, lw=3.7, solid_capstyle="round", zorder=4)
for x, y in joints[:3]:
    ax.add_patch(Circle((x, y), .085, facecolor="white", edgecolor=INK, lw=1.2, zorder=5))
box(6.99, 6.37, .62, .10, fill=INK, edge=INK, radius=.02)
ax.plot([7.93, 8.10, 8.21], [6.85, 6.91, 6.80], color=INK, lw=1.5)
ax.plot([7.93, 8.00, 8.13], [6.85, 6.68, 6.67], color=INK, lw=1.5)
text(7.00, 7.58, r"$\tau_f$", size=12, color=RED)
arrow([(7.00, 7.37), (6.97, 7.07)], color=RED)
text(7.46, 6.24, "robot", size=6.7, color=GRAY)
arrow([(8.32, 6.84), (9.10, 6.84)])
text(9.88, 6.84, r"$\mathbf{y}_{t+1}$", size=17)
text(9.88, 7.39, "measured motion", size=6.7, color=GRAY)

# The predictor sees only known pre-fault commands. Seven history slots are a
# memory glyph, not an invented measured coefficient plot.
arrow([(3.82, 6.84), (3.82, 5.16), (4.79, 5.16)], color=TEAL)
text(4.01, 6.00, "pre-fault", size=6.7, color=TEAL, ha="left")
box(4.80, 4.78, 1.87, .77, edge=TEAL)
for i in range(7):
    box(4.93+i*.23, 5.42, .17, .22,
        fill=matplotlib.colors.to_rgba(TEAL, 1-i*.10), edge="white", radius=.025, lw=.6)
text(5.73, 5.10, r"$\mathcal{P}_h[u]$", size=17, color=TEAL)
text(5.73, 4.61, "FIR history", size=6.7, color=TEAL)
arrow([(6.67, 5.16), (8.62, 5.16)], color=TEAL)
text(7.51, 5.43, r"$\hat y_{t+1}$", size=10, color=TEAL)
circle(8.80, 5.16, r"$\Sigma$", color=TEAL)
text(8.56, 5.40, "−", size=10, color=TEAL)
text(9.04, 5.40, "+", size=9, color=TEAL)
arrow([(9.88, 6.55), (9.88, 5.16), (8.98, 5.16)], color=TEAL)
arrow([(8.80, 4.99), (8.80, 4.38), (1.05, 4.38), (1.05, 4.78)], color=TEAL)
text(7.40, 4.59, r"$r_{t+1}$", size=10, color=TEAL)

# An estimator plus a bounded correction rule; the latter may be masking with
# projection or the weighted allocator. The purple addition is optional.
box(.43, 4.78, 1.24, .77, edge=TEAL)
text(1.05, 5.18, r"$\mathcal{E}_M$", size=17, color=TEAL)
text(1.05, 5.76, "observer", size=6.7, color=TEAL)
arrow([(1.67, 5.16), (1.91, 5.16)], color=TEAL)
circle(2.05, 5.16, "+", color=PURPLE, radius=.115, size=10, fill=PURPLE_BG)
arrow([(2.19, 5.16), (2.45, 5.16)], color=TEAL)
box(2.45, 4.78, 1.08, .77, edge=TEAL)
text(2.99, 5.18, r"$\mathcal{A}_{\mathcal{U}}$", size=16, color=TEAL)
text(2.99, 4.61, "bound / allocate", size=6.7, color=TEAL)
arrow([(2.99, 5.55), (2.99, 6.09), (2.60, 6.09), (2.60, 6.65)], color=TEAL)
text(3.18, 6.28, r"$c_{t+1}$", size=10, color=TEAL)
text(2.03, 5.94, "next action", size=6.7, color=TEAL)
text(2.36, 6.45, r"$c_t$", size=9, color=TEAL)

# Optional composite branch; the matching purple plus supplies the added term
# at the estimator output, before the bounded correction. Definitions are in
# the paper caption, keeping the algorithm graphic primarily symbolic.
box(.12, 3.55, 10.74, .59, fill=PURPLE_BG, edge=PURPLE_BG, lw=0)
text(.17, 4.19, "optional composite", size=6.7, color=PURPLE, ha="left")
text(.66, 3.85, r"$a_t$", size=11, color=PURPLE)
arrow([(1.01, 3.85), (1.61, 3.85)], color=PURPLE)
text(2.20, 3.85, r"$\mathcal{R}_{A,B}$", size=14, color=PURPLE)
arrow([(2.91, 3.85), (3.47, 3.85)], color=PURPLE)
text(4.91, 3.85, r"$e=q[J]-q^r$", size=11, color=PURPLE)
arrow([(6.29, 3.85), (6.71, 3.85)], color=PURPLE)
text(8.20, 3.85, r"$\Delta t\,\eta P^+D^\top Le$", size=12, color=PURPLE)
arrow([(9.74, 3.85), (10.28, 3.85)], color=PURPLE)
circle(10.44, 3.85, "+", color=PURPLE, radius=.115, size=10)
text(5.50, 3.34, r"$M$: calibrated sensitivity     $\cdot$     $D$: correction response",
     size=6.7, color=GRAY)

# B. Exact historical scene pixels and a separately sourced failure-condition
# chart. The photographs do not depict the joint-5 torque experiment.
ax.plot([.15, 10.85], [3.14, 3.14], color=LINE, lw=.6)
title("B", "Recorded examples and a measured limitation", 2.96)
text(2.75, 2.64, "Panda  /  command offset", size=7.0, weight="bold")
text(6.90, 2.64, "ALOHA", size=7.0, weight="bold")
text(9.62, 2.64, "Joint 5 torque", size=7.0, weight="bold")

for name, x, caption, color in (
        ("panda_off_matched", .15, "Off: later timeout", GRAY),
        ("panda_on_success", 2.85, "Online: success", TEAL),
        ("aloha_on_success", 5.65, "Held: success", TEAL)):
    # Native aspect ratio is retained: no further scene crop or annotations.
    pane = (x, .49, 2.50, 1.98)
    box(*pane, fill=PALE, edge=LINE, radius=.06, lw=.7)
    im = images[name]
    scale = min((pane[2]-.04)/im.shape[1], (pane[3]-.04)/im.shape[0])
    w, h = im.shape[1]*scale, im.shape[0]*scale
    left, bottom = x+(pane[2]-w)/2, pane[1]+(pane[3]-h)/2
    ax.imshow(im, extent=(left, left+w, bottom, bottom+h),
              interpolation="none", aspect="auto", zorder=3)
    text(x+1.25, .26, caption, size=6.7, color=color)

# Two bars from validated per-episode counts, with a percent axis and exact n.
chart = fig.add_axes([8.98/11, .80/8.2, 1.64/11, 1.52/8.2])
chart.bar([0, 1], np.asarray(rates)*100, width=.56, color=[GRAY, RED], zorder=3)
chart.set(ylim=(0, 100), xlim=(-.6, 1.6), yticks=[0, 50, 100],
          yticklabels=["0", "50", "100%"], xticks=[0, 1], xticklabels=["Off", "Legacy"])
chart.tick_params(axis="both", labelsize=6.5, length=0, pad=2)
chart.set_axisbelow(True)
chart.grid(axis="y", color=LINE, lw=.5)
for spine in ("top", "right", "left"):
    chart.spines[spine].set_visible(False)
chart.spines["bottom"].set_color(LINE)
chart.spines["bottom"].set_linewidth(.6)
for i, ((count, n), rate) in enumerate(zip(counts, rates)):
    chart.text(i, rate*100+4, f"{count}/{n}", fontsize=7, ha="center", color=INK)
text(9.62, .26, "Legacy: 60% → 15%", size=6.7, color=RED)

# Source binding is retained inside the PDF metadata and printed by the
# reproduction script. Unchanged scenes are embedded as native raster images;
# every other figure element remains vector.
for name, expected in source_hashes.items():
    if sha(ROOT / name) != expected:
        raise RuntimeError("A figure source changed while rendering: " + name)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for axis in fig.axes:
    for item in axis.texts:
        bounds = item.get_window_extent(renderer)
        if bounds.x0 < 0 or bounds.x1 > fig.bbox.width or bounds.y0 < 0 or bounds.y1 > fig.bbox.height:
            raise RuntimeError("Figure text extends outside canvas: " + item.get_text())
metadata = dict(Title="Frozen-policy interface repair: method and recorded examples",
                Creator="make_interface_figure.py", CreationDate=None,
                Subject=json.dumps(dict(sources_sha256=source_hashes, joint5_counts=counts,
                                        scope="Selected historical scenes; joint5 bars are a separate translation-only torque experiment.")))
fig.savefig(OUT/"fig_interface.pdf", metadata=metadata)
fig.savefig(OUT/"fig_interface.png", dpi=240)
plt.close(fig)

# The PDF backend stores image rows in the opposite order to PNG and applies
# the corresponding placement transform. Check the embedded pixel content,
# not only the input file hashes, without altering the source scenes.
embedded_checks = {}
with tempfile.TemporaryDirectory(prefix="interface-native-pixels-") as directory:
    prefix = Path(directory)/"image"
    subprocess.run(["pdfimages", "-png", str(OUT/"fig_interface.pdf"), str(prefix)], check=True)
    extracted = sorted(Path(directory).glob("image-*.png"))
    if len(extracted) != len(ASSETS):
        raise ValueError("Unexpected raster objects in the method figure")
    for name, path in zip(ASSETS, extracted):
        embedded = np.asarray(Image.open(path))
        if np.array_equal(embedded, images[name]):
            order = "original row order"
        elif np.array_equal(embedded[::-1], images[name]):
            order = "reversed storage rows; PDF placement restores source orientation"
        else:
            raise ValueError("PDF image pixels differ from source scene: " + name)
        embedded_checks[name] = dict(native_shape=list(images[name].shape),
                                     exact_pixel_content=True, storage=order)

receipt = dict(
    schema_version=1,
    interpretation="Symbolic method schematic and selected historical illustrations; photographs do not depict torque faults, composite adaptation, or weighted allocation.",
    sources_sha256=source_hashes,
    selected_scenes={name: provenance["assets"][name] for name in ASSETS},
    video_methods={key: {field: provenance["sources"][key][field]
                        for field in ("robot", "policy", "fault", "method", "timing")}
                   for key in ("panda", "aloha")},
    joint5=dict(source=str(CELL.relative_to(ROOT)), joint_fault=cell["joint_fault"],
                continuous_update=cell["args"]["static_corr"] is None,
                correction_dimensions=cell["args"]["corr_dims"],
                law=cell["args"]["law"],
                counts=dict(off=dict(successes=counts[0][0], n=counts[0][1]),
                            legacy=dict(successes=counts[1][0], n=counts[1][1])),
                rates=dict(off=rates[0], legacy=rates[1]),
                scope="Separate historical translation-only torque experiment; no photo represents this condition."),
    rendering=dict(width_inches=5.5, height_inches=4.10, preview_dpi=240,
                   native_image_embedding=embedded_checks,
                   all_nonimage_marks_are_vector=True),
    outputs={str(path.relative_to(ROOT)): dict(sha256=sha(path), bytes=path.stat().st_size)
             for path in (OUT/"fig_interface.pdf", OUT/"fig_interface.png")},
    reproduce="python paper/make_interface_figure.py (Matplotlib, NumPy, Pillow, and Poppler pdfimages required)")
(OUT/"interface_figure_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps(dict(pdf=str(OUT/"fig_interface.pdf"), preview=str(OUT/"fig_interface.png"),
                      receipt=str(OUT/"interface_figure_receipt.json"),
                      source_hashes=source_hashes, joint5_counts=counts), indent=2))
