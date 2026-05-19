"""Figure 1: Stage-I Demo Workflow Diagram.
Nature-style schematic with dark blue / grey palette.
Distinguishes implemented Stage-I from future interfaces.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

FIG_W = 10
FIG_H = 3.8

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.8)
ax.axis("off")

# ── Color palette ──
C_DARK = "#1a1a2e"
C_BLUE = "#2c5f8a"
C_LIGHT_BLUE = "#4a90c4"
C_GREY = "#5a5a5a"
C_LIGHT_GREY = "#b0b0b0"
C_BG_BOX = "#e8f0f8"
C_FUTURE_BG = "#f0f0f0"
C_WHITE = "#ffffff"
C_ARROW = "#3a3a3a"

def draw_box(ax, x, y, w, h, text, color=C_BLUE, text_color="white", fontsize=7.5, bold=True):
    """Draw a rounded box with text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor="none", alpha=0.92, zorder=3
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight=weight, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5):
    """Draw an arrow between two points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                               connectionstyle="arc3,rad=0"), zorder=2)

def draw_label(ax, x, y, text, color=C_DARK, fontsize=6.5, bold=False):
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=color, fontweight=weight)

# ── Title ──
ax.text(5, 3.5, "Stage-I Demo Workflow", ha="center", va="center",
        fontsize=10, fontweight="bold", color=C_DARK)

# ── Implemented Stage-I pipeline boxes ──
boxes_main = [
    (0.65, 2.1, 1.1, 0.52, "NIST IR\nSpectrum", C_DARK),
    (1.95, 2.1, 1.1, 0.52, "Preprocess", C_BLUE),
    (3.25, 2.1, 1.1, 0.52, "Coord-Aware\nEncoder", C_BLUE),
    (4.55, 2.1, 1.1, 0.52, "Distance-Bias\nAttention", C_BLUE),
    (5.85, 2.1, 1.1, 0.52, "Candidate\nRetrieval", C_BLUE),
    (7.15, 2.1, 1.1, 0.52, "Forward\nReranking", C_BLUE),
    (8.50, 2.1, 1.1, 0.52, "Top-K\nMolecules", C_DARK),
]

for x, y, w, h, text, color in boxes_main:
    draw_box(ax, x, y, w, h, text, color=color, fontsize=6.8)

# Arrows between main boxes
arrow_y = 2.1
for i in range(len(boxes_main) - 1):
    x1 = boxes_main[i][0] + boxes_main[i][2]/2 + 0.05
    x2 = boxes_main[i+1][0] - boxes_main[i+1][2]/2 - 0.05
    draw_arrow(ax, x1, arrow_y, x2, arrow_y)

# ── Stage-I label ──
stage1_bg = FancyBboxPatch((0.05, 2.42), 9.9, 0.55,
    boxstyle="round,pad=0.08", facecolor="#e0ecf5", edgecolor=C_BLUE,
    linewidth=0.8, linestyle="--", alpha=0.5, zorder=0)
ax.add_patch(stage1_bg)
ax.text(9.0, 2.82, "Implemented Stage-I", fontsize=6.5, color=C_BLUE,
        fontweight="bold", ha="right")

# ── Future interfaces (lighter, below) ──
ax.text(5, 1.1, "Future Interfaces (Stage-II / Stage-III)", ha="center",
        fontsize=7, fontweight="bold", color=C_GREY, style="italic")

future_boxes = [
    (2.0, 0.5, 1.4, 0.4, "Graph\nDiffusion"),
    (4.2, 0.5, 1.4, 0.4, "Pairwise\nDistance"),
    (6.4, 0.5, 1.6, 0.4, "TFN / SE(3)\nRefinement"),
]

for x, y, w, h, text in future_boxes:
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.1",
        facecolor="white", edgecolor=C_LIGHT_GREY,
        linewidth=0.8, linestyle="--", alpha=0.7, zorder=3
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=6.5,
            color=C_GREY, zorder=4)

# Dashed arrows between future boxes
for i in range(len(future_boxes) - 1):
    x1 = future_boxes[i][0] + future_boxes[i][2]/2 + 0.08
    x2 = future_boxes[i+1][0] - future_boxes[i+1][2]/2 - 0.08
    ax.annotate("", xy=(x2, 0.5), xytext=(x1, 0.5),
                arrowprops=dict(arrowstyle="->", color=C_LIGHT_GREY, lw=0.8,
                               linestyle="--"), zorder=1)

# Dashed connector from Stage-I to future
ax.annotate("", xy=(2.0, 0.9), xytext=(5.0, 1.58),
            arrowprops=dict(arrowstyle="->", color=C_LIGHT_GREY, lw=0.6,
                           linestyle=":", connectionstyle="arc3,rad=0.2"), zorder=1)
ax.text(3.7, 1.25, "planned\nreplacement", fontsize=5.5, color=C_LIGHT_GREY,
        ha="center", style="italic")

# ── Legend ──
legend_y = 2.78
# Solid box
p1 = mpatches.Patch(facecolor=C_BLUE, edgecolor="none", alpha=0.9,
                     label="Implemented module")
# Dashed box
p2 = mpatches.Patch(facecolor="white", edgecolor=C_LIGHT_GREY,
                     linewidth=0.8, linestyle="--",
                     label="Future interface (stub)")
leg = ax.legend(handles=[p1, p2], loc="upper left", fontsize=6,
                frameon=True, facecolor="white", edgecolor="#ddd",
                bbox_to_anchor=(0.05, 0.97))
leg.set_zorder(5)

fig.tight_layout(pad=0.3)
out = "02_demo_document/latex/figures/fig1_workflow.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white",
            edgecolor="none")
fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved {out}")
