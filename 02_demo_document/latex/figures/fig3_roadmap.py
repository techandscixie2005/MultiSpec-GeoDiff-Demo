"""Figure 3: Development Roadmap.
Three-stage compact roadmap: Completed -> Planned -> Future.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

FIG_W = 9.5
FIG_H = 3.0

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 9.5)
ax.set_ylim(0, 3.0)
ax.axis("off")

C_DARK = "#1a1a2e"
C_BLUE = "#2c5f8a"
C_GREEN = "#2d7a4f"
C_AMBER = "#c8880a"
C_GREY = "#666666"
C_LIGHT_GREY = "#aaaaaa"
C_BG_CARD = "#f5f8fc"

# ── Three stage cards ──
card_data = [
    {
        "x": 0.3, "w": 2.8, "title": "Stage-I  Completed",
        "color": C_GREEN, "bg": "#e8f5ec",
        "items": [
            "IR retrieval / reranking",
            "Coord-aware encoding",
            "Distance-bias attention",
            "Forward-spectrum check",
            "200 NIST spectra validated",
        ]
    },
    {
        "x": 3.4, "w": 2.8, "title": "Stage-II  Planned",
        "color": C_AMBER, "bg": "#fef8ec",
        "items": [
            "Graph diffusion generator",
            "Size-adaptive generation",
            "Pairwise distance matrix",
            "Classical MDS 2D->3D",
        ]
    },
    {
        "x": 6.5, "w": 2.8, "title": "Stage-III  Future",
        "color": C_GREY, "bg": "#f2f2f2",
        "items": [
            "TFN / SE(3) refinement",
            "Parity channels (0e,0o,1e,1o)",
            "IR / Raman / NMR / MS",
            "Top-K with phys. consistency",
        ]
    },
]

for c in card_data:
    # Card background
    card = FancyBboxPatch(
        (c["x"], 0.15), c["w"], 2.55,
        boxstyle="round,pad=0.15",
        facecolor=c["bg"], edgecolor=c["color"], linewidth=0.9, alpha=0.7, zorder=1
    )
    ax.add_patch(card)
    # Card header
    header = FancyBboxPatch(
        (c["x"] + 0.08, 2.22), c["w"] - 0.16, 0.42,
        boxstyle="round,pad=0.06",
        facecolor=c["color"], edgecolor="none", alpha=0.9, zorder=3
    )
    ax.add_patch(header)
    ax.text(c["x"] + c["w"]/2, 2.43, c["title"], ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="white", zorder=4)
    # Items
    for j, item in enumerate(c["items"]):
        y = 2.0 - j * 0.35
        ax.text(c["x"] + 0.25, y, "▸ " + item, fontsize=6.5, color=C_DARK,
                va="center", zorder=4)

# Arrows between cards
for x_src, x_dst in [(3.1, 3.4), (6.2, 6.5)]:
    ax.annotate("", xy=(x_dst, 1.45), xytext=(x_src, 1.45),
                arrowprops=dict(arrowstyle="->", color=C_GREY, lw=1.2), zorder=2)

# ── Title ──
ax.text(4.75, 2.9, "Development Roadmap", ha="center", va="center",
        fontsize=10, fontweight="bold", color=C_DARK)

fig.tight_layout(pad=0.3)
out = "02_demo_document/latex/figures/fig3_roadmap.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved {out}")
