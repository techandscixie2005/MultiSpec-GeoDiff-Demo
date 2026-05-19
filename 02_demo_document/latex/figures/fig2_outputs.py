"""Figure 2: Demo Outputs — three-panel composite.
Combines spectrum_overlay, molecule_grid, and attention_heatmap
from existing 05_outputs/ into a single multi-panel figure.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 9,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

C_DARK = "#1a1a2e"

# Load images
img_spec = PILImage.open("05_outputs/spectrum_overlay.png")
img_mol = PILImage.open("05_outputs/molecule_grid.png")
img_attn = PILImage.open("05_outputs/attention_heatmap.png")

FIG_W = 9.0
FIG_H = 7.5
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

# ── Panel (a): Spectrum overlay ──
ax_a = fig.add_axes([0.03, 0.52, 0.94, 0.44])
ax_a.imshow(img_spec)
ax_a.axis("off")
ax_a.text(0.01, 1.01, "a", transform=ax_a.transAxes, fontsize=11,
          fontweight="bold", color=C_DARK, va="bottom")
ax_a.text(0.5, 1.01, "Query IR Spectrum vs Top-K Candidates",
          transform=ax_a.transAxes, fontsize=9, fontweight="bold",
          color=C_DARK, ha="center", va="bottom")

# ── Panel (b): Molecule grid ──
ax_b = fig.add_axes([0.03, 0.28, 0.94, 0.21])
ax_b.imshow(img_mol)
ax_b.axis("off")
ax_b.text(0.01, 1.05, "b", transform=ax_b.transAxes, fontsize=11,
          fontweight="bold", color=C_DARK, va="bottom")
ax_b.text(0.5, 1.05, "Top-K Candidate Molecular Structures",
          transform=ax_b.transAxes, fontsize=9, fontweight="bold",
          color=C_DARK, ha="center", va="bottom")

# ── Panel (c): Attention heatmap ──
ax_c = fig.add_axes([0.03, 0.02, 0.44, 0.23])
ax_c.imshow(img_attn)
ax_c.axis("off")
ax_c.text(0.02, 1.06, "c", transform=ax_c.transAxes, fontsize=11,
          fontweight="bold", color=C_DARK, va="bottom")
ax_c.text(0.5, 1.06, "Spectral Distance-Bias Attention",
          transform=ax_c.transAxes, fontsize=9, fontweight="bold",
          color=C_DARK, ha="center", va="bottom")

# ── Caption text panel ──
ax_text = fig.add_axes([0.5, 0.04, 0.47, 0.20])
ax_text.axis("off")
caption = (
    "Panel (a): Query spectrum (black) overlaid with Top-K candidates (colored). "
    "Panel (b): 2D structures of the top candidates with final scores. "
    "Panel (c): Attention weight matrix showing enhanced diagonal structure "
    "indicating wavenumber-local correlation bias."
)
ax_text.text(0, 0.95, caption, transform=ax_text.transAxes, fontsize=7.5,
             color="#444444", va="top", wrap=True)

out = "02_demo_document/latex/figures/fig2_outputs.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved {out}")
