"""Visualization utilities for the demo pipeline.

RDKit is optional; when Cairo backend is unavailable, molecules are rendered
via SVG + cairosvg fallback.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False

# cairosvg is used as a fallback when RDKit's Cairo backend is missing.
try:
    import cairosvg
    _CAIROSVG_AVAILABLE = True
except Exception:
    _CAIROSVG_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")


def _rdkit_mol_to_image(mol, size: tuple[int, int] = (300, 200)):
    """Render an RDKit Mol to a PIL Image.

    Tries Cairo first; falls back to SVG + cairosvg; returns None on failure.
    """
    w, h = size
    # Try Cairo backend first.
    try:
        from rdkit.Chem.Draw import MolDraw2DCairo
        drawer = MolDraw2DCairo(w, h)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return PILImage.open(io.BytesIO(drawer.GetDrawingText()))
    except Exception:
        pass

    # Fallback: SVG via cairosvg.
    if _CAIROSVG_AVAILABLE:
        try:
            drawer = Draw.MolDraw2DSVG(w, h)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            png_data = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
            return PILImage.open(io.BytesIO(png_data))
        except Exception:
            pass

    return None


def _ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_spectrum_overlay(
    query_record: dict[str, Any],
    top_candidates: list[dict[str, Any]],
    output_path: str,
    title: str = "Query vs Top-K Candidate Spectra",
) -> str:
    """Plot query spectrum overlaid with top-K candidate spectra.

    Parameters
    ----------
    query_record : dict
        Preprocessed query with 'wavenumber' and 'intensity'.
    top_candidates : list[dict]
        Ranked candidate records (with 'wavenumber', 'intensity', and 'smiles').
    output_path : str
        Path to save the figure.

    Returns
    -------
    output_path : str
        Path to saved figure.
    """
    output_path = str(_ensure_parent(output_path))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        query_record["wavenumber"],
        query_record["intensity"],
        color="black",
        lw=2.5,
        label="Query",
    )
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.1, 0.9, len(top_candidates)))
    for i, cand in enumerate(top_candidates):
        label = f"#{i + 1} {cand['smiles'][:20]}..."
        ax.plot(
            cand["wavenumber"],
            cand["intensity"],
            lw=1.1,
            alpha=0.8,
            color=colors[i],
            label=label,
        )
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    output_path: str,
    title: str = "Spectral-Distance-Bias Attention",
) -> str:
    """Plot an attention heatmap.

    Parameters
    ----------
    attention_matrix : ndarray, shape (n, n)
        Attention weight matrix.
    output_path : str
        Path to save the figure.
    title : str
        Plot title.

    Returns
    -------
    output_path : str
        Path to saved figure.
    """
    output_path = str(_ensure_parent(output_path))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(attention_matrix, aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Token index")
    ax.set_ylabel("Token index")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_molecule_grid(
    smiles_list: list[str],
    scores: list[float],
    output_path: str,
) -> str:
    """Plot a grid of molecule structures.

    Uses RDKit Cairo when available; falls back to SVG+cairosvg;
    text-only labels as last resort.
    """
    output_path = str(_ensure_parent(output_path))
    n = len(smiles_list)
    # Use a clean 1-row layout for up to 5 molecules.
    if n <= 5:
        cols = n
        rows = 1
    else:
        cols = 3
        rows = max(1, int(np.ceil(n / cols)))

    fig_width = min(14, cols * 3.5)
    fig_height = rows * 3.2
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = np.array(axes).reshape(-1)

    if RDKIT_AVAILABLE:
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    else:
        mols = [None] * n

    for i, (ax_i, smiles, score, mol) in enumerate(zip(axes, smiles_list, scores, mols)):
        ax_i.axis("off")
        ax_i.set_title(f"#{i + 1}  Score={score:.4f}", fontsize=10, pad=4)

        img = None
        if mol is not None:
            img = _rdkit_mol_to_image(mol, size=(300, 200))

        if img is not None:
            ax_i.imshow(img)
        else:
            # Text-only fallback when RDKit or rendering is unavailable.
            ax_i.text(
                0.5, 0.55, smiles,
                ha="center", va="center",
                fontsize=9, family="monospace",
                transform=ax_i.transAxes,
            )
            ax_i.text(
                0.5, 0.15, "(structure rendering unavailable)",
                ha="center", va="center",
                fontsize=7, color="#999",
                transform=ax_i.transAxes,
            )

    for ax_i in axes[n:]:
        ax_i.axis("off")

    fig.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
