"""Visualization utilities for the demo pipeline.

RDKit is optional; if unavailable, molecule grid falls back to text panels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")


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
    """Plot a grid of molecule structures/SMILES.

    Uses RDKit if available; otherwise falls back to text panels.
    """
    output_path = str(_ensure_parent(output_path))
    n = len(smiles_list)
    cols = min(2, n)
    rows = max(1, int(np.ceil(n / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(4, rows * 3.5)))
    axes = np.array(axes).reshape(-1)

    if RDKIT_AVAILABLE:
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    else:
        mols = [None] * n

    for i, (ax_i, smiles, score, mol) in enumerate(zip(axes, smiles_list, scores, mols)):
        ax_i.axis("off")
        ax_i.set_title(f"#{i + 1}  Score={score:.3f}", fontsize=11)
        if RDKIT_AVAILABLE and mol is not None:
            try:
                img = Draw.MolToImage(mol, size=(250, 250))
                ax_i.imshow(img)
            except Exception:
                ax_i.text(0.5, 0.5, smiles, ha="center", va="center",
                          fontsize=9, family="monospace", transform=ax_i.transAxes)
                ax_i.text(0.5, 0.1, "RDKit Cairo unavailable (text fallback)",
                          ha="center", va="center", fontsize=7, color="#888",
                          transform=ax_i.transAxes)
        else:
            # Fallback: SMILES text panel
            ax_i.text(
                0.5, 0.5, smiles,
                ha="center", va="center",
                fontsize=9, family="monospace",
                transform=ax_i.transAxes,
            )
            status = "RDKit available" if RDKIT_AVAILABLE else "RDKit unavailable (text fallback)"
            ax_i.text(
                0.5, 0.1, status,
                ha="center", va="center",
                fontsize=7, color="#888",
                transform=ax_i.transAxes,
            )

    for ax_i in axes[n:]:
        ax_i.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
