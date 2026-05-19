"""Similarity metrics for spectral comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def spectral_l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L1 distance between two normalized spectra."""
    return float(np.mean(np.abs(a - b)))


def peak_match_score(
    query_wavenumber: np.ndarray,
    query_intensity: np.ndarray,
    candidate_wavenumber: np.ndarray,
    candidate_intensity: np.ndarray,
    tolerance: float = 30.0,
) -> float:
    """Match peaks between query and candidate spectra.

    For each query peak, find the nearest candidate peak within tolerance.
    Returns a score in [0, 1] weighted by intensity agreement.
    """
    if len(query_wavenumber) == 0 or len(candidate_wavenumber) == 0:
        return 0.0
    scores = []
    for qw, qi in zip(query_wavenumber, query_intensity):
        diffs = np.abs(candidate_wavenumber - qw)
        min_idx = np.argmin(diffs)
        if diffs[min_idx] <= tolerance:
            ci = candidate_intensity[min_idx]
            match = 1.0 - min(1.0, diffs[min_idx] / tolerance)
            intensity_agreement = 1.0 - abs(qi - ci) / max(abs(qi) + abs(ci), 1e-8)
            scores.append(0.5 * match + 0.5 * intensity_agreement)
    return float(np.mean(scores)) if scores else 0.0


def topk_hit(ranked_smiles: list[str], ground_truth_smiles: str, k: int) -> bool:
    """Check if ground truth appears in top-k ranked SMILES."""
    return ground_truth_smiles in ranked_smiles[:k]
