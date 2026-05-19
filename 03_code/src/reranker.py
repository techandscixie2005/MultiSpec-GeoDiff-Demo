"""Forward-spectrum similarity reranking.

After candidate retrieval, compute fine-grained spectral similarity
between the query and each candidate using multiple metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from metrics import cosine_similarity, peak_match_score, spectral_l1_distance


def compute_forward_score(
    query_intensity: np.ndarray,
    candidate_intensity: np.ndarray,
    query_peaks_wavenumber: np.ndarray,
    query_peaks_intensity: np.ndarray,
    candidate_peaks_wavenumber: np.ndarray,
    candidate_peaks_intensity: np.ndarray,
) -> dict[str, float]:
    """Compute forward-spectrum similarity metrics between query and candidate.

    Parameters
    ----------
    query_intensity : ndarray
        Query intensity vector.
    candidate_intensity : ndarray
        Candidate intensity vector.
    ...peaks_* : ndarrays
        Peak positions and intensities.

    Returns
    -------
    scores : dict with 'cosine', 'l1', 'peak_match', 'forward_score'.
    """
    cos = cosine_similarity(query_intensity, candidate_intensity)
    l1 = spectral_l1_distance(query_intensity, candidate_intensity)
    l1_score = max(0.0, 1.0 - l1 * 2.0)
    peak = peak_match_score(
        query_peaks_wavenumber, query_peaks_intensity,
        candidate_peaks_wavenumber, candidate_peaks_intensity,
    )
    forward_score = 0.4 * cos + 0.3 * l1_score + 0.3 * peak
    return {
        "cosine_similarity": cos,
        "l1_similarity": l1_score,
        "peak_match": peak,
        "forward_score": forward_score,
    }


def rerank_candidates(
    query_record: dict[str, Any],
    candidate_records: list[dict[str, Any]],
    candidate_indices: list[int],
    retrieval_scores: list[float],
    top_k: int = 5,
) -> pd.DataFrame:
    """Rerank retrieved candidates by forward-spectrum similarity.

    Parameters
    ----------
    query_record : dict
        Preprocessed query spectrum.
    candidate_records : list[dict]
        All preprocessed library records.
    candidate_indices : list[int]
        Indices of top-M candidates (from retrieval).
    retrieval_scores : list[float]
        Retrieval similarity scores.
    top_k : int
        Final number of candidates to return.

    Returns
    -------
    df : pd.DataFrame
        Ranked table with columns: rank, smiles, retrieval_score,
        forward_score, final_score, is_ground_truth.
    """
    rows: list[dict[str, Any]] = []
    q_intensity = query_record["intensity"]
    q_px = query_record["peaks_wavenumber"]
    q_py = query_record["peaks_intensity"]
    query_smiles = query_record["original_record"]["smiles"]

    for idx, ret_score in zip(candidate_indices, retrieval_scores):
        cand = candidate_records[idx]
        forward = compute_forward_score(
            q_intensity, cand["intensity"],
            q_px, q_py,
            cand["peaks_wavenumber"], cand["peaks_intensity"],
        )
        final_score = 0.3 * ret_score + 0.7 * forward["forward_score"]
        is_gt = cand["original_record"]["smiles"] == query_smiles
        rows.append({
            "smiles": cand["original_record"]["smiles"],
            "retrieval_score": round(ret_score, 4),
            "cosine_similarity": round(forward["cosine_similarity"], 4),
            "l1_similarity": round(forward["l1_similarity"], 4),
            "peak_match": round(forward["peak_match"], 4),
            "forward_score": round(forward["forward_score"], 4),
            "final_score": round(final_score, 4),
            "is_ground_truth": is_gt,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.head(top_k)
