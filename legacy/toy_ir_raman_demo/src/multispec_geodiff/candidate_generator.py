from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_io import load_optional_spectrum
from .forward_scorer import modality_similarity
from .molecular_filters import mass_similarity
from .spectra_encoder import CoordinateAwareSpectralEncoder
from .spectra_preprocess import ProcessedSpectrum, preprocess_optional_spectrum


@dataclass
class RetrievedCandidate:
    row: dict[str, Any]
    query_similarity: float
    embedding_similarity: float
    retrieval_score: float
    candidate_ir: ProcessedSpectrum | None
    candidate_raman: ProcessedSpectrum | None


def embedding_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _relative_to(base_path: str | Path, target_path: str | None) -> str | None:
    if not target_path:
        return None
    try:
        return str(Path(target_path).resolve().relative_to(Path(base_path).resolve()))
    except Exception:
        return str(target_path)


def retrieve_candidates(
    candidate_library: pd.DataFrame,
    query_ir: ProcessedSpectrum | None,
    query_raman: ProcessedSpectrum | None,
    query_embedding: np.ndarray,
    encoder: CoordinateAwareSpectralEncoder,
    common_grid: np.ndarray,
    preprocess_kwargs: dict[str, Any],
    retrieval_cfg: dict[str, Any],
    query_metadata: dict[str, Any] | None = None,
) -> tuple[list[RetrievedCandidate], list[str]]:
    query_metadata = query_metadata or {}
    weights = retrieval_cfg.get("similarity_weights", {"ir": 0.5, "raman": 0.5})
    embedding_weight = float(retrieval_cfg.get("embedding_weight", 0.2))
    top_m = int(retrieval_cfg.get("top_m", 8))
    mass_tolerance = float(retrieval_cfg.get("mass_tolerance", 2.5))
    filtering_steps: list[str] = []
    rows: list[RetrievedCandidate] = []

    for _, row in candidate_library.iterrows():
        candidate = row.to_dict()
        if query_metadata.get("approximate_mass") is not None:
            msim = mass_similarity(query_metadata.get("approximate_mass"), candidate.get("approximate_mass"), tolerance=mass_tolerance)
            if msim < 0.12:
                filtering_steps.append(f"mass_filter_penalty_only:{candidate['molecule_id']}")
        ir_df = load_optional_spectrum(candidate.get("ir_path"))
        raman_df = load_optional_spectrum(candidate.get("raman_path"))
        candidate_ir = preprocess_optional_spectrum(ir_df, modality="ir", common_grid=common_grid, **preprocess_kwargs) if ir_df is not None else None
        candidate_raman = preprocess_optional_spectrum(raman_df, modality="raman", common_grid=common_grid, **preprocess_kwargs) if raman_df is not None else None
        encoding = encoder.encode(candidate_ir, candidate_raman)
        embedding_similarity = max(0.0, embedding_cosine(query_embedding, encoding.fused_embedding))
        ir_similarity = modality_similarity(query_ir, candidate_ir)
        raman_similarity = modality_similarity(query_raman, candidate_raman)
        raw_similarity = float(weights.get("ir", 0.5) * ir_similarity + weights.get("raman", 0.5) * raman_similarity)
        retrieval_score = (1 - embedding_weight) * raw_similarity + embedding_weight * embedding_similarity
        rows.append(
            RetrievedCandidate(
                row=candidate,
                query_similarity=raw_similarity,
                embedding_similarity=embedding_similarity,
                retrieval_score=retrieval_score,
                candidate_ir=candidate_ir,
                candidate_raman=candidate_raman,
            )
        )

    rows.sort(key=lambda item: item.retrieval_score, reverse=True)
    return rows[:top_m], filtering_steps
