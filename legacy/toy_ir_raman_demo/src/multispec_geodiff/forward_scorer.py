from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .molecular_filters import ValidityResult, compute_candidate_validity, formula_match_score, mass_similarity
from .spectra_preprocess import ProcessedSpectrum


@dataclass
class CandidateScore:
    ir_similarity: float
    raman_similarity: float
    peak_score: float
    mass_score: float
    validity: float
    validity_result: ValidityResult
    notes: list[str]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def peak_match_score(query: ProcessedSpectrum | None, candidate: ProcessedSpectrum | None) -> float:
    if query is None or candidate is None:
        return 0.5
    if len(query.peaks) == 0 or len(candidate.peaks) == 0:
        return 0.0
    scores = []
    for peak in query.peaks:
        nearest = float(np.min(np.abs(candidate.peaks - peak)))
        scores.append(np.exp(-((nearest / 60.0) ** 2)))
    return float(np.mean(scores))


def modality_similarity(query: ProcessedSpectrum | None, candidate: ProcessedSpectrum | None) -> float:
    if query is None and candidate is None:
        return 1.0
    if query is None or candidate is None:
        return 0.0
    corr = float(np.corrcoef(query.intensity, candidate.intensity)[0, 1]) if np.std(query.intensity) > 0 and np.std(candidate.intensity) > 0 else 0.0
    cos = cosine_similarity(query.intensity, candidate.intensity)
    return max(0.0, min(1.0, 0.5 * (corr + 1.0) * 0.5 + 0.5 * max(cos, 0.0)))


def score_candidate(
    candidate: dict[str, Any],
    query_ir: ProcessedSpectrum | None,
    query_raman: ProcessedSpectrum | None,
    candidate_ir: ProcessedSpectrum | None,
    candidate_raman: ProcessedSpectrum | None,
    query_metadata: dict[str, Any] | None = None,
) -> CandidateScore:
    query_metadata = query_metadata or {}
    ir_similarity = modality_similarity(query_ir, candidate_ir)
    raman_similarity = modality_similarity(query_raman, candidate_raman)
    peak_score = 0.5 * peak_match_score(query_ir, candidate_ir) + 0.5 * peak_match_score(query_raman, candidate_raman)
    mass_score = 0.7 * mass_similarity(query_metadata.get("approximate_mass"), candidate.get("approximate_mass")) + 0.3 * formula_match_score(query_metadata.get("formula"), candidate.get("formula"))
    validity_result = compute_candidate_validity(candidate)
    notes = [validity_result.notes]
    return CandidateScore(
        ir_similarity=ir_similarity,
        raman_similarity=raman_similarity,
        peak_score=peak_score,
        mass_score=mass_score,
        validity=validity_result.score,
        validity_result=validity_result,
        notes=notes,
    )
