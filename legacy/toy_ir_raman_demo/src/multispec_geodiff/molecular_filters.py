from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math


try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw
except Exception:  # pragma: no cover - dependency optional
    Chem = None
    Descriptors = None
    Draw = None


@dataclass
class ValidityResult:
    available: bool
    is_valid: bool | None
    score: float
    notes: str


def rdkit_available() -> bool:
    return Chem is not None


def validate_smiles(smiles: str) -> ValidityResult:
    if not rdkit_available():
        return ValidityResult(False, None, 0.6, "RDKit unavailable; fallback validity score applied.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ValidityResult(True, False, 0.0, "Invalid SMILES according to RDKit.")
    return ValidityResult(True, True, 1.0, "RDKit sanitized candidate.")


def mass_similarity(query_mass: float | None, candidate_mass: float | None, tolerance: float = 2.5) -> float:
    if query_mass is None or candidate_mass is None:
        return 0.5
    delta = abs(float(query_mass) - float(candidate_mass))
    return float(math.exp(-((delta / max(tolerance, 1e-6)) ** 2)))


def formula_match_score(query_formula: str | None, candidate_formula: str | None) -> float:
    if not query_formula:
        return 0.5
    return 1.0 if str(query_formula).strip() == str(candidate_formula).strip() else 0.0


def parse_tags(raw_tags: str | None) -> set[str]:
    if raw_tags is None:
        return set()
    return {token.strip().lower() for token in str(raw_tags).split(",") if token.strip()}


def functional_group_overlap(query_tags: str | None, candidate_tags: str | None) -> float:
    q = parse_tags(query_tags)
    c = parse_tags(candidate_tags)
    if not q or not c:
        return 0.5
    overlap = len(q & c)
    denom = len(q | c)
    return overlap / denom if denom else 0.5


def compute_candidate_validity(candidate: dict[str, Any]) -> ValidityResult:
    return validate_smiles(candidate.get("smiles", ""))
