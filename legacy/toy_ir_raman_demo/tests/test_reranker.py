from __future__ import annotations

from multispec_geodiff.reranker import combine_scores, rerank_candidates


def test_combine_scores_applies_requested_weights() -> None:
    score = combine_scores(
        {"ir_similarity": 0.8, "raman_similarity": 0.5, "validity": 1.0},
        {"ir_similarity": 0.5, "raman_similarity": 0.25, "validity": 0.25},
    )

    assert score == 0.775


def test_rerank_candidates_orders_by_final_score_then_similarity_tiebreakers() -> None:
    ranked = rerank_candidates(
        rows=[
            {"molecule_id": "mol_b", "name": "B", "ir_similarity": 0.7, "raman_similarity": 0.9, "peak_score": 0.5, "mass_score": 0.5, "validity": 1.0},
            {"molecule_id": "mol_a", "name": "A", "ir_similarity": 0.8, "raman_similarity": 0.8, "peak_score": 0.5, "mass_score": 0.5, "validity": 1.0},
            {"molecule_id": "mol_c", "name": "C", "ir_similarity": 0.2, "raman_similarity": 0.2, "peak_score": 0.1, "mass_score": 0.2, "validity": 0.1},
        ],
        weights={"ir_similarity": 0.5, "raman_similarity": 0.5},
        top_k=2,
    )

    assert ranked["molecule_id"].tolist() == ["mol_a", "mol_b"]
    assert ranked["rank"].tolist() == [1, 2]
