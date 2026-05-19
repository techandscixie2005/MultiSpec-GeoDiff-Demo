from __future__ import annotations

from typing import Any

import pandas as pd


def combine_scores(score_row: dict[str, Any], weights: dict[str, float]) -> float:
    total = 0.0
    for key, weight in weights.items():
        total += float(score_row.get(key, 0.0)) * float(weight)
    return total


def rerank_candidates(rows: list[dict[str, Any]], weights: dict[str, float], top_k: int) -> pd.DataFrame:
    ranked = []
    for row in rows:
        row = dict(row)
        row["final_score"] = combine_scores(row, weights)
        ranked.append(row)
    df = pd.DataFrame(ranked).sort_values(["final_score", "ir_similarity", "raman_similarity"], ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.head(top_k).copy()
