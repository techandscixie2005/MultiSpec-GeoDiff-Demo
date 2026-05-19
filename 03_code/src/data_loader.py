"""JSONL data loader for NIST IR spectra."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_ir_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load all records from a JSONL file.

    Each line is a JSON object with keys:
      - smiles: str
      - value: dict with 'x' (list[float]) and 'y' (list[float])
    """
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_record(record: dict[str, Any]) -> bool:
    """Check that a record has the required fields with valid data."""
    if "smiles" not in record or not isinstance(record["smiles"], str):
        return False
    value = record.get("value")
    if not isinstance(value, dict):
        return False
    x = value.get("x")
    y = value.get("y")
    if not isinstance(x, list) or not isinstance(y, list):
        return False
    if len(x) < 10 or len(y) < 10:
        return False
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if np.any(np.isnan(x_arr)) or np.any(np.isnan(y_arr)):
        return False
    if np.any(np.isinf(x_arr)) or np.any(np.isinf(y_arr)):
        return False
    return True


def split_query_library(
    records: list[dict[str, Any]], query_id: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Split records into one query and the rest as library.

    Parameters
    ----------
    records : list[dict]
        Full list of spectral records.
    query_id : int
        Index of the query molecule.

    Returns
    -------
    query_record : dict
        The selected query record.
    library_records : list[dict]
        All other records (candidate library).
    """
    if query_id < 0 or query_id >= len(records):
        raise ValueError(f"query_id {query_id} out of range [0, {len(records) - 1}]")
    query_record = records[query_id]
    library_records = [r for i, r in enumerate(records) if i != query_id]
    return query_record, library_records


def get_unique_smiles(records: list[dict[str, Any]]) -> list[str]:
    """Return list of unique SMILES strings (preserving order)."""
    seen: set[str] = set()
    result: list[str] = []
    for r in records:
        s = r["smiles"]
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result
