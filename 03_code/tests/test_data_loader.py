"""Tests for data_loader module."""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[2] / "04_data"
JSONL_PATH = DATA_DIR / "IR_nist_200.jsonl"


@pytest.fixture(scope="module")
def records():
    from data_loader import load_ir_jsonl
    return load_ir_jsonl(str(JSONL_PATH))


def test_load_jsonl(records):
    assert len(records) == 200


def test_validate_record(records):
    from data_loader import validate_record
    for rec in records[:10]:
        assert validate_record(rec), f"Record failed validation: {rec['smiles']}"


def test_validate_record_invalid():
    from data_loader import validate_record
    assert not validate_record({})
    assert not validate_record({"smiles": "C"})
    assert not validate_record({"smiles": "C", "value": {"x": [1, 2], "y": [1]}})


def test_split_query_library(records):
    from data_loader import split_query_library
    query, library = split_query_library(records, 0)
    assert query["smiles"] == records[0]["smiles"]
    assert len(library) == 199
    assert all(r["smiles"] != query["smiles"] for r in library)


def test_split_query_library_out_of_range(records):
    from data_loader import split_query_library
    with pytest.raises(ValueError):
        split_query_library(records, 999)


def test_get_unique_smiles(records):
    from data_loader import get_unique_smiles
    unique = get_unique_smiles(records)
    assert len(unique) <= len(records)
    assert len(unique) == len(set(unique))
