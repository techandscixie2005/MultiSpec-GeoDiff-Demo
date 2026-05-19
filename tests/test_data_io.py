from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from multispec_geodiff.data_io import load_candidate_library, load_optional_spectrum, validate_candidate_library, validate_spectrum_frame


def test_validate_spectrum_frame_renames_and_sorts_xy_columns() -> None:
    frame = pd.DataFrame(
        {
            "x": [1100.0, 900.0, 1000.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0],
            "y": [0.2, 0.8, 0.4, 0.1, 0.3, 0.5, 0.6, 0.7],
        }
    )

    validated = validate_spectrum_frame(frame)

    assert list(validated.columns) == ["wavenumber", "intensity"]
    assert validated["wavenumber"].tolist() == [900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0]


def test_validate_spectrum_frame_rejects_missing_required_columns() -> None:
    frame = pd.DataFrame({"wavenumber": [1, 2, 3, 4, 5, 6, 7, 8]})

    with pytest.raises(ValueError, match="missing required columns"):
        validate_spectrum_frame(frame, path="broken.csv")


def test_validate_candidate_library_rejects_invalid_mass_values() -> None:
    frame = pd.DataFrame(
        {
            "molecule_id": ["m1"],
            "name": ["mol"],
            "smiles": ["CCO"],
            "formula": ["C2H6O"],
            "approximate_mass": ["not-a-number"],
            "ir_path": ["ir.csv"],
            "raman_path": ["raman.csv"],
        }
    )

    with pytest.raises(ValueError, match="invalid approximate_mass"):
        validate_candidate_library(frame)


def test_load_candidate_library_resolves_relative_spectrum_paths(tmp_path: Path) -> None:
    library_dir = tmp_path / "sample_data"
    library_dir.mkdir()
    (library_dir / "ir.csv").write_text("wavenumber,intensity\n400,0.1\n500,0.2\n600,0.3\n700,0.4\n800,0.5\n900,0.6\n1000,0.7\n1100,0.8\n", encoding="utf-8")
    (library_dir / "raman.csv").write_text("wavenumber,intensity\n400,0.1\n500,0.2\n600,0.3\n700,0.4\n800,0.5\n900,0.6\n1000,0.7\n1100,0.8\n", encoding="utf-8")
    pd.DataFrame(
        {
            "molecule_id": ["m1"],
            "name": ["mol"],
            "smiles": ["CCO"],
            "formula": ["C2H6O"],
            "approximate_mass": [46.07],
            "ir_path": ["ir.csv"],
            "raman_path": ["raman.csv"],
            "functional_group_tags": ["alcohol"],
            "notes": ["fixture"],
        }
    ).to_csv(library_dir / "candidate_library.csv", index=False)

    loaded = load_candidate_library(library_dir / "candidate_library.csv")

    assert loaded.loc[0, "ir_path"] == str((library_dir / "ir.csv").resolve())
    assert loaded.loc[0, "raman_path"] == str((library_dir / "raman.csv").resolve())


def test_load_optional_spectrum_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert load_optional_spectrum(tmp_path / "missing.csv") is None
