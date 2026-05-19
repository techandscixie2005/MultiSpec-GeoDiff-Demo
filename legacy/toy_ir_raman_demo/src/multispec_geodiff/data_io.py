from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SPECTRUM_REQUIRED_COLUMNS = ("wavenumber", "intensity")
CANDIDATE_REQUIRED_COLUMNS = (
    "molecule_id",
    "name",
    "smiles",
    "formula",
    "approximate_mass",
    "ir_path",
    "raman_path",
)


def _ensure_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def load_config_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with _ensure_path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def validate_spectrum_frame(df: pd.DataFrame, path: str | Path | None = None) -> pd.DataFrame:
    rename_map = {"x": "wavenumber", "y": "intensity"}
    df = df.rename(columns=rename_map)
    missing = [col for col in SPECTRUM_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        location = f" in {path}" if path else ""
        raise ValueError(
            f"Spectrum CSV{location} is missing required columns {missing}. "
            f"Expected columns: {list(SPECTRUM_REQUIRED_COLUMNS)}"
        )
    out = df.loc[:, ["wavenumber", "intensity"]].copy()
    out["wavenumber"] = pd.to_numeric(out["wavenumber"], errors="coerce")
    out["intensity"] = pd.to_numeric(out["intensity"], errors="coerce")
    if out.isna().any().any():
        raise ValueError(f"Spectrum CSV {path or ''} contains non-numeric or missing values.")
    out = out.sort_values("wavenumber").reset_index(drop=True)
    if len(out) < 8:
        raise ValueError(f"Spectrum CSV {path or ''} must contain at least 8 rows.")
    return out


def load_spectrum_csv(path: str | Path) -> pd.DataFrame:
    path = _ensure_path(path)
    df = pd.read_csv(path)
    return validate_spectrum_frame(df, path)


def load_optional_spectrum(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    candidate = _ensure_path(path)
    if not candidate.exists():
        return None
    return load_spectrum_csv(candidate)


def validate_candidate_library(df: pd.DataFrame, path: str | Path | None = None) -> pd.DataFrame:
    missing = [col for col in CANDIDATE_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        location = f" in {path}" if path else ""
        raise ValueError(
            f"Candidate library CSV{location} is missing required columns {missing}. "
            f"Expected at least: {list(CANDIDATE_REQUIRED_COLUMNS)}"
        )
    out = df.copy()
    out["approximate_mass"] = pd.to_numeric(out["approximate_mass"], errors="coerce")
    if out["approximate_mass"].isna().any():
        raise ValueError("Candidate library contains invalid approximate_mass values.")
    out["functional_group_tags"] = out.get("functional_group_tags", "").fillna("")
    out["notes"] = out.get("notes", "").fillna("")
    return out


def _resolve_relative(base_dir: Path, raw_path: str | None) -> str | None:
    if raw_path is None or str(raw_path).strip() == "":
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


def load_candidate_library(path: str | Path) -> pd.DataFrame:
    path = _ensure_path(path)
    df = pd.read_csv(path)
    df = validate_candidate_library(df, path)
    base = path.parent
    for col in ["ir_path", "raman_path"]:
        df[col] = df[col].apply(lambda value: _resolve_relative(base, value))
    return df


def read_json(path: str | Path) -> dict[str, Any]:
    with _ensure_path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = _ensure_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


def ensure_output_dir(path: str | Path) -> Path:
    out = _ensure_path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_environment_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {"python": None, "packages": {}}
    import platform
    import sys

    summary["python"] = {
        "version": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    for name in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "yaml", "rdkit", "torch"]:
        try:
            module = __import__(name)
            summary["packages"][name] = getattr(module, "__version__", "available")
        except Exception:
            summary["packages"][name] = None
    return summary
