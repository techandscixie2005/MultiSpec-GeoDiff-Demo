from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "samples"
LIB_DIR = DATA_DIR / "library_spectra"


def gaussian_spectrum(grid: np.ndarray, peaks: list[tuple[float, float, float]], baseline: float = 0.0) -> np.ndarray:
    signal = np.full_like(grid, baseline, dtype=float)
    for center, width, height in peaks:
        signal += height * np.exp(-0.5 * ((grid - center) / width) ** 2)
    return signal


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.min()
    return arr / max(arr.max(), 1e-8)


def library_definitions() -> list[dict]:
    return [
        {"molecule_id": "mol_01", "name": "benzene", "smiles": "c1ccccc1", "formula": "C6H6", "approximate_mass": 78.11, "functional_group_tags": "aromatic", "ir_peaks": [(674, 40, 0.45), (1038, 55, 0.35), (1485, 70, 0.55), (3030, 65, 0.55)], "raman_peaks": [(606, 30, 0.35), (992, 35, 1.0), (1178, 35, 0.45), (3065, 55, 0.35)]},
        {"molecule_id": "mol_02", "name": "toluene", "smiles": "Cc1ccccc1", "formula": "C7H8", "approximate_mass": 92.14, "functional_group_tags": "aromatic,methyl", "ir_peaks": [(728, 35, 0.42), (1030, 55, 0.28), (1450, 60, 0.58), (2920, 75, 0.52), (3030, 60, 0.35)], "raman_peaks": [(786, 28, 0.4), (1003, 28, 0.92), (1210, 35, 0.35), (3055, 55, 0.32)]},
        {"molecule_id": "mol_03", "name": "phenol", "smiles": "c1ccc(cc1)O", "formula": "C6H6O", "approximate_mass": 94.11, "functional_group_tags": "aromatic,hydroxyl", "ir_peaks": [(760, 35, 0.4), (1215, 55, 0.72), (1500, 60, 0.48), (1600, 55, 0.52), (3340, 120, 1.0)], "raman_peaks": [(621, 25, 0.28), (1001, 30, 0.88), (1170, 35, 0.52), (1605, 40, 0.62), (3060, 55, 0.25)]},
        {"molecule_id": "mol_04", "name": "aniline", "smiles": "Nc1ccccc1", "formula": "C6H7N", "approximate_mass": 93.13, "functional_group_tags": "aromatic,amine", "ir_peaks": [(690, 35, 0.38), (1280, 50, 0.5), (1495, 55, 0.45), (1615, 45, 0.58), (3460, 110, 0.78)], "raman_peaks": [(618, 25, 0.24), (1000, 30, 0.85), (1165, 35, 0.38), (1598, 40, 0.56), (3062, 55, 0.22)]},
        {"molecule_id": "mol_05", "name": "ethanol", "smiles": "CCO", "formula": "C2H6O", "approximate_mass": 46.07, "functional_group_tags": "alcohol", "ir_peaks": [(1050, 45, 0.7), (1090, 35, 0.48), (1450, 55, 0.42), (2930, 80, 0.52), (3335, 130, 0.88)], "raman_peaks": [(880, 30, 0.35), (1045, 28, 0.62), (1092, 25, 0.4), (2885, 60, 0.48)]},
        {"molecule_id": "mol_06", "name": "dimethyl ether", "smiles": "COC", "formula": "C2H6O", "approximate_mass": 46.07, "functional_group_tags": "ether", "ir_peaks": [(1110, 40, 0.84), (1195, 38, 0.35), (1460, 55, 0.45), (2830, 65, 0.58), (2950, 75, 0.46)], "raman_peaks": [(922, 25, 0.28), (1098, 28, 0.74), (1455, 40, 0.38), (2840, 60, 0.52)]},
        {"molecule_id": "mol_07", "name": "acetone", "smiles": "CC(=O)C", "formula": "C3H6O", "approximate_mass": 58.08, "functional_group_tags": "ketone", "ir_peaks": [(1220, 45, 0.38), (1365, 50, 0.4), (1715, 50, 1.0), (2920, 70, 0.42)], "raman_peaks": [(790, 30, 0.24), (1220, 35, 0.3), (1712, 35, 0.58), (2925, 60, 0.32)]},
        {"molecule_id": "mol_08", "name": "acetaldehyde", "smiles": "CC=O", "formula": "C2H4O", "approximate_mass": 44.05, "functional_group_tags": "aldehyde", "ir_peaks": [(1160, 40, 0.3), (1388, 45, 0.34), (1730, 45, 0.92), (2720, 70, 0.62), (2820, 70, 0.58)], "raman_peaks": [(778, 28, 0.18), (1355, 35, 0.35), (1730, 35, 0.52), (2930, 60, 0.26)]},
        {"molecule_id": "mol_09", "name": "acetic acid", "smiles": "CC(=O)O", "formula": "C2H4O2", "approximate_mass": 60.05, "functional_group_tags": "carboxylic_acid", "ir_peaks": [(1285, 45, 0.52), (1410, 55, 0.4), (1710, 50, 0.82), (3000, 150, 0.38), (3400, 140, 0.95)], "raman_peaks": [(633, 25, 0.2), (894, 30, 0.22), (1415, 35, 0.4), (1710, 35, 0.48)]},
        {"molecule_id": "mol_10", "name": "ethyl acetate", "smiles": "CCOC(=O)C", "formula": "C4H8O2", "approximate_mass": 88.11, "functional_group_tags": "ester", "ir_peaks": [(1035, 45, 0.55), (1240, 45, 0.62), (1368, 50, 0.28), (1742, 45, 0.95), (2980, 70, 0.45)], "raman_peaks": [(848, 26, 0.26), (1032, 28, 0.54), (1245, 30, 0.34), (1740, 35, 0.42), (2935, 55, 0.28)]},
        {"molecule_id": "mol_11", "name": "propanol", "smiles": "CCCO", "formula": "C3H8O", "approximate_mass": 60.10, "functional_group_tags": "alcohol", "ir_peaks": [(1060, 45, 0.62), (1120, 35, 0.42), (1460, 55, 0.38), (2940, 75, 0.48), (3340, 125, 0.9)], "raman_peaks": [(860, 28, 0.32), (1058, 28, 0.54), (1452, 36, 0.35), (2888, 58, 0.44)]},
        {"molecule_id": "mol_12", "name": "methyl benzoate", "smiles": "COC(=O)c1ccccc1", "formula": "C8H8O2", "approximate_mass": 136.15, "functional_group_tags": "aromatic,ester", "ir_peaks": [(710, 35, 0.34), (1110, 42, 0.42), (1275, 45, 0.48), (1718, 45, 0.9), (2948, 70, 0.25), (3050, 55, 0.22)], "raman_peaks": [(618, 25, 0.22), (1002, 30, 0.85), (1175, 35, 0.32), (1608, 40, 0.52), (1715, 35, 0.4)]},
    ]


def write_spectrum(path: Path, grid: np.ndarray, intensity: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"wavenumber": grid, "intensity": intensity}).to_csv(path, index=False)


def build_sample_dataset(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    grid = np.arange(400.0, 4000.0 + 4.0, 4.0)
    rows = []
    target_id = "mol_03"
    for entry in library_definitions():
        ir = normalize(gaussian_spectrum(grid, entry["ir_peaks"], baseline=0.01))
        raman = normalize(gaussian_spectrum(grid, entry["raman_peaks"], baseline=0.008))
        ir_path = LIB_DIR / f"{entry['molecule_id']}_ir.csv"
        raman_path = LIB_DIR / f"{entry['molecule_id']}_raman.csv"
        write_spectrum(ir_path, grid, ir)
        write_spectrum(raman_path, grid, raman)
        row = {k: v for k, v in entry.items() if not k.endswith('_peaks')}
        row["ir_path"] = str(ir_path.relative_to(DATA_DIR))
        row["raman_path"] = str(raman_path.relative_to(DATA_DIR))
        row["notes"] = "Toy Gaussian-template spectra for demo only."
        rows.append(row)
        if entry["molecule_id"] == target_id:
            shifted_ir = normalize(gaussian_spectrum(grid, [(c + rng.normal(0, 8), w * rng.uniform(0.95, 1.08), h * rng.uniform(0.92, 1.05)) for c, w, h in entry["ir_peaks"]], baseline=0.012) + rng.normal(0, 0.01, size=grid.shape))
            shifted_raman = normalize(gaussian_spectrum(grid, [(c + rng.normal(0, 7), w * rng.uniform(0.95, 1.08), h * rng.uniform(0.92, 1.05)) for c, w, h in entry["raman_peaks"]], baseline=0.008) + rng.normal(0, 0.008, size=grid.shape))
            write_spectrum(DATA_DIR / "query_ir.csv", grid, shifted_ir)
            write_spectrum(DATA_DIR / "query_raman.csv", grid, shifted_raman)
    pd.DataFrame(rows).to_csv(DATA_DIR / "candidate_library.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate built-in toy spectra for the MultiSpec-GeoDiff demo.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    build_sample_dataset(seed=args.seed)
    print(f"Sample data written under {DATA_DIR}")
