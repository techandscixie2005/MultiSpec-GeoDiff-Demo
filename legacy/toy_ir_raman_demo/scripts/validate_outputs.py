from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

REQUIRED_FILES = [
    "topk_candidates.csv",
    "query_spectra.png",
    "spectrum_match_top1.png",
    "topk_spectrum_overlay.png",
    "molecule_grid.png",
    "attention_heatmap_ir.png",
    "attention_heatmap_raman.png",
    "demo_summary_panel.png",
    "trace_log.json",
]
CSV_COLUMNS = [
    "rank",
    "molecule_id",
    "name",
    "smiles",
    "formula",
    "final_score",
    "ir_similarity",
    "raman_similarity",
    "peak_score",
    "mass_score",
    "validity",
    "notes",
]
TRACE_KEYS = [
    "timestamp",
    "git_commit_hash",
    "input_files",
    "config_file",
    "preprocessing_parameters",
    "modalities_used",
    "candidate_library_size",
    "filtering_steps",
    "scoring_weights",
    "topk_candidates",
    "generated_output_files",
    "environment",
    "warnings",
    "fallback_modes",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate required MultiSpec-GeoDiff demo outputs.")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()
    out = ROOT / args.output_dir
    missing = [name for name in REQUIRED_FILES if not (out / name).exists()]
    if missing:
        raise SystemExit(f"Missing required outputs: {missing}")
    df = pd.read_csv(out / "topk_candidates.csv")
    missing_cols = [col for col in CSV_COLUMNS if col not in df.columns]
    if missing_cols:
        raise SystemExit(f"Missing required CSV columns: {missing_cols}")
    if df.empty:
        raise SystemExit("topk_candidates.csv must contain at least one row")
    trace = json.loads((out / "trace_log.json").read_text(encoding="utf-8"))
    missing_keys = [key for key in TRACE_KEYS if key not in trace]
    if missing_keys:
        raise SystemExit(f"Missing required trace keys: {missing_keys}")
    print("Validation passed for", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
