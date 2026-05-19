from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REQUIRED_TRACE_KEYS = {
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
    "config_snapshot",
}


def test_run_demo_generates_valid_outputs(demo_config_path: Path, tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "demo_outputs"

    run = subprocess.run(
        [sys.executable, "scripts/run_demo.py", "--config", str(demo_config_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    validate = subprocess.run(
        [sys.executable, "scripts/validate_outputs.py", "--output_dir", str(output_dir)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    topk = pd.read_csv(output_dir / "topk_candidates.csv")
    trace = json.loads((output_dir / "trace_log.json").read_text(encoding="utf-8"))

    assert "MultiSpec-GeoDiff demo complete" in run.stdout
    assert validate.stdout.strip().endswith(str(output_dir))
    assert topk.iloc[0]["name"] == "phenol"
    assert (output_dir / "demo_summary.md").exists()
    assert REQUIRED_TRACE_KEYS <= set(trace)
