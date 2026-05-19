from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path
from typing import Any

from .data_io import get_environment_summary, write_json


def git_commit_hash(cwd: str | Path = ".") -> str | None:
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), stderr=subprocess.DEVNULL, text=True).strip()
        return output or None
    except Exception:
        return None


def build_trace_log(
    *,
    project_root: str | Path,
    config_path: str | Path,
    config: dict[str, Any],
    input_files: dict[str, Any],
    preprocessing_params: dict[str, Any],
    modalities_used: list[str],
    candidate_library_size: int,
    filtering_steps: list[str],
    scoring_weights: dict[str, float],
    topk_candidates: list[dict[str, Any]],
    output_files: list[str],
    warnings: list[str],
    fallback_modes: list[str],
) -> dict[str, Any]:
    return {
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "git_commit_hash": git_commit_hash(project_root),
        "input_files": input_files,
        "config_file": str(config_path),
        "preprocessing_parameters": preprocessing_params,
        "modalities_used": modalities_used,
        "candidate_library_size": candidate_library_size,
        "filtering_steps": filtering_steps,
        "scoring_weights": scoring_weights,
        "topk_candidates": topk_candidates,
        "generated_output_files": output_files,
        "environment": get_environment_summary(),
        "warnings": warnings,
        "fallback_modes": fallback_modes,
        "config_snapshot": config,
    }


def write_trace_log(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
