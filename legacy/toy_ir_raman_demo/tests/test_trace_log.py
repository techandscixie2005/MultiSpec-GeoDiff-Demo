from __future__ import annotations

from multispec_geodiff.trace_logger import build_trace_log


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


def test_build_trace_log_contains_required_fields(monkeypatch) -> None:
    monkeypatch.setattr("multispec_geodiff.trace_logger.git_commit_hash", lambda cwd: "abc123")
    monkeypatch.setattr(
        "multispec_geodiff.trace_logger.get_environment_summary",
        lambda: {"python": {"version": "3.11.0"}, "packages": {}},
    )

    trace = build_trace_log(
        project_root=".",
        config_path="configs/demo.yaml",
        config={"seed": 7},
        input_files={"query_ir": "query_ir.csv"},
        preprocessing_params={"normalize": "max", "grid_size": 4},
        modalities_used=["ir"],
        candidate_library_size=12,
        filtering_steps=["mass_filter_penalty_only:mol_99"],
        scoring_weights={"ir_similarity": 1.0},
        topk_candidates=[{"molecule_id": "mol_03"}],
        output_files=["trace_log.json"],
        warnings=["numpy_encoder"],
        fallback_modes=["no_rdkit"],
    )

    assert REQUIRED_TRACE_KEYS <= set(trace)
    assert trace["git_commit_hash"] == "abc123"
    assert trace["config_snapshot"] == {"seed": 7}
