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

from multispec_geodiff.candidate_generator import retrieve_candidates
from multispec_geodiff.data_io import ensure_output_dir, load_candidate_library, load_config_yaml, load_optional_spectrum
from multispec_geodiff.forward_scorer import score_candidate
from multispec_geodiff.reranker import rerank_candidates
from multispec_geodiff.spectra_encoder import CoordinateAwareSpectralEncoder
from multispec_geodiff.spectra_preprocess import make_common_grid, preprocess_optional_spectrum
from multispec_geodiff.trace_logger import build_trace_log, write_trace_log
from multispec_geodiff.visualization import (
    plot_attention_heatmap,
    plot_candidate_grid,
    plot_demo_summary_panel,
    plot_query_spectra,
    plot_spectrum_match,
    plot_topk_overlay,
)
from generate_sample_data import build_sample_dataset


def write_demo_summary_markdown(path: Path, topk_df: pd.DataFrame, trace_log: dict) -> None:
    top = topk_df.iloc[0].to_dict() if not topk_df.empty else {}
    text = f"""# Demo Summary

- Timestamp: {trace_log['timestamp']}
- Modalities used: {', '.join(trace_log['modalities_used'])}
- Candidate library size: {trace_log['candidate_library_size']}
- Best candidate: {top.get('name', 'N/A')} ({top.get('formula', 'N/A')})
- Final score: {top.get('final_score', 'N/A')}
- Fallback modes: {', '.join(trace_log['fallback_modes']) if trace_log['fallback_modes'] else 'none'}
- Warnings: {', '.join(trace_log['warnings']) if trace_log['warnings'] else 'none'}
"""
    path.write_text(text, encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MultiSpec-GeoDiff demo pipeline.")
    parser.add_argument("--config", default="configs/demo.yaml")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = load_config_yaml(config_path)
    build_sample_dataset(seed=int(config.get("seed", 7)))

    output_dir = ensure_output_dir(ROOT / config["outputs"]["output_dir"])
    input_cfg = config["input"]
    preprocess_cfg = config["preprocess"].copy()
    query_metadata = input_cfg.get("query_metadata", {})

    query_ir_df = load_optional_spectrum(ROOT / input_cfg["query_ir"])
    query_raman_df = load_optional_spectrum(ROOT / input_cfg["query_raman"])
    if query_ir_df is None and query_raman_df is None:
        raise SystemExit("At least one query modality must be present.")

    common_grid = make_common_grid(
        float(preprocess_cfg.pop("wavenumber_min")),
        float(preprocess_cfg.pop("wavenumber_max")),
        float(preprocess_cfg.pop("wavenumber_step")),
    )
    preprocess_cfg.pop("allow_missing_modalities", None)
    query_ir = preprocess_optional_spectrum(query_ir_df, modality="ir", common_grid=common_grid, **preprocess_cfg) if query_ir_df is not None else None
    query_raman = preprocess_optional_spectrum(query_raman_df, modality="raman", common_grid=common_grid, **preprocess_cfg) if query_raman_df is not None else None

    encoder = CoordinateAwareSpectralEncoder(**config["encoder"], seed=int(config.get("seed", 7)))
    query_encoding = encoder.encode(query_ir, query_raman)

    candidate_library = load_candidate_library(ROOT / input_cfg["candidate_library"])
    retrieved, filtering_steps = retrieve_candidates(
        candidate_library=candidate_library,
        query_ir=query_ir,
        query_raman=query_raman,
        query_embedding=query_encoding.fused_embedding,
        encoder=encoder,
        common_grid=common_grid,
        preprocess_kwargs=preprocess_cfg,
        retrieval_cfg=config["retrieval"],
        query_metadata=query_metadata,
    )

    scoring_rows = []
    for item in retrieved:
        score = score_candidate(
            candidate=item.row,
            query_ir=query_ir,
            query_raman=query_raman,
            candidate_ir=item.candidate_ir,
            candidate_raman=item.candidate_raman,
            query_metadata=query_metadata,
        )
        scoring_rows.append(
            {
                "molecule_id": item.row["molecule_id"],
                "name": item.row["name"],
                "smiles": item.row["smiles"],
                "formula": item.row["formula"],
                "final_score": 0.0,
                "ir_similarity": score.ir_similarity,
                "raman_similarity": score.raman_similarity,
                "peak_score": score.peak_score,
                "mass_score": score.mass_score,
                "validity": score.validity,
                "notes": " | ".join(score.notes + [f"retrieval={item.retrieval_score:.3f}", f"embed={item.embedding_similarity:.3f}"]),
                "candidate_ir": item.candidate_ir,
                "candidate_raman": item.candidate_raman,
            }
        )

    topk_df = rerank_candidates(scoring_rows, weights=config["rerank"]["weights"], top_k=int(config["rerank"]["top_k"]))
    csv_columns = ["rank", "molecule_id", "name", "smiles", "formula", "final_score", "ir_similarity", "raman_similarity", "peak_score", "mass_score", "validity", "notes"]
    topk_csv_path = output_dir / "topk_candidates.csv"
    topk_df.loc[:, csv_columns].to_csv(topk_csv_path, index=False)

    generated_files = []
    generated_files.append(plot_query_spectra(query_ir, query_raman, output_dir / "query_spectra.png"))
    if query_encoding.ir is not None:
        generated_files.append(plot_attention_heatmap(query_encoding.ir.attention, "IR intra-modal attention", output_dir / "attention_heatmap_ir.png"))
    if query_encoding.raman is not None:
        generated_files.append(plot_attention_heatmap(query_encoding.raman.attention, "Raman intra-modal attention", output_dir / "attention_heatmap_raman.png"))

    best = topk_df.iloc[0]
    best_row = next(row for row in scoring_rows if row["molecule_id"] == best["molecule_id"])
    generated_files.append(plot_spectrum_match(query_ir, best_row["candidate_ir"], f"Top-1 IR match: {best['name']}", output_dir / "spectrum_match_top1.png"))
    overlay_candidates = [(f"#{int(row['rank'])} {row['name']}", next(item for item in scoring_rows if item['molecule_id'] == row['molecule_id'])["candidate_ir"]) for _, row in topk_df.iterrows()]
    generated_files.append(plot_topk_overlay(query_ir, overlay_candidates, "Top-K IR overlay", output_dir / "topk_spectrum_overlay.png"))
    generated_files.append(plot_candidate_grid(topk_df.loc[:, csv_columns], output_dir / "molecule_grid.png"))
    generated_files.append(plot_candidate_grid(topk_df.loc[:, csv_columns], output_dir / "topk_candidates.png"))
    generated_files.append(plot_demo_summary_panel(topk_df.loc[:, csv_columns], output_dir / "demo_summary_panel.png"))

    warnings = []
    fallback_modes = []
    env_packages = []
    try:
        import rdkit  # noqa: F401
        env_packages.append("rdkit")
    except Exception:
        warnings.append("RDKit unavailable: using text-only molecule visualization and fallback validity scoring.")
        fallback_modes.append("no_rdkit")
    try:
        import torch  # noqa: F401
        env_packages.append("torch")
    except Exception:
        warnings.append("PyTorch unavailable: using deterministic NumPy encoder.")
        fallback_modes.append("numpy_encoder")

    trace_log = build_trace_log(
        project_root=ROOT,
        config_path=config_path,
        config=config,
        input_files={
            "query_ir": str(ROOT / input_cfg["query_ir"]),
            "query_raman": str(ROOT / input_cfg["query_raman"]),
            "candidate_library": str(ROOT / input_cfg["candidate_library"]),
        },
        preprocessing_params={**preprocess_cfg, "grid_size": int(len(common_grid))},
        modalities_used=[m for m, spec in [("ir", query_ir), ("raman", query_raman)] if spec is not None],
        candidate_library_size=int(len(candidate_library)),
        filtering_steps=filtering_steps,
        scoring_weights=config["rerank"]["weights"],
        topk_candidates=topk_df.loc[:, csv_columns].to_dict(orient="records"),
        output_files=[str(topk_csv_path)] + generated_files + [str(output_dir / "trace_log.json")],
        warnings=warnings,
        fallback_modes=fallback_modes,
    )
    trace_path = output_dir / "trace_log.json"
    write_trace_log(trace_path, trace_log)
    generated_files.append(str(trace_path))

    if config["outputs"].get("write_demo_summary_md", True):
        summary_md = output_dir / "demo_summary.md"
        write_demo_summary_markdown(summary_md, topk_df.loc[:, csv_columns], trace_log)
        generated_files.append(str(summary_md))

    print("MultiSpec-GeoDiff demo complete")
    print(json.dumps({
        "top_candidate": topk_df.iloc[0]["name"],
        "top_score": round(float(topk_df.iloc[0]["final_score"]), 4),
        "output_dir": str(output_dir),
        "generated_files": generated_files,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
