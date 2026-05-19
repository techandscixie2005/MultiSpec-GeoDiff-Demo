# Figure placeholders

This directory intentionally contains **compilable placeholder `.tex` files** instead of external image assets.

Current placeholders:
- `placeholder_overall_architecture.tex`
- `placeholder_multimodal_encoder.tex`
- `placeholder_demo_pipeline.tex`
- `placeholder_future_geometry.tex`
- `placeholder_output_visualization.tex`

## How to replace later
1. Export your final figure as PDF/PNG/SVG-to-PDF.
2. Replace the corresponding `\input{figures/placeholder_*.tex}` call in the section file with `\includegraphics[width=...]`.
3. Keep the existing captions unless the figure semantics change.

## Suggested final figure mapping
- overall architecture
- coordinate-aware multi-modal encoder
- MVP candidate generation and reranking pipeline
- future 2D-to-3D + TFN-Transformer extension
- expected demo output panel
