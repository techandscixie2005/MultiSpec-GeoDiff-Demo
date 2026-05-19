# MultiSpec-GeoDiff Demo

MultiSpec-GeoDiff 是一个面向 **谱图驱动分子结构反演** 的可追踪 demo 文档包。当前仓库的**已实现 MVP** 只覆盖 IR/Raman 双模态闭环：

> **谱图预处理 → 坐标感知编码 → 候选召回 → 前向谱图评分 → 重排 → 可视化与 trace log**

`graph diffusion`、`pairwise distance` 和 `TFN-Transformer` 只保留为 **roadmap-only** 接口，不应写成已训练完成的能力。

---

## Reviewer Quick Start

```bash
git clone https://github.com/techandscixie2005/MultiSpec-GeoDiff-Demo.git
cd MultiSpec-GeoDiff-Demo
python -m pip install -r requirements.txt
python scripts/run_demo.py --config configs/demo.yaml
python scripts/validate_outputs.py --output_dir outputs
python -m pytest -q
```

只需以上五条命令即可完成克隆、运行、验证与测试。依赖为轻量数值库；RDKit 和 PyTorch **不是必需的**——缺失时会自动降级。

---

## What this demo proves / What it does not prove

| Proves ✅ | Does not prove ❌ |
|---|---|
| Runnable IR/Raman → Top-K candidate ranking workflow | Real experimental benchmark performance |
| Coordinate-aware spectral representation with $\Delta\nu$ relative-distance bias attention | Trained graph diffusion for de-novo molecule generation |
| Forward-spectrum consistency reranking (peak matching, mass/formula scoring) | Trained pairwise distance prediction as 2D-to-3D bridge |
| Traceable, inspectable outputs (CSV, JSON, figures, trace log) | Trained TFN-Transformer parity-aware geometry refinement |

---

## Central design narrative

**This demo does not treat spectrum-to-structure as one-shot SMILES translation. It treats it as candidate generation/retrieval plus forward-spectrum consistency reranking in a traceable scientific workflow.**

The current **candidate generator** is the interface-level substitute for future graph diffusion; the current **forward scorer** is the interface-level substitute for future 3D tensor-spectrum scoring; the **trace log** is the audit interface for future agentic scientific workflows.

---

## Reviewer-facing outputs

| File | Description |
|---|---|
| `outputs/topk_candidates.csv` | Top-K ranked candidates with per-component scores |
| `outputs/trace_log.json` | Structured execution trace: git hash, config, environment, warnings |
| `outputs/query_spectra.png` | Input IR and Raman query spectra |
| `outputs/spectrum_match_top1.png` | Top-1 candidate spectrum vs query overlay |
| `outputs/topk_spectrum_overlay.png` | All Top-K candidate spectra overlaid with query |
| `outputs/molecule_grid.png` | Preview grid of candidate molecules |
| `outputs/topk_candidates.png` | Ranked candidate grid with scores |
| `outputs/attention_heatmap_ir.png` | IR intra-modal attention heatmap |
| `outputs/attention_heatmap_raman.png` | Raman intra-modal attention heatmap |
| `outputs/demo_summary_panel.png` | Consolidated visual summary panel |
| `outputs/demo_summary.md` | One-line markdown summary of results |

---

## 当前状态

| 层级 | 状态 | 说明 |
|---|---|---|
| IR/Raman 预处理与重采样 | ✅ 已实现 | `src/multispec_geodiff/spectra_preprocess.py` |
| 坐标感知编码器 | ✅ 已实现 | `src/multispec_geodiff/spectra_encoder.py` |
| 候选召回与软过滤 | ✅ 已实现 | `src/multispec_geodiff/candidate_generator.py` |
| 前向谱图评分 | ✅ 已实现 | `src/multispec_geodiff/forward_scorer.py` |
| 重排与 Top-K 输出 | ✅ 已实现 | `src/multispec_geodiff/reranker.py` |
| 可视化与 trace log | ✅ 已实现 | `visualization.py` + `trace_logger.py` |
| Toy geometry MDS preview | 🧪 仅示意 | `geometry.py` — classical MDS 仅供预览，非 3D 恢复 |
| Graph diffusion | 🔜 roadmap-only | `future_modules/graph_diffusion_stub.py` |
| Pairwise distance bridge | 🔜 roadmap-only | `future_modules/pairwise_distance_stub.py` |
| TFN-Transformer refinement | 🔜 roadmap-only | `future_modules/tfn_transformer_stub.py` |

---

## 仓库结构

```text
.
├── README.md
├── proposal.md              # 项目提议（~500 字中文）
├── demo_report.md           # Demo 技术报告
├── design_overview.md       # 实现边界与模块设计
├── FINAL_DEMO_REPORT.md     # 最终审计报告
├── configs/demo.yaml        # 唯一默认配置
├── scripts/
│   ├── run_demo.py          # 一键运行全流程
│   └── validate_outputs.py  # 输出完整性校验
├── src/multispec_geodiff/   # 核心模块源码
├── tests/                   # pytest 测试（17 passed）
├── outputs/                 # 生成输出目录
├── notebooks/               # Jupyter notebook
├── docs/                    # 补充文档
└── latex/                   # 提交用 PDF 源文件
```

## 参考文档

- `proposal.md` — 提交用项目提议（中文）
- `design_overview.md` — 实现边界与模块设计
- `demo_report.md` — 提交用简版技术报告
- `docs/demo_usage.md` — 运行说明
- `docs/limitations_and_roadmap.md` — 当前限制与路线图
- `docs/implementation_notes.md` — 实现备注与输出约定
- `latex/submission.pdf` — submission 版 PDF brief
