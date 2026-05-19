# MultiSpec-GeoDiff Demo

MultiSpec-GeoDiff 是一个面向 **谱图驱动分子结构反演** 的可追踪 demo 文档包。当前仓库的**已实现 MVP** 只覆盖 IR/Raman 双模态闭环：谱图预处理 → 坐标感知编码 → 候选召回 → 前向谱图评分 → 重排 → 可视化与 trace log。  
`graph diffusion`、`pairwise distance` 和 `TFN-Transformer` 只保留为 **roadmap-only** 接口，不应写成已训练完成的能力。

## 当前状态

| 层级 | 状态 | 说明 |
|---|---|---|
| IR/Raman 预处理与重采样 | 已实现 | 见 `src/multispec_geodiff/spectra_preprocess.py` |
| 坐标感知编码器 | 已实现 | 见 `src/multispec_geodiff/spectra_encoder.py` |
| 候选召回与软过滤 | 已实现 | 见 `src/multispec_geodiff/candidate_generator.py` |
| 前向谱图评分 | 已实现 | 见 `src/multispec_geodiff/forward_scorer.py` |
| 重排与 Top-K 输出 | 已实现 | 见 `src/multispec_geodiff/reranker.py` |
| 可视化与 trace log | 已实现 | 见 `src/multispec_geodiff/visualization.py` 和 `trace_logger.py` |
| 3D 恢复 / 图扩散 / TFN 精修 | roadmap-only | 见 `src/multispec_geodiff/future_modules/` |

## 仓库结构

```text
.
├── README.md
├── proposal.md
├── demo_report.md
├── design_overview.md
├── docs/
│   ├── implementation_notes.md
│   ├── demo_usage.md
│   └── limitations_and_roadmap.md
├── configs/demo.yaml
├── data/samples/
├── outputs/
├── scripts/
├── src/multispec_geodiff/
└── latex/
```

## 一键快速开始

先安装依赖：

```bash
python -m pip install -r requirements.txt
```

依赖装好后，真正的一键运行命令就是：

```bash
python scripts/run_demo.py --config configs/demo.yaml
```

验证输出是否齐全：

```bash
python scripts/validate_outputs.py --output_dir outputs
```

## 运行后会生成什么

`scripts/run_demo.py` 会自动生成或刷新 `data/samples/` 和 `outputs/`，并写出以下关键产物：

- `outputs/topk_candidates.csv`
- `outputs/trace_log.json`
- `outputs/query_spectra.png`
- `outputs/spectrum_match_top1.png`
- `outputs/topk_spectrum_overlay.png`
- `outputs/molecule_grid.png`
- `outputs/demo_summary.md`

## 这版 demo 只承诺什么

- 只承诺一个**可运行、可验证、可追踪**的 IR/Raman 候选排序闭环。
- 只承诺当前仓库里的实现模块，不把 roadmap stubs 写成训练完成的模型。
- 只把 `geometry.py` 里的 classical MDS 当作**toy preview**，不把它当作真实 3D 恢复系统。

## 参考文档

- `proposal.md`：提交用项目提议
- `design_overview.md`：实现边界与模块设计
- `demo_report.md`：提交用简版技术报告
- `docs/demo_usage.md`：运行说明
- `docs/limitations_and_roadmap.md`：当前限制与路线图
- `docs/implementation_notes.md`：实现备注与输出约定
- `latex/submission.pdf`：submission 版 PDF brief
