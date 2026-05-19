# Implementation Notes

本文档记录实现侧的真实约定，目的是避免把 demo 的 **MVP** 和 **roadmap-only** 混写。

## 1. 运行链路

默认入口是：

```bash
python scripts/run_demo.py --config configs/demo.yaml
```

这个命令会按顺序做四件事：

1. 生成或刷新 `data/samples/`
2. 读取 `configs/demo.yaml`
3. 运行 IR/Raman 预处理、编码、召回、评分和重排
4. 写出 `outputs/` 下的 CSV、PNG、JSON 和 summary markdown

## 2. 默认数据

- `scripts/generate_sample_data.py` 生成合成 Gaussian spectra。
- `data/samples/query_ir.csv` 和 `data/samples/query_raman.csv` 是查询样本。
- `data/samples/candidate_library.csv` 是候选库。
- 当前示例不是实验数据集，因此不要把结果写成实验 benchmark 结论。

## 3. 模块责任边界

| 文件 | 责任 | 备注 |
|---|---|---|
| `spectra_preprocess.py` | 读入后预处理 | 负责重采样、平滑、归一化、峰提取 |
| `spectra_encoder.py` | 特征编码 | 负责坐标感知 token、attention 和融合 |
| `candidate_generator.py` | 候选召回 | 负责谱图相似度、embedding 相似度和软过滤 |
| `forward_scorer.py` | 前向打分 | 负责峰、质量、formula 和 validity 组合 |
| `reranker.py` | 最终排序 | 负责 weighted score 和 Top-K |
| `visualization.py` | 可视化 | 负责谱图、叠图、候选面板和 summary panel |
| `trace_logger.py` | 追踪信息 | 负责 git hash、环境、配置快照、输出文件列表 |
| `future_modules/*` | roadmap 接口 | 只保留未来模块的签名和说明 |

## 4. fallback 行为

当前环境如果没有 `rdkit` 或 `torch`，demo 仍然成功运行，但会自动切换到：

- `no_rdkit`：使用文本化结构展示和 fallback validity
- `numpy_encoder`：使用确定性的 NumPy 编码器

这不是错误，而是设计好的降级路径。  
文档里不要把这些 fallback 写成失败。

## 5. 输出约定

当前验证脚本要求存在：

- `outputs/topk_candidates.csv`
- `outputs/query_spectra.png`
- `outputs/spectrum_match_top1.png`
- `outputs/topk_spectrum_overlay.png`
- `outputs/molecule_grid.png`
- `outputs/attention_heatmap_ir.png`
- `outputs/attention_heatmap_raman.png`
- `outputs/demo_summary_panel.png`
- `outputs/trace_log.json`

其中 `topk_candidates.csv` 至少应包含：

`rank, molecule_id, name, smiles, formula, final_score, ir_similarity, raman_similarity, peak_score, mass_score, validity, notes`

## 6. 当前 sanity check 结果

在固定种子和 bundled toy data 上，当前 Top-1 是 `phenol`。  
这个结果只是验证流程正确，不应被写成泛化性能结论。

## 7. 修改建议

- 如果改 `configs/demo.yaml`，要同步检查 `validate_outputs.py` 和 README 里的 quick start。
- 如果增加新输出文件，要同步更新验证脚本和文档。
- 如果未来用真实实验谱图替换 toy data，要明确重新分开“样例数据”和“实验数据”。
