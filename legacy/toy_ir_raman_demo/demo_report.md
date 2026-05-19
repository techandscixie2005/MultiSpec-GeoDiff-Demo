# MultiSpec-GeoDiff Demo Report

## 1. 报告范围
本文档只描述**已实现 MVP** 和**roadmap-only 模块**的真实状态。它不把当前 demo 写成已训练的 diffusion 系统，也不把 toy 3D preview 写成正式三维恢复结果。

## 2. 已实现的 MVP

| 模块 | 文件 | 状态 | 说明 |
|---|---|---|---|
| 输入与预处理 | `spectra_preprocess.py`、`data_io.py` | 已实现 | 重采样、平滑、归一化、峰提取、CSV 校验 |
| 坐标感知编码 | `spectra_encoder.py` | 已实现 | 真实横坐标特征 + 模态内 attention + 跨模态融合 |
| 候选召回 | `candidate_generator.py` | 已实现 | 谱图相似度、embedding 相似度、质量软过滤 |
| 前向评分 | `forward_scorer.py` | 已实现 | 峰匹配、质量匹配、RDKit 兼容 validity |
| 重排 | `reranker.py` | 已实现 | 加权 Top-K 排序 |
| 可视化 | `visualization.py` | 已实现 | 输入谱图、重叠图、候选网格、summary panel |
| 追踪记录 | `trace_logger.py` | 已实现 | 记录 git hash、配置快照、环境、警告和输出文件 |
| 运行编排 | `scripts/run_demo.py` | 已实现 | 一键生成样本数据并完成全流程 |

当前 MVP 的输入是 `data/samples/query_ir.csv` 与 `data/samples/query_raman.csv`，候选库是 `data/samples/candidate_library.csv`。  
当前 MVP 的输出是 `outputs/topk_candidates.csv`、`outputs/trace_log.json`、若干图像文件和 `outputs/demo_summary.md`。

## 3. Roadmap-only 模块

| 模块 | 文件 | 真实状态 | 说明 |
|---|---|---|---|
| Size-adaptive graph diffusion | `future_modules/graph_diffusion_stub.py` | 仅接口 | 只返回计划信息，不做采样训练 |
| Pairwise distance bridge | `future_modules/pairwise_distance_stub.py` | 仅接口 | 只返回启发式矩阵形状，未训练预测器 |
| Parity-aware TFN-Transformer | `future_modules/tfn_transformer_stub.py` | 仅接口 | 只返回元数据，不做几何精修 |
| Toy geometry preview | `geometry.py` | 仅示意 | classical MDS 只是可视化预览，不是 3D 恢复模块 |

## 4. 运行结果与验证

已验证的运行命令：

```bash
python -m pip install -r requirements.txt
python scripts/run_demo.py --config configs/demo.yaml
python scripts/validate_outputs.py --output_dir outputs
```

在当前 bundled toy data 上，demo 的 Top-1 候选为 **phenol**；其余名次会随权重配置与候选库设置而变化，但会稳定输出可审查的 Top-K 排名表。  
这只是一个**样例闭环 sanity check**，不是 benchmark 结论，也不是实验域性能声明。

当前环境缺少 `rdkit` 和 `torch` 时，demo 会自动降级为：

- `no_rdkit`：使用文本化结构面板和 fallback validity 分数
- `numpy_encoder`：使用确定性的 NumPy 编码器

## 5. 结论
这版仓库已经能交付一个**可运行、可验证、边界清晰**的 demo。后续扩展必须继续保留“已实现 MVP”和“roadmap-only”之间的明确分界，不能把 stub、示意图或未来模块写成已完成模型。
