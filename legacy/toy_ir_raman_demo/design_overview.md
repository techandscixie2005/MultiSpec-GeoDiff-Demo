# Design Overview

## 1. 设计边界
MultiSpec-GeoDiff 当前的真实系统边界是：

```text
IR/Raman query spectra
-> preprocessing
-> coordinate-aware encoder
-> candidate retrieval
-> forward scoring
-> reranking
-> visual outputs + trace log
```

这条链路已经在 `scripts/run_demo.py` 中闭合。  
`graph diffusion`、`pairwise distance` 和 `TFN-Transformer` 只保留接口，不应描述为已训练完成的生成系统。

## 2. 已实现模块地图

| 模块 | 文件 | 作用 | 状态 |
|---|---|---|---|
| 数据读取与校验 | `data_io.py` | 读 CSV / YAML，检查谱图和候选库字段 | 已实现 |
| 谱图预处理 | `spectra_preprocess.py` | 重采样、平滑、归一化、峰提取 | 已实现 |
| 坐标感知编码器 | `spectra_encoder.py` | 真实横坐标特征、RBF bias、intra-modal attention、cross-modal fusion | 已实现 |
| 候选召回 | `candidate_generator.py` | 结合谱图相似度和 embedding 相似度召回候选 | 已实现 |
| 前向评分 | `forward_scorer.py` | 峰匹配、质量相似度、formula match、validity | 已实现 |
| 重排 | `reranker.py` | 计算最终分数并输出 Top-K | 已实现 |
| 可视化 | `visualization.py` | 查询谱图、叠图、候选网格、summary panel | 已实现 |
| trace 记录 | `trace_logger.py` | 生成带环境和配置快照的 JSON trace | 已实现 |
| 运行脚本 | `scripts/run_demo.py` | 串起全流程并写入 `outputs/` | 已实现 |
| 输出校验 | `scripts/validate_outputs.py` | 检查文件、CSV 列和 trace 键 | 已实现 |
| 样本数据 | `scripts/generate_sample_data.py` | 生成 toy Gaussian spectra 和候选库 | 已实现 |

## 3. Roadmap-only 模块地图

| 模块 | 文件 | 状态 | 备注 |
|---|---|---|---|
| Size-adaptive graph diffusion | `future_modules/graph_diffusion_stub.py` | roadmap-only | 只定义可插入/删除节点等操作名 |
| Pairwise distance head | `future_modules/pairwise_distance_stub.py` | roadmap-only | 只提供接口和启发式矩阵 |
| Parity-aware TFN refiner | `future_modules/tfn_transformer_stub.py` | roadmap-only | 只保留 parity channel 与 forward 接口 |
| 3D 恢复 | `geometry.py` | toy helper | classical MDS 仅用于示意或预览 |

## 4. 设计原则

1. **候选优先，不做过度承诺**  
   当前系统输出 Top-K 候选，而不是唯一答案。

2. **真实坐标优先**  
   谱图横坐标进入 encoder，而不是被当作普通 token 序号。

3. **后验重排优先于“更大的生成器”**  
   当前 MVP 的实际收益来自检索、前向评分和重排闭环。

4. **失败要可见**  
   trace log 必须记录 warnings、fallback modes、环境和配置快照。

5. **stub 只能是 stub**  
   roadmap 模块可以被引用和说明，但不能被写成已实现能力。

## 5. 当前实现的可复现性
`configs/demo.yaml` 是当前 demo 的唯一默认配置入口。它会触发样本数据生成、候选库加载、编码、召回、重排和输出写入。  
当前样例数据是合成 Gaussian spectra，不是实验 benchmark 数据；因此任何结果都应被解释为**demo sanity check**，不是论文级性能宣称。
