# MultiSpec-GeoDiff：谱图驱动分子结构反演引擎 — Demo 说明

## 一、任务目标、定位与价值

本 Demo 是 MultiSpec-GeoDiff 项目的 Stage-I 基线演示，目标为验证：一个规模可控、可追溯、可执行的最小管线，能否在真实实验 IR 光谱上支撑"候选检索 + 前向谱图一致性重排"闭环，并为后续图扩散生成与三维几何细化提供可集成的编码器基础。

完整提议覆盖从多模态谱图编码、图扩散生成、pairwise 距离预测到 SE(3) 等变细化的端到端管线。本次提交仅聚焦 Stage-I，不包含训练完备的生成模型或生产级未知物鉴定系统。

## 二、已完成的核心能力

### 1. 实验 IR 谱图读取与预处理能力

从 JSONL 加载 200 条 NIST 实验 IR 光谱，含 SMILES、波数、强度、数据来源字段。预处理包括公共网格插值（400–4000 cm⁻¹，1800 点）、强度归一化、峰检测。

### 2. 坐标感知谱图编码能力

将 IR 波数轴视为物理坐标而非无结构序列索引。嵌入层将强度投影与傅里叶波数位置编码相加：h_i = W_I · I_i + p(ν_i)，使模型天然感知每个数据点在光谱中的物理位置。

### 3. 距离偏置 attention 能力

在自注意力中注入波数间距依赖的 RBF 偏置 b(|ν_i − ν_j|)，使物理邻近谱区的注意力权重增强、远离谱区被抑制。这一先验不依赖大规模训练数据，直接编码了红外光谱的物理结构。

### 4. 候选分子检索与排序能力

对查询光谱与候选库（199 个分子）做余弦相似度检索，返回 Top-M 候选。

### 5. 前向谱图一致性重排能力

对检索候选进行三指标前向谱图对比（余弦相似度 + L1 距离 + 峰匹配），以加权得分重排，验证"候选提出 + 前向一致性校验"闭环。

### 6. Top-K 可视化与 trace log 输出能力

输出 Top-K 候选排序表、谱图叠加图、分子结构网格、注意力热图及完整 JSON 执行日志。

### 7. 测试与 CI 验证能力

22 个 pytest 测试全部通过，GitHub Actions CI 配置就绪，所有输出可复现。

## 三、Demo 输出示例说明

### 输入

- **数据文件：** `04_data/IR_nist_200.jsonl`（200 条 NIST 实验 IR 光谱）
- **查询方式：** 按 `query_id` 从库中选取一条作为查询，其余 199 条构成候选库
- **参数：** `top_k = 5`, `top_m = 20`

### 运行命令

```bash
# 安装依赖
pip install -r 03_code/requirements.txt

# 运行 Stage-I Demo
python 03_code/run_demo.py --query_id 0 --data 04_data/IR_nist_200.jsonl --top_k 5

# 运行测试
PYTHONPATH=03_code/src pytest 03_code/tests -q
```

### 结构化输出示例

```json
{
  "meta": {
    "demo_version": "stage1-ir-rerank-v1",
    "is_baseline": true,
    "note": "当前 Demo 实现 Stage-I IR 检索与重排；图扩散与 TFN-Transformer 为 roadmap-only。"
  },
  "input": {
    "query_id": 0,
    "modality": ["IR"],
    "data_source": "NIST IR subset",
    "num_library_spectra": 200
  },
  "topk_candidates": [
    {
      "rank": 1,
      "molecule_id": "example_candidate",
      "smiles": "COC(=O)C(Cl)(CCl)CCl",
      "formula": "C5H7Cl3O2",
      "retrieval_score": 1.0,
      "cosine_similarity": 0.9747,
      "forward_score": 0.8797,
      "final_score": 0.9158,
      "is_ground_truth": false,
      "reason": "经前向谱图重排后的最优候选，综合余弦、L1 与峰匹配得分。"
    }
  ],
  "outputs": {
    "topk_table": "05_outputs/topk_candidates.csv",
    "spectrum_overlay": "05_outputs/spectrum_overlay.png",
    "molecule_grid": "05_outputs/molecule_grid.png",
    "attention_heatmap": "05_outputs/attention_heatmap.png",
    "ablation_results": "05_outputs/ablation_results.csv",
    "trace_log": "05_outputs/trace_log.json"
  },
  "boundary": {
    "implemented": [
      "IR 谱图预处理（插值、归一化、峰检测）",
      "坐标感知谱图编码",
      "距离偏置 attention",
      "候选分子检索",
      "前向谱图重排",
      "Top-K 可视化",
      "trace log",
      "CI 测试"
    ],
    "roadmap_only": [
      "图扩散生成",
      "pairwise 距离预测",
      "TFN-Transformer 三维细化",
      "宇称/手性感知张量通道"
    ]
  }
}
```

## 四、核心原理

### 4.1 谱图坐标感知编码

IR 光谱每个数据点同时携带强度信息 I_i 与物理波数坐标 ν_i。嵌入层将强度投影与傅里叶波数位置编码相加，使模型天然感知每个点的物理位置。这是区别于普通 Transformer 将光谱视为无结构序列的核心设计。

### 4.2 谱图距离偏置注意力

物理邻近的波数区域具有更强的本征相关性。在注意力中注入波数间距 RBF 偏置，使得远离谱区之间的注意力权重被物理先验抑制。这与 Graph Transformer 中注入拓扑距离偏置的思路一致，但应用对象是光谱物理坐标。

### 4.3 候选检索与前向谱图重排

当前 Demo 使用检索（而非训练完成的图扩散模型）产生候选——这是 Stage-I 的刻意简化。候选经前向谱图三指标对比重排，验证"候选提出 + 前向一致性校验"闭环在真实实验数据上的可行性。

### 4.4 Roadmap：图扩散 → pairwise 距离 → TFN-Transformer 细化（未来阶段）

- **Stage-II：** 大小自适应图扩散生成二维分子图，pairwise 距离矩阵预测实现 2D→3D 桥接
- **Stage-III：** EGNN / TFN-Transformer 在三维空间同步更新原子坐标与多阶张量特征，引入宇称分辨通道（0e, 0o, 1e, 1o, 2e, 2o）为手性敏感模态预留接口

以上为 roadmap 规划，当前 Demo 未包含训练完成的图扩散或 TFN-Transformer 模型。

## 五、当前 Baseline 的能力边界

- 当前仅使用 IR 单一模态；Raman、NMR、MS 等未接入
- 候选生成使用检索/重排而非训练完成的图扩散模型
- 无训练完成的 TFN-Transformer 三维细化模型
- 无法在手性敏感光谱（CD/VCD/ROA）缺失时可靠推断手性
- 候选质量受限于候选库覆盖范围（199 个分子）
- 200 条数据仅支撑 Demo 级可行性验证，不构成统计性能声明

## 六、Baseline 的价值与验证结论

本基线证明：

- 真实实验 IR 光谱可进入可复现的检索/重排闭环
- 坐标感知谱图编码与距离偏置注意力可实现，并可作为后续阶段的编码器基础
- 检索 → 重排形成最小谱图→候选分子环路
- 代码接口为检索 → 图扩散 → 几何细化的逐步替换提供了清晰基础

## 七、下一步可演进方向

| 阶段 | 内容 |
|------|------|
| Stage-II | 大小自适应图扩散（insert/delete）实现 de-novo 二维分子图生成 |
| Stage-II/III 桥接 | pairwise 距离矩阵预测 + distance geometry 三维初始化 |
| Stage-III | EGNN / TFN-Transformer 坐标与张量细化，宇称分辨通道 |
| 多模态扩展 | Raman, NMR, MS, UV, CD/VCD/ROA |
| 后验评分 | 不确定性感知的重排与置信度估计 |

## 当前效果

| 项目 | 状态 |
|------|------|
| 数据规模 | 200 条 NIST 实验 IR 光谱 |
| Top-K 设置 | K = 5 |
| 前向重排 | 已实现（余弦 + L1 + 峰匹配） |
| trace log | 已实现（`05_outputs/trace_log.json`） |
| 可视化 | 已实现（谱图叠加、分子网格、注意力热图） |
| pytest | 22 个测试全部通过 |
| CI | GitHub Actions 通过 |
| 图扩散 | roadmap-only |
| TFN-Transformer | roadmap-only |

## 验证数据集

- **文件：** `04_data/IR_nist_200.jsonl`
- **规模：** 200 条 NIST 实验 IR 光谱
- **格式：** 每行 JSON，含 SMILES / 波数 / 强度 / 来源
- **候选库：** `04_data/candidate_library.csv`
- **数据卡片：** `04_data/data_card.md`

## 问题分析

### 1. 数据规模有限

200 条光谱仅支撑 Demo 级验证，不足以得出统计显著的检索性能结论。扩展至 QM9S 及公开多模态谱图数据库是后续阶段的必要步骤。

### 2. 谱图反演一谱多解

同一 IR 光谱可对应多个合理分子结构。当前检索/重排策略无法区分这些等价解，需图扩散生成与后验评分共同应对。

### 3. 当前不是完整生成模型

候选来自预定义库的检索，非 de-novo 图扩散生成。当目标分子不在候选库中时，系统无法提出正确候选。

### 4. 当前 3D 信息不足

IR 光谱对三维构象敏感，但当前管线仅使用一维谱图，未接入三维几何建模。

### 5. 候选库覆盖限制

检索质量完全取决于候选库的化学空间覆盖。199 个分子的库远不足以覆盖有意义的化学多样性。
