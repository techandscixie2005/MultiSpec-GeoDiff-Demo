# MultiSpec-GeoDiff — Demo 说明

> **DP Technology 实习最终提交 | 2026-05**
>
> GitHub 仓库：[https://github.com/techandscixie2005/MultiSpec-GeoDiff-Demo](https://github.com/techandscixie2005/MultiSpec-GeoDiff-Demo)

---

## 一、任务目标、定位与价值

本 Demo 对应提交的项目提议 **"MultiSpec-GeoDiff: 多模态谱图驱动的几何分子结构反演"**（详见 `01_project_proposal/proposal.md`）。

当前实现为 **Stage-I 可行性闭环**：给定一张实验红外（IR）光谱，系统通过坐标感知谱图编码、谱图距离偏置注意力、候选检索与前向谱图一致性重排，从 200 个 NIST 实验光谱库中输出 Top-K 候选分子结构。

核心流程：

```
实验 IR 光谱 → 坐标感知谱图编码 → 候选检索 → 前向谱图重排 → Top-K 分子结构
```

本 Demo 的定位是**工作流可行性验证**，而非训练完备的生成模型或生产级未知物鉴定系统。GitHub 仓库是主要提交入口。

---

## 二、已完成的核心能力

以下能力已在 Stage-I 中实现并可通过命令行复现：

| # | 能力 | 说明 |
|---|------|------|
| 1 | NIST 实验 IR JSONL 数据加载 | 200 条实验光谱，含 SMILES、波数、强度 |
| 2 | 谱图插值与归一化 | 统一插值至 400–4000 cm^{-1}，1800 点公共网格 |
| 3 | 坐标感知谱图编码 | 强度投影 + 傅里叶波数位置编码 |
| 4 | 谱图距离偏置注意力 | 自注意力注入波数间距 RBF 偏置 |
| 5 | 候选分子检索 | 全局嵌入池化 + 余弦相似度 Top-M 检索 |
| 6 | 前向谱图一致性重排 | 余弦 + L1 + 峰匹配三指标加权重排 |
| 7 | Top-K 候选输出与可视化 | CSV 表格 + 谱图叠加 + 分子网格 |
| 8 | 注意力热图可视化 | 波数维度的注意力权重矩阵 |
| 9 | Trace log 生成 | 完整执行路径与元数据 JSON 记录 |
| 10 | 本地测试与 GitHub Actions CI | 22 个测试，CI 配置于 `.github/workflows/tests.yml` |

**关于数据规模：** 当前使用 200 张 NIST 实验 IR 光谱进行 Demo 级可行性验证。这些数据**未用于训练完整的生成模型**，仅用于检索/重排闭环演示。

---

## 三、Demo 输入输出示例说明

### 输入

| 项目 | 内容 |
|------|------|
| 数据文件 | `04_data/IR_nist_200.jsonl`（200 条 NIST IR 光谱） |
| 查询参数 | `query_id = 0`，`top_k = 5` |

查询分子从库中按 `query_id` 选取，其余 199 个分子构成候选库。

### 运行命令

```bash
# 运行 Demo
python 03_code/run_demo.py --query_id 0 --data 04_data/IR_nist_200.jsonl --top_k 5

# 运行测试
PYTHONPATH=03_code/src pytest 03_code/tests -q
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `05_outputs/topk_candidates.csv` | Top-K 候选分子排序表（含各指标得分） |
| `05_outputs/spectrum_overlay.png` | 查询光谱与 Top-K 候选光谱叠加对比 |
| `05_outputs/molecule_grid.png` | Top-K 候选分子的二维结构图 |
| `05_outputs/attention_heatmap.png` | 谱图距离偏置注意力权重热图 |
| `05_outputs/ablation_results.csv` | 消融对比结果（Baseline vs Full） |
| `05_outputs/trace_log.json` | 完整执行追踪日志 |

### 输出示例图

![谱图叠加对比](../05_outputs/spectrum_overlay.png)

*图 1：查询光谱（黑色虚线）与 Top-K 候选光谱（彩色实线）叠加对比*

![候选分子结构](../05_outputs/molecule_grid.png)

*图 2：Top-K 候选分子的二维结构图*

![注意力热图](../05_outputs/attention_heatmap.png)

*图 3：谱图距离偏置自注意力权重矩阵*

---

## 四、核心原理：Stage-I 最小闭环

### 4.1 坐标感知谱图嵌入

与普通 Transformer 将输入视为无序 token 序列不同，IR 光谱的每个点同时携带**强度**与**波数物理坐标**。为此，嵌入层将强度投影与波数位置编码相加：

$$h_i = W_I \cdot I_i + p(\nu_i)$$

其中 $I_i$ 为归一化强度，$p(\nu_i)$ 为波数 $\nu_i$ 的傅里叶位置编码，$W_I$ 为可学习线性投影。

### 4.2 谱图距离偏置注意力

普通 Transformer 的 token 关系主要是语义层面的。但在 IR 光谱中，**物理上邻近的波数区域具有更强的相互作用**，这一先验不应由模型从头学习。因此，自注意力注入波数间距依赖的偏置项：

$$A_{ij} = \operatorname{softmax}_j\left(\frac{Q_i K_j^T}{\sqrt{d}} + b(|\nu_i - \nu_j|)\right)$$

其中 $b(\Delta\nu) = \exp(-(\Delta\nu / \sigma)^2)$ 为 RBF 核，随波数间距增大而衰减，使得远离的谱区之间的注意力权重被物理先验抑制。

这一设计的本质与 Graph Transformer 中注入拓扑距离偏置类似，但应用对象从图结构变为**光谱坐标**。

### 4.3 检索与重排评分

候选分子经编码与池化后，与查询嵌入计算余弦相似度进行初筛（Top-M），随后通过前向谱图对比进行精细重排（Top-K）：

$$\operatorname{Score}(G_k) = \lambda_1 \cdot \operatorname{CosSim}(S_q, S_k) + \lambda_2 \cdot \operatorname{PeakMatch}(S_q, S_k)$$

其中：
- $\operatorname{CosSim}(\cdot,\cdot)$ 为全谱向量余弦相似度
- $\operatorname{PeakMatch}(\cdot,\cdot)$ 为谱峰位置与强度的匹配得分
- $\lambda_1, \lambda_2$ 为加权系数

这一前向谱图一致性校验的设计思路，对应完整提议中"生成模块提出候选，前向谱图模块物理校验"的闭环思想。

---

## 五、当前 Demo 的能力边界

### 已实现

- Stage-I 检索/重排可行性闭环（200 条 NIST IR 光谱）

### 未作为训练模块实现

| 模块 | 说明 |
|------|------|
| 训练完成的图扩散生成器 | 候选生成当前使用检索替代，非 de-novo 生成 |
| 训练完成的 pairwise 距离预测器 | 距离预测头为接口桩 |
| 训练完成的宇称分辨 TFN-Transformer | TFN 模块为接口桩 |
| 完整多模态系统 | 当前仅使用 IR，Raman/NMR/MS/UV 未接入 |
| 可靠的未知物鉴定基准 | 库规模与评估协议不足以支撑生产级鉴定 |

### 未来模块的代码桩

为保持架构完整性，Stage-II 和 Stage-III 的关键模块已作为可导入的接口桩存在于代码库中：

- `03_code/src/distance_head.py` — 2D 分子图 → pairwise 距离矩阵（Stage-II）
- `03_code/src/tfn_transformer_stub.py` — SE(3) 等变张量场细化（Stage-III）

这些文件定义了清晰的接口签名与数据结构，为后续开发提供了明确的集成点。

---

## 六、验证结果与可复现性

### 执行环境

| 项目 | 值 |
|------|-----|
| Python | 3.10.18 |
| PyTorch | 2.8.0+cu126 |
| RDKit | 2025.03.6 |
| NumPy | 2.2.5 |
| SciPy | 1.15.3 |
| 平台 | Linux (WSL2) |

### 验证结果

| 项目 | 状态 |
|------|------|
| 本地 Demo 运行 | 成功 |
| 测试（22 个） | 全部通过 |
| GitHub Actions CI | 通过 |
| trace_log.json | 完整记录 |
| 输出文件 | 已提交至 `05_outputs/` |

**最新已知提交:** `main` HEAD: `2bc955d`

### Trace Log 摘要（`05_outputs/trace_log.json`）

| 字段 | 值 |
|------|-----|
| 光谱数量 | 200 |
| 查询 ID | 5 |
| 查询分子 | C/C(=C/CCC1(C)C2CC3C(C2)C31C)COC(O)=Nc1ccccc1[N+](=O)[O-] |
| Top-K | 3 |
| 已实现模块 | 8 个（data_loader, spectra_preprocess, spectra_encoder, spectral_attention, candidate_generator, reranker, visualization, metrics） |

### 消融对比（来自 `05_outputs/ablation_results.csv`）

| 策略 | Top-1 CosSim | 说明 |
|------|-------------|------|
| Baseline（原始余弦相似度） | ~0.97 | 无编码、无注意力 |
| Full（完整管线） | ~0.97 + 重排 | 编码 + 距离偏置注意力 + 前向重排 |

消融显示 Full 管线在前向重排后提供了除纯余弦相似度之外的额外排序信号（L1 距离、峰匹配），使候选排序更具物理可解释性。

---

## 七、下一步可演进方向

1. **图扩散候选生成（Stage-II）** — 将当前检索式候选生成替换为大小自适应图扩散模型，实现 de-novo 候选分子生成
2. **Pairwise 距离预测** — $G_{2D} \rightarrow D \rightarrow R^{(0)}$，以距离矩阵为 2D 拓扑到 3D 几何的桥梁
3. **EGNN/TFN-Transformer 3D 细化（Stage-III）** — 在三维空间中同时更新坐标与多阶张量特征
4. **多模态扩展** — 从单一 IR 扩展至 IR / Raman / NMR / MS / UV / CD / VCD
5. **手性敏感通道** — 若 CD/VCD/ROA 数据可用，引入 $0o$ 宇称通道编码手性信息

---

## 八、提交材料清单

| # | 材料 | 路径/链接 |
|---|------|----------|
| 1 | **GitHub 仓库** | [https://github.com/techandscixie2005/MultiSpec-GeoDiff-Demo](https://github.com/techandscixie2005/MultiSpec-GeoDiff-Demo) |
| 2 | 项目提议 | `01_project_proposal/proposal.md` |
| 3 | Demo 说明 PDF | `02_demo_document/MultiSpec-GeoDiff_Demo_说明.pdf` |
| 4 | Demo 说明 Markdown | `02_demo_document/demo_report.md`（本文件） |
| 5 | 代码 | `03_code/`（src + tests + run_demo.py） |
| 6 | 数据 | `04_data/IR_nist_200.jsonl`（200 条 NIST IR 光谱） |
| 7 | 输出 | `05_outputs/`（CSV, PNG, JSON） |
| 8 | Notebook | `06_notebook/demo_pipeline.ipynb` |
| 9 | CI | `.github/workflows/tests.yml`（GitHub Actions，已通过） |

**主命令：**
```bash
python 03_code/run_demo.py --query_id 0 --data 04_data/IR_nist_200.jsonl --top_k 5
```

**测试命令：**
```bash
PYTHONPATH=03_code/src pytest 03_code/tests -q
```
