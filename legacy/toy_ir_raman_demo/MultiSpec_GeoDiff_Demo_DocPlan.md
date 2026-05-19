# MultiSpec-GeoDiff Demo 文档层面规划

> 目标：先完成项目 demo 的文档体系设计，后续再据此补齐 LaTeX/Markdown 文档、图示、README、报告和代码说明。本文档不是最终提交稿，而是整个 demo 文档工程的写作蓝图。

---

## 0. 总体定位

### 项目名称

**MultiSpec-GeoDiff：多模态谱图驱动的分子结构反演 Demo**

### 一句话定位

将实验 IR/Raman/NMR/MS 等多模态谱图作为输入，通过横坐标感知的谱图编码、候选分子生成/检索、前向谱图一致性重排，输出 Top-K 候选分子结构，并为未来扩展到 graph diffusion、pairwise distance 和 TFN-Transformer 三维几何精修提供清晰路线。

### 文档叙事主线

本文档体系不要把重点放在“完整训练大模型”，而应突出：

1. **问题重要**：实验谱图到分子结构反演是化学分析中的核心问题。
2. **现有不足**：直接谱图到 SMILES 翻译忽略图结构、几何约束和一谱多解。
3. **核心思路**：谱图侧尊重真实横坐标；分子侧尊重图结构和几何对称性；结果侧用前向谱图一致性验证。
4. **Demo 可落地**：先实现可运行小闭环，而不是一开始训练完整扩散模型。
5. **未来可扩展**：从 2D 候选检索逐步升级到可增删节点图扩散、pairwise distance、TFN-Transformer 和手性表达。

---

## 1. 最终提交文档包规划

建议最终文档层面至少包含以下文件：

```text
MultiSpec-GeoDiff-Demo/
├── proposal.md                 # 500 字以内项目提议
├── README.md                   # demo 运行说明与项目入口
├── demo_report.md              # 主要说明文档，2-4 页等价内容
├── design_overview.md          # 从 Design.md 精炼出的技术方案说明
├── assets/
│   ├── fig1_overall_pipeline.png
│   ├── fig2_multispec_encoder.png
│   ├── fig3_demo_workflow.png
│   ├── fig4_future_geodiff_tfn.png
│   ├── fig5_result_visualization_placeholder.png
│   └── fig6_traceable_workflow.png
└── outputs/
    ├── topk_candidates.csv
    ├── spectrum_match.png
    ├── molecule_grid.png
    └── trace_log.json
```

其中真正给评审看的核心是：

1. `proposal.md`
2. `README.md`
3. `demo_report.md`
4. `notebooks/demo_pipeline.ipynb`，此文件在代码阶段完成
5. `outputs/` 中的结果图和 trace log，代码阶段生成

---

## 2. proposal.md 规划

### 目标

满足深势科技项目提议要求：不超过 500 字，包含三部分：

1. 具体目标
2. 核心价值
3. 可行性简析

### 写作风格

- 简洁、清晰、偏工程科研项目提案风格。
- 不展开过多数学公式。
- 不写过度宏大的完整大模型训练承诺。
- 强调“Demo 阶段验证最小可行闭环”。

### 建议结构

```markdown
# 项目提议：MultiSpec-GeoDiff

## 具体目标
...

## 核心价值
...

## 可行性简析
...
```

### 内容要点

#### 具体目标

写清楚：

- 输入：IR/Raman/NMR/MS 等多模态谱图，Demo 先支持 IR/Raman。
- 输出：Top-K 候选分子结构、谱图一致性评分和可视化结果。
- Demo 产出：可运行 notebook、小数据样例、候选重排结果、trace log。

#### 核心价值

写清楚：

- 谱图反演是实验化学中的关键任务。
- 相比谱图到 SMILES，图/几何表示更符合分子本体。
- 前向谱图重排提供可验证闭环。
- 后续可扩展到 graph diffusion、pairwise distance 和 TFN-Transformer。

#### 可行性简析

写清楚：

- 数据：公开 IR/Raman/QM9S/NIST 小样例。
- 方法：先检索候选 + 前向谱图重排，后续再替换为生成模型。
- 风险：实验噪声、一谱多解、候选覆盖不足。
- 验证：Top-K 命中率、谱图相似度、分子式/官能团匹配率、重排增益。

### 不建议写的内容

- 不要承诺完整训练大型多模态扩散模型。
- 不要过多堆砌 TFN/Equiformer 公式。
- 不要把 demo 写成论文计划，而要写成能跑的最小原型。

---

## 3. README.md 规划

### 目标

让评审能够快速理解并运行 demo。

### 推荐结构

```markdown
# MultiSpec-GeoDiff Demo

## 1. What is this demo?
## 2. Core idea
## 3. Repository structure
## 4. Quick start
## 5. Input and output
## 6. Demo workflow
## 7. Expected results
## 8. Future extensions
## 9. References
```

### 各部分写法

#### 1. What is this demo?

用 3-5 句话说明：

- 本 demo 关注多模态谱图到小分子结构反演。
- 当前版本实现 IR/Raman 输入、候选生成/检索、前向谱图重排和 Top-K 可视化。
- 当前不是完整生产系统，而是验证核心技术路线。

#### 2. Core idea

建议用简短流程式表达：

```text
Input spectra -> Spectral encoder -> Candidate generator -> Forward spectrum scorer -> Top-K molecules
```

强调三点：

- spectral coordinate-aware encoding
- candidate-level structure reasoning
- forward spectrum consistency reranking

#### 3. Repository structure

列出目录树。

#### 4. Quick start

预留如下命令：

```bash
pip install -r requirements.txt
python scripts/run_demo.py --config configs/demo.yaml
```

或：

```bash
jupyter notebook notebooks/demo_pipeline.ipynb
```

#### 5. Input and output

说明输入 CSV 格式：

```csv
wavenumber,intensity
400,0.02
401,0.03
...
```

说明输出：

- `outputs/topk_candidates.csv`
- `outputs/spectrum_match.png`
- `outputs/molecule_grid.png`
- `outputs/trace_log.json`

#### 6. Demo workflow

插入 `fig3_demo_workflow.png`。

#### 7. Expected results

展示预期截图占位符。

#### 8. Future extensions

用三阶段写：

1. Phase I：2D graph diffusion
2. Phase II：pairwise distance + 3D initialization
3. Phase III：TFN-Transformer + chirality-aware spectra

---

## 4. demo_report.md 规划

### 目标

这是最重要的文档。它应当像一个简短技术白皮书，说明：

1. 为什么这个问题值得做；
2. 为什么当前方案有科学原则；
3. demo 如何落地；
4. 结果如何验证；
5. 后续如何扩展。

建议控制在 2-4 页等价内容，Markdown 中约 2500-4000 中文字。

---

## 5. demo_report.md 详细章节规划

### 标题

```markdown
# MultiSpec-GeoDiff: A Traceable Demo for Multi-modal Spectra-to-Molecule Inference
```

副标题：

```markdown
A minimal executable workflow for spectrum-conditioned molecular structure identification
```

---

### 1. Background and Motivation

#### 写什么

说明分子谱图是结构识别的重要实验信息来源。IR/Raman/NMR/MS/UV 等模态分别携带不同结构证据：

- IR/Raman：振动指纹、官能团、骨架信息；
- NMR：局部化学环境；
- MS：分子量和碎片；
- UV/CD/VCD：共轭和手性相关信息。

然后指出：传统谱图库检索覆盖有限，量化计算 trial-and-error 成本高，端到端 SMILES 翻译缺少几何和物理约束。

#### 怎样写

采用“问题—不足—机会”的结构：

```text
Molecular spectra are experimental fingerprints of molecular structures. However, structure identification from spectra remains an inverse problem: noisy, underdetermined, and often one-to-many. Existing spectrum-to-SMILES translation pipelines are useful but do not explicitly model molecular graph topology, 3D geometry, or forward spectral consistency. This motivates a geometry-aware and traceable workflow.
```

#### 插入图片

**Figure 1: Overall problem setting**

建议图像内容：

左侧：多模态谱图 IR/Raman/NMR/MS；
中间：MultiSpec-GeoDiff；
右侧：Top-K candidate molecules + spectrum matching scores。

文件名：

```text
assets/fig1_overall_pipeline.png
```

图注：

```text
Figure 1. MultiSpec-GeoDiff treats spectrum-to-structure inference as a multi-modal, geometry-aware, and verifiable inverse problem.
```

---

### 2. Core Design Principle

#### 写什么

强调四条原则：

1. 谱图侧尊重真实横坐标；
2. 分子侧用图而不是字符串；
3. 2D 到 3D 用 pairwise distance 作为中间语言；
4. 最终结果用前向谱图一致性验证。

#### 怎样写

建议用四个小段落，每段 3-5 句话。

核心句：

```text
The central design principle is to preserve the native geometry of each representation: spectra live on physical coordinate axes, molecules live on graphs and 3D Euclidean space, and predictions should be verified by forward spectral simulation.
```

#### 插入图片

**Figure 2: Design principles**

建议做成四格图：

1. Spectral coordinate encoding
2. Molecular graph generation
3. Pairwise distance to 3D
4. Forward spectrum consistency

文件名：

```text
assets/fig2_design_principles.png
```

---

### 3. Multi-modal Spectral Encoder

#### 写什么

这是文档中的重点之一。需要把你的核心创新写清楚：

- 每个模态先用 CNN 提取峰形局部特征；
- 加入由真实横坐标决定的位置编码；
- 模态内 self-attention 加横坐标距离偏置；
- 跨模态 cross-attention 不加横坐标偏置，因为不同模态坐标体系不同；
- 输出 global condition 和 token-level condition。

#### 公式

可以放少量公式：

```math
H_i^{(m)} = \operatorname{CNN}_m(S_i^{(m)}) + e_m + p_m(\nu_i^{(m)})
```

```math
A_{ij}^{(m)} = \operatorname{softmax}_j\left(\frac{Q_iK_j^T}{\sqrt d} + b_m(|\nu_i-\nu_j|)\right)
```

跨模态：

```math
A_{ij}^{(m\leftarrow n)} = \operatorname{softmax}_j\left(\frac{Q_i^{(m)}K_j^{(n)T}}{\sqrt d} + \gamma_{mn}\right)
```

#### 怎样写

强调一句：

```text
Intra-modal attention models physical proximity along a spectral axis; cross-modal attention models evidence compatibility across different measurement spaces.
```

#### 插入图片

**Figure 3: Multi-modal spectral encoder**

内容：

- IR/Raman/NMR/MS 四条分支；
- 每条分支：CNN stem -> coordinate PE -> intra-modal attention with Δν bias；
- 跨分支：cross-modal attention without coordinate bias；
- 输出：global condition + condition tokens。

文件名：

```text
assets/fig3_multispec_encoder.png
```

图注：

```text
Figure 3. Multi-modal spectral encoder. Coordinate-aware attention is used within each modality, while cross-modal attention only learns evidence compatibility without forcing heterogeneous coordinate systems into a shared distance metric.
```

---

### 4. Demo-stage Molecular Candidate Generation

#### 写什么

说明 demo 阶段不训练完整 diffusion，而采用候选库检索/轻量生成作为替代，以验证闭环。

内容：

- 输入 query spectrum；
- 在候选库中检索 Top-M；
- 用分子量、分子式、官能团规则过滤；
- 用 forward spectrum reranker 重新排序；
- 正式阶段替换为可增删节点 graph diffusion。

#### 怎样写

要主动解释为什么这样做不是“偷懒”：

```text
For the demo, candidate retrieval is used as a controlled proxy for the future graph diffusion generator. This allows us to isolate and validate the core inference loop: condition extraction, candidate evaluation, and forward-consistency reranking.
```

#### 插入图片

**Figure 4: Demo workflow**

内容：

Input spectra -> encoder -> candidate retrieval -> forward spectrum scorer -> Top-K candidates.

文件名：

```text
assets/fig4_demo_workflow.png
```

---

### 5. Future Generator: Size-adaptive Graph Diffusion

#### 写什么

这是未来扩展部分，简洁写。不要写得像已经完成。

说明：

- 分子图状态：原子类型和键类型；
- 扩散过程允许 insert/delete/change atom/change bond；
- 适合未知分子大小的谱图反演；
- 注意化学约束，如价态、连通性、芳香性。

#### 公式

```math
G_t=(X_t,E_t),\quad a_t\in\{\text{insert},\text{delete},\text{change atom},\text{change bond}\}
```

#### 插入图片

**Figure 5: Future graph diffusion module**

内容：

随机/噪声图 -> 多步 denoising -> 合法候选分子图；中间标注 insert/delete node。

文件名：

```text
assets/fig5_graph_diffusion_future.png
```

---

### 6. Pairwise Distance as the 2D-to-3D Bridge

#### 写什么

这是你的重要观点，要重点写。

说明：

- 直接预测坐标不如预测 pairwise distance 合理；
- 距离矩阵对旋转和平移不变；
- 后续等变层只需要相对坐标和距离；
- 可由 MDS/距离几何恢复初始坐标。

#### 公式

```math
\hat d_{ij}=f_D(h_i,h_j,e_{ij},c)
```

```math
D=[\hat d_{ij}]\rightarrow R^{(0)}
```

MDS 可简写：

```math
B=-\frac{1}{2}JD^{\circ 2}J,\quad R^{(0)}=U_{1:3}\Lambda_{1:3}^{1/2}
```

#### 插入图片

**Figure 6: Pairwise distance bridge**

内容：

2D molecular graph -> distance matrix heatmap -> MDS -> initial 3D conformer。

文件名：

```text
assets/fig6_pairwise_distance_bridge.png
```

---

### 7. Future TFN-Transformer Refinement

#### 写什么

这是文档中的第二个重点。但要写成未来高级模块，不要让 demo 显得无法完成。

要点：

- 节点携带多阶张量特征：0e, 0o, 1e, 1o, 2e, 2o；
- attention 权重用张量内积得到 invariant scalar；
- 距离偏置和边偏置作为额外 bias；
- 坐标用 EGNN 式相对向量更新；
- 谱图条件通过 scalar gate 注入，不破坏等变性；
- odd-parity channel 表达手性。

#### 公式

张量 attention：

```math
s_{ij}=\sum_{l,p,c}\langle Q_{i,c}^{(l,p)},K_{j,c}^{(l,p)}\rangle + b_D(\|r_i-r_j\|)+b_E(e_{ij})
```

坐标更新：

```math
r_i^{\ell+1}=r_i^{\ell}+\sum_j \alpha_{ij}(r_i-r_j)\phi_x(\|r_i-r_j\|^2,e_{ij},c)
```

手性伪标量：

```math
\chi_{ijkl}=(r_j-r_i)\cdot[(r_k-r_i)\times(r_l-r_i)]
```

#### 插入图片

**Figure 7: TFN-Transformer refinement**

内容：

3D molecular graph；每个原子旁边画 tensor channels；边上标注 tensor inner-product attention；坐标箭头表示 EGNN update；旁边标注 chirality 0o channel。

文件名：

```text
assets/fig7_tfn_transformer_refinement.png
```

---

### 8. Forward Spectrum Consistency and Posterior Reranking

#### 写什么

说明输出不是一个未经验证的分子，而是经过前向谱图一致性重排的 Top-K 候选。

Demo 阶段：

- 用候选库谱图或轻量前向模型预测谱图；
- 计算输入谱图和候选谱图相似度；
- 输出 Top-K。

未来阶段：

- 2D：GNN 直接预测谱图；
- 3D：先预测物理量，再组合谱图。

#### 公式

```math
s_k = \operatorname{PosteriorNet}(S,\hat S_k,G_k,f_{chem,k})
```

或简化：

```math
s_k=-d(S,\hat S_k)+\lambda C_{chem}(G_k)
```

#### 插入图片

**Figure 8: Spectrum consistency reranking**

内容：

Top-M 候选分子 -> forward spectrum prediction -> 与 query spectrum 叠图比较 -> Top-K reranking。

文件名：

```text
assets/fig8_forward_reranking.png
```

---

### 9. Traceable Scientific Workflow

#### 写什么

这部分用来贴合深势科技/Bohrium/SciMaster 的风格。

说明 demo 会记录：

- 输入谱图路径；
- 参数配置；
- 候选分子；
- 过滤规则；
- 相似度分数；
- 输出图像；
- 运行时间；
- 软件版本。

输出到：

```text
outputs/trace_log.json
```

#### 插入图片

**Figure 9: Traceable workflow**

内容：

Input -> modules -> outputs，每一步下方有 trace log 记录图标。

文件名：

```text
assets/fig9_traceable_workflow.png
```

---

### 10. Evaluation Plan

#### 写什么

分 demo 指标和未来指标。

Demo 阶段指标：

- Top-K retrieval accuracy；
- spectrum cosine similarity；
- peak matching score；
- molecular formula match；
- functional group recall；
- reranking gain；
- trace completeness。

未来阶段指标：

- graph validity；
- uniqueness；
- novelty；
- distance matrix MAE；
- 3D RMSD；
- chirality consistency；
- multimodal spectrum similarity。

#### 建议表格

| Evaluation target | Metric | Demo stage | Future stage |
|---|---|---|---|
| Candidate quality | Top-K hit rate | yes | yes |
| Spectrum consistency | cosine / peak score | yes | yes |
| Chemistry validity | RDKit sanitize | partial | yes |
| 3D geometry | distance MAE / RMSD | no | yes |
| Chirality | enantiomer consistency | no | optional |
| Traceability | trace log completeness | yes | yes |

---

## 6. architecture.png 规划

最终至少要画 4 张关键图。若时间有限，优先级如下：

### 必画图 1：总体 Pipeline 图

文件名：

```text
fig1_overall_pipeline.png
```

画面结构：

```text
IR/Raman/NMR/MS spectra
        ↓
MultiSpec Encoder
        ↓
Candidate Generation / Retrieval
        ↓
Forward Spectrum Reranking
        ↓
Top-K Molecules + Spectrum Matching
```

风格：

- 白底；
- 科技蓝灰主色；
- 橙色强调谱图横坐标 PE；
- 绿色强调 reranking 输出；
- 英文标签；
- 不要太花。

### 必画图 2：多模态谱图编码图

文件名：

```text
fig3_multispec_encoder.png
```

画面结构：

```text
IR branch: CNN + coord PE + intra attention
Raman branch: CNN + coord PE + intra attention
NMR branch: CNN + coord PE + intra attention
MS branch: CNN + coord PE + intra attention
        ↘ cross-modal attention ↙
Condition tokens + global condition
```

重点标注：

```text
Intra-modal: relative coordinate bias
Cross-modal: no coordinate bias
```

### 必画图 3：Demo 工作流图

文件名：

```text
fig4_demo_workflow.png
```

画面结构：

```text
query spectra -> encoder -> candidate library -> spectrum scorer -> Top-K table
```

强调：这是当前可运行 demo，不是未来大模型。

### 必画图 4：未来 GeoDiff-TFN 扩展图

文件名：

```text
fig7_tfn_transformer_refinement.png
```

画面结构：

```text
2D graph -> pairwise distance -> initial 3D -> TFN-Transformer -> tensor physical properties -> spectra
```

重点标注：

- tensor inner-product attention；
- relative coordinate update；
- parity/chirality channels；
- forward spectral physics。

---

## 7. demo_report.md 的推荐文字风格

### 总体风格

- 中文为主，关键模块名用英文；
- 每节开头先给一句结论；
- 不要长篇堆论文综述；
- 公式少而精；
- 每张图下都写“这张图说明什么”。

### 推荐句式

```text
本 demo 的目标不是直接构建完整生产级模型，而是验证谱图反演任务中最关键的可执行闭环：多模态谱图条件提取、候选结构生成/检索、前向谱图一致性重排和可追踪结果输出。
```

```text
与直接生成 SMILES 的方法不同，本方案将分子视为图与几何对象，并将谱图横坐标、分子拓扑、pairwise distance 和前向谱图验证统一到同一条可复现实验路径中。
```

```text
Demo 阶段用候选检索近似 graph diffusion generator，使得系统可以在较短时间内形成可运行闭环；未来可将该模块替换为 size-adaptive graph diffusion。
```

```text
Pairwise distance 是连接 2D 图和 3D 几何的自然中间表示，因为它对整体旋转和平移不变，并且正是后续等变层计算相对几何关系的核心输入。
```

---

## 8. 文档之间的关系

### proposal.md

作用：投递时快速说明项目。

长度：不超过 500 字。

### README.md

作用：让评审快速运行 demo。

长度：800-1500 字。

### demo_report.md

作用：让评审理解技术深度和创新性。

长度：2500-4000 字。

### design_overview.md

作用：从 Design.md 中提炼长期技术路线。

长度：1500-2500 字。

建议关系：

```text
proposal.md = 最短版项目提议
README.md = 怎么运行
demo_report.md = 为什么做、怎么做、结果怎么看
design_overview.md = 长期模型架构和未来路线
```

---

## 9. 最终写作顺序建议

不要先写最长的 `demo_report.md`。推荐顺序：

1. 先写 `proposal.md`，固定项目边界。
2. 再画 `fig1_overall_pipeline.png`，固定整体叙事。
3. 写 `README.md`，固定 demo 入口。
4. 写 `demo_report.md`，展开技术细节。
5. 写 `design_overview.md`，承接 Design.md 的长期架构。
6. 最后统一检查图、术语、文件名和结果路径。

---

## 10. 文档审查清单

提交前逐项检查：

### 项目提议检查

- [ ] 是否不超过 500 字？
- [ ] 是否包含具体目标、核心价值、可行性简析？
- [ ] 是否避免承诺完整大模型训练？
- [ ] 是否明确 demo 产出？

### README 检查

- [ ] 是否 3 分钟内能读懂项目？
- [ ] 是否有一键运行命令？
- [ ] 是否说明输入输出格式？
- [ ] 是否说明 outputs 中每个文件的意义？

### demo_report 检查

- [ ] 是否有清晰问题背景？
- [ ] 是否突出多模态谱图编码？
- [ ] 是否突出前向谱图一致性重排？
- [ ] 是否解释为什么 pairwise distance 合理？
- [ ] 是否把 TFN-Transformer 写成未来扩展，而不是 demo 已完成？
- [ ] 是否有风险和验证方式？

### 图示检查

- [ ] 是否所有图都是英文标签？
- [ ] 是否白底、清晰、科技风？
- [ ] 是否每张图都有图注？
- [ ] 是否图中没有未解释缩写？

### 风格检查

- [ ] 是否避免过度夸大？
- [ ] 是否避免空泛口号？
- [ ] 是否体现 AI4S、可追踪、可验证、可落地？
- [ ] 是否能体现你对谱图、分子图和几何深度学习的理解？

---

## 11. 后续交给 Claude Code / Codex 的文档任务拆分

后续可以让代码助手按以下顺序执行：

```text
Task 1: Read Design.md and this planning file.
Task 2: Create proposal.md under 500 Chinese characters.
Task 3: Create README.md with repository structure and quick-start instructions.
Task 4: Create demo_report.md with figure placeholders.
Task 5: Create design_overview.md summarizing the long-term model architecture.
Task 6: Add figure placeholder blocks with filenames and captions.
Task 7: Check consistency among all document files.
```

---

## 12. 推荐最终文档标题体系

### proposal.md

```markdown
# 项目提议：MultiSpec-GeoDiff 多模态谱图驱动的分子结构反演 Demo
```

### README.md

```markdown
# MultiSpec-GeoDiff Demo
```

### demo_report.md

```markdown
# MultiSpec-GeoDiff: A Traceable Demo for Multi-modal Spectra-to-Molecule Inference
```

### design_overview.md

```markdown
# Design Overview: From Spectral Evidence to Geometry-aware Molecular Inference
```

---

## 13. 最终文档层面目标

完成文档后，评审应能在 5 分钟内明白：

1. 你要解决什么问题；
2. 为什么谱图到分子结构反演值得做；
3. 为什么不能只做谱图到 SMILES；
4. 你的 demo 能跑什么；
5. 你的长期方案如何从 2D 图扩展到 3D 几何和张量特征；
6. 你的方案如何体现 AI4S 的可追踪、可验证、可复现实验工作流。

最终文档应形成一句清晰印象：

> MultiSpec-GeoDiff 不是一个空泛的大模型设想，而是一个从多模态实验谱图出发，经候选结构生成与前向谱图验证，逐步走向几何扩散分子反演的可执行 AI4S demo。
