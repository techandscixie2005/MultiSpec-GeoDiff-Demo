# MultiSpec-GeoDiff Demo 层面规划

> 目标：本文件规划 MultiSpec-GeoDiff demo 的工程实现方案，包括项目架构、代码模块、数据组织、可运行流程、可行性边界、测试方式、输出材料和后续升级路径。它面向“可提交、可运行、可展示”的实习申请 demo，而不是完整论文级训练系统。

---

## 0. Demo 总体定位

### 项目名称

**MultiSpec-GeoDiff Demo：多模态谱图驱动的分子结构反演最小闭环**

### Demo 一句话目标

输入 IR/Raman 谱图，经过谱图预处理、横坐标感知编码、候选分子召回、前向谱图一致性重排，输出 Top-K 候选分子、匹配分数、谱图叠图、分子结构图和完整 trace log。

### Demo 必须证明什么

当前 demo 不需要证明完整 graph diffusion 或 TFN-Transformer 已经训练成功，而要证明以下核心闭环成立：

```text
Query spectra
    -> coordinate-aware spectral representation
    -> candidate molecule retrieval / lightweight generation
    -> forward spectrum consistency scoring
    -> Top-K molecular candidates
    -> traceable output package
```

### Demo 的边界

本 demo 分为两层：

1. **MVP 可运行层**：必须能在普通电脑上跑通，生成真实文件和可视化结果。
2. **Advanced prototype 层**：保留 graph diffusion、pairwise distance、TFN-Transformer 的代码接口和轻量 forward-only 原型，用于展示长期技术路径。

---

## 1. Demo 与文档方案的对应关系

你的文档中已经把方案定位为：

```text
多模态谱图编码 -> 候选召回/轻量生成 -> 前向谱图一致性重排 -> Top-K 分子输出
```

Demo 层面应该直接服务这个叙事，不要偏离到过度复杂的完整模型训练。

| 文档层面概念 | Demo 层面落地方式 | 完成优先级 |
|---|---|---|
| 多模态谱图输入 | IR/Raman CSV 输入 | 必须 |
| 横坐标感知编码 | coordinate PE + optional attention bias prototype | 必须 |
| 候选召回/轻量生成 | 候选库检索 + 分子量/官能团过滤 | 必须 |
| 前向谱图一致性重排 | 候选谱图相似度评分 / 轻量 forward predictor | 必须 |
| Top-K 输出 | CSV + molecule grid + spectrum overlay | 必须 |
| traceable workflow | JSON trace log | 必须 |
| graph diffusion | 接口 + stub + 说明 | 可选但建议 |
| pairwise distance | RDKit conformer + distance matrix demo | 建议 |
| TFN-Transformer | 接口 + forward-only tensor attention sketch | 可选 |
| chirality | 文档和接口说明，非 MVP 必做 | 可选 |

---

## 2. 最终项目目录架构

建议项目目录如下：

```text
MultiSpec-GeoDiff-Demo/
├── README.md
├── proposal.md
├── demo_report.md
├── design_overview.md
├── requirements.txt
├── environment.yml
├── pyproject.toml                         # 可选
├── configs/
│   ├── demo.yaml
│   ├── encoder.yaml
│   └── retrieval.yaml
├── data/
│   ├── raw/
│   │   └── README.md
│   ├── samples/
│   │   ├── query_ir.csv
│   │   ├── query_raman.csv
│   │   ├── candidate_library.csv
│   │   └── spectra_library/
│   │       ├── mol_0001_ir.csv
│   │       ├── mol_0001_raman.csv
│   │       └── ...
│   └── processed/
│       ├── candidate_features.parquet
│       └── spectra_embeddings.npy
├── notebooks/
│   └── demo_pipeline.ipynb
├── scripts/
│   ├── run_demo.py
│   ├── build_sample_data.py
│   ├── build_candidate_index.py
│   └── smoke_test.py
├── src/
│   └── multispec_geodiff/
│       ├── __init__.py
│       ├── data/
│       │   ├── spectra_io.py
│       │   ├── molecule_io.py
│       │   └── dataset.py
│       ├── spectra/
│       │   ├── preprocess.py
│       │   ├── peaks.py
│       │   ├── similarity.py
│       │   └── encoder.py
│       ├── chemistry/
│       │   ├── rdkit_utils.py
│       │   ├── descriptors.py
│       │   ├── filters.py
│       │   └── functional_groups.py
│       ├── retrieval/
│       │   ├── candidate_index.py
│       │   ├── retriever.py
│       │   └── lightweight_generator.py
│       ├── scoring/
│       │   ├── forward_spectrum.py
│       │   ├── reranker.py
│       │   └── posterior.py
│       ├── geometry/
│       │   ├── distance_matrix.py
│       │   ├── conformer.py
│       │   └── mds.py
│       ├── models/
│       │   ├── multispec_encoder.py
│       │   ├── graph_diffusion_stub.py
│       │   ├── distance_head_stub.py
│       │   └── tfn_transformer_stub.py
│       ├── visualization/
│       │   ├── spectra_plot.py
│       │   ├── molecule_plot.py
│       │   ├── attention_plot.py
│       │   └── report_figures.py
│       ├── tracing/
│       │   ├── trace.py
│       │   └── schema.py
│       └── utils/
│           ├── config.py
│           ├── seed.py
│           └── logging.py
├── tests/
│   ├── test_spectra_io.py
│   ├── test_similarity.py
│   ├── test_rdkit_utils.py
│   ├── test_reranker.py
│   └── test_trace.py
├── assets/
│   ├── architecture.png
│   └── demo_screenshot.png
└── outputs/
    ├── topk_candidates.csv
    ├── spectrum_match.png
    ├── molecule_grid.png
    ├── attention_heatmap.png
    ├── distance_matrix.png
    ├── trace_log.json
    └── demo_summary.md
```

---

## 3. MVP 的最小可运行流程

### 3.1 输入

MVP 支持两个输入谱图：

```text
data/samples/query_ir.csv
data/samples/query_raman.csv
```

CSV 格式：

```csv
x,intensity
400,0.01
401,0.02
...
4000,0.03
```

字段说明：

| 字段 | 含义 |
|---|---|
| `x` | 谱图横坐标，IR/Raman 中为 wavenumber cm^-1 |
| `intensity` | 谱图强度，建议归一化到 [0, 1] |

候选库：

```text
data/samples/candidate_library.csv
```

格式：

```csv
mol_id,smiles,name,formula,mass,ir_path,raman_path
mol_0001,CCO,ethanol,C2H6O,46.07,spectra_library/mol_0001_ir.csv,spectra_library/mol_0001_raman.csv
...
```

### 3.2 输出

运行后必须生成：

```text
outputs/topk_candidates.csv
outputs/spectrum_match.png
outputs/molecule_grid.png
outputs/trace_log.json
outputs/demo_summary.md
```

其中：

- `topk_candidates.csv`：Top-K 候选分子表。
- `spectrum_match.png`：输入谱图与 Top-1/Top-3 候选谱图叠图。
- `molecule_grid.png`：Top-K 分子结构网格图。
- `trace_log.json`：完整运行记录。
- `demo_summary.md`：自动生成的简短结果报告。

### 3.3 一键运行命令

推荐：

```bash
python scripts/run_demo.py --config configs/demo.yaml
```

或 notebook：

```bash
jupyter notebook notebooks/demo_pipeline.ipynb
```

---

## 4. MVP 主流程详细设计

### Step 1：谱图读取与预处理

文件：

```text
src/multispec_geodiff/spectra/preprocess.py
src/multispec_geodiff/data/spectra_io.py
```

功能：

1. 读取 CSV；
2. 统一横坐标网格；
3. baseline correction，可选；
4. intensity normalization；
5. smoothing，可选；
6. peak extraction，可选；
7. 输出标准化谱图对象。

建议数据结构：

```python
@dataclass
class Spectrum:
    modality: str
    x: np.ndarray
    intensity: np.ndarray
    unit: str = "cm^-1"
    source_path: str | None = None
```

关键函数：

```python
def load_spectrum(path: str, modality: str) -> Spectrum:
    ...

def resample_spectrum(spec: Spectrum, x_min: float, x_max: float, step: float) -> Spectrum:
    ...

def normalize_spectrum(spec: Spectrum, method: str = "max") -> Spectrum:
    ...
```

MVP 中只需要支持：

- IR；
- Raman；
- CSV 输入；
- max normalization；
- interpolation to common grid。

---

### Step 2：谱图表示与相似度计算

文件：

```text
src/multispec_geodiff/spectra/similarity.py
src/multispec_geodiff/spectra/encoder.py
```

MVP 中先做非训练式 encoder：

```python
class HandcraftedSpectralEncoder:
    def encode(self, spec: Spectrum) -> np.ndarray:
        ...
```

建议特征：

1. resampled intensity vector；
2. peak positions；
3. peak intensities；
4. low-dimensional PCA/SVD projection，可选；
5. coordinate-weighted moments：

```text
sum I_i, sum x_i I_i, sum x_i^2 I_i
```

相似度：

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    ...

def peak_match_score(query: Spectrum, target: Spectrum, tolerance: float) -> float:
    ...

def weighted_spectral_score(query_ir, cand_ir, query_raman, cand_raman, weights) -> float:
    ...
```

MVP 推荐评分：

```text
score_spectrum = w_ir * cos(IR_query, IR_candidate)
               + w_raman * cos(Raman_query, Raman_candidate)
               + w_peak * peak_match_score
```

默认权重：

```yaml
weights:
  ir: 0.45
  raman: 0.35
  peak: 0.20
```

---

### Step 3：横坐标感知谱图编码原型

文件：

```text
src/multispec_geodiff/models/multispec_encoder.py
```

目的：展示你的核心技术点，不一定用于最终 Top-K 评分。

实现一个轻量 PyTorch 模块：

```python
class CoordinatePositionalEncoding(nn.Module):
    def forward(self, x_coord: torch.Tensor) -> torch.Tensor:
        ...

class RelativeCoordinateBias(nn.Module):
    def forward(self, x_coord: torch.Tensor) -> torch.Tensor:
        ...

class SpectralSelfAttention(nn.Module):
    def forward(self, h: torch.Tensor, x_coord: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

class MultiSpecEncoderPrototype(nn.Module):
    def forward(self, batch: dict) -> dict:
        ...
```

模态内 attention：

```text
attention_score = QK^T / sqrt(d) + b(|x_i - x_j|)
```

跨模态 attention：

```text
attention_score = Q_m K_n^T / sqrt(d) + modality_pair_bias
```

注意：跨模态不加横坐标距离偏置。

MVP 中该模块输出：

1. `global_condition`；
2. `condition_tokens`；
3. `attention_heatmap`。

即使不训练，也可以用随机初始化或小规模自监督方式生成 heatmap，用于展示架构可运行。

---

### Step 4：候选分子库构建

文件：

```text
scripts/build_sample_data.py
scripts/build_candidate_index.py
src/multispec_geodiff/retrieval/candidate_index.py
```

候选库来源可以有三种：

1. 人工构造 20-100 个常见小分子；
2. 从公开小数据集中截取；
3. 从已有实验/模拟数据中整理。

MVP 最稳做法：人工构造一个小候选库，每个分子配一个简化谱图或公开/模拟谱图。候选库不需要很大，重点是流程清楚。

候选分子字段：

```python
@dataclass
class CandidateMolecule:
    mol_id: str
    smiles: str
    name: str | None
    formula: str
    mass: float
    ir_spectrum: Spectrum | None
    raman_spectrum: Spectrum | None
    rdkit_mol: Any | None
```

候选索引：

```python
class CandidateIndex:
    def build(self, library_csv: str) -> None:
        ...

    def search(self, query: QuerySpectra, top_m: int) -> list[CandidateHit]:
        ...
```

候选召回策略：

```text
1. 计算 query IR/Raman 与库中 IR/Raman 的相似度。
2. 选出 Top-M。
3. 如果输入给出 molecular mass，则做 mass filter。
4. 如果输入给出 formula，则做 formula filter。
5. 如果检测到特征峰，则做 functional group hint filter。
```

---

### Step 5：化学过滤与 RDKit 工具

文件：

```text
src/multispec_geodiff/chemistry/rdkit_utils.py
src/multispec_geodiff/chemistry/filters.py
src/multispec_geodiff/chemistry/descriptors.py
```

功能：

1. SMILES -> RDKit Mol；
2. sanitize；
3. formula/mass 计算；
4. Morgan fingerprint；
5. functional group matching；
6. molecule drawing。

关键函数：

```python
def smiles_to_mol(smiles: str):
    ...

def is_valid_molecule(smiles: str) -> bool:
    ...

def compute_formula(smiles: str) -> str:
    ...

def compute_exact_mass(smiles: str) -> float:
    ...

def draw_molecule_grid(smiles_list: list[str], legends: list[str], out_path: str) -> None:
    ...
```

可行性说明：RDKit 已经支持分子读写、fingerprint 和相似度计算，因此 MVP 可以快速完成化学工具层。

---

### Step 6：前向谱图一致性重排

文件：

```text
src/multispec_geodiff/scoring/forward_spectrum.py
src/multispec_geodiff/scoring/reranker.py
src/multispec_geodiff/scoring/posterior.py
```

MVP 不强制训练 forward model，先用候选库已有谱图作为 forward spectrum：

```text
candidate molecule -> stored candidate IR/Raman spectrum -> compare with query
```

可选升级：训练轻量模型：

```text
Morgan fingerprint / RDKit descriptors -> MLP -> binned spectrum
```

Posterior score：

```text
final_score = alpha * spectral_similarity
            + beta  * chemical_validity_score
            + gamma * mass_match_score
            + delta * functional_group_score
```

默认：

```yaml
rerank:
  alpha: 0.75
  beta: 0.10
  gamma: 0.10
  delta: 0.05
```

输出 Top-K：

```python
@dataclass
class RankedCandidate:
    rank: int
    mol_id: str
    smiles: str
    name: str
    final_score: float
    spectral_score: float
    ir_score: float
    raman_score: float
    mass: float
    formula: str
    valid: bool
```

---

### Step 7：可视化输出

文件：

```text
src/multispec_geodiff/visualization/spectra_plot.py
src/multispec_geodiff/visualization/molecule_plot.py
src/multispec_geodiff/visualization/attention_plot.py
```

必须生成三类图：

#### 1. 谱图叠图

```text
outputs/spectrum_match.png
```

内容：

- Query IR vs Top-1/Top-3 candidate IR；
- Query Raman vs Top-1/Top-3 candidate Raman；
- 标注 similarity score。

#### 2. Top-K 分子图

```text
outputs/molecule_grid.png
```

内容：

- 分子结构；
- rank；
- score；
- formula；
- mass。

#### 3. Attention heatmap / coordinate bias heatmap

```text
outputs/attention_heatmap.png
```

内容：

- 横坐标距离 bias；
- 或 spectral self-attention map；
- 用于展示“谱图横坐标感知编码”。

可选图：

```text
outputs/distance_matrix.png
```

展示 Top-1 候选的 RDKit conformer pairwise distance matrix。

---

### Step 8：trace log

文件：

```text
src/multispec_geodiff/tracing/trace.py
```

目标：体现 demo 是可执行、可追踪、可复查的 AI4S workflow。

`trace_log.json` 应包含：

```json
{
  "run_id": "2026-05-xx_xxxxxx",
  "timestamp": "...",
  "config_path": "configs/demo.yaml",
  "input": {
    "ir_path": "data/samples/query_ir.csv",
    "raman_path": "data/samples/query_raman.csv"
  },
  "preprocessing": {
    "normalization": "max",
    "grid": [400, 4000, 1]
  },
  "candidate_generation": {
    "library": "data/samples/candidate_library.csv",
    "top_m": 20,
    "filters": ["validity", "mass_optional"]
  },
  "scoring": {
    "weights": {
      "ir": 0.45,
      "raman": 0.35,
      "peak": 0.20
    }
  },
  "outputs": {
    "topk_csv": "outputs/topk_candidates.csv",
    "spectrum_plot": "outputs/spectrum_match.png",
    "mol_grid": "outputs/molecule_grid.png"
  },
  "topk": [
    {
      "rank": 1,
      "mol_id": "mol_0001",
      "smiles": "...",
      "score": 0.91
    }
  ]
}
```

---

## 5. Advanced Prototype 层规划

Advanced Prototype 的作用不是训练完整模型，而是在代码结构中给出未来升级接口，让评审看到你知道如何从 demo 走向真正模型。

---

### 5.1 Graph diffusion stub

文件：

```text
src/multispec_geodiff/models/graph_diffusion_stub.py
```

类：

```python
class SizeAdaptiveGraphDiffusionStub(nn.Module):
    """Prototype interface for future insert/delete graph diffusion."""

    def forward(self, graph_state, condition):
        ...

    def propose_actions(self, graph_state, condition):
        ...
```

需要表达的概念：

```text
actions = insert_atom / delete_atom / change_atom / change_bond
```

不要求真实训练，只要接口清楚。

---

### 5.2 Pairwise distance bridge

文件：

```text
src/multispec_geodiff/geometry/distance_matrix.py
src/multispec_geodiff/geometry/conformer.py
src/multispec_geodiff/geometry/mds.py
```

MVP 可实现真实功能：

1. 用 RDKit 生成 conformer；
2. 计算 pairwise distance matrix；
3. 保存 distance matrix heatmap；
4. 可选实现 classical MDS。

函数：

```python
def generate_rdkit_conformer(smiles: str, seed: int = 0) -> np.ndarray:
    ...

def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    ...

def classical_mds(distance_matrix: np.ndarray, dim: int = 3) -> np.ndarray:
    ...
```

这部分建议真实实现，因为成本不高，而且很能体现“2D -> distance -> 3D”的思想。

---

### 5.3 TFN-Transformer stub

文件：

```text
src/multispec_geodiff/models/tfn_transformer_stub.py
```

MVP 不实现完整 e3nn/TFN，只实现概念级 forward：

```python
class TensorFeature:
    scalar_even: torch.Tensor
    scalar_odd: torch.Tensor | None
    vector_even: torch.Tensor | None
    vector_odd: torch.Tensor | None

class TFNTransformerBlockStub(nn.Module):
    def forward(self, node_features, coords, edge_index, edge_attr, condition):
        ...
```

重点展示：

1. tensor inner-product attention 的接口；
2. distance bias；
3. edge bias；
4. relative coordinate update；
5. parity/chirality channel 的预留。

不建议在 demo 阶段安装复杂依赖，如 e3nn，除非时间充足。

---

## 6. 配置文件规划

`configs/demo.yaml`：

```yaml
seed: 42

input:
  ir_path: data/samples/query_ir.csv
  raman_path: data/samples/query_raman.csv
  candidate_library: data/samples/candidate_library.csv

spectra:
  x_min: 400
  x_max: 4000
  step: 1
  normalization: max
  smoothing: false

retrieval:
  top_m: 20
  top_k: 5
  use_mass_filter: false
  use_functional_group_filter: true

scoring:
  weights:
    ir: 0.45
    raman: 0.35
    peak: 0.20
  posterior:
    spectral: 0.75
    validity: 0.10
    mass: 0.10
    functional_group: 0.05

visualization:
  save_attention_heatmap: true
  save_distance_matrix: true
  dpi: 300

output:
  dir: outputs
  trace: outputs/trace_log.json
```

---

## 7. 代码实现顺序

推荐按以下顺序写代码，不要从模型开始：

### Phase A：跑通数据和输出

1. `spectra_io.py`
2. `preprocess.py`
3. `candidate_index.py`
4. `similarity.py`
5. `reranker.py`
6. `run_demo.py`
7. `topk_candidates.csv`

目标：先能输出 Top-K。

### Phase B：补齐化学与可视化

1. `rdkit_utils.py`
2. `molecule_plot.py`
3. `spectra_plot.py`
4. `trace.py`
5. `demo_summary.md`

目标：生成可展示结果。

### Phase C：加入核心技术展示模块

1. `multispec_encoder.py`
2. `attention_plot.py`
3. `distance_matrix.py`
4. `conformer.py`
5. `mds.py`

目标：展示横坐标 attention 和 pairwise distance。

### Phase D：加入高级接口

1. `graph_diffusion_stub.py`
2. `distance_head_stub.py`
3. `tfn_transformer_stub.py`

目标：展示未来扩展路线，不追求训练。

---

## 8. `run_demo.py` 伪代码

```python
def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg.seed)
    trace = TraceLogger(cfg)

    # 1. Load query spectra
    query_ir = load_spectrum(cfg.input.ir_path, modality="IR")
    query_raman = load_spectrum(cfg.input.raman_path, modality="Raman")

    # 2. Preprocess spectra
    query_ir = preprocess_spectrum(query_ir, cfg.spectra)
    query_raman = preprocess_spectrum(query_raman, cfg.spectra)

    # 3. Load candidate library
    index = CandidateIndex.from_csv(cfg.input.candidate_library)

    # 4. Retrieve Top-M candidates
    hits = index.search(
        query_ir=query_ir,
        query_raman=query_raman,
        top_m=cfg.retrieval.top_m,
    )

    # 5. Chemical filtering
    hits = apply_chemical_filters(hits, cfg.retrieval)

    # 6. Forward spectrum consistency reranking
    ranked = rerank_candidates(
        hits=hits,
        query_ir=query_ir,
        query_raman=query_raman,
        cfg=cfg.scoring,
    )

    topk = ranked[: cfg.retrieval.top_k]

    # 7. Optional coordinate-aware encoder visualization
    attention_map = run_encoder_prototype(query_ir, query_raman, cfg)

    # 8. Optional pairwise distance demo for Top-1
    coords = generate_rdkit_conformer(topk[0].smiles)
    distance_matrix = compute_distance_matrix(coords)

    # 9. Save outputs
    save_topk_csv(topk, "outputs/topk_candidates.csv")
    plot_spectrum_match(query_ir, query_raman, topk, "outputs/spectrum_match.png")
    draw_molecule_grid(topk, "outputs/molecule_grid.png")
    plot_attention_heatmap(attention_map, "outputs/attention_heatmap.png")
    plot_distance_matrix(distance_matrix, "outputs/distance_matrix.png")

    # 10. Save trace and summary
    trace.record_topk(topk)
    trace.save("outputs/trace_log.json")
    write_demo_summary(topk, cfg, "outputs/demo_summary.md")
```

---

## 9. 数据准备方案

### 9.1 最稳方案：内置小样例库

准备 20-50 个小分子：

- ethanol；
- acetone；
- acetic acid；
- benzene；
- toluene；
- phenol；
- aniline；
- ethyl acetate；
- cyclohexane；
- pyridine；
- furan；
- thiophene；
- benzaldehyde；
- anisole；
- nitrobenzene；
- 等。

每个分子配 IR/Raman 简化谱图。谱图来源可以是：

1. 公开数据；
2. 手工构造峰形；
3. 用已有谱图库片段整理；
4. 用小型模拟生成假数据。

为了 demo 可运行，允许先使用 synthetic spectra，但需要在 README 中明确说明：

```text
The bundled sample data are for demonstrating the executable workflow, not for reporting final scientific accuracy.
```

### 9.2 synthetic spectra 生成方法

可以用高斯峰叠加：

```text
I(x)=sum_k A_k exp(-(x-mu_k)^2/(2 sigma_k^2)) + noise
```

为不同官能团设置峰区：

| 官能团 | IR peak region 示例 |
|---|---|
| C=O | 1650-1800 cm^-1 |
| O-H | 3200-3600 cm^-1 |
| C-H | 2800-3100 cm^-1 |
| C=C aromatic | 1450-1650 cm^-1 |
| C-O | 1000-1300 cm^-1 |

这可以让 demo 的检索结果直观合理。

---

## 10. 可行性分析

### 10.1 为什么 MVP 可行

MVP 不依赖大规模训练，主要依赖：

1. CSV 谱图处理；
2. 候选库检索；
3. 谱图相似度计算；
4. RDKit 分子处理；
5. 可视化；
6. trace log。

这些都是短周期可以完成的工程任务。

### 10.2 为什么仍能体现创新性

虽然 MVP 不训练完整 diffusion，但它展示了关键科学思路：

1. 谱图 token 使用真实横坐标；
2. 多模态谱图不是简单拼接；
3. 候选结构通过前向谱图验证；
4. 输出 Top-K，而不是单一分子；
5. workflow 可追踪；
6. 未来能自然替换候选检索为 graph diffusion。

### 10.3 主要风险与规避

| 风险 | 表现 | 规避方案 |
|---|---|---|
| 数据不足 | 检索结果不稳定 | 先用 synthetic + small curated library |
| 真实谱图噪声 | 匹配分数不可靠 | 平滑、归一化、peak-based score |
| 候选库覆盖不足 | 正确分子不在库中 | Demo 明确是候选库闭环，未来换 graph diffusion |
| RDKit 依赖安装 | 环境问题 | 提供 environment.yml 和 fallback no-RDKit mode |
| 过度复杂 | 做不完 | MVP 和 Advanced prototype 分离 |
| 评审误解为完整系统 | 期望不一致 | README 中明确 demo scope |

---

## 11. 测试计划

### 11.1 单元测试

```text
tests/test_spectra_io.py
```

检查：

- CSV 能读；
- x/intensity 长度一致；
- normalization 后最大值为 1。

```text
tests/test_similarity.py
```

检查：

- 相同谱图 cosine = 1；
- 不同谱图分数更低；
- peak score 不报错。

```text
tests/test_rdkit_utils.py
```

检查：

- 合法 SMILES 可转 Mol；
- 非法 SMILES 返回 None；
- formula/mass 可计算。

```text
tests/test_reranker.py
```

检查：

- final score 可排序；
- Top-K 数量正确；
- 分数字段完整。

```text
tests/test_trace.py
```

检查：

- trace log 可写；
- JSON 可解析；
- 关键字段存在。

### 11.2 smoke test

```bash
python scripts/smoke_test.py
```

要求：

1. 能在 1-2 分钟内跑完；
2. outputs 中所有关键文件存在；
3. topk_candidates.csv 不为空；
4. trace_log.json 可解析。

---

## 12. 评估指标

MVP 指标：

| 指标 | 含义 | 是否必须 |
|---|---|---|
| Top-K hit | query 是否能召回目标分子 | 是 |
| IR similarity | IR 谱图相似度 | 是 |
| Raman similarity | Raman 谱图相似度 | 是 |
| reranking gain | 重排前后目标分子 rank 是否提升 | 是 |
| validity | RDKit sanitize 成功率 | 是 |
| trace completeness | trace log 字段完整度 | 是 |
| runtime | demo 运行时间 | 是 |

Advanced 指标：

| 指标 | 含义 | 阶段 |
|---|---|---|
| distance matrix MAE | 预测距离矩阵误差 | Phase II |
| conformer RMSD | 3D 构象误差 | Phase II |
| chirality consistency | 手性一致性 | Phase III |
| spectrum reconstruction error | forward model 谱图误差 | Phase III |

---

## 13. Demo 展示方式

### 13.1 README 中的展示

放三张图：

```text
assets/architecture.png
outputs/spectrum_match.png
outputs/molecule_grid.png
```

### 13.2 notebook 中的展示顺序

```text
1. Load query IR/Raman spectra
2. Plot query spectra
3. Load candidate library
4. Run retrieval
5. Show initial Top-M
6. Run forward spectrum reranking
7. Show Top-K candidate table
8. Draw molecule grid
9. Plot query vs candidate spectra
10. Show trace log
11. Show future modules: distance matrix and TFN stub
```

### 13.3 demo_summary.md

自动生成：

```markdown
# Demo Summary

## Input
- IR: ...
- Raman: ...

## Top-K Candidates
| rank | SMILES | formula | score | IR score | Raman score |

## Generated Files
- spectrum_match.png
- molecule_grid.png
- trace_log.json

## Notes
This demo validates the core executable workflow; future versions will replace retrieval with graph diffusion.
```

---

## 14. 和长期方案的接口设计

| 当前 MVP 模块 | 未来替换模块 | 接口是否保持 |
|---|---|---|
| CandidateIndex.search | SizeAdaptiveGraphDiffusion.sample | 是 |
| Stored candidate spectra | 2D/3D forward spectrum model | 是 |
| Handcrafted encoder | Trainable MultiSpec Encoder | 是 |
| RDKit conformer distance | Learned pairwise distance head | 是 |
| Reranker weighted score | PosteriorNet | 是 |
| TFN stub | real e3nn / Equiformer-style module | 是 |

这意味着代码设计要遵循接口稳定原则：

```text
QuerySpectra -> CandidateSet -> RankedCandidateSet -> OutputPackage
```

只要中间对象格式稳定，后续可以逐步替换模块。

---

## 15. 依赖规划

### 15.1 最小依赖

```text
numpy
pandas
scipy
scikit-learn
matplotlib
pyyaml
tqdm
pytest
```

### 15.2 推荐依赖

```text
rdkit
pillow
seaborn optional
jupyter
```

注意：如果使用 matplotlib 画图，不一定需要 seaborn。

### 15.3 可选高级依赖

```text
torch
e3nn optional
networkx optional
```

MVP 不强制 e3nn。

---

## 16. environment.yml 建议

```yaml
name: multispec-geodiff-demo
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - matplotlib
  - pyyaml
  - tqdm
  - pytest
  - jupyter
  - rdkit
  - pip
  - pip:
      - torch
```

如果 RDKit 安装困难，README 中应提供 fallback：

```text
python scripts/run_demo.py --config configs/demo.yaml --no-rdkit
```

fallback 模式只输出 SMILES 文本和谱图，不画分子图。

---

## 17. 里程碑计划

### Milestone 1：能跑通 Top-K

产出：

- data/samples；
- run_demo.py；
- topk_candidates.csv。

验收：

```bash
python scripts/run_demo.py --config configs/demo.yaml
```

可以成功输出 Top-K。

### Milestone 2：可视化完整

产出：

- spectrum_match.png；
- molecule_grid.png；
- demo_summary.md。

验收：

README 中能展示结果图。

### Milestone 3：traceable workflow

产出：

- trace_log.json；
- config snapshot；
- runtime info。

验收：

trace log 字段完整。

### Milestone 4：核心技术展示

产出：

- coordinate-aware encoder prototype；
- attention_heatmap.png；
- distance_matrix.png。

验收：

notebook 中能展示横坐标 bias 和 pairwise distance。

### Milestone 5：高级接口

产出：

- graph_diffusion_stub.py；
- tfn_transformer_stub.py；
- design notes。

验收：

代码结构能说明未来如何替换模块。

---

## 18. 提交物清单

最终提交建议包含：

```text
1. README.md
2. proposal.md
3. demo_report.md
4. notebooks/demo_pipeline.ipynb
5. scripts/run_demo.py
6. src/multispec_geodiff/
7. data/samples/
8. outputs/topk_candidates.csv
9. outputs/spectrum_match.png
10. outputs/molecule_grid.png
11. outputs/trace_log.json
12. assets/architecture.png
```

压缩包或 GitHub repo 中应保证：

```bash
python scripts/smoke_test.py
```

可以直接通过。

---

## 19. 不建议在第一版做的事

第一版 demo 不建议：

1. 训练完整可增删节点图扩散；
2. 训练完整 TFN-Transformer；
3. 引入 e3nn 复杂张量网络作为硬依赖；
4. 做大规模 NIST 数据清洗；
5. 做 Web 前端；
6. 做严格概率校准；
7. 承诺真实未知物鉴定准确率。

这些都可以作为未来路线，而不是当前 demo 的验收内容。

---

## 20. 最终 demo 叙事

评审打开 demo 后，应看到如下逻辑：

```text
1. 这是一个多模态谱图到分子结构反演任务。
2. Demo 支持 IR/Raman 输入。
3. 系统先对谱图做横坐标感知表示。
4. 再从候选库中召回可能分子。
5. 再用前向谱图一致性进行重排。
6. 最后输出 Top-K 分子、谱图匹配图和 trace log。
7. 未来可自然升级为 graph diffusion + pairwise distance + TFN-Transformer。
```

核心印象应是：

> 这个 demo 不是空泛的模型设想，而是一个能运行、能追踪、能可视化、能逐步替换为更强生成模型的 AI4S 小型科学工作流。

---

## 21. 给后续 Claude Code / Codex 的实现任务拆分

后续可以把任务拆成以下 prompt：

```text
Task 1: Create repository skeleton exactly following MultiSpec_GeoDiff_Demo_ImplementationPlan.md.
Task 2: Implement spectra_io.py, preprocess.py, similarity.py and tests.
Task 3: Implement candidate library loading and retrieval.
Task 4: Implement RDKit molecule utilities and molecule visualization.
Task 5: Implement reranking and output CSV.
Task 6: Implement run_demo.py and smoke_test.py.
Task 7: Implement trace_log.json and demo_summary.md generation.
Task 8: Implement coordinate-aware encoder prototype and attention heatmap.
Task 9: Implement distance matrix demo using RDKit conformer.
Task 10: Add graph_diffusion_stub.py and tfn_transformer_stub.py as future extension interfaces.
Task 11: Update README with quick start and generated figures.
Task 12: Run smoke test and fix all errors.
```

---

## 22. 最终检查清单

### 工程检查

- [ ] `python scripts/run_demo.py --config configs/demo.yaml` 可运行；
- [ ] `python scripts/smoke_test.py` 可运行；
- [ ] `outputs/topk_candidates.csv` 存在且非空；
- [ ] `outputs/spectrum_match.png` 存在；
- [ ] `outputs/molecule_grid.png` 存在；
- [ ] `outputs/trace_log.json` 存在且可解析；
- [ ] README 的 quick start 与实际命令一致。

### 科学叙事检查

- [ ] 明确说明 demo 是最小闭环；
- [ ] 明确说明候选检索未来可替换为 graph diffusion；
- [ ] 明确说明前向谱图重排的必要性；
- [ ] 明确展示横坐标感知谱图编码；
- [ ] 明确展示 pairwise distance 作为 2D-to-3D 桥梁；
- [ ] 明确保留 TFN-Transformer 和手性表达作为未来升级。

### 提交观感检查

- [ ] repo 文件不乱；
- [ ] 数据样例小而完整；
- [ ] 图片清晰；
- [ ] trace log 体现可追踪；
- [ ] 不夸大结果；
- [ ] 能在 3-5 分钟内看懂 demo 核心价值。

