下面给出一版更“宏大”、更聚焦核心卖点的最终方案。它不再是“谱图翻译成 SMILES”，而是一个面向科学反演的统一框架：

[
\boxed{
\textbf{MultiSpec-GeoDiff-TFN：多模态谱图驱动的几何扩散分子反演模型}
}
]

它的基本思想是：

[
\boxed{
\text{多模态谱图}
\rightarrow
\text{条件表示}
\rightarrow
\text{可增删节点的 2D 图扩散}
\rightarrow
\text{pairwise distance 预测}
\rightarrow
\text{距离几何恢复 3D}
\rightarrow
\text{TFN-Transformer 等变精修}
\rightarrow
\text{前向谱图一致性重排}
\rightarrow
\text{Top-}K\text{候选分子}
}
]

这个方案和你给的 Ref.md 是一致的：Ref.md 已经把问题定位为“多模态谱图输入、分子图生成、距离矩阵/三维坐标恢复、EGNN/TFN 等变建模，以及基于前向谱图与物理约束的后验重排”的小分子二维/三维结构反演任务。 Ref.md 也强调，最值得优先优化的是实验域数据、候选重排、距离几何约束和前向谱图模型的闭环训练。

---

# 1. 总体图景：从“谱图语言”到“几何分子世界”的反演

传统方案通常是：

[
\text{Spectrum}
\rightarrow
\text{SMILES}
]

但这个路径有三个根本问题：

1. **SMILES 是人为序列，不是分子本体。**
   同一个分子有多个 SMILES，序列顺序不具备物理意义。

2. **谱图到结构是 one-to-many 反问题。**
   一个谱图可能对应多个近似合理的候选分子，不能强行压成单一序列输出。

3. **谱图的物理来源是三维电子结构、振动模式、张量响应，而不是字符串。**
   如果模型只输出 SMILES，它很难自然利用 IR/Raman/NMR/UV/CD/VCD 背后的几何和张量物理。

所以我们的目标应升级为：

[
\boxed{
\mathcal S_{\mathrm{multi}}
\mapsto
{
G_{2D}^{(k)},D^{(k)},R^{(k)},\mathcal T^{(k)},\hat S^{(k)},s_k
}_{k=1}^{K}
}
]

其中：

* (\mathcal S_{\mathrm{multi}})：多模态谱图；
* (G_{2D}^{(k)}=(X^{(k)},E^{(k)}))：第 (k) 个候选二维分子图；
* (D^{(k)})：候选分子的 pairwise distance matrix；
* (R^{(k)})：由距离几何恢复并经等变层精修的三维坐标；
* (\mathcal T^{(k)})：每个原子的多阶张量特征；
* (\hat S^{(k)})：候选分子的前向预测谱图；
* (s_k)：后验排序分数；
* 最终输出 Top-(K) 候选，不强求严格概率分布。

这版模型的核心卖点不是“用了扩散模型”本身，而是：

[
\boxed{
\text{把谱图反演问题从序列翻译提升为几何约束下的多模态条件生成问题。}
}
]

---

# 2. 模块总览

最终架构分为六个大模块：

[
\mathcal S_{\mathrm{multi}}
\xrightarrow{\text{MultiSpec Encoder}}
(c,C)
\xrightarrow{\text{Insert/Delete 2D Graph Diffusion}}
G_{2D}
\xrightarrow{\text{Pairwise Distance Head}}
D
\xrightarrow{\text{Distance Geometry}}
R^{(0)}
\xrightarrow{\text{TFN-Transformer Refiner}}
(R,\mathcal T)
\xrightarrow{\text{Forward Spectrum + PosteriorNet}}
\text{Top-}K.
]

其中最重要的两个模块是：

1. **多模态谱图编码层**
   负责把 IR/Raman/NMR/MS/UV/CD/VCD 等谱图转化为统一条件表示。

2. **TFN-Transformer 几何精修层**
   负责在 3D 几何空间中更新坐标和张量特征，表达方向性、角分布、张量响应与手性。

---

# 3. 多模态谱图编码层：从多个实验观测中抽取结构证据

## 3.1 输入表示

设输入谱图为：

[
\mathcal S
==========

{
S^{\mathrm{IR}},
S^{\mathrm{Raman}},
S^{\mathrm{NMR}},
S^{\mathrm{MS}},
S^{\mathrm{UV}},
S^{\mathrm{CD}},
S^{\mathrm{VCD}}
}.
]

第 (m) 个模态写成：

[
S^{(m)}
=======

{(\nu_i^{(m)},I_i^{(m)})}_{i=1}^{L_m}.
]

其中：

* IR/Raman：(\nu) 是 (\mathrm{cm}^{-1})；
* NMR：(\nu) 是 ppm；
* MS：(\nu) 是 (m/z)；
* UV/CD：(\nu) 是 wavelength 或 energy；
* VCD/ROA：(\nu) 仍是振动频率，但强度带有手性响应信息。

每个模态先经过 CNN stem：

[
Z^{(m)}
=======

\operatorname{CNN}_m(S^{(m)}).
]

CNN 的作用不是“理解分子”，而是提取局部谱图结构：

[
\text{peak shape},\quad
\text{peak width},\quad
\text{shoulder peak},\quad
\text{local baseline},\quad
\text{noise pattern}.
]

然后加入三类 embedding：

[
H_i^{(m,0)}
===========

Z_i^{(m)}
+
e_m
+
p_m(\nu_i^{(m)}).
]

其中：

* (e_m)：模态 embedding；
* (p_m(\nu))：真实谱图横坐标位置编码；
* (Z_i^{(m)})：CNN 提取的局部峰特征。

关键是：**位置编码必须由真实横坐标决定，而不是由 token index 决定。**

[
p_m(\nu)
========

\operatorname{MLP}_m
\left[
\nu,
\sin(\omega_1\nu),
\cos(\omega_1\nu),
\dots,
\sin(\omega_K\nu),
\cos(\omega_K\nu)
\right].
]

这样模型知道：

[
1700\ \mathrm{cm}^{-1}
\neq
3000\ \mathrm{cm}^{-1},
]

因为前者可能对应 C=O stretching，后者可能对应 C-H stretching。谱图 token 的物理含义来自横坐标，而不只是序列位置。

---

## 3.2 模态内 attention：谱图横坐标偏置

每个模态内部先做 self-attention：

[
A_{ij}^{(m)}
============

\operatorname{softmax}*j
\left(
\frac{
Q_i^{(m)}K_j^{(m)\top}
}{\sqrt d}
+
b_m(\Delta \nu*{ij}^{(m)})
\right).
]

其中：

[
\Delta \nu_{ij}^{(m)}
=====================

|\nu_i^{(m)}-\nu_j^{(m)}|.
]

这里的 (b_m) 不需要复杂 MLP，建议使用 RBF 展开：

[
b_m(\Delta\nu)
==============

w_m^\top \operatorname{RBF}_m(\Delta\nu).
]

这有很强物理意义：在同一个谱图模态内部，两个峰的相对位置差是有意义的。例如 IR 中 (1700\ \mathrm{cm}^{-1}) 附近峰和 (1200\ \mathrm{cm}^{-1}) 附近峰之间的关系，可能共同指向某类官能团组合。

这个设计与 Graphormer 的思想一致：普通 self-attention 只计算语义相似性，不考虑图结构；Graphormer 的关键就是把结构信息编码进 attention，使 Transformer 能感知节点间关系。 在我们的模型里，谱图横坐标距离就是“谱图 token 的结构关系”。

---

## 3.3 跨模态 attention：不加横坐标位置偏置

你指出 cross-attention 不加位置偏置，这是正确的。

跨模态 attention 写成：

[
A_{ij}^{(m\leftarrow n)}
========================

\operatorname{softmax}*j
\left(
\frac{
Q_i^{(m)}K_j^{(n)\top}
}{\sqrt d}
+
\gamma*{mn}
\right).
]

不加入：

[
|\nu_i^{(m)}-\nu_j^{(n)}|.
]

原因是：

[
\nu^{\mathrm{IR}},\quad
\nu^{\mathrm{NMR}},\quad
\nu^{\mathrm{MS}},\quad
\nu^{\mathrm{UV}}
]

不处于同一个物理坐标系。IR 的 (1700\ \mathrm{cm}^{-1}) 和 NMR 的 7 ppm 之间没有自然距离。跨模态 attention 应该学习的是“证据互补关系”，而不是“横坐标邻近关系”。

例如：

* IR 的 C=O peak 与 MS 的分子量/碎片峰共同支持羰基结构；
* NMR 的 aromatic proton chemical shift 与 UV 的吸收红移共同支持共轭芳香体系；
* Raman 与 IR 对振动模式选择定则不同，互补提供对称性信息；
* CD/VCD/ROA 与手性构型有关，可为后续 chirality head 提供条件证据。

因此跨模态 attention 的本质是：

[
\boxed{
\text{不同谱图模态之间的证据融合，而不是不同横坐标空间之间的几何注意力。}
}
]

---

## 3.4 Encoder block 结构

一个完整的 MultiSpec Encoder block 为：

[
H^{(m)}
\leftarrow
H^{(m)}
+
\operatorname{MHSA}_{\Delta\nu}^{(m)}(H^{(m)}),
]

[
H^{(m)}
\leftarrow
H^{(m)}
+
\sum_{n\neq m}
\operatorname{CrossAttn}^{(m\leftarrow n)}(H^{(m)},H^{(n)}),
]

[
H^{(m)}
\leftarrow
H^{(m)}
+
\operatorname{FFN}_m(H^{(m)}).
]

最后输出：

[
c_{\mathrm{global}}
===================

\operatorname{Pool}\left({H^{(m)}}_m\right),
]

[
C_{\mathrm{tokens}}
===================

\operatorname{Concat}\left(H^{(1)},\dots,H^{(M)}\right).
]

其中：

* (c_{\mathrm{global}})：整体分子条件，如分子大小、官能团组合、芳香性、共轭程度、手性可能性；
* (C_{\mathrm{tokens}})：细粒度谱图 token 条件，用于扩散过程和 3D 精修过程中的 cross-attention/gating。

---

# 4. 2D 图扩散：大小自适应的候选图生成

我们不固定 (N_{\max})，而采用可增删节点的 graph diffusion。原因很明确：谱图反演时，分子大小本身就是未知量。固定节点数会引入大量空节点，也会削弱模型对分子式、分子量和骨架复杂度的自适应能力。

GRIDDD 这类工作指出，传统图扩散的核心限制是不能在扩散过程中改变图大小，而分子性质往往与分子大小相关；因此允许插入和删除节点，是条件分子生成的重要方向。

二维图状态写成：

[
G_t=(X_t,E_t).
]

其中：

[
X_t={x_i}_{i=1}^{N_t},
]

[
E_t={e_{ij}}_{i,j=1}^{N_t}.
]

反向去噪动作包括：

[
a_t
\in
{
\text{insert atom},
\text{delete atom},
\text{change atom type},
\text{change bond type}
}.
]

denoiser 为条件 Graph Transformer：

[
D_\theta(G_t,t,c_{\mathrm{global}},C_{\mathrm{tokens}})
\rightarrow
a_t.
]

---

## 4.1 2D Graph Transformer attention

二维阶段没有 3D 坐标，但有图拓扑距离和边类型。因此 attention 写成：

[
A_{ij}
======

\operatorname{softmax}*j
\left(
\frac{Q_iK_j^\top}{\sqrt d}
+
\beta*{\operatorname{SPD}(i,j)}
+
\eta_{e_{ij}}
\right).
]

其中：

* (\operatorname{SPD}(i,j))：图最短路距离；
* (\beta_{\operatorname{SPD}(i,j)})：拓扑距离偏置；
* (\eta_{e_{ij}})：边类型偏置。

这和 Graphormer 的精神一致：Transformer 用于图时，必须引入 structural encoding / structure-aware attention。Graph Transformer survey 也把图 Transformer 的结构信息处理分为图 tokenization、positional encoding、structure-aware attention 和 GNN-Transformer ensemble 等路线。

但我们的设计保持克制：

[
\boxed{
\text{2D 阶段只加入最短路距离偏置和边偏置，不引入过多手工化学规则。}
}
]

这样既有结构感，又不过度硬编码。

---

# 5. 从 2D 到 3D：直接预测 pairwise distance matrix

你这次指出“直接给出 pairwise distance 更合理”，这是非常关键的优化。

因为后面的等变层真正依赖的是：

[
r_{ij}=r_i-r_j,
\qquad
|r_{ij}|.
]

而绝对坐标 (R) 的整体旋转和平移没有意义。因此与其直接预测坐标，不如先预测距离矩阵：

[
D
=

[d_{ij}],
\qquad
d_{ij}
======

|r_i-r_j|.
]

距离矩阵天然满足：

[
D(R)=D(QR+t),
]

也就是对旋转和平移不变。

所以 2D→3D 的桥梁改成：

[
G_{2D},c_{\mathrm{global}},C_{\mathrm{tokens}}
\rightarrow
\hat D.
]

而不是直接：

[
G_{2D}\rightarrow \hat R.
]

---

## 5.1 Pairwise distance head

先用 2D Graph Transformer 得到节点表示：

[
h_i
===

\operatorname{GraphTransformer}(X,E,c).
]

然后预测 pairwise distance：

[
\hat d_{ij}
===========

\operatorname{Softplus}
\left(
\phi_D
[
h_i+h_j,\ |h_i-h_j|,\ h_i\odot h_j,\ e_{ij},\ c_{\mathrm{global}}
]
\right).
]

为保证对称性：

[
\hat d_{ij}=\hat d_{ji}.
]

为保证基本几何合理性，可以加入软约束：

[
\hat d_{ij}>0,
]

[
|\hat d_{ij}-\hat d_{ik}|\leq \hat d_{jk}\leq \hat d_{ij}+\hat d_{ik}.
]

训练损失：

[
\mathcal L_D
============

\sum_{i<j}
\left(
\hat d_{ij}-d_{ij}^{\mathrm{ref}}
\right)^2.
]

如果没有真实 3D 构象，可以用 RDKit/ETKDG 或量化模拟结构作为弱监督。

---

## 5.2 距离几何恢复初始坐标

由 (\hat D) 得到初始坐标：

[
\hat D
\rightarrow
R^{(0)}.
]

可以用 Classical MDS：

[
B
=

-\frac{1}{2}
J \hat D^{\circ 2} J,
]

[
J=I-\frac{1}{N}\mathbf 1\mathbf 1^\top,
]

对 (B) 做特征分解：

[
B=U\Lambda U^\top,
]

取前三个最大正特征值：

[
R^{(0)}
=======

U_{:,1:3}\Lambda_{1:3}^{1/2}.
]

这一步的优势是：

[
\boxed{
\text{先预测旋转平移不变量，再恢复任意 gauge 下的 3D 坐标。}
}
]

后续 EGNN/TFN 层只需要相对坐标和相对距离，因此初始坐标的绝对朝向不重要。

---

# 6. TFN-Transformer 层：模型的核心卖点

TFN-Transformer 层负责把粗略 3D 坐标和 2D 图结构精修成带有物理响应能力的三维分子表示：

[
(G_{2D},R^{(0)},c,C)
\rightarrow
(R,\mathcal T).
]

它同时更新：

1. 相对坐标；
2. 标量特征；
3. 向量特征；
4. 二阶/高阶张量特征；
5. 手性相关的奇宇称特征。

这部分是模型最重要的几何深度学习创新。

---

## 6.1 节点张量特征

每个原子节点 (i) 保存一组 irreps 特征：

[
\mathcal T_i
============

{
T_i^{(l,p)}
}.
]

其中：

* (l=0,1,2,\dots,l_{\max})：角动量阶数；
* (p\in{e,o})：宇称，even/odd parity；
* (T_i^{(l,p)}\in \mathbb R^{C_{l,p}\times(2l+1)})。

建议第一版使用：

[
\boxed{
0e,\ 0o,\ 1e,\ 1o,\ 2e,\ 2o
}
]

其中：

* (0e)：普通标量，如原子环境、局部电负性、质量相关特征；
* (0o)：伪标量，可表达手性；
* (1e/1o)：极向量/轴向量；
* (2e/2o)：二阶张量，可表达极化率、局部角分布、振动响应相关特征。

几何 GNN 综述指出，EGNN/PaiNN 通过引入几何向量保留方向信息，而 TFN、SE(3)-Transformer、SEGNN 则把标量和向量推广到高阶 steerable spherical tensors。 这正是我们把 TFN 放在 3D 精修层的理由。

---

## 6.2 张量内积 attention：用 invariant scalar 做注意力权重

普通 Transformer attention 是：

[
\frac{Q_iK_j^\top}{\sqrt d}.
]

但在 TFN-Transformer 中，(Q_i) 和 (K_j) 不是普通向量，而是多阶张量特征。如果直接拼接后做普通 dot product，会破坏几何变换规律。

因此 attention 权重必须是旋转不变量 scalar。

定义每阶张量的 query/key：

[
Q_i^{(l,p)}
===========

W_Q^{(l,p)}T_i^{(l,p)},
]

[
K_j^{(l,p)}
===========

W_K^{(l,p)}T_j^{(l,p)}.
]

张量内积 attention score 为：

[
s_{ij}^{\mathrm{tensor}}
========================

\sum_{l,p}
\sum_{c=1}^{C_{l,p}}
\left\langle
Q_{i,c}^{(l,p)},
K_{j,c}^{(l,p)}
\right\rangle.
]

由于同阶同宇称 irreps 的内积是旋转不变量，所以：

[
s_{ij}^{\mathrm{tensor}}
]

是 invariant scalar。

这就是我们的核心卖点之一：

[
\boxed{
\text{attention 权重不是普通向量点积，而是多阶张量特征的 invariant inner product。}
}
]

几何 GNN 综述中对 SE(3)-Transformer 的描述也指出，其 attention coefficient 由 query 和 key 的内积得到，并保证旋转不变性；随后用该不变量权重聚合 steerable value，从而保持 SE(3)-equivariance。

---

## 6.3 距离偏置 + 边偏置

你的设想里，(b_{\mathrm{graph}}) 分成两部分：

[
b_{\mathrm{graph}}
==================

b_{\mathrm{dist}}
+
b_{\mathrm{edge}}.
]

我建议最终 attention score 写成：

[
s_{ij}
======

s_{ij}^{\mathrm{tensor}}
+
b_{\mathrm{dist}}(|r_i-r_j|)
+
b_{\mathrm{edge}}(e_{ij}).
]

其中：

[
b_{\mathrm{dist}}(|r_i-r_j|)
============================

w_d^\top \operatorname{RBF}(|r_i-r_j|),
]

[
b_{\mathrm{edge}}(e_{ij})
=========================

w_e^\top \operatorname{Embed}(e_{ij}).
]

然后：

[
\alpha_{ij}
===========

\operatorname{softmax}*j(s*{ij}).
]

这里没有复杂 MLP，只有：

1. 张量内积；
2. 相对距离 RBF；
3. 边类型 embedding。

这很重要，因为它让模型有明确物理含义：

[
\boxed{
\text{谁和谁交互，主要由张量态相似性、空间距离和化学键关系共同决定。}
}
]

这比“把所有东西丢进 MLP 学一个 attention bias”更干净，也更符合你希望“不引入过多归纳偏置”的要求。

---

## 6.4 TFN tensor message

相对坐标：

[
r_{ij}=r_i-r_j.
]

相对方向：

[
\hat r_{ij}
===========

\frac{r_{ij}}{|r_{ij}|}.
]

球谐函数：

[
Y_l(\hat r_{ij}).
]

对每条边 ((i,j))，构造 TFN 消息：

[
M_{ij}^{(l_3,p_3)}
==================

\sum_{l_1,p_1,l_2,p_2}
\phi_{l_1,l_2,l_3}^{p_1,p_2,p_3}
(|r_{ij}|,e_{ij})
\left(
T_j^{(l_1,p_1)}
\otimes
Y_{l_2}^{(p_2)}(\hat r_{ij})
\right)_{l_3,p_3}.
]

角动量耦合满足：

[
|l_1-l_2|
\leq
l_3
\leq
l_1+l_2.
]

宇称满足：

[
p_3=p_1p_2.
]

然后用 attention 权重聚合：

[
\Delta T_i^{(l,p)}
==================

\sum_j
\alpha_{ij}
M_{ij}^{(l,p)}.
]

更新：

[
T_i^{(l,p)}
\leftarrow
T_i^{(l,p)}
+
\Delta T_i^{(l,p)}.
]

Equiformer 的核心正是把 Transformer 操作替换为等变操作，使用 irreps features 和 tensor products，使 Transformer 能适配 3D atomistic graphs。 EquiformerV2 进一步指出，提高 irreps degree 会带来更强表达能力，但 tensor product 计算成本会成为瓶颈，因此第一版采用 (l_{\max}=2) 是合理折中。

---

# 7. 相对坐标更新机制：EGNN 思想与张量调制结合

坐标更新使用 EGNN 风格：

[
r_i^{(\ell+1)}
==============

r_i^{(\ell)}
+
\sum_{j\neq i}
(r_i^{(\ell)}-r_j^{(\ell)})
\cdot
\psi_{ij}^{(\ell)}.
]

其中：

[
\psi_{ij}^{(\ell)}
==================

\phi_x
\left(
T_i^{(0e)},
T_j^{(0e)},
e_{ij},
|r_i-r_j|^2,
c_{\mathrm{global}}
\right).
]

更进一步，可以让张量 attention 也参与坐标更新：

[
\psi_{ij}^{(\ell)}
==================

\alpha_{ij}
\cdot
\phi_x
\left(
|r_i-r_j|^2,
e_{ij},
\langle T_i,T_j\rangle_{\mathrm{tensor}},
c_{\mathrm{global}}
\right).
]

于是：

[
r_i^{(\ell+1)}
==============

r_i^{(\ell)}
+
\sum_{j\neq i}
\alpha_{ij}
(r_i^{(\ell)}-r_j^{(\ell)})
\phi_x
\left(
|r_i-r_j|^2,
e_{ij},
\langle T_i,T_j\rangle_{\mathrm{tensor}},
c
\right).
]

这有几个好处：

1. 坐标更新方向来自相对向量 (r_i-r_j)，保证旋转等变；
2. 更新幅度由距离、边类型、张量相似性和谱图条件决定；
3. 坐标和张量特征相互耦合，而不是分离更新；
4. 仍然不依赖绝对坐标方向。

EGNN 的核心优势正是同时更新节点特征和坐标，并在不使用昂贵高阶表示的情况下保持 (E(n)) 等变；EGNN 文献中也强调，坐标和节点 embedding 在 edge operation 中交换信息，从而保持灵活性和等变性。

我们的方案是在 EGNN 的坐标更新上加入 TFN 的高阶张量特征，使其同时具备：

[
\boxed{
\text{EGNN 的稳定坐标更新}
+
\text{TFN 的高阶角向/张量表达}
+
\text{Transformer 的全局交互能力}
}
]

---

# 8. 谱图条件如何注入 TFN-Transformer

谱图条件 (C_{\mathrm{tokens}}) 本身是标量序列，不随三维旋转变化。因此它不能直接加到 (1e,1o,2e,2o) 等张量通道上，否则会破坏等变性。

正确做法是：谱图条件只通过 scalar gate 调制张量通道。

先用 scalar channel 作为 query：

[
g_i
===

\operatorname{CrossAttn}
\left(
Q=T_i^{(0e)},
K=C_{\mathrm{tokens}},
V=C_{\mathrm{tokens}}
\right).
]

然后生成各阶 gate：

[
\Gamma_{l,p}(g_i)
=================

\operatorname{MLP}_{l,p}(g_i).
]

更新：

[
T_i^{(l,p)}
\leftarrow
T_i^{(l,p)}
+
\Gamma_{l,p}(g_i)
\odot
T_i^{(l,p)}.
]

这样谱图条件只改变每个通道的幅度，不改变张量的旋转变换规律。

这是第二个核心卖点：

[
\boxed{
\text{多模态谱图不是粗暴拼接到 3D 特征里，而是通过 scalar gates 调制等变张量场。}
}
]

---

# 9. 手性表达：通过 odd parity 与 pseudoscalar 通道实现

手性是你方案里很有辨识度的卖点。关键是要讲清楚：**模型结构可以表达手性，但能否从数据中判别手性取决于输入谱图是否含手性敏感信息。**

普通 IR/Raman/MS/NMR 在非手性环境中通常难以区分一对对映体；但 CD、VCD、ROA 或手性环境下的 NMR 可以提供手性信息。因此我们在模型结构上保留 chirality capacity。

手性对应反射变换下符号改变的量。典型例子是三重积：

[
\chi_{ijkl}
===========

(r_j-r_i)\cdot
\left[
(r_k-r_i)\times(r_l-r_i)
\right].
]

在旋转下：

[
\chi_{ijkl}\mapsto \chi_{ijkl},
]

在反射下：

[
\chi_{ijkl}\mapsto -\chi_{ijkl}.
]

因此它是一个 pseudoscalar：

[
0o.
]

也就是说，手性不是普通标量 (0e)，而是奇宇称标量 (0o)。

我们的 TFN-Transformer 保留：

[
0e,\ 0o,\ 1e,\ 1o,\ 2e,\ 2o.
]

张量积满足宇称乘法：

[
p_3=p_1p_2.
]

因此模型可以从向量和轴向量的组合中形成 (0o) 手性通道。例如：

[
1e \otimes 1e \rightarrow 1o,
]

[
1o \otimes 1e \rightarrow 0o.
]

最终手性读出：

[
\chi_{\mathrm{mol}}
===================

\operatorname{Pool}
\left(
{T_i^{(0o)}}_{i=1}^N
\right).
]

如果输入包含 CD/VCD/ROA，则后验评分网络可以使用：

[
\chi_{\mathrm{mol}}
]

来区分对映体。如果没有手性敏感模态，则模型可以输出两个镜像候选，或者把绝对构型视为不确定。

这个设计的宏观意义是：

[
\boxed{
\text{模型不是只恢复连接关系，而是恢复分子的拓扑、几何、张量响应和手性结构。}
}
]

---

# 10. 前向谱图闭环：从几何张量到谱图

最终候选分子不能只靠生成模型判断，还要通过前向谱图模型检验。

Ref.md 也明确指出，方案不应只是端到端翻译，而应强调前向模型和物理一致性重排；它还建议把多模态 forward heads 做成“共享几何骨干 + 模态特异不确定度头”。

因此我们设计：

[
(R,\mathcal T,G_{2D})
\rightarrow
\mathcal P
\rightarrow
\hat{\mathcal S}_{\mathrm{multi}}.
]

其中 (\mathcal P) 是中间物理量：

[
\mathcal P
==========

{
\omega_k,
\partial\mu/\partial Q_k,
\partial\alpha/\partial Q_k,
\sigma_{\mathrm{NMR}},
E_{\mathrm{exc}},
f_{\mathrm{osc}},
R_{\mathrm{rot}},
\chi_{\mathrm{chiral}}
}.
]

然后不同谱图由不同 composer 得到：

[
\hat S_{\mathrm{IR}}
====================

\operatorname{Composer}_{\mathrm{IR}}
(\omega_k,\partial\mu/\partial Q_k),
]

[
\hat S_{\mathrm{Raman}}
=======================

\operatorname{Composer}_{\mathrm{Raman}}
(\omega_k,\partial\alpha/\partial Q_k),
]

[
\hat S_{\mathrm{NMR}}
=====================

\operatorname{Composer}*{\mathrm{NMR}}
(\sigma*{\mathrm{NMR}}),
]

[
\hat S_{\mathrm{UV}}
====================

\operatorname{Composer}*{\mathrm{UV}}
(E*{\mathrm{exc}},f_{\mathrm{osc}}),
]

[
\hat S_{\mathrm{CD/VCD}}
========================

\operatorname{Composer}*{\mathrm{chiral}}
(\chi*{\mathrm{chiral}},R_{\mathrm{rot}},\dots).
]

DetaNet 正是这一路线的重要参考：它结合 (E(3))-equivariance 和 self-attention，通过高阶几何张量消息预测 scalars、vectors、二阶和三阶张量，并用于 IR、Raman、UV-Vis、(^1)H/(^{13})C NMR 等谱图预测。

这说明我们的前向谱图闭环不是随意加的，而是有明确文献基础的。

---

# 11. 后验评分：不输出概率，只输出 Top-K

你不需要显式预测一个严格概率分布 (p(G\mid S))，这很合理。因为严格概率需要归一化、校准和大规模负样本，复杂度过高。

我们只输出 Top-(K)：

[
{\hat G_1,\dots,\hat G_K}.
]

每个候选有一个后验评分：

[
s_k
===

\operatorname{PosteriorNet}*\psi
\left(
\mathcal S*{\mathrm{input}},
\hat{\mathcal S}*k,
G*{2D,k},
D_k,
R_k,
\mathcal T_k,
f_{\mathrm{chem},k}
\right).
]

其中：

[
f_{\mathrm{chem},k}
]

包括：

* valence validity；
* ring validity；
* aromatic consistency；
* molecular mass consistency；
* formula consistency；
* functional group evidence；
* stereochemistry/chirality consistency；
* forward spectrum similarity；
* denoising trajectory confidence。

排序：

[
\hat G_{(1)},\dots,\hat G_{(K)}
===============================

\operatorname{TopK}
\left(
s_1,\dots,s_M
\right).
]

TranSpec + SpecGNN 的工作已经说明，“先生成候选，再用前向谱图模型模拟并重排”对实验 IR 识别很重要；该工作通过多源信息、增强、迁移、融合、质量过滤和候选重排，把实验 IR 任务表现显著提高。 我们的方案是把这个思路从 SMILES 层面提升到图、距离、3D 几何和张量响应层面。

---

# 12. 三阶段落地路线

## 阶段 I：2D 可增删节点图扩散

目标：

[
\mathcal S_{\mathrm{multi}}
\rightarrow
G_{2D}
]

先不做 3D，不做 TFN。重点验证：

1. 多模态 Encoder 是否能提取结构证据；
2. insert/delete graph diffusion 是否能生成合法分子图；
3. Top-(K) 候选是否优于 SMILES 直接生成；
4. 2D forward GNN reranker 是否能提升排序。

核心模块：

[
\text{MultiSpec Encoder}
+
\text{Insert/Delete Graph Diffusion}
+
\text{2D GNN Forward Scorer}
+
\text{PosteriorNet}_{2D}.
]

输出：

[
{\hat G_{2D,1},\dots,\hat G_{2D,K}}.
]

指标：

* Top-1 / Top-5 / Top-10 exact match；
* molecular formula accuracy；
* functional group recall；
* validity；
* uniqueness；
* novelty；
* scaffold accuracy；
* 2D Tanimoto similarity；
* forward spectrum similarity。

---

## 阶段 II：pairwise distance + 3D 初始化

目标：

[
G_{2D}
\rightarrow
D
\rightarrow
R^{(0)}.
]

重点验证：

1. 直接预测距离矩阵是否比直接预测坐标更稳定；
2. 距离几何恢复的 3D 构象是否合理；
3. 3D 信息是否改善 forward spectrum reranking。

核心模块：

[
\text{Pairwise Distance Head}
+
\text{Distance Geometry / MDS}
+
\text{Geometry Validity Filter}
+
\text{3D-aware PosteriorNet}.
]

损失：

[
\mathcal L_D
============

\sum_{i<j}
(\hat d_{ij}-d_{ij})^2.
]

可加入：

[
\mathcal L_{\mathrm{triangle}},
\quad
\mathcal L_{\mathrm{bond-length}},
\quad
\mathcal L_{\mathrm{angle}},
\quad
\mathcal L_{\mathrm{chirality}}.
]

指标：

* pairwise distance MAE；
* RMSD；
* conformer coverage；
* chirality consistency；
* spectrum reranking improvement；
* Top-(K) exact match improvement。

这一阶段是整个方案的关键桥梁：

[
\boxed{
\text{2D 图生成不是终点，距离矩阵才是通向 3D 几何和物理谱图的中间语言。}
}
]

---

## 阶段 III：TFN-Transformer 等变精修 + 张量谱图闭环

目标：

[
(G_{2D},R^{(0)},D,\mathcal S)
\rightarrow
(R,\mathcal T)
\rightarrow
\hat{\mathcal S}
\rightarrow
\text{Top-}K.
]

核心模块：

[
\text{EGNN Coordinate Update}
+
\text{TFN Tensor Message}
+
\text{Tensor Inner-product Attention}
+
\text{Parity-aware Chirality Channel}
+
\text{Physics Spectrum Composer}.
]

这一阶段是模型的最终形态。

TFN-Transformer block 可以概括为：

[
\mathcal T_i
\rightarrow
Q_i,K_i,V_i,
]

[
s_{ij}
======

\langle Q_i,K_j\rangle_{\mathrm{tensor}}
+
b_{\mathrm{dist}}(|r_i-r_j|)
+
b_{\mathrm{edge}}(e_{ij}),
]

[
\alpha_{ij}
===========

\operatorname{softmax}*j(s*{ij}),
]

[
\Delta \mathcal T_i
===================

\sum_j
\alpha_{ij}
\operatorname{TFNMessage}(\mathcal T_j,r_{ij},e_{ij}),
]

[
r_i
\leftarrow
r_i
+
\sum_j
\alpha_{ij}
(r_i-r_j)
\phi_x(|r_i-r_j|^2,e_{ij},\langle T_i,T_j\rangle,c).
]

最终：

[
(R,\mathcal T)
\rightarrow
\mathcal P
\rightarrow
\hat{\mathcal S}
\rightarrow
s_k.
]

指标：

* 3D RMSD；
* distance MAE；
* tensor property MAE；
* IR/Raman/NMR/UV/CD/VCD spectrum similarity；
* chirality accuracy；
* Top-(K) exact match；
* reranking gain；
* invalid molecule rate；
* runtime。

---

# 13. 最终方案的核心创新点

可以把这个方案的卖点浓缩为六句话：

## 1. 多模态谱图不是简单拼接，而是分层证据融合

模态内 attention 使用真实横坐标距离偏置：

[
b_m(|\nu_i-\nu_j|).
]

跨模态 attention 不加横坐标偏置，只学习模态间证据关系。

---

## 2. 分子不是字符串，而是可变大小图

Decoder 不输出 SMILES，而是通过可增删节点图扩散生成：

[
G_{2D}=(X,E).
]

这避免了 SMILES 顺序问题，也更适合 one-to-many 结构反演。

---

## 3. 2D 到 3D 的桥梁不是坐标，而是 pairwise distance

直接预测：

[
D=[d_{ij}].
]

再通过距离几何恢复：

[
D\rightarrow R^{(0)}.
]

这符合旋转平移不变性，也更贴近后续等变层的真实输入需求。

---

## 4. TFN-Transformer 的 attention 是张量内积 attention

attention 权重为：

[
s_{ij}^{\mathrm{tensor}}
========================

\sum_{l,p,c}
\langle Q_{i,c}^{(l,p)},K_{j,c}^{(l,p)}\rangle.
]

这是 invariant scalar，不破坏等变性。

---

## 5. 坐标和张量特征共同更新

坐标用 EGNN 式相对向量更新：

[
r_i
\leftarrow
r_i+\sum_j(r_i-r_j)\psi_{ij}.
]

张量用 TFN 式球谐张量积更新：

[
T_j^{(l_1,p_1)}\otimes Y_{l_2}^{(p_2)}(\hat r_{ij})
\rightarrow
T_i^{(l_3,p_3)}.
]

两者通过 attention 和 scalar gates 耦合。

---

## 6. 手性通过 odd-parity tensor channels 表达

保留：

[
0e,\ 0o,\ 1e,\ 1o,\ 2e,\ 2o.
]

其中 (0o) 是 chirality pseudoscalar 通道。模型可以在 CD/VCD/ROA 等手性敏感谱图存在时区分对映体。

---

# 14. 最终一句话版本

[
\boxed{
\textbf{MultiSpec-GeoDiff-TFN 是一个从多模态谱图出发，通过可变大小图扩散生成二维分子图，再以 pairwise distance 作为几何中间语言恢复三维结构，并用 parity-aware TFN-Transformer 更新坐标、张量特征和手性响应，最后通过前向谱图闭环重排 Top-}K\textbf{候选分子的谱图结构反演模型。}
}
]

这个图景足够宏大，但仍然可落地：阶段 I 做 2D 图扩散，阶段 II 做距离矩阵和 3D 初始化，阶段 III 做 TFN-Transformer、张量谱图和手性表达。它的核心不只是“更复杂”，而是每一层都对应一个清晰的科学原则：

[
\boxed{
\text{谱图侧尊重实验坐标；图生成侧尊重置换对称；几何侧尊重 }SE(3)/O(3)\text{ 表示；光谱侧尊重张量物理响应。}
}
]
