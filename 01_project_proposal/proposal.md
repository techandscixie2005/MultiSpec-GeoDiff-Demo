# 项目提议：MultiSpec-GeoDiff：多模态谱图驱动的几何分子结构反演

## 具体目标

拟构建多模态谱图驱动的几何分子结构反演模型，将反演形式化为几何约束下的多模态条件生成问题：

$$
\mathcal S_{\mathrm{multi}} \rightarrow (c,C) \rightarrow G_{2D} \rightarrow D \rightarrow R^{(0)} \rightarrow (R,\mathcal T) \rightarrow \hat{\mathcal S} \rightarrow \operatorname{TopK}.
$$

### 1. 多模态谱图编码：从实验坐标到条件表示

谱图反演的输入不是无结构序列：每个点都携带物理横坐标与强度。多模态谱图表示为峰集合 $S^{(m)} = \{(\nu_i^{(m)}, I_i^{(m)})\}_{i=1}^{L_m}$。CNN 提取局部峰形后，叠加模态编码与物理坐标位置编码：

$$
H_i^{(m,0)} = \operatorname{CNN}_m(I^{(m)})_i + e_m + p_m(\nu_i^{(m)}).
$$

模态内自注意力引入真实横坐标距离偏置：

$$
A_{ij}^{(m)} = \operatorname{softmax}_j\left(\frac{Q_i^{(m)}K_j^{(m)\top}}{\sqrt d} + b_m(|\nu_i^{(m)}-\nu_j^{(m)}|)\right).
$$

跨模态注意力不引入横坐标偏置（不同模态坐标系不可公度）：

$$
A_{ij}^{(m\leftarrow n)} = \operatorname{softmax}_j\left(\frac{Q_i^{(m)}K_j^{(n)\top}}{\sqrt d} + \gamma_{mn}\right).
$$

编码器输出全局条件 $c$ 与细粒度 token 条件 $C$。得到 $(c,C)$ 后，谱图不再只是信号，而成为约束分子生成过程的条件场。

### 2. 二维图扩散：从条件表示到候选分子图

谱图反演是一对多问题，生成比直接回归更自然。采用大小自适应图扩散，每一步允许原子增删与键变更：

$$
G_t = (X_t, E_t),\quad a_t \in \{\mathrm{insert},\mathrm{delete},\mathrm{atom},\mathrm{bond}\},\quad
D_\theta(G_t, t, c, C) \rightarrow a_t.
$$

反扩散 Graph Transformer 引入拓扑距离偏置与键类型偏置：

$$
\alpha_{ij} = \operatorname{softmax}_j\left(\frac{Q_iK_j^\top}{\sqrt d} + \beta_{\operatorname{SPD}(i,j)} + \eta_{e_{ij}}\right).
$$

输出 $G_{2D}=(X,E)$。但光谱响应还依赖三维几何；因此下一步不直接预测绝对坐标，而预测旋转平移不变的距离矩阵。

### 3. Pairwise distance：2D 图到 3D 几何的桥梁

绝对坐标具有旋转平移规范歧义，而 pairwise 距离矩阵天然不变：$D_{ij}=\|r_i-r_j\|$、$D(R)=D(QR+t)$。先预测距离矩阵：

$$
\hat D = f_\theta(G_{2D}, c, C),
$$

再经 Classical MDS 恢复初始坐标：

$$
R^{(0)} = \operatorname{DistGeom}(\hat D),\qquad
\mathcal L_D = \sum_{i<j} |\hat D_{ij} - D_{ij}^{\mathrm{ref}}|^2.
$$

距离几何只给出一个初始构象；真正的物理响应需在三维空间中同时更新坐标与张量特征。

### 4. TFN-Transformer：坐标、张量特征与等变注意力

各原子携带宇称分辨的多阶张量特征 $\mathcal T_i = \{T_i^{(l,p)}\}$（$l\in\{0,1,2\},\ p\in\{e,o\}$）。张量内积注意力为纯旋转不变量：

$$
s_{ij} = \sum_{l,p,c} \langle Q_{i,c}^{(l,p)}, K_{j,c}^{(l,p)}\rangle + b_{\mathrm{dist}}(\|r_i-r_j\|) + b_{\mathrm{edge}}(e_{ij}).
$$

TFN 消息通过球谐张量积实现等变交互：

$$
M_{ij}^{(l_3,p_3)} = \sum_{l_1,p_1,l_2,p_2} \phi(\|r_{ij}\|,e_{ij}) \bigl(T_j^{(l_1,p_1)} \otimes Y_{l_2}^{(p_2)}(\hat r_{ij})\bigr)_{l_3,p_3},\quad p_3 = p_1p_2.
$$

坐标以 EGNN 风格相对更新：

$$
r_i^{(\ell+1)} = r_i^{(\ell)} + \sum_{j \neq i} \alpha_{ij} (r_i^{(\ell)}-r_j^{(\ell)})\, \phi_x(\|r_i-r_j\|^2, e_{ij}, \langle \mathcal T_i,\mathcal T_j\rangle_{\mathrm{tensor}}, c).
$$

输出 $(R,\mathcal T)$。此时模型不再只是生成分子拓扑，而是在三维空间中形成可用于预测光谱响应的等变张量场。

### 5. 手性与宇称：从几何对称性到 $0o$ 通道

三重向量积 $\chi_{ijkl} = (r_j - r_i) \cdot [(r_k - r_i) \times (r_l - r_i)]$ 在旋转下不变、反射下反号，因此属于 $0o$（奇宇称标量）通道：

$$
\chi \mapsto \chi\ (\mathrm{SO(3)}),\qquad \chi \mapsto -\chi\ (\mathrm{reflection}) \;\Longrightarrow\; \chi \in 0o.
$$

$0o$ 通道可在 CD/VCD/ROA 等手性敏感模态监督下编码手性信息。当前 MVP 尚未实现手性区分。

### 6. 前向谱图一致性与 Top-K 后验重排

候选分子需经前向谱图验证：

$$
\hat{\mathcal S}_k = F_{\mathrm{spec}}(G_{2D,k}, R_k, \mathcal T_k),\quad
s_k = -\operatorname{Dist}(\mathcal S, \hat{\mathcal S}_k) + \lambda_{\mathrm{chem}} S_{\mathrm{chem}}(G_k) + \lambda_{\mathrm{traj}} S_{\mathrm{diff}}(G_k).
$$

最终 $\operatorname{TopK} = \operatorname{argsort}_k(s_k)$。生成模块负责提出候选，前向谱图模块负责物理校验，二者共同形成谱图反演的闭环。

## 核心价值

谱图反演本质是一谱多解的逆问题——横坐标携带物理含义，分子具有图拓扑与三维几何，而直接谱图到 SMILES 翻译忽视这两层结构。本方案将反演定义为几何约束下的多模态条件生成：以横坐标偏置注意力尊重物理坐标，以可增删节点图扩散自适应分子大小，以 pairwise distance 矩阵为二维到三维桥梁，以 TFN-Transformer 多阶张量特征与张量内积注意力实现 SE(3) 等变建模与手性响应，以前向谱图一致性验证候选。从而将谱图反演从序列翻译提升至几何物理约束下的生成与验证闭环。

## 可行性简析

分三阶段推进：阶段一实现 IR 谱图编码与候选重排闭环（已实现）；阶段二引入大小自适应图扩散与 pairwise distance 预测；阶段三加入 parity-aware TFN-Transformer。当前 Demo 使用 200 条 NIST 实验 IR 光谱，验证 IR 编码—候选召回—前向谱图重排—Top-K 输出闭环；完整阶段可扩展至 QM9S 与公开多模态谱图数据库。主要风险为实验噪声、一谱多解与数据规模有限；验证方式包括 Top-K 命中率、谱图相似度、候选重排增益、分子式/官能团匹配率、trace log 与 CI 测试。
