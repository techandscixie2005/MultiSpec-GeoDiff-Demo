# Limitations and Roadmap

## 1. 当前限制

### 1.1 输入范围有限
当前 MVP 只验证 **IR/Raman** 双模态。  
虽然设计叙事里保留了更广的多模态扩展接口，但仓库里没有把 NMR/MS/CD/VCD 做成已运行的主流程。

### 1.2 数据是 toy 数据
`scripts/generate_sample_data.py` 生成的是合成 Gaussian spectra。  
它适合做流程验证，但**不能**被写成实验 benchmark 或真实化学发现结果。

### 1.3 生成器不是训练好的 diffusion
当前候选召回主要依赖检索、软过滤和前向评分，不是训练好的 graph diffusion sampler。  
因此不要把当前 demo 表述成“已实现可增删节点的分子图扩散模型”。

### 1.4 3D 不是正式恢复
`geometry.py` 里的 classical MDS 只是 toy preview。  
它能帮助展示“距离矩阵 → 坐标”的直觉，但不能替代训练好的 pairwise distance head 或 3D refinement network。

### 1.5 依赖是可选降级的
`rdkit` 和 `torch` 在当前仓库里都属于可选依赖。  
缺失时 demo 会降级运行，所以文档里要把 fallback 说成设计选择，而不是 bug。

## 2. Roadmap-only 模块

| 模块 | 文件 | 说明 |
|---|---|---|
| Size-adaptive graph diffusion | `src/multispec_geodiff/future_modules/graph_diffusion_stub.py` | 只定义插入/删除节点等操作名 |
| Pairwise distance bridge | `src/multispec_geodiff/future_modules/pairwise_distance_stub.py` | 只返回启发式矩阵和 MDS 入口 |
| Parity-aware TFN-Transformer | `src/multispec_geodiff/future_modules/tfn_transformer_stub.py` | 只返回元数据和张量形状 |

## 3. 推荐升级顺序

1. **先增强数据与评测**  
   把 toy spectra 换成真实实验数据，并补齐统一评测协议。

2. **再强化检索和后验重排**  
   在当前候选排序闭环里增加更强的 forward model 和 calibration。

3. **再替换 graph diffusion stub**  
   把 `graph_diffusion_stub.py` 升级成真正的条件生成器。

4. **再接 pairwise distance**  
   让 2D 图和 3D 几何之间有明确的中间表征。

5. **最后接 TFN refinement**  
   在三维坐标恢复后再做 parity-aware 几何精修。

## 4. 文档写作规则

- 看到 `stub`、`roadmap-only`、`toy preview` 时，就要明确它不是已完成模型。
- 看到 `current demo` 时，只能指向已运行的 IR/Raman MVP。
- 看到 `future extension` 时，只能描述计划，不要写结果。

## 5. 最重要的一条

**不要把路线图写成结果。**  
这是这份仓库文档最需要守住的边界。
