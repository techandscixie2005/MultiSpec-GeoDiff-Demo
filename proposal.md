# 项目提议：MultiSpec-GeoDiff

## 具体目标
构建一个运行在 IR/Raman 双模态上的最小可执行谱图反演 demo：输入实验谱图，输出 Top-K 候选分子、谱图一致性分数、可视化结果与 trace log。当前只验证"谱图证据→候选排序"闭环，不声称已完成多模态大模型训练。graph diffusion、pairwise distance 与 TFN-Transformer 均为 roadmap-only 接口。

## 核心价值
谱图反演更适合表述为**候选集合排序问题**而非单点翻译为 SMILES。将真实横坐标编码、候选召回与前向一致性重排串成可审计闭环，同时提升可解释性与后续几何扩展能力。

## 可行性简析
仓库已包含样本数据生成、预处理、坐标感知编码、候选召回、前向评分、重排、可视化与 trace log，可直接运行验证。`rdkit` 与 `torch` 缺失时会自动降级，主流程仍可完成。pytest 测试覆盖核心模块；`validate_outputs.py` 可校验输出完整性。
