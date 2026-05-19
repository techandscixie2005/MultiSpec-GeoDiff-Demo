# Demo Usage

## 1. 安装依赖

```bash
python -m pip install -r requirements.txt
```

## 2. 一键运行 demo

```bash
python scripts/run_demo.py --config configs/demo.yaml
```

依赖安装完成后，这就是唯一需要记住的运行命令。它会自动生成样本数据、运行 demo 流水线，并刷新 `outputs/`。

## 3. 验证输出

```bash
python scripts/validate_outputs.py --output_dir outputs
```

## 4. 你会看到什么

运行成功后，`outputs/` 里至少会出现：

| 文件 | 作用 |
|---|---|
| `topk_candidates.csv` | Top-K 候选表 |
| `trace_log.json` | 配置、环境、候选和输出的运行记录 |
| `query_spectra.png` | 查询谱图 |
| `spectrum_match_top1.png` | Top-1 候选与查询谱图叠图 |
| `topk_spectrum_overlay.png` | Top-K 叠图 |
| `molecule_grid.png` | 候选结构面板 |
| `demo_summary.md` | 简短结果摘要 |

## 5. 当前环境的常见提示

- 如果没有 `rdkit`，结构图会降级成文本面板，validity 也会使用 fallback 分数。
- 如果没有 `torch`，编码器会使用确定性的 NumPy 实现。
- 这些提示是预期行为，不是运行失败。

## 6. 使用建议

- 先看 `outputs/topk_candidates.csv`，再看 `outputs/trace_log.json`。
- 如果你要改数据或配置，先重新跑 `scripts/run_demo.py`，再跑验证脚本。
- 不要把当前 demo 说成“已完成图扩散训练”或“已完成 3D 精修”，那些模块现在仍是 roadmap-only。
