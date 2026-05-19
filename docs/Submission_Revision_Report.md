# Submission Revision Report

## What was changed
为保留原有详细技术报告版本，本次没有删除或覆盖 `latex/main.tex` / `latex/main.pdf`，而是新增一套更适合投递的 submission 版本。

本次 revision 的核心变化如下：
- 新建 `latex/submission.tex`，采用简洁的单页眉式开头，移除了 title page、目录、长摘要页。
- 将原来的长技术报告压缩为 6 个紧凑章节：项目提议、问题定义、核心技术路径、价值、可行性与验证、后续路线。
- 保留了最关键的内容：500 字内项目提议、Top-K 输出、forward-spectrum reranking、trace log、当前/未来边界、graph diffusion + pairwise distance + TFN-Transformer roadmap。
- 大幅压缩或删除了以下内容：长背景综述、长公式链、多个占位图、长参考文献区、重复性的 future module 描述、较花哨的视觉容器。
- 将 submission 版本中的图示限制为 2 个紧凑占位图：
  1. pipeline placeholder
  2. roadmap placeholder
- 将参考依据改为一行紧凑文字说明，而不是完整 bibliography，使 PDF 更像项目 brief 而不是论文草稿。

## Files created
- `latex/submission.tex`
- `latex/submission.pdf`
- `latex/figures/submission_pipeline_placeholder.tex`
- `latex/figures/submission_roadmap_placeholder.tex`
- `docs/Submission_Revision_Report.md`

## PDF path
- `latex/submission.pdf`

## Page count
- 2 pages

## Compile status
- Success
- Command used:
  - `cd latex`
  - `latexmk -xelatex -interaction=nonstopmode -halt-on-error submission.tex`

## Remaining warnings
- None in final verified build
- Checked: undefined citations, undefined references, float warnings, overfull boxes, font warnings, missing characters, xdvipdfmx warnings

## 500-character proposal check
- Checked
- Rough Chinese-character count for Section 1 proposal body: `188`
- Clearly below the 500-character limit

## Content removed or compressed
以下内容被移除或显著压缩，以增强“submission demo proposal”风格：
- decorative title page
- table of contents
- standalone long abstract
- long background and motivation discussion
- multi-section technical report style decomposition
- long formula blocks
- full bibliography section
- excessive placeholder figures
- repeated future-module explanations
- heavy box-driven page layout

## Quality-check summary
- `submission.pdf` compiles successfully: Yes
- page count is between 2 and 4: Yes (2 pages)
- no table of contents: Yes
- no decorative title page: Yes
- 500-character proposal checked: Yes
- current demo vs future extension clearly distinguished: Yes
- no claim that full diffusion / TFN training is completed: Yes
- at most 2 figure placeholders used: Yes
- tone is concise and submission-oriented: Yes

## Suggested next steps
1. Replace pipeline placeholder with real architecture figure
2. Add one screenshot of Top-K demo output after code is ready
3. Final manual proofreading before submission

## 本轮微调说明
- 本轮未改动 submission 的大结构、页数目标或图示数量，只做了版式与“正式投递 brief 感”微调。
- 将首页保持为标题 / subtitle / positioning 置顶，确认 pipeline 图仅出现在“3. 核心技术路径”标题之后。
- 在核心技术路径后加入了 5 行以内的 `Demo 交付物` 紧凑列表。
- 将文中 `候选生成/检索` 统一调整为 `候选召回/轻量生成`，进一步强调当前 demo 的边界。
- 在核心技术路径开头增加了当前 demo 与完整版本之间的技术定位句。
- 在后续扩展路线中补充说明：pairwise distance matrix 是 2D 图与 3D 几何之间的中间语言，更符合旋转/平移不变性。

## 最终微调（排版与术语一致性）
- 本轮未改动 submission 的结构、章节数量、图数量和页数目标。
- 将首页标题区保持为标题 / subtitle / positioning 置顶，并将 pipeline 图从浮动体改为固定内嵌显示，确保不会出现在标题区之前。
- 将相关表述统一为 `编码--候选召回/轻量生成--一致性重排`。
- 将 `后续三维扩展能力` 统一改为 `后续 3D 几何扩展能力`。
- 再次检查全文术语，将 `候选生成/检索` 统一为 `候选召回/轻量生成`。

## 图编号修复
- 删除了标题 `1. 项目提议` 后的 `（500 字内）` 标注，仅保留简洁节标题。
- 修复了 submission 版两张图的编号衔接：第一张 pipeline 图作为 `图 1` 参与计数，第二张 roadmap 图现在正确显示为 `图 2`。
- 本轮未改动正文内容、页数、结构与图表数量，仅修复图编号显示。
