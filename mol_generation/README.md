# Corrole molecule generation

本项目仅以咔咯（corrole）分子为例，展示基于 RDKit 的候选分子生成流程，并不涵盖论文中全部分子的合成过程。示例包含以下两条独立路线：

- **Free-base**：模板匹配 → 筛选种子 → 提取 meso / β 取代基及频率 → 重组生成 → 模板约束的 2D 重绘。
- **含金属 corrole**：读取种子 → 按芳香 C–H 取代反应规则生成候选 → 描述符和相似度计算 → 导出候选库。

仓库仅提供脚本和必要的 MOL 模板，不包含原始 SDF 数据、已生成的分子库、历史报告或检查点。使用者需根据实际研究需求，自行准备相应的 SDF 文件后再运行脚本；数据可存放于仓库外，并通过输入目录参数指定其位置。

## 安装

已在 Python **3.13.12**、RDKit **2025.09.6** 环境验证；`requirements.txt` 固定了验证时使用的第三方包版本。`pandas` / `openpyxl` 用于金属路线的表格输出，`tqdm` 用于进度显示。

在本目录执行：

```bash
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux 激活环境：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

如已有包含上述依赖的 Conda 环境，可直接使用其中的 Python。macOS / Linux 未在本次整理中实际运行验证。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `Mol_classification.py` | 可选预处理，按文件分出含金属、free-base 和读取失败的 SDF |
| `00_diagnose_template_match.py` | 诊断核心匹配及位点外部邻居数量 |
| `01_auto_fix_template_dummy_maps.py` | 可选：根据数据集中取代频率重新分配 dummy map |
| `02_filter_free_base_corrole_by_fixed_template.py` | 筛选符合模板、位点及 meso 一致性要求的 free-base 种子 |
| `03_generate_corrole_from_matched.py` | 从通过筛选的种子建立片段库并随机重组 |
| `04_optimize_generated_corroles_template_constrained.py` | 按模板校正核心键并重画 2D 坐标，导出合并及单分子 SDF |
| `metal_generate.py` | 含金属路线：反应枚举、过滤及 CSV / Excel / SDF 导出 |
| `common.py` | 公共路径、位点定义、SDF 读取和模板匹配工具 |
| `templates/template_raw_mapped.mol` | 自动修正脚本的初始映射模板 |
| `templates/template_raw_mapped_auto_fixed.mol` | 从原项目复制的已修正模板，默认使用 |
| `tests/test_workflow.py` | 使用临时合成结构的回归检查，不依赖原始 SDF |

原项目的 `Test.py`、模板绘制草稿和通用统计脚本不属于当前运行依赖，因此未纳入。金属生成脚本已能导出单分子 SDF，无需额外拆分脚本。

所有脚本支持 `--help`。默认路径相对于本仓库目录解析；用户传入的相对路径相对于命令执行时的工作目录解析。建议在本目录执行下列命令。输出默认集中于 `outputs/`，不同实验请指定独立的输出目录；重复使用输出目录可能覆盖同名报告，并留下上次的单分子文件。

## Free-base 快速开始

### 1. 提供种子数据

将 free-base SDF 放入 `data/free_base/`，或在步骤 2 中传入外部目录。每个文件可以包含一个或多个分子，只读取目录顶层的 `*.sdf`。

如果输入尚未分类，可先运行：

```bash
python Mol_classification.py --input-dir /path/to/raw_sdf
```

结果在 `outputs/classified/free_base/`、`outputs/classified/metal/` 和 `outputs/classified/failed_read/`，同时输出分类日志。分类沿用原规则：一个多分子文件中只要存在含金属分子，整个文件归入 metal；这一步不验证咔咯核心。

### 2. 筛选 → 生成 → 2D 重绘

若使用 `data/free_base/`：

```bash
python 02_filter_free_base_corrole_by_fixed_template.py
python 03_generate_corrole_from_matched.py --count 1000 --seed 2026
python 04_optimize_generated_corroles_template_constrained.py
```

若使用分类结果或仓库外数据，仅替换筛选步骤的输入，例如：

```bash
python 02_filter_free_base_corrole_by_fixed_template.py --input-dir outputs/classified/free_base
```

在原项目旁整理本目录时，也可直接读取父目录数据，不需要复制 SDF：

```bash
python 02_filter_free_base_corrole_by_fixed_template.py --input-dir ../SDF_corrole_free_base
```

主要输出：

| 目录 | 内容 |
| --- | --- |
| `outputs/filter/` | `filter_report.csv`、`matched/`、`rejected/` |
| `outputs/generated/` | meso / β 片段与模式 CSV、提取及生成报告、`generated_corroles.smi`、`generated_corroles.sdf` |
| `outputs/optimized/` | `generated_corroles_core_fixed_2d.sdf`、`optimized_2d_sdf/`、`optimization_report.csv` |

`--count` 是目标唯一结构数；尝试次数上限仍为目标数量的 100 倍。片段组合不足或候选重复时，实际数量可以少于目标，程序会打印实际数量。此路线对生成结果去重，但沿用原逻辑，不排除与种子相同的结构。

### 3. 可选诊断和模板重映射

已有修正模板可直接运行步骤 2，无需按文件编号从 00 开始。

```bash
python 00_diagnose_template_match.py --input-dir data/free_base
```

诊断报告默认位于 `outputs/diagnose_template_match.csv`。诊断只检查核心匹配和外部邻居数量，不检查三处 meso 片段是否相同，因此诊断通过不等于筛选通过。

需要重新推断位点映射时：

```bash
python 01_auto_fix_template_dummy_maps.py --input-dir data/free_base
```

新模板写到 `outputs/template_raw_mapped_auto_fixed.mol`，不会覆盖随仓库提供的模板。检查转换表和结构后，在筛选、生成、重绘中使用同一个新模板，例如：

```bash
python 02_filter_free_base_corrole_by_fixed_template.py --template outputs/template_raw_mapped_auto_fixed.mol
python 03_generate_corrole_from_matched.py --template outputs/template_raw_mapped_auto_fixed.mol --count 1000
python 04_optimize_generated_corroles_template_constrained.py --template outputs/template_raw_mapped_auto_fixed.mol
```

自动修正沿用原有启发式：将外部非氢取代出现最频繁的三个位置视为 meso，并尽量保留原编号。它依赖种子分布，不能替代对模板的人工确认。

## 含金属路线

将种子放入 `data/metal/`，或使用外部路径：

```bash
python metal_generate.py --input-dir ../SDF_corrole_metal --count 1500 --conf-method 2D
```

原默认构象模式仍是 `ETKDG`，省略 `--conf-method` 即使用该模式；`2D` 可用于较快地检查生成结构。`--max-products` 控制每个种子的产物上限（默认 200），`--second-round` 开启第二轮取代（默认关闭）。

结果默认写入 `outputs/metal/`：种子 Excel、包含全部/过滤后候选两个工作表的 Excel、DMPNN 输入 CSV、合并 SDF 和单分子 SDF。SDF 导出对象沿用原脚本的**全部候选**；DMPNN CSV 使用**过滤后的候选**。ETKDG 嵌入失败的结构会跳过 SDF 导出，因此 SDF 数量可能少于表格中的候选数量。检查点仅为中间快照，不支持自动恢复进度。

反应规则保持为芳香 C–H 替换成 F、Cl、Br、I、OMe、Me、CN、COOH。规则作用于匹配的芳香 C–H，不限定为 meso 苯环上的某个位置。候选按 canonical SMILES 去重并排除种子，按规则交替收集产物。过滤阈值保持为 MW ≤ 1300、I 数 ≤ 6、Br 数 ≤ 8、LogP ≤ 12、最近种子 Morgan/Tanimoto 相似度 ≥ 0.50。

## 保留的逻辑与整理范围

- 位点编号保持为 meso `[1, 2, 3]`，β `[102, 103, 107, 108, 112, 113, 117, 118]`。
- 模板筛选和片段提取保留 exact → relaxed 的匹配逻辑；要求三个 meso 位置各有一个且相同的非氢取代基，每个 β 位置最多一个非氢取代基。
- 按观测频率抽取 meso 片段、β 取代模式和各位置片段；同一 meso 片段连接三个位点；未占用的 dummy 被删除。默认随机种子仍为 2026。
- 片段库保留严格 sanitize；完整生成结构保留“sanitize 失败仍可输出”的默认行为，并在报告中记录。可使用 `--strict-generated-sanitize` 改为拒绝这些候选。
- 04 脚本执行模板约束的 **2D 重绘**，不进行 3D 能量最小化，也不能自动修复所有错误连接。金属路线的 ETKDG 嵌入后会尝试 MMFF 优化，但原逻辑不强制要求参数完整或优化收敛。
- 整理主要合并重复读取、核心构建和邻居遍历函数；去掉固定机器路径；添加参数、输出目录创建和正整数校验；移除重复分隔注释。SDF 读取保留空白标题行、记录属性与原始记录索引，避免原先整块 `strip()` 导致的格式错位。通过 Python 文件句柄读写 SDF，兼容当前 Windows 环境的中文路径。

## 验证

无需原始数据即可运行：

```bash
python -m unittest discover -s tests -v
```

整理时额外使用原项目数据对比了新旧代码：141 个 free-base 分子的筛选结果一致（59 个通过）；片段频率、取代模式和提取报告一致；随机种子 2026 下生成的 20 个结构及生成报告一致；5 个金属种子的 40 个取代产物一致。原始数据未随这些验证结果发布。

4 项合成数据回归检查全部通过。7 个脚本均通过命令行运行验证，测试从仓库外工作目录执行并向包含中文的临时路径写入结果，完成了 10 个 free-base 结构的生成和 2D 重绘，以及 10 个金属候选的 2D SDF 导出；本次未运行完整 ETKDG / MMFF 批量验证。

## 上传 GitHub

将本目录作为仓库根目录即可。`.gitignore` 排除了 SDF、`data/` 中的数据、`outputs/`、虚拟环境及 IDE 缓存，两个必要的 `.mol` 模板会保留。上传前可用 `git status --short` 检查暂存内容；本次整理没有自动创建远程仓库或上传文件。
