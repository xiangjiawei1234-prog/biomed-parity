# biomed-parity

用纯 Python（NumPy + SciPy）实现常用生物医学统计方法，并为每一项保留可复现的 R 参考用例、数值证据与已知差异。

[![CI](https://github.com/xiangjiawei1234-prog/biomed-parity/actions/workflows/ci.yml/badge.svg)](https://github.com/xiangjiawei1234-prog/biomed-parity/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

> **项目状态：Alpha（v0.1.0，迁移进行中）。** 16 个 R 方法入口均已建立至少一组数值对照，但每个入口仍有未覆盖的接口或统计路径，公开 API 可能随验收进展调整。已验证范围与剩余任务见 [迁移进展](catalog/progress.json)。

**biomed-parity** reimplements selected biomedical statistical methods in pure Python (NumPy/SciPy) and validates each implementation against recorded R references. Every method ships with reproducible fixtures, documented tolerances, and explicit notes on remaining gaps. The project is an early-stage research prototype: APIs may change and coverage is still growing. Documentation is currently written in Chinese; docstrings and code comments are in English.

## 项目定位

R 生态里许多经典统计方法没有等价、可逐项核对的 Python 实现。本项目不是给 R 包套 Python 包装层，而是：

- 只依赖 NumPy 与 SciPy，不引入 R 运行时；
- 为每个方法记录估计目标、输入约定、结果字段和不可估计时的行为；
- 用固定输入、R oracle 脚本和入库快照做数值对照，容差与已知差异写进文档；
- 明确列出尚未覆盖的路径，而不是假装与 R 完全等价。

## 入口

| 内容 | 位置 |
| --- | --- |
| 调研与优先级 | [docs/research.md](docs/research.md) |
| 方法目录（分类、R 入口、候选与依据） | [catalog/methods.json](catalog/methods.json) |
| 迁移进展（已验证范围与剩余任务） | [catalog/progress.json](catalog/progress.json) |
| 方法文档（接口、统计口径、验证与限制） | [docs/methods](docs/methods) |
| 开发与验收流程 | [docs/development.md](docs/development.md) |
| 本地验证记录 | [docs/validation.md](docs/validation.md) |

覆盖的 R 入口按生物医学分类组织：

| 分类 | R 入口 |
| --- | --- |
| 生存与竞争风险 | `survival::survdiff`、`survRM2::rmst2`、`cmprsk`、`riskRegression::Score`、`mstate` |
| 诊断准确性 | `pROC::roc`、`auc`、`ci.auc`、`roc.test`（DeLong、部分 AUC、Bootstrap） |
| 临床试验设计 | `gsDesign`（边界、越界概率、条件把握度） |
| 临床预测与回归 | `rms::lrm`、`calibrate`、`validate`、`glmmTMB`、`variancePartition::dream` |
| 流行病学与证据综合 | `survey`（`svycoxph`、`twophase`）、`mice`、`metafor` |
| 人体表达统计 | `limma`、`edgeR`、`DESeq2`（含原生负二项 LRT） |

## 安装

仓库尚未发布到 PyPI，可从源码安装：

```sh
git clone https://github.com/xiangjiawei1234-prog/biomed-parity.git
cd biomed-parity
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

或直接从 GitHub 安装：

```sh
python3 -m pip install "git+https://github.com/xiangjiawei1234-prog/biomed-parity.git"
```

默认测试只依赖已入库的 R 参考快照，可在没有 R 的环境中运行。当前版本默认套件为 366 项通过、16 项实时 R 对照按配置跳过。

## 实时 R 对照

需要本地 R（开发时使用 4.5.3）。R 依赖只安装到仓库内的 `.r-lib`：

```sh
Rscript --vanilla validation/install_r.R
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error
```

实时套件共 382 项测试，会调用对应 R 包并逐项比较数值行为。

## 项目结构

| 目录 | 内容 |
| --- | --- |
| `src/biomed_parity/` | 纯 Python 统计实现 |
| `catalog/` | 方法调研、优先级与迁移进展 |
| `docs/methods/` | 每个入口的估计目标、数据约定、结果字段、示例和限制 |
| `validation/<方法>/` | 人工构造输入、R oracle 脚本、参考输出与候选评估 |
| `tests/` | Python 回归、边界条件与可选实时 R 对照 |

## 范围与限制

- 数值基线对应固定版本的 R 与 R 包（版本记录在方法文档和各 `validation/` 快照中）；升级上游包后需重新生成快照并复核差异。
- 不追求 R 对象、公式接口、绘图和 S3 方法的等价；只对齐数值行为与可验证的统计口径。
- 参考输入均为人工构造或匿名数据，不包含真实患者数据。
- 上游 R 包仅作外部数值参考，未复制或翻译其源码；实现依据论文与公开接口描述独立完成，来源记录在 `catalog/methods.json`。
- 每个方法尚未覆盖的路径见 [迁移进展](catalog/progress.json) 与对应方法页。
- 本项目是研究原型，不构成经监管验证的临床统计软件；用于正式分析前请自行核对结果。

## 贡献

欢迎提交 Issue 报告数值差异，或按 [贡献指南](CONTRIBUTING.md) 提交 PR。新增或扩展方法时，请同时提供 R 参考脚本、入库快照、测试与文档，并更新 `catalog/progress.json`。

## 许可

[MIT](LICENSE) © 2026 xiangjiawei1234-prog
