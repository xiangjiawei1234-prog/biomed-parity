# biomed-parity

生物医学统计方法的独立 Python 实现，附可复现的 R 参考用例与数值对照。

[![PyPI](https://img.shields.io/pypi/v/biomed-parity.svg)](https://pypi.org/project/biomed-parity/)
[![CI](https://github.com/xiangjiawei1234-prog/biomed-parity/actions/workflows/ci.yml/badge.svg)](https://github.com/xiangjiawei1234-prog/biomed-parity/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

> **项目状态：Alpha（v0.1.1，迁移进行中）。** 方法目录中的 16 组 R 统计入口均已建立数值对照。下表描述当前源码中的方法范围。各入口仍存在未覆盖的接口与统计路径，公开 API 可能随验收进展调整。已验证范围与剩余任务见[迁移进展](catalog/progress.json)。

**biomed-parity** provides independent Python implementations of selected biomedical statistical methods, accompanied by reproducible R reference cases and numerical comparisons. The implementations depend only on NumPy and SciPy and do not require an R runtime. The project is an early-stage research prototype: public APIs may change as validation continues. Documentation is written in Chinese; docstrings and code comments are in English.

## 概述

本项目提供以下内容：

- 不依赖 R 运行时的 Python 统计实现，运行时仅使用 NumPy 与 SciPy；
- 每个方法的估计目标、输入约定、结果字段，以及不可估计时的返回行为；
- 基于固定输入、R oracle 脚本与版本化快照的数值对照，容差与已知差异记录在文档中；
- 方法文档与迁移进展中列出的未覆盖路径。

## 入口

| 内容 | 位置 |
| --- | --- |
| 调研与优先级 | [docs/research.md](docs/research.md) |
| 方法目录（分类、R 入口、候选与依据） | [catalog/methods.json](catalog/methods.json) |
| 迁移进展（已验证范围与剩余任务） | [catalog/progress.json](catalog/progress.json) |
| 方法文档（接口、统计口径、验证与限制） | [docs/methods](docs/methods) |
| 开发与验收流程 | [docs/development.md](docs/development.md) |
| 本地验证记录 | [docs/validation.md](docs/validation.md) |

方法目录覆盖的 R 入口按生物医学分类如下：

| 分类 | R 入口 |
| --- | --- |
| 生存与竞争风险 | `survival::survdiff`、`survRM2::rmst2`、`cmprsk`、`riskRegression::Score/plotCalibration`、`mstate::msprep/msfit/probtrans` |
| 诊断准确性 | `pROC::roc`、`auc`、`ci.auc`、`roc.test`（DeLong、部分 AUC、Bootstrap） |
| 临床试验设计 | `gsDesign`（边界、越界概率、条件把握度） |
| 临床预测与回归 | `rms::calibrate/validate/rcs`、`glmmTMB`、`variancePartition::dream` |
| 流行病学与证据综合 | `survey`（`svycoxph`、`twophase`）、`mice`（含插补链和 D1/D2）、`metafor::rma.uni/rma.mv` |
| 人体表达统计 | `limma`、`edgeR`、`DESeq2`（负二项 LRT 和给定趋势的局部 VST） |

## 安装与验证

从 PyPI 安装：

```sh
python3 -m pip install biomed-parity
```

开发或验证时从源码安装并运行默认测试：

```sh
git clone https://github.com/xiangjiawei1234-prog/biomed-parity.git
cd biomed-parity
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

也可直接从 GitHub 安装：

```sh
python3 -m pip install "git+https://github.com/xiangjiawei1234-prog/biomed-parity.git"
```

默认测试仅依赖入库的 R 参考快照，无需安装 R。当前源码及运行环境的结果见 [本地验证记录](docs/validation.md)。

## 实时 R 对照

需要本地 R 环境（开发使用 R 4.5.3），R 依赖安装到仓库内的 `.r-lib`：

```sh
Rscript --vanilla validation/install_r.R
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error
```

实时对照调用对应 R 包并逐项比较数值行为。

## 项目结构

| 目录 | 内容 |
| --- | --- |
| `src/biomed_parity/` | Python 统计实现 |
| `catalog/` | 方法调研、优先级与迁移进展 |
| `docs/methods/` | 每个入口的估计目标、数据约定、结果字段、示例和限制 |
| `validation/<方法>/` | 人工构造输入、R oracle 脚本、参考输出与候选评估 |
| `tests/` | Python 回归、边界条件与可选实时 R 对照 |

## 范围与限制

- 数值基线对应固定版本的 R 与 R 包，版本记录于方法文档和各 `validation/` 快照。升级上游包后需重新生成快照并复核差异。
- 本项目对齐数值行为与统计口径，不提供 R 对象结构、公式接口、绘图及 S3 方法的等价实现。
- 参考输入均为人工构造或匿名数据，不包含真实患者数据。
- 上游 R 包仅作外部数值参考，未复制或翻译其源码。实现依据论文与公开接口描述独立完成，来源记录在 `catalog/methods.json`。
- 各方法尚未覆盖的路径见[迁移进展](catalog/progress.json)与对应方法页。
- 本项目为研究原型，不构成经监管验证的临床统计软件。用于正式分析前应自行核对结果。

## 贡献

问题反馈与代码贡献通过 Issue 和 Pull Request 提交，流程见[贡献指南](CONTRIBUTING.md)。新增或扩展方法时，需同时提供 R 参考脚本、版本化快照、测试与文档，并更新 `catalog/progress.json`。

## 许可

[MIT](LICENSE) © 2026 xiangjiawei1234-prog
