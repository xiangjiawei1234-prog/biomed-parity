# 生物医学统计方法迁移

本地 Python 包和可复现的 R 参考用例。调研按生存、诊断、试验设计、临床预测、流行病学及人体表达统计组织。每项方法独立记录兼容范围和数值证据。

| 入口 | 内容 |
| --- | --- |
| [调研与优先级](docs/research.md) | R 方法与 Python 候选的功能级比较 |
| [方法目录](catalog/methods.json) | 分类、优先级、R 入口、候选和依据 |
| [迁移进展](catalog/progress.json) | 实现、替代验证和待补范围 |
| [开发文档](docs/development.md) | 工作包、接口和验收流程 |
| [本地验证记录](docs/validation.md) | 本次环境、通过范围和复现方式 |

## 安装与验证

在仓库根目录运行以下命令。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

Python 测试使用入库的 R 参考输出。以下命令启用本地 R 实时对照。依赖只补装到仓库内的 `.r-lib`。

```sh
Rscript --vanilla validation/install_r.R
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q
```

外部 Python 候选的对照使用各自文档中的隔离环境。入口见 [本地验证记录](docs/validation.md)。
