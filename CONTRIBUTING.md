# 贡献指南

感谢关注 biomed-parity。本项目把生物医学统计方法从 R 迁移为可独立运行的 Python 实现，并保留可复现的 R 参考用例。欢迎通过 Issue 报告数值差异、文档问题或覆盖缺口，也欢迎提交 PR。

## 开发环境

```sh
git clone https://github.com/xiangjiawei1234-prog/biomed-parity.git
cd biomed-parity
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

默认测试只依赖入库快照。如需实时 R 对照，本地需要 R 与对应 R 包：

```sh
Rscript --vanilla validation/install_r.R
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error
```

## 新增或扩展方法

1. 在 `catalog/methods.json` 中登记 R 入口、优先级、候选实现与官方来源，并补齐 `catalog/progress.json` 的状态字段。
2. 明确估计目标与输入假设：参照组与因子顺序、删失与并列值、权重、区间、自由度、缺失值，以及不可估计时的返回行为。
3. 在 `src/biomed_parity/` 实现，仅依赖 NumPy 与 SciPy；公共接口需要简洁的英文 docstring。
4. 在 `validation/<方法>/` 保存人工构造输入、`oracle.R` 参考脚本和 JSON/TSV 快照。快照应由脚本生成，不手改数值。
5. 在 `tests/` 增加默认回归测试（读取快照）与 `BIOMED_R_PARITY=1` 的实时对照测试。
6. 在 `docs/methods/` 新增或更新方法页，说明接口、统计口径、验证范围、已知差异和剩余限制，并更新 `docs/validation.md`。
7. 用逻辑独立的提交保存调研、实现、验证和文档。

验收细节见 [开发与验收流程](docs/development.md)。

## 快照与容差约定

- 快照文件头记录生成时的 R 与 R 包版本；升级上游包后需重新生成并逐项复核。
- 容差按输出尺度和数值算法确定，写在对应测试与方法页中，不随意放宽。
- 若 Python 与 R 存在有意差异（例如上游实现的历史行为），用独立用例显式记录，并在测试中单独断言，不要把差异算作数值一致。

## 数据与隐私

不要提交真实患者数据、未脱敏数据、下载的临时文件、R 库、虚拟环境或构建产物。参考输入应使用人工构造或公开的匿名数据。

## 提交与许可

- 提交信息用一句话说明改动目的，正文按需列出验证证据。
- 提交 PR 前请确保 `pytest -q` 通过；改动数值路径时另行运行实时 R 对照。
- 贡献按项目的 [MIT 许可证](LICENSE) 授权。
