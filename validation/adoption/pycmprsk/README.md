# cmprsk 与 pycmprsk 本地对照

2026-11-19 使用 R 4.5.3、cmprsk 2.2-12、Python 3.13.15、pycmprsk 1.0.0、NumPy 2.5.5 和 SciPy 1.18.1 运行。数据为固定随机种子生成的 144 条模拟记录，包含三组、三层、两个事件类型、并列时间及一行协变量缺失。删失编码为 9，目标事件编码为 2。

比较 82 个输出字段，数组按字段逐元素核对。相对容差为 `1e-7`，绝对容差为 `1e-9`，缺失值位置必须相同。

| 接口 | 直接调用 pycmprsk 1.0.0 | 采用仓库适配函数 |
| --- | --- | --- |
| `cuminc` | 曲线结点、累计发生率、方差及分层 Gray 检验一致，包含 `rho=0` 和 `rho=1` | 同左 |
| `timepoints` | 12 个字段有差异。查询发生率跃点时，原生调用可返回跳变前值 | `cmprsk_timepoints` 返回跃点后的估计及方差 |
| `crr` 与 `predict.crr` | 系数、方差、对数似然及预测一致，包含时变效应与分组删失权重。残差输出含 66 个事件行，R 输出 26 个唯一事件时间行 | `cmprsk_residuals_by_time` 按目标事件时间合计残差 |
| 合计 | 68/82 个字段一致 | 82/82 个字段一致，最大绝对差为 `5.68e-14` |

组 A、事件 2 在时间 7 的累计发生率为 R `0.19319989395546125`，pycmprsk 1.0.0 `0.17139448568398727`。适配调用返回与 R 一致的跃点后值。残差适配函数需要从模型输入提供完整病例的目标事件时间，并与 Python 每事件残差的行次序对齐。

该验证覆盖所列的估计与预测场景。pycmprsk 还支持其他用法，绘图、任意缺失值处理、`subset` 和退化风险集另需对照。pycmprsk 1.0.0 需要 Python 3.13 或更高版本，本仓库的适配函数仅依赖 NumPy。项目的核心环境无需安装 pycmprsk。

## 复现

从仓库根目录运行。装有 `uv` 时，可在独立环境安装候选。使用现有 R 环境，对照包安装到临时目录。

```bash
adoption_dir="$(mktemp -d)"
mkdir -p "$adoption_dir/r"
uv venv --python 3.13 "$adoption_dir/python"
uv pip install --python "$adoption_dir/python/bin/python" -e . pycmprsk==1.0.0 numpy==2.5.5 scipy==1.18.1
R_LIBS_USER="$adoption_dir/r" Rscript --vanilla -e 'install.packages("cmprsk", repos="https://cloud.r-project.org", lib=Sys.getenv("R_LIBS_USER"))'
R_LIBS_USER="$adoption_dir/r" "$adoption_dir/python/bin/python" validation/adoption/pycmprsk/compare.py
```

脚本会列出原生候选的差异，并在适配输出超出容差时以非零状态退出。固定复核版本为 [cmprsk 2.2-12](https://cran.r-project.org/web/packages/cmprsk/index.html) 和 [pycmprsk 1.0.0](https://pypi.org/project/pycmprsk/1.0.0/)。
