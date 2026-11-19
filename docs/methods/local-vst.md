# DESeq2 的局部方差稳定化变换

`biomed_parity.vst.local_vst` 接收已拟合的离散度趋势，复现 DESeq2 的 `getVarianceStabilizedData` 局部分支。输入为特征 × 样本计数矩阵，离散度趋势可用函数或采样表表示。输出是同样形状的变换值，可用于样本探索和质量检查。DESeq2 的差异表达检验使用原始计数和负二项模型。 [DESeq2 方法手册](https://github.com/thelovelab/DESeq2/blob/devel/man/varianceStabilizingTransformation.Rd) · [DESeq2 分析指南](https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html)

## 输入

```python
import numpy as np

from biomed_parity.vst import DispersionGrid, local_vst, local_vst_grid

counts = np.array([[0, 1, 0], [8, 6, 11], [34, 37, 51], [300, 310, 413]])
size_factors = np.array([0.83, 0.94, 1.25])

def fitted_curve(mean):
    return 0.13 + 0.6 / (mean + 2)

means = local_vst_grid(counts / size_factors)
trend = DispersionGrid(means, fitted_curve(means))
transformed = local_vst(counts, trend, size_factors=size_factors)
```

- 对原始计数可传样本尺度因子 `size_factors`、逐特征和逐样本的归一化矩阵 `normalization_factors`，或两者兼有。矩阵用于除计数，样本尺度因子优先用于 Poisson 项。只有矩阵时，Poisson 项使用各列几何均值的倒数平均。
- 传入已归一化的计数时用 `normalized=True`。尺度因子仍可用于方差表达式中的 Poisson 项。两类因子均省略时使用单位尺度。
- `DispersionGrid` 对采样点作线性插值，并要求采样范围覆盖积分网格。通过 `local_vst_grid` 取得实际计算点后，从外部拟合曲线上导出同一批点，可避免引入离散度插值误差。也可传入支持数组查询的趋势函数。

## 数值方法

令 `q` 为归一化计数，局部方差为 `α(q) q² + b q`，其中 `b` 为样本尺度因子倒数的均值。实现对方差平方根的倒数作梯形积分，在 `asinh` 坐标上用 Forsythe–Malcolm–Moler 三次插值。端点三阶差商决定边界。该细节影响零计数的外推值。结果在特征平均归一化计数的 95% 和 99.9% 分位点锚定到 `log2` 尺度。两个锚点必须为正且互异。 [R 插值样条手册](https://search.r-project.org/R/refmans/stats/help/splinefun.html)

当前 Python 路径的计算起点为已拟合趋势和相应归一化信息。逐特征离散度估计、`locfit` 自适应拟合及盲态设计处理属于拟合阶段。

## 本地验证

用 R 4.5.3、DESeq2 1.50.2 和 locfit 1.5.9.12 验证五组匿名合成计数。覆盖非线性趋势、用户采样网格、归一化矩阵、显式样本因子和零计数。最后一组在 R 拟合局部离散度趋势，将 999 个拟合值交给 Python，再对照变换结果。五组变换值的最大绝对差为 `1.11e-13`。 [合成输入](../../validation/vst/cases.json) · [R 参考输出](../../validation/vst/expected.json)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_vst.py
```

## Python 候选

PyDESeq2 0.6.0 的已发布代码仅接受参数化和均值两种 VST，既有本地审计记录了局部分支的异常。上游主干的 `81bb3ca` 已实现局部趋势拟合和 VST。该提交在 96 × 6 合成矩阵上的样本尺度因子与 R 最大绝对差为 `3.56e-15`，端到端变换的最大绝对差为 `0.01235`，绝对差中位数为 `0.01168`。将 Python 候选自己拟合的曲线同时给两个 Python 实现后，最大绝对差降至 `2.67e-14`。保留既有 R 拟合曲线的复现可以使用采样表，新拟合结果的跨语言对照仍需把拟合和变换分别核对。 [已发布版本的审计](deseq2-adoption.md) · [上游提交](https://github.com/scverse/PyDESeq2/commit/81bb3ca7205d601dea21543d4c9adab160dbeb2e) · [对照程序](../../validation/vst/audit_pydeseq2.py) · [对照输出](../../validation/vst/pydeseq2-dev.json)
