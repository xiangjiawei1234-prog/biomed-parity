# survey 的复制权重设计

`biomed_parity.survey_replicates.svrepdesign` 使用预先给定的复制权重，计算与 R `survey::svrepdesign` 配套的均值、总量和设计协方差。适用于同时发布全样本权重和复制权重的复杂抽样调查。支持 BRR、Fay、JK1、JKn、bootstrap、ACS、successive-difference、JK2 和自定义 `other`。 [R 复制权重设计手册](https://search.r-project.org/CRAN/refmans/survey/html/svrepdesign.html) · [survey 4.5 手册](https://cran.r-project.org/web/packages/survey/refman/survey.html)

```python
from biomed_parity.survey_replicates import svrepdesign

design = svrepdesign(
    repweights=[[2, 2, 0, 0], [0, 0, 2, 2], [2, 0, 2, 0], [0, 2, 0, 2]],
    weights=[2, 3, 2.5, 1.8],
    type="BRR",
    combined_weights=False,
)
outcomes = [[144, 1], [123, 0], [129, 1], [117, 0]]  # 收缩压和二元结局
mean = design.mean(outcomes)
total = design.total(outcomes)
print(mean.estimate, mean.covariance, total.standard_errors)
```

输入矩阵每行对应一个观测，每列对应一次复制。`combined_weights=True` 表示矩阵中的列已包含抽样权重。`False` 表示矩阵中的列是乘数，分析时逐行乘以抽样权重。省略 `weights` 时使用等权。结局可传一维数值向量，也可传多列数值矩阵。每次复制的均值用该次复制的权重和归一化，二元结局的均值即比例。总量保留权重的原始尺度。

协方差以 `scale` 乘上各次复制偏差外积的加权和，每次复制的乘数为 `rscales`。默认中心为 `rscales` 大于零的有效复制估计的算术平均。`mse=True` 使用全样本估计作中心，ACS 默认应用此规则。Fay 需提供 `rho`。合并后的 JKn 权重需提供 `rscales`。平均 bootstrap 权重可提供 `bootstrap_average`。JK1、JKn 和 `other` 可提供每次复制的 `fpc`，用 `fpctype="fraction"` 指明抽样比例，或用 `fpctype="correction"` 直接传入方差乘数。BRR 和 Fay 固定总体 `scale`，survey 4.5 仍应用显式传入的 `rscales`。 [R 对照脚本](../../validation/survey_replicates/oracle.R) · [数值基线](../../validation/survey_replicates/expected.json)

`design.subset(mask)` 对布尔掩码选择的行做调查域估计。把选中行的结局传给其 `.mean()` 或 `.total()`。`na_rm=True` 在分析前删除任何结局缺失的行。两种操作都保留原有复制列和方差缩放。权重和为零的复制均值以 `NaN` 保留在 `replicate_estimates` 中，并从协方差计算中剔除。某次复制的所有权重为零时，复制总量为零。`design_df` 支持显式设置 `degf`，否则按实际分析行的复制权重数值秩减一计算。

给定设计也可用于 Gaussian 或二元 quasibinomial 回归。仓库另提供一阶段 JK1/JKn 权重生成，以及线性和 raking 校准。 [重复权重回归](survey-glm.md) · [Jackknife 权重生成](survey-jackknife.md) · [调查校准](survey-calibration.md)

用 R 4.5.3 和 survey 4.5 核对了 14 个人工方案中的抽样权重、复制权重、均值、总量、复制估计、中心、协方差、标准误和自由度。 [验证用例](../../validation/survey_replicates/cases.json)

```sh
BIOMED_R_PARITY=1 R_LIBS_USER="$PWD/.r-lib" .venv/bin/python -m pytest -q tests/test_survey_replicates.py
```
