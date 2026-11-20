# survey 的复制权重广义线性模型

`biomed_parity.survey_glm.svyglm` 使用与 R `survey::svyglm` 对应的复制权重设计拟合高斯恒等链接回归，或 `family="quasibinomial"` 的二元逻辑回归。可使用调查发布的复制权重，也可使用 `as_svrepdesign` 生成的一阶段 JK1 或 JKn 复制权重。 [R 手册](https://search.r-project.org/CRAN/refmans/survey/html/svyglm.html) · [复制权重设计](survey-replicates.md) · [Jackknife 权重构造](survey-jackknife.md)

```python
from biomed_parity.survey_glm import svyglm
from biomed_parity.survey_jackknife import as_svrepdesign

design = as_svrepdesign(
    weights=[1.1, 1.1, 0.8, 0.8, 1.3, 1.3, 1.0, 1.0],
    psu=["a", "a", "b", "b", "c", "c", "d", "d"],
)
fit = svyglm(
    values=[126, 134, 119, 142, 128, 137, 123, 140],
    covariates=[42, 57, 38, 64, 49, 60, 44, 62],
    design=design,
)
print(fit.coefficients, fit.standard_errors)
```

每行对应设计中的一个观测。`covariates` 可传一维数值向量或多列矩阵，分类变量须预先编码。模型默认添加截距，截距放在系数向量首位。可用 `covariates=None` 拟合仅含截距的模型，用 `intercept=False` 去掉截距，用 `offset` 指定线性预测量上已知的偏移项。`subset` 接收表示调查域的布尔掩码。默认删除结局、协变量或偏移项为 `NaN` 的行。`na_rm=False` 会在缺失时抛出错误。`included` 是按原设计行序返回的掩码，`fitted_values` 按该掩码选出的行序返回。

全样本权重默认在删除缺失值之前按调查域的权重均值归一化。`rescale=True` 使该域的权重和归一为 1，`rescale=False` 使用原始权重。各复制用各自归一到权重和为 1 的权重重新拟合。在二元模型中，若某行全样本权重为零而合并复制权重为正，复制拟合沿用 R 初始化后的结局 0。

设计协方差使用原设计的 `scale` 和 `rscales`，并按 `mse` 选取全样本系数或正方差乘数的有效复制系数均值为中心。`replicate_estimates` 按复制列的原始顺序返回系数。权重全为零、模型矩阵秩不足或无法得到有限收敛解的复制会产生警告，其整行系数记为 `NaN` 并从协方差计算中剔除。部分秩不足的复制在 R 中可能保留单个可估计系数。全样本模型须有足够的正权重行并找到有限的满秩解，且至少有一个有效复制。

标准误取自设计协方差的对角线。`naive_covariance` 是未乘离散参数的全样本模型协方差。`dispersion` 为全样本加权 Pearson 统计量除以完整观测数，对应 survey 4.5 的复制设计模型摘要。设计自由度使用显式 `degf`，或在应用 `subset` 后、删除模型缺失行之前计算。默认残差自由度为设计自由度加 1 再减去系数个数。`df_resid` 可覆盖区间与检验的自由度，设为 `np.inf` 时使用正态极限。默认残差自由度不为正时，p 值和 Wald 区间为 `NaN`。

目前覆盖数值矩阵的两种模型、训练样本拟合值和系数的 Wald 推断。Taylor 线性化、其他链接函数或分布、分组二项结局、预测标准误及似然区间仍待验证。

R 4.5.3 与 survey 4.5 的七个人工用例覆盖合并权重及乘数、bootstrap、JK1、Fay、调查域及缺失行、零权重、复制秩不足，以及 17 个输出字段。 [验证用例](../../validation/survey_glm/cases.json) · [R 对照脚本](../../validation/survey_glm/oracle.R) · [对照结果](../../validation/survey_glm/expected.json)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_survey_glm.py
```
