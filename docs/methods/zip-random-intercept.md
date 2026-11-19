# 零膨胀 Poisson 随机截距模型

用于分析同一单位重复观察的事件计数。Python 实现对应 `glmmTMB(..., family=poisson, ziformula=...)` 的一个子集。计数部分使用 Gaussian 分组随机截距和 log 链接，零膨胀部分使用固定效应和 logit 链接，模型通过一阶 Laplace 近似极大化边际似然。[glmmTMB 参考手册](https://cran.r-project.org/web/packages/glmmTMB/refman/glmmTMB.html) [glmmTMB 指南](https://cran.r-project.org/web/packages/glmmTMB/vignettes/glmmTMB.pdf)

## Python 接口

| 函数 | 输出 |
| --- | --- |
| `fit_zip_random_intercept(counts, groups, ...)` | 两组固定效应系数、随机截距标准差、近似边际对数似然、迭代数与优化器状态 |
| `zip_random_intercept_loglikelihood(counts, groups, count_coefficients, inflation_coefficients, random_intercept_sd, ...)` | 在给定参数下积分消去随机截距后的一阶 Laplace 对数似然 |

计数为非负整数，`groups` 为逐行分组标签。可分别提供二维矩阵 `count_design` 和 `inflation_design`，列顺序决定系数顺序。未提供设计矩阵时仅使用截距。计数侧 `offset` 是直接加到 log 均值上的逐行向量，例如暴露时间的对数。

```python
import numpy as np
from biomed_parity.zero_inflated import fit_zip_random_intercept

rows = np.genfromtxt(
    "validation/zero_inflated/observations.tsv",
    delimiter="\t", names=True, dtype=None, encoding="utf-8",
)
data = rows[rows["dataset"] == "covariates"]
ones = np.ones(len(data))
fit = fit_zip_random_intercept(
    data["count"],
    data["group"],
    count_design=np.column_stack((ones, data["x"])),
    inflation_design=np.column_stack((ones, data["z"])),
    offset=data["offset"],
)
print(fit.count_coefficients, fit.inflation_coefficients)
print(fit.random_intercept_sd, fit.log_likelihood, fit.converged)
```

示例计数系数约为 (0.642407, 0.535433)，结构零 logit 系数约为 (−1.561381, 1.310782)，随机截距标准差约为 0.849084，对数似然约为 −1124.985571。

## 计算与边界

在给定随机截距下，零计数的概率是结构零和 Poisson 零的概率之和，正计数只来自 Poisson 成分。以标准正态尺度计算每组的条件模式，用条件对数似然曲率做 Laplace 校正，并将各组的结果相加。拟合用 L-BFGS-B 对固定效应和 log 标准差优化。给定参数的接口接受随机效应标准差为零，对应独立 ZIP 似然。

拟合要求两个以上分组、同时存在零和正计数以及满列秩设计。log 标准差的数值搜索区间为 [−12, 3]，到达边界时结果状态标记为未收敛。当前收敛字段对应数值优化状态。近边界、分离、小分组与多峰条件似然的统计推断需分别验证。

首版为单层随机截距模型。标准误和区间、诊断及预测、其他计数分布、零膨胀部分的随机项和交叉或嵌套随机效应属于后续范围。

## Python 候选

核查的 statsmodels 接口分别提供固定效应 ZIP 和 Bayesian Poisson 混合模型。Bambi 提供可建模两个成分的 Bayesian ZIP。Pymer4 的混合模型接口调用 R 的 `lme4`。这些公开接口覆盖同类计数建模需求。本次迁移针对 glmmTMB 的频率学派随机截距 ZIP 及其按给定参数计算的 Laplace 对数似然。[statsmodels ZIP](https://www.statsmodels.org/stable/generated/statsmodels.discrete.count_model.ZeroInflatedPoisson.html) [statsmodels 混合模型](https://www.statsmodels.org/stable/mixed_glm.html) [Bambi ZIP](https://bambinos.github.io/bambi/notebooks/zero_inflated_regression.html) [Pymer4](https://eshinjolly.com/pymer4/pages/comingfromR.html)

## 本地验证

R 4.5.3 和 glmmTMB 1.1.14 生成三组参考结果，覆盖纯截距、两子模型不同设计加偏置项、较少组内观测。三个 R 拟合均由 BFGS 收敛并返回正定 Hessian。对每组数据也在一组指定参数上核对边际似然，独立于优化器返回的系数。拟合系数和标准差的绝对差小于 10⁻⁶，拟合及给定参数的对数似然绝对差小于 4 × 10⁻⁸。[合成观测](../../validation/zero_inflated/observations.tsv) [R 对照](../../validation/zero_inflated/expected.tsv)

本机 R 加载时报告 glmmTMB 编译使用 TMB 1.9.19、实际加载 TMB 1.9.23。警告保留在直接运行参考脚本的终端输出。这次结果使用上述实际版本，兼容性证据限于三个已验证的拟合场景。

```sh
python -m pytest -q tests/test_zero_inflated.py
BIOMED_R_PARITY=1 python -m pytest -q tests/test_zero_inflated.py
Rscript --vanilla validation/zero_inflated/generate.R
```
