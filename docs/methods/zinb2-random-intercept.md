# 零膨胀负二项随机截距模型

`biomed_parity.zero_inflated.fit_zinb2_random_intercept` 拟合 `glmmTMB(family=nbinom2)` 对应的零膨胀随机截距模型。计数均值、结构零概率与离散度分别使用 log、logit 和 log 链接。条件计数方差为 `mu + mu**2 / shape`。簇随机截距只加到计数均值上。传入每行的对数暴露量可比较不同观测时间下的计数率。 [glmmTMB 建模入口](https://glmmtmb.github.io/glmmTMB/reference/glmmTMB.html) · [NB2 方差定义](https://glmmtmb.github.io/glmmTMB/reference/nbinom2.html)

```python
from biomed_parity.zero_inflated import fit_zinb2_random_intercept

fit = fit_zinb2_random_intercept(
    counts, hospitals, count_design=mean_design, inflation_design=zero_design,
    dispersion_design=shape_design, offset=log_exposure,
)
print(fit.count_coefficients, fit.inflation_coefficients, fit.dispersion_coefficients)
```

设计矩阵每行对应一个观测，须显式加入所需截距。省略一个设计矩阵时，该子模型使用单独的截距。返回各子模型回归系数、随机截距标准差、Laplace 对数似然、迭代计数和收敛信息。用 `zinb2_random_intercept_loglikelihood` 可在给定参数上计算相同似然。当随机效应标准差为零时，退化为逐行独立的结构零与负二项混合。 [计数模型的预测与模拟](count-predictions.md)

用 R 4.5.3 和 glmmTMB 1.1.14 核对两组人工数据，包含仅截距，以及三个子模型各有协变量且计数模型带 offset 的情况。比较三组系数、随机截距标准差、拟合及指定参数的对数似然。系数最大绝对差低于 `1e-6`，对数似然最大绝对差低于 `1e-10`。本机 glmmTMB 使用 TMB 1.9.19 编译，运行时 TMB 为 1.9.23，R 报出版本提示。另用 SciPy 逐行概率核对零随机方差特例。目前支持一个随机截距和 NB2 离散度模型，标准误、随机斜率和截断计数模型仍待迁移。 [参考脚本](../../validation/zinb2/oracle.R) · [用例](../../validation/zinb2/cases.json)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_zinb2.py
```
