# 半参数区间删失比例风险与比例优势

`fit_interval_regression` 对应 `icenReg::ic_sp` 的 `model="ph"` 和 `model="po"`。它用准确时间、有限区间、左删失和右删失共同估计回归系数和基线生存。基线概率质量位于观测区间最大交集，可指定开闭端点和病例权重。 [icenReg 半参数接口](https://search.r-project.org/CRAN/refmans/icenReg/html/ic_sp.html) · [方法论文](https://www.jstatsoft.org/article/view/v081i12) · [端点定义](interval-censoring.md)

```python
import numpy as np
from biomed_parity.interval_regression import fit_interval_regression

fit = fit_interval_regression(
    [0, 1, 1, 2, 3, 3, 4, 5],
    [1, 2, 1, 4, 4, np.inf, 4, np.inf],
    [0, 1, 0, 1, 0, 0, 1, 1],
    model="ph",
)
print(fit.coefficients, fit.log_likelihood)
print(fit.distribution([1]).survival_bounds([1, 2.5, 4]))
```

## 统计口径

- `moderators` 可以是一列或数值设计矩阵。基线吸收截距。分类列和其他设计列由调用方编码。
- PH 使用 `S(t|x) = S0(t)**exp(x @ beta)`。比例优势采用 icenReg 的生存优势方向，`logit(S(t|x)) = logit(S0(t)) + x @ beta`。
- 每行似然是该人的事件区间概率。基线的有限累积风险增量限制为非负。优化使用解析导数和多个初值，并检查投影梯度。
- `baseline` 为协变量全零时的概率分布，`distribution(row)` 为指定一行的分布。两者均返回每个最大交集的质量和生存概率界。就给定的拟合质量而言，界限来自每个交集内事件位置尚未确定的部分。

## 对照与范围

R 4.5.3 和 icenReg 2.0.16 的六组人工数据对照包括一至两个协变量、分数权重、端点组合及多种删失。最大绝对偏差为系数 `2.1e-6`、对数似然 `5.6e-11`、生存概率 `9.0e-7`。生存对照包括全零及非零协变量。 [本地 R 脚本](../../validation/interval_regression/oracle.R) · [参考输出](../../validation/interval_regression/expected.json)

当前提供固定设计、点估计和预测分布。系数协方差和区间见[病例 Bootstrap 推断](interval-bootstrap.md)。时变协变量、左截断和多事件流程另行验证。

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 python -m pytest -q -W error tests/test_interval_regression.py
```
