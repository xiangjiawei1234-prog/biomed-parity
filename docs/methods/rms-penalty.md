# rms::pentrace 二元 logistic 惩罚强度选择

`trace_logistic_penalties` 对同一组已展开的数值设计与 0/1 结局逐个拟合惩罚模型。它返回每个网格点的有效自由度、未惩罚似然比和三种信息准则，并返回所选模型。默认用修正 Akaike 准则 `criterion="aicc"`，也可指定 `"aic"` 或 `"bic"`。 [rms::pentrace 手册](https://search.r-project.org/CRAN/refmans/rms/html/pentrace.html)

```python
import numpy as np

from biomed_parity.rms_penalty import trace_logistic_penalties

x = np.array([-2.1, -1.5, -0.8, -0.1, 0.4, 1.0, 1.6, 2.3])
design = np.column_stack((x, x**2))
y = [0, 1, 0, 0, 0, 1, 1, 1]
trace = trace_logistic_penalties(
    design, y, {"simple": [0, 0.5, 2], "nonlinear": [0, 1, 4, 8]},
    term_types=["simple", "nonlinear"],
)
print(trace.best_penalty, trace.best_fit.coefficients)
```

## 网格与选择准则

- 对一列共同惩罚倍数，默认在最前面加上零并保留其余正值的原有次序。设 `include_zero=False` 可直接用输入次序。命名网格只用给定的值，其第一个输入轴变化最快。
- 命名网格使用 `simple`、`nonlinear`、`interaction`、`nonlinear.interaction`，通过 `term_types` 标注每个斜率列。省略的类型依上述顺序继承上一个类型的倍数，`simple` 默认零。设 `complex_more=True` 时，输入轴按从简单到复杂的次序给出，仅保留给定倍数非递减的组合。
- 基础矩阵用设计列的样本方差。分类主效应可用 `categorical_blocks` 标注处理编码列，也可通过 `penalty_matrix` 传入自定义矩阵。它们沿用[二元 logistic 拟合](rms-logistic.md)的规则。

设 `LR` 为用未惩罚偏差计算的似然比，`d` 为斜率的有效自由度，`n` 为样本行数。与 rms 一致，三种得分分别为 `LR - 2*d`、`LR - log(n)*d`、`LR - 2*d*(1 + (d+1)/(n-d-1))`，按得分较大者选取。分母非正时该网格点不参与 AICc 选择。得分完全相等时采用最先出现的行。 [rms::pentrace 源码](https://github.com/harrelfe/rms/blob/master/R/pentrace.s)

`best_index` 为所选行的 0 基下标，`best_fit` 保存拟合系数和未调整的惩罚信息逆。`adjusted_covariance` 是 Gray 调整后的 `V I V`，其中 `V` 是所选模型的惩罚信息矩阵逆，`I` 是在该拟合点计算的未惩罚信息矩阵。设 `keep_coefficients=True` 可保留每个网格点的系数。当前范围为无权重和偏移的二元固定设计网格搜索。

## 本地验证

```bash
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_rms_penalty.py
R_LIBS_USER="$PWD/.r-lib" .venv/bin/python validation/rms_penalty/verify.py
```

R 4.5.3 和 rms 8.2-1 的四个合成案例涵盖 25 个网格点，核对零惩罚、自定义相关惩罚矩阵、样条和类别交互项、倍数继承及三种选择准则。比较选中行、系数、得分、有效自由度和两种系数协方差，绝对容差为 `3e-9`，相对容差为 `2e-9`。 [R 验证脚本](../../validation/rms_penalty/oracle.R)
