# rms 二元 logistic 惩罚似然拟合

`fit_penalized_logistic` 接受展开后的数值斜率设计和 0/1 结局，估计截距和斜率。它在二元 logistic 对数似然中扣除 `beta.T @ P @ beta / 2`，其中 `beta` 只有斜率。默认 `penalty=0` 时为普通最大似然拟合。 [rms::lrm 手册](https://search.r-project.org/CRAN/refmans/rms/html/lrm.html)

```python
import numpy as np
from scipy.special import expit

from biomed_parity.rms_logistic import fit_penalized_logistic
from biomed_parity.rms_prediction import wald_predict

x = np.array([[-2, 0], [-1, 1], [0, 0], [1, 1], [2, 0], [3, 1]])
y = [0, 1, 0, 0, 1, 1]
fit = fit_penalized_logistic(x, y, penalty=3, categorical_blocks=[[1]])
design = np.column_stack((np.ones(len(x)), x))
prediction = wald_predict(design, fit.coefficients, fit.covariance, transform=expit)
```

## 惩罚矩阵

- 默认对角元是各斜率设计列的样本方差。用 `categorical_blocks` 指定采用处理编码的分类主效应，每个列表包含同一个因子的 0 基列下标。一个有 `k` 列的因子块采用 `I - 1 / (k + 1)`，即以类别平均效应为中心对系数平方和施加惩罚。
- `penalty_matrix` 可传入自定义的对称半正定基础矩阵，行列与斜率顺序一致。`penalty` 可以为共同的非负倍数，也可以为逐列倍数。逐列倍数的平方根同时缩放基础矩阵的行和列。因子取零时对应列不参与惩罚。可按设计列顺序给样条非线性项和交互项指定不同倍数。
- `fit.penalty_matrix` 是实际施加到斜率的矩阵。截距相应的惩罚恒为零。默认惩罚受设计列的变化范围影响，分类主效应的惩罚在同一块各列采用共同倍数时与参考水平选择无关。 [rms 惩罚矩阵源码](https://github.com/harrelfe/rms/blob/master/R/rmsMisc.s)

## 结果与验证

`coefficients` 和 `covariance` 使用原始设计尺度，截距位于第一项。协方差是拟合点的惩罚信息矩阵逆。`effective_df_diagonal` 是未惩罚信息矩阵与该协方差乘积的对角线，`effective_df` 是斜率项贡献的总和。`deviance`、`penalized_deviance` 和 `null_deviance` 分别是拟合点的未惩罚、惩罚和空模型偏差。`likelihood_ratio` 使用未惩罚偏差，`pvalue` 用有效自由度计算卡方右尾概率。输出也包含训练行的线性预测和响应概率。 [rms::lrm 源码](https://github.com/harrelfe/rms/blob/master/R/lrm.s)

当前接口覆盖完整的二元响应和数值斜率设计。调用者负责分类变量编码、样条基构造和数据缺失处理。序数响应、偏移量、观测权重、惩罚强度搜索和惩罚模型重抽样为后续范围。惩罚须能约束造成分离或共线的方向，才能得到有限且可识别的拟合。

```bash
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_rms_logistic.py
R_LIBS_USER="$PWD/.r-lib" .venv/bin/python validation/rms_logistic/verify.py
```

R 4.5.3 和 rms 8.2-1 的六个合成案例覆盖普通拟合、连续列方差惩罚、相关自定义矩阵与不惩罚的列、多水平因子和样条交互、完全分离及共线设计。对照拟合系数、协方差、实际惩罚、预测、有效自由度和偏差统计量，绝对容差为 `3e-9`，相对容差为 `2e-9`。 [R 验证脚本](../../validation/rms_logistic/oracle.R)
