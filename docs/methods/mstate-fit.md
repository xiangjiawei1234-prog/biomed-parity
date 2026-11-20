# mstate 转移风险拟合

`biomed_parity.mstate_fit.msfit()` 从计数过程长表估计各转移的累积风险。每条边拥有单独的 Breslow 基线风险。可以只估计基线，也可以拟合各层共享的 Cox 系数，或传入外部系数进行预测。返回的时间轴、转移边和累积风险方差可直接用于 Aalen–Johansen 预测。对需要独立系数的协变量，可以先按转移展开。 [mstate::msfit 手册](https://hputter.github.io/mstate/reference/msfit.html) · [mstate 分析教程](https://www.jstatsoft.org/article/view/v038i07) · [按转移展开协变量](mstate-covariates.md)

## 从宽表到状态概率

```python
from biomed_parity.mstate_prep import msprep
from biomed_parity.mstate_fit import msfit

records = msprep(
    time=[[None, 2, 6], [None, 4, 7], [None, 6, 5]],
    status=[[None, 1, 1], [None, 1, 0], [None, 0, 1]],
    transitions=[(0, 1), (0, 2), (1, 2)],
)

fit = msfit(records, vartype="greenwood")
prediction = fit.predict(predt=0, covariance=True)
print(prediction.times)
print(prediction.probabilities[:, 0, :])
```

已在长表中的数据可以直接传入 `t_start`、`t_stop`、`status`、`transition` 和 `transitions`。边和状态均从 0 编号，允许有回访的长表。区间为 `(t_start, t_stop]`，恰在事件时点进入的行当时尚不属于风险集。默认做近并列修正。真实或修正后的零长度区间需先按研究方案处理，有事件的区间终点必须有限。

## 系数与不确定性

- `covariates` 为与长表逐行对齐的数值设计矩阵。默认拟合各转移共享的 Cox 系数。要为某些转移单独设定效应，可在设计矩阵中用转移指示量乘以相应协变量。
- `new_covariates` 可为一行共同的设计向量，也可给每条转移一行。默认全零。若传入 `coefficients`，计算采用外部系数。可附送同一尺度的 `coefficient_covariance`。
- `vartype="aalen"` 的累计协方差包含各边基线的计数方差，以及跨转移共享系数的梯度传播项。外部系数未附协方差时，方差条件于这些系数。
- `vartype="greenwood"` 用于无协变量的估计。它在共享同一风险集的出边之间保留竞争事件的负协方差。对于外部长表，函数会核对同一来源状态的风险集人数和事件总数，风险集人员一致性由调用者保证。
- 输出网格为事件时点和最后一个有限随访终点。没有观察到事件的边保留零风险轨迹。`variance=False` 可跳过风险协方差。样本权重、稳健聚类方差、随机效应、其它并列算法及独立的多层基线映射仍待扩展。 [Breslow 跳跃及方差推导](https://journal.r-project.org/articles/RJ-2017-062/) · [mstate::probtrans 手册](https://hputter.github.io/mstate/reference/probtrans.html)

## 本地对照

R 4.5.3、survival 3.8-6 和 mstate 0.3.3 直接拟合 7 个场景。覆盖无协变量 Aalen 或 Greenwood、协变量拟合、外部系数和条件方差、回访模型及关闭方差。3 个方案同时验证从宽表到状态概率、标准误及完整协方差。固定场景使用相对容差 `1e-10`、绝对容差 `1e-12`。时间边界、无事件、无限删失时间和不可估输出另有独立用例。

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_mstate_fit.py
```
