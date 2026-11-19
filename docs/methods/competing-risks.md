# cmprsk 竞争风险适配

`pycmprsk` 提供 Fine–Gray 模型、累积发生率和 Gray 检验。仓库补齐本地对照中发现的两种输出行为。两个接口接受 NumPy 可读数组，可独立用于已有的累积发生率曲线或事件残差。

## 给定时点的累计发生率与方差

`cmprsk_timepoints(curve_times, estimates, variances, times)` 接收一条阶梯曲线。曲线时间允许重复，以分别保存跃点前后的值。返回排序并去重的查询时间、右连续估计和方差。在曲线的时点范围之外返回 `NaN`，并省略输入中的 `NaN` 时间。无穷查询时间抛出 `ValueError`。

```python
from biomed_parity.competing_risks import cmprsk_timepoints

times, incidence, variance = cmprsk_timepoints(
    [0, 3, 3, 5],
    [0, 0, 0.25, 0.25],
    [0, 0, 0.04, 0.04],
    [2, 3, 5, 6],
)
```

此例在时间 3 取跳变后的累计发生率 0.25，在时间 6 返回 `NaN`。也可以遍历 `pycmprsk.cuminc(...).curves`，将每条曲线的 `time`、`est` 和 `var` 传入该函数。

## 按唯一目标事件时间合计残差

`cmprsk_residuals_by_time(event_times, event_residuals)` 输入一行一个目标事件的残差矩阵及对应时点。先按照时点稳定排序，再将并列事件的残差相加。返回升序的唯一时点及合计矩阵。映射输入前先应用模型拟合时的完整病例、事件编码和 `subset`，然后按原生残差实际顺序配齐时点。

```python
from biomed_parity.competing_risks import cmprsk_residuals_by_time

times, residuals = cmprsk_residuals_by_time(
    [2, 3, 3],
    [[0.1, -0.2], [0.4, 0.2], [-0.1, 0.3]],
)
```

输出与 R 的 `crr()` 一样在时间 3 使用一行残差。若来自 `pycmprsk` 拟合，返回时点可与 `fit.uftime` 比对。

本地对照涵盖三组、三层分层检验、两个事件类型、并列时点和一个缺失病例，以及目标事件非默认编码、时变效应和分组删失。两种适配后 82 个字段均符合容差。 [验证记录与命令](../../validation/adoption/pycmprsk/README.md)
