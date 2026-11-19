# 固定时点的竞争风险校准

`biomed_parity.competing_calibration.calibrate_competing_risks` 接收对齐的随访时间、事件编码和预测累积发生风险。状态 0 表示右删失，正整数标记事件类型。观察端按指定 `cause` 估计。输出包含横轴预测值、纵轴实际发生风险、每个横轴值的样本数，以及可选的删失伪值。调用者可直接用这两列绘制曲线。 [riskRegression 校准接口](https://search.r-project.org/CRAN/refmans/riskRegression/html/plotCalibration.html) · [Gerds 等方法论文](https://doi.org/10.1002/sim.6152)

```python
from biomed_parity.competing_calibration import calibrate_competing_risks

curve = calibrate_competing_risks(
    time=[1, 2, 2, 3, 4, 5],
    status=[1, 0, 2, 1, 0, 2],
    risks=[.13, .22, .33, .52, .64, .77],
    horizon=3,
    method="nne",
    estimator="jackknife",
    bandwidth=.35,
)
print(curve.predicted, curve.observed)
```

分位数组使用 `groups` 指定组数或完整的分组边界。近邻法 `method="nne"` 使用明确给出的 `bandwidth` 作为经验秩的半径比例，默认将概率保留两位便于合并并列值。局部法在每组单独估计 Aalen–Johansen 累积发生率，超过该组最大随访时间时返回 NaN。无竞争事件时退化为 Kaplan–Meier 的事件风险。Jackknife 法在全队列计算逐行伪值，再按组或近邻求均值；伪值允许超出 [0, 1]。`jackknife_cumulative_incidence` 也可单独调用。 [prodlim 的伪值定义](https://search.r-project.org/CRAN/refmans/prodlim/html/jackknife.html)

按少数离散预测值直接分组时，Python 保留样本和预测的原始对应关系。自动带宽选择、信赖带、模型重拟合与图形样式仍待扩展。

## 本地对照

R 4.5.3、riskRegression 2026.03.11 和 prodlim 2026.03.11 验证七组人工输入。五组经过 `Score` 及 `plotCalibration` 核对预测坐标、校准结果、人数和伪值，包含竞争事件、单事件、显式分组边界、并列随访与并列预测。两组直接核对末尾唯一受试者的删失和事件。最大绝对差低于 `1.6e-14`。另用独立的逐个删除重估验证排序后按风险集线性遍历的伪值计算。

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest tests/test_competing_calibration.py -q
```
