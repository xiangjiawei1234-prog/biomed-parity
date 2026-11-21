# Royston–Parmar 样条回归

`fit_royston_parmar` 对应 flexsurv 的 `flexsurvspline()` / `flexsurvreg(dist = "survspline")` 拟合：在给定结点上估计 Royston–Parmar 样条系数（基线），以及主设计（flexsurv 位置参数）和 ancillary 设计的回归系数。支持右删失、精确时间、左删失和区间删失，支持左截断、频数权重和按样条系数分别指定的 ancillary 回归。`royston_parmar_log_likelihood` 对给定系数计算同一加权似然。`RoystonParmarFit.predict` 复用 [Royston–Parmar 分布](royston-parmar.md) 的计算入口。实现只依赖 NumPy 和 SciPy。[flexsurvspline 手册](https://chjackson.github.io/flexsurv/reference/flexsurvspline.html) · [flexsurvreg 手册](https://chjackson.github.io/flexsurv/reference/flexsurvreg.html) · [flexsurv 样条分布](https://chjackson.github.io/flexsurv/reference/Survspline.html) · [Royston–Parmar 方法论文](https://doi.org/10.1002/sim.1203)

```python
import numpy as np
from biomed_parity.royston_parmar_regression import (
    fit_royston_parmar,
    royston_parmar_log_likelihood,
)

time = np.array([0.42, 0.68, 1.05, 1.47, 1.92, 2.55, 3.28, 4.16, 5.30, 6.75, 8.60, 11.02,
                 0.55, 0.88, 1.31, 1.79, 2.40, 3.07, 4.02, 5.12, 6.53, 8.32, 10.61, 13.49])
event = np.array([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1,
                  1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1])
right = np.where(event == 1, time, np.inf)

fit = fit_royston_parmar(time, right, knots=[-0.9, 1.2, 2.7])
print(fit.gamma, fit.log_likelihood, fit.covariance)
print(royston_parmar_log_likelihood(time, right, fit.gamma, knots=fit.knots))  # 同一个似然
print(fit.predict([0.5, 1.5, 4.0]).survival)

x = np.linspace(-1.5, 1.5, time.size)[:, None]  # 不含截距的设计矩阵
regression = fit_royston_parmar(time, right, x, knots=fit.knots)
print(regression.coefficients, regression.predict([1.0, 3.0], design=[[0.5]]).hazard)
```

## 统计口径

- `left` 与 `right` 描述每个观测的事件区间，端点约定与 [区间删失端点定义](interval-censoring.md) 一致：`left == right` 为精确事件，`left == 0` 为左删失，`right = inf` 为右删失，`0 < left < right < inf` 为区间删失，区间取左开右闭 `(left, right]`。
- `knots` 是一维、严格递增的完整结点向量，首尾为边界结点，中间为内部结点，按 `timescale` 选择的尺度给出（默认 `"log"` 为对数时间）。默认 `(-10, 10)` 没有内部结点，此时模型在变换后时间上是线性的。`timescale="identity"` 的结点按原始时间给出。
- 样条预测量 `η` 的前两个系数为截距和线性项，之后每个内部结点一个系数；`hazard`、`odds`、`normal` 三种尺度分别为 `log H(t)`、`log((1-S)/S)` 和 `-Φ⁻¹(S)`。无内部结点、`hazard`、对数时间时为 Weibull：`shape = gamma1`，`scale = exp(-gamma0 / gamma1)`。[flexsurv 基底与分布实现](https://github.com/chjackson/flexsurv/blob/master/R/spline.R)
- `design` 的列作为回归作用在第一个样条系数（flexsurv 的位置参数 `gamma0`）上；`ancillary` 是以样条系数下标（从 1 开始）为键的映射，键 `j` 的设计作用在 `gamma_j` 上。设计矩阵不含截距，同一组变量可以同时出现在主设计和 ancillary 设计中。参数估计按基线、位置系数、ancillary 系数（下标升序）排列。主设计和每个 ancillary 设计都必须满秩（连同类间截距），否则拒绝拟合。
- `entry` 为左截断的进入时间。每个观测的似然是在 `T > entry` 条件下的概率，即用无条件概率除以 `S(entry)`；进入时间必须晚于 0、严格早于区间上界（右删失时不得晚于观测时间）。`predict` 传入 `entry` 时返回同样的条件分布。
- `weights` 为非负频数权重：似然为 `sum(w_i * log p_i)`，协方差为加权观测信息矩阵的逆，权重为 0 的行被丢弃且不计入 `n_observations`。
- 优化使用解析梯度与 BFGS，从线性时间的多个初值启动，要求缩放后的梯度足够小；协方差由解析梯度中心差分得到的观测信息求逆。`initial` 可按上述参数顺序提供固定初值。
- `covariance` 是自然参数尺度上的估计协方差；结点、`scale` 和 `timescale` 原样保存在拟合结果中，供 `predict` 复用。

`RoystonParmarFit` 字段：`gamma`、`coefficients`、`ancillary_coefficients`、`covariance`、`knots`、`scale`、`timescale`、`log_likelihood`、`n_observations`。`predict(time, design=None, *, ancillary=None, entry=None)` 返回 [RoystonParmarResult](royston-parmar.md)，包含生存率、累积风险、瞬时风险、密度、累积分布及对应对数值。

`predict` 的 `time` 与设计、ancillary 的末维遵循 NumPy 广播规则：`design` 的每行给出一个预测剖面，与 `time` 的前导维逐行对应；若要对同一组时间输出多条剖面，逐条调用 `predict`。`predict` 只接受拟合过的 ancillary 下标。

## 数值对照

R 4.5.3 和 flexsurv 2.3.2 的 17 个合成用例与入库快照/实时 R 对照，其中 10 个拟合用例、7 个给定系数下的似然用例。拟合覆盖：无内部结点的 Weibull 特例（同时对照 `flexsurvreg(Surv(time, event) ~ 1, dist = "weibull")` 的形状/尺度到 gamma 的映射及 delta 方法协方差）、一至两个内部结点、`hazard`/`odds`/`normal` 三种尺度、对数与恒等时间尺度、右删失、精确事件、区间删失、左截断（counting 响应）、频数权重、单列主协变量，以及作用在 `gamma1` 上的单列 ancillary 回归。似然用例覆盖精确/右删失、区间删失、左截断、权重、三种尺度和逐观测系数。预测对照核对 12 个预测块（19 条剖面曲线）上的生存率、累积风险、瞬时风险、密度和累积分布；不含 `entry` 的生存率/累积风险/风险由 `summary(fit, type = ...)` 给出，密度和累积分布由 `dsurvspline`/`psurvspline` 给出，带 `entry` 的条件预测按同一条件公式计算。

| 对照项 | 容差 | 实测最大偏差 |
| --- | --- | --- |
| 系数/回归系数（绝对） | `2e-6` | `1.3e-6`（ancillary 拟合） |
| 对数似然（绝对） | `1e-9` | `1.9e-11`（ancillary 拟合） |
| 协方差（相对） | `1e-2` | `3.6e-3`（区间删失拟合） |
| 参考参数下的独立似然（绝对） | `1e-9` | `4.3e-14` |
| Weibull 映射：gamma / 对数似然 / 协方差 | `1e-7` / `1e-10` / `1e-8` | `5.1e-9` / `2.8e-14` / `2.3e-10` |
| 分布预测（相对，绝对下限 `1e-9`） | `2e-4` | `2.6e-5` |
| 给定系数的似然用例（绝对） | `1e-9` | `5.1e-13` |

[合成用例](../../validation/royston_parmar_regression/cases.json) · [R 对照脚本](../../validation/royston_parmar_regression/oracle.R) · [版本化快照](../../validation/royston_parmar_regression/expected.json)

```sh
.venv/bin/python -m pytest -q tests/test_royston_parmar_regression.py
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_royston_parmar_regression.py
```

## 已知差异和未覆盖范围

- 系数偏差来自两个优化器的停止准则而非估计目标差异：系数量级的最大差为 `1.3e-6`，此时对数似然差 `1.9e-11`。对照脚本把 R `optim` 的 `reltol` 收紧到 `1e-14`、有限差分步长收紧到 `1e-6`，使区间删失和 `normal` 尺度（R 侧无解析导数）也收敛到同一最优点。
- 协方差的最大差异出现在区间删失拟合（相对 `3.6e-3`）和 `normal` 尺度拟合（相对 `4.3e-4`）：flexsurv 在这两种情形用数值 Hessian 计算协方差，Python 用解析梯度的差分构造观测信息。右删失的 `hazard`/`odds` 尺度、权重、协变量、ancillary 与 Weibull 特例的协方差相对差异在 `1e-7` 量级或更小。Weibull 对照中 `flexsurvreg` 的协方差在对数形状/对数尺度上估计，oracle 用 delta 方法映射到 gamma 参数化后再比较。
- 无内部结点的 Weibull 特例中，Python 的 `knots=(-10, 10)` 只影响边界而不进入预测量；flexsurv 侧用显式边界结点拟合零内部结点样条，结果与 `flexsurvreg(dist = "weibull")` 的映射一致。
- 带 `entry` 的条件预测无法由 `summary.flexsurvreg` 直接给出，oracle 用 `psurvspline`/`hsurvspline`/`dsurvspline` 按条件公式构造；非条件预测已与 `summary()` 逐值核对（本脚本生成的快照中两条路径完全一致）。
- 未覆盖：超过两个内部结点；多列主设计矩阵；除 `gamma1` 以外的 ancillary 下标、多个 ancillary 设计或多个 ancillary 协变量；结点自动选择（`flexsurvspline` 的 `k=` 分位数结点）；`splines2ns` 基底与 `spline="ns"`；`initial` 只做了“到达同一最优点”的自洽检查，没有单独的 R 对照；左截断与区间删失同时出现的组合；删失/截断端点与 flexsurv 差异的边界情形；真实数据；优化失败、分离和近奇异设计；`bhazard` 相对生存模型。
- 当前只验收点估计、协方差与分布预测，不包含 `summary()` 的置信区间、均值、中位数、RMST 或分位数；这些入口不在本模块范围内。
