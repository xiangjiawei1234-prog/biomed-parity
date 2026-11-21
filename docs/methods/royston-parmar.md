# Royston–Parmar 样条生存分布

给定结点和系数，`biomed_parity.royston_parmar` 计算 flexsurv 的 Royston–Parmar 生存分布与分位数。`royston_parmar()` 返回生存率、累积分布、累积风险、瞬时风险及密度。`royston_parmar_quantile()` 支持下尾、上尾和对数概率。计算只依赖 NumPy 和 SciPy。[flexsurv 分布手册](https://chjackson.github.io/flexsurv/reference/Survspline.html)

```python
from biomed_parity.royston_parmar import royston_parmar, royston_parmar_quantile

gamma = [-0.4, 1.25, 0.015, -0.01]
settings = dict(knots=[-3, -1, 0.5, 2], scale="hazard")
curve = royston_parmar([0.5, 1, 2, 5], gamma, **settings)
median = royston_parmar_quantile(0.5, gamma, **settings)
```

`gamma` 的末轴依次为截距、线性斜率和每个内部结点的系数。`knots` 的末轴从小到大包含左边界、内部结点和右边界，长度与 `gamma` 相同。默认结点 `(-10, 10)` 只有边界，此时模型在变换后的时间上为线性。`timescale="log"` 为默认值，结点按对数时间给出。`timescale="identity"` 的结点按原始时间给出。时间或概率数组与系数、结点的前导维度遵循 NumPy 广播规则，因此可以为不同预测行传入不同参数。[flexsurv 模型参数化](https://chjackson.github.io/flexsurv/reference/flexsurvspline.html)

| `scale` | 样条预测量 `η` | 无内部结点且使用对数时间时的模型 |
| --- | --- | --- |
| `"hazard"`（默认） | `log(H(t))` | Weibull |
| `"odds"` | `log((1 - S(t)) / S(t))` | 对数逻辑斯蒂 |
| `"normal"` | `-Φ⁻¹(S(t))` | 对数正态 |

每个内部结点使用 Royston–Parmar 的自然三次样条基底。边界之外用解析的线性形式计算，避免在远端相减两个较大的三次项。瞬时风险和密度用解析导数及时间变换的链式法则计算。结果的 `log_survival`、`log_hazard` 和 `log_density` 在对数尺度上计算，可用于尾部概率和似然。[flexsurv 基底及分布实现](https://github.com/chjackson/flexsurv/blob/master/R/spline.R)

输入时间小于或等于零时，生存率为 1，累积风险、瞬时风险和密度为 0，沿用 flexsurv。原始时间尺度下，零点右侧的累积概率可能大于零，落在该概率区间内的分位数归到零。对适定分布，正无穷处返回解析极限，flexsurv 对带内部结点的累积风险、瞬时风险和密度可能返回 `NaN`。系数导致局部下降时，点估计仍采用 flexsurv 的规则，将该处的瞬时风险和密度截为零。[flexsurv 分布实现](https://github.com/chjackson/flexsurv/blob/master/R/spline.R)

分位数先反解对应的概率链接，再确定目标所在的时间区间。线性尾部直接求根，内部三次区间以区间端点为界求根。该入口要求预测量在时间支持内非降，且右尾斜率为正。累积概率为 0 或 1 时，沿用 `qsurvspline()` 的负无穷和正无穷端点。极端上尾可直接传入 `lower_tail=False, log_p=True` 的对数生存概率。[flexsurv 分位数手册](https://chjackson.github.io/flexsurv/reference/Survspline.html)

## 数值对照

10 个合成场景与 R 4.5.3 和 flexsurv 2.3.2 对照，覆盖三种链接、两种时间尺度、无内部结点和多个结点、逐行结点与系数，以及局部下降时的截断。9 个场景还核对分位数。分布输出的最大绝对偏差为 `4.8e-14`，非零分位数的最大相对偏差为 `2.8e-13`。额外验证包括三种简化分布、远端线性预测量和极端对数尾概率的往返计算。[合成用例](../../validation/royston_parmar/cases.json) · [R 对照脚本](../../validation/royston_parmar/oracle.R) · [版本化快照](../../validation/royston_parmar/expected.json)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 python -m pytest -q -W error tests/test_royston_parmar.py
```

当前入口接收已给定的系数和结点。回归拟合、自动选结点及 `splines2ns` 基底属于后续范围。
