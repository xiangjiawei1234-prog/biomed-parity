# 两臂 Kaplan–Meier 曲线与 RMST 面积

`biomed_parity.rmst_curves.rmst_curve_data` 返回两臂 Kaplan–Meier 阶梯曲线、截断时间以内的 RMST/RMTL 填色多边形和未调整估计。数据可以直接交给绘图库，也可以保存为 JSON。几何定义对应 survRM2 1.0-4 的 `plot.rmst2()`。[survRM2 绘图手册](https://search.r-project.org/CRAN/refmans/survRM2/html/plot.rmst2.html) · [绘图实现](https://github.com/uno1lab/survRM2/blob/master/R/plot.rmst2.R)

```python
import json
from pathlib import Path

from biomed_parity.rmst_curves import rmst_curve_data

data = rmst_curve_data(
    time=[0, 2, 4, 4, 7, 1, 3, 5, 7, 8],
    status=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    arm=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    tau=6,
    pointwise_confidence_level=0.95,
)
Path("rmst-curves.json").write_text(json.dumps(data.to_dict(), allow_nan=False), encoding="utf-8")
```

## 结果结构

输入及 `tau` 的默认选择与 [`rmst2`](rmst2.md) 相同，状态 1 为事件，0 为右删失，臂 0 为参照。结果的 `summary` 包含标准未调整 `rmst2` 表，包括 `tau`、逐臂 RMST/RMTL、置信区间和三种组间对比。参数 `alpha` 只用于这些 RMST/RMTL 估计。`arm0` 和 `arm1` 各含下列可绘图数组。

| 字段 | 数据 |
| --- | --- |
| `time`、`survival` | 各观测时点及该时点之后的 Kaplan–Meier 概率，保留纯删失时点。近并列时间沿用 `survival::survfit()` 的合并规则。 |
| `n_risk`、`n_event`、`n_censor` | 每个时点的在险、事件和删失计数。事件与删失同期时均计入该时点之前的在险人数。 |
| `log_survival_standard_error` | 对数生存概率的 Greenwood 标准误，最后一次事件把生存概率降至零时为无穷。 |
| `steps` | 两列 `(time, survival)`，从 `(0, 1)` 出发，每个观测时点重复两次以画出右连续阶梯。曲线延续到该臂最后观测时点。 |
| `rmst_polygons`、`rmtl_polygons` | 每个正宽区间一行、四个 `(time, survival)` 顶点。RMST 区域从 0 到曲线高度，RMTL 区域从曲线高度到 1。两者都只覆盖 `[0, tau]`，顶点积分分别等于该臂的 RMST 和 RMTL。 |
| `confidence_band` | 指定 `pointwise_confidence_level` 时返回 `(time, lower, upper)` 阶梯坐标，使用 `survival::survfit(conf.type="log")` 的逐点对数尺度区间。原版 `plot.rmst2()` 隐藏该区间，因此默认值为 `None`。生存概率降到零时的区间在 NumPy 中是 `NaN`。 |

`to_dict()` 将数组转为嵌套列表，无穷或缺失的数值转为 JSON `null`。使用曲线绘图时可把 `steps` 的两列传给 `plot`，并按顺序逐个填充 `rmst_polygons` 和 `rmtl_polygons` 的四点坐标。分组、颜色、线宽和填色样式由调用者决定。[survival::survfit 手册](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survfit.formula.html)

## 验证

14 个人工场景覆盖了 28 条臂曲线。参考脚本运行已安装的 `plot.rmst2()` 并记录传给 `lines` 与 `polygon` 的参数，去掉重复的原点和零宽多边形后与 Python 逐项比较。逐点区间由 `survfit()` 另行核验。场景含零时点事件、事件恰在 `tau`、`tau` 早于首个观测、两臂末次随访长短不一、并列和近并列、完全删失及生存降为零。[对照用例](../../validation/rmst_curves/cases.json) · [R 参考脚本](../../validation/rmst_curves/oracle.R) · [R 4.5.3 / survRM2 1.0-4 / survival 3.8-6 快照](../../validation/rmst_curves/expected.json)

时间和在险、事件及删失计数逐项一致。阶梯、区间与多边形坐标的最大绝对误差为 `4.5e-16`，RMST/RMTL 估计的最大误差为 `3.6e-15`。

```sh
python -m pytest -q tests/test_rmst_curves.py
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 python -m pytest -q -W error tests/test_rmst_curves.py
R_LIBS_USER="$PWD/.r-lib" Rscript --vanilla validation/rmst_curves/oracle.R validation/rmst_curves/cases.json
```
