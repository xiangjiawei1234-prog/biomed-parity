# 按类拟合分布的参数化平滑 ROC

`biomed_parity.roc_parametric.smooth_parametric_roc` 对病例和对照分别拟合分布，返回平滑 ROC、完整或部分 AUC，以及每组参数估计。实现覆盖 pROC `smooth(method="fitdistr")` 调用 MASS 时，带闭式极大似然估计的三个连续分布。[pROC 平滑手册](https://cran.r-project.org/web/packages/pROC/refman/pROC.html#smooth) · [MASS 拟合手册](https://cran.r-project.org/web/packages/MASS/refman/MASS.html#fitdistr)

```python
from biomed_parity.roc_parametric import smooth_parametric_roc

roc = smooth_parametric_roc(
    labels=[0, 0, 0, 0, 1, 1, 1, 1],
    scores=[0.1, 0.4, 0.8, 1.8, 0.2, 0.7, 1.5, 3.9],
    control_distribution="lognormal", case_distribution="exponential",
    direction="auto", partial_auc=(0.65, 1.0),
)
print(roc.auc, roc.control_fit.estimates)
```

## 分布、尺度和返回值

`labels` 是和分数等长的一维向量，`0` 为对照，`1` 为病例。两组分布可独立选择，默认均为 `"normal"`。`control_fit` 与 `case_fit` 提供分布名、参数名 `parameter_names`、参数估计 `estimates`、标准误 `standard_errors`、协方差 `covariance`、对数似然 `log_likelihood` 与人数 `n_observations`。

| 分布 | 参数与输入 |
| --- | --- |
| `"normal"` | `mean`、`sd`。每组至少两个观测，分数为有限实数且有正方差。标准差的极大似然估计以组内人数为分母。 |
| `"lognormal"` 或 `"log-normal"` | `meanlog`、`sdlog`。每组至少两个严格为正的分数，在自然对数尺度上有正方差。 |
| `"exponential"` | `rate`，为组内均值的倒数。允许零值，要求分数非负且组内均值为正。 |

`direction="<"` 表示高分偏向病例，`">"` 表示低分偏向病例，`"auto"` 使用两组中位数。`n` 是合并分数最小和最大值间的等距网格大小，默认为 512。两组拟合后的密度在同一网格上计算并各自归一化。每个网格阈值依据对应方向累计密度，灵敏度包含该阈值的病例质量。返回按特异度升序排列的 `specificity` 和 `sensitivity`，含 `n` 个平滑点和 `(0,1)`、`(1,0)` 两个锚点。[pROC 平滑实现](https://github.com/xrobin/pROC/blob/master/R/smooth.R) · [密度坐标实现](https://github.com/xrobin/pROC/blob/master/R/roc.utils.R)

完整 AUC 使用含锚点曲线的梯形积分。`partial_auc=(a,b)` 可用 `focus="specificity"` 或 `"sensitivity"` 计算部分面积，可用 `standardize=True` 做 McClish 校正并用 `allow_below_chance=True` 保留低于机会水平的校正值。`percent=True` 将坐标、AUC 和部分面积端点改用 0–100 尺度。分布参数保留原始尺度。[pROC AUC 手册](https://cran.r-project.org/web/packages/pROC/refman/pROC.html#auc)

## 验证与范围

17 个场景覆盖 7 组人工数据。12 个数值场景比较两组拟合参数、标准误、协方差、对数似然、整条曲线与 AUC，并覆盖不同支持范围、两种方向、单点网格、重复网格值、两种部分面积轴、校正和百分制。另有 3 个双方均拒绝的场景，2 个 pROC 返回缺失 AUC 而 Python 报错的场景。数值容差为相对 `3e-14`、绝对 `5e-15`，百分制坐标和 AUC 的绝对容差乘以 100。实测最大绝对误差为归一化曲线坐标 `3.6e-15`、AUC `2.5e-15`、参数 `4.9e-15`、对数似然 `4.9e-14`。[用例](../../validation/roc_parametric/cases.json) · [R 4.5.3 / pROC 1.19.1 / MASS 7.3-65 快照](../../validation/roc_parametric/expected.json) · [R 参考脚本](../../validation/roc_parametric/oracle.R)

零方差正态样本或网格内没有某一组的正密度时会抛出 `ValueError`。密度计算在对数尺度归一化，以避免普通密度求值中的下溢。支持范围为上述闭式模型的未加权二分类点估计。其它 MASS 参数分布、用户提供密度函数、拟合起始值及该方法的重抽样推断属于独立工作。

```sh
python -m pytest -q tests/test_roc_parametric.py
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 python -m pytest -q -W error tests/test_roc_parametric.py
R_LIBS_USER="$PWD/.r-lib" Rscript --vanilla validation/roc_parametric/oracle.R validation/roc_parametric/cases.json
```
