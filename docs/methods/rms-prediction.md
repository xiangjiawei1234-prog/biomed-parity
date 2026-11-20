# rms 变量分布与逐点预测

`rms_prediction` 提供 `datadist` 风格的变量分布摘要，以及从已有系数和协方差计算逐点均值预测及 Wald 置信区间的接口。可单独生成预测设置，也可将设置传给已有模型的设计函数。 [rms datadist](https://cran.r-project.org/web/packages/rms/refman/rms.html#datadist) [rms Predict](https://cran.r-project.org/web/packages/rms/refman/rms.html#Predict)

```python
import numpy as np

from biomed_parity.rms_prediction import datadist, predict

distribution = datadist(
    {"age": [30, 40, 45, 50, 60], "arm": ["control", "control", "active", "control", "active"]},
    factor_levels={"arm": ["control", "active"]},
)

def design(grid):
    age = grid["age"]
    return np.column_stack((np.ones(len(age)), age, grid["arm"] == "active"))

result = predict(
    distribution, {"age": [35, 45, 55], "arm": None}, design,
    coefficients=[0.4, 0.03, -0.2],
    covariance=np.diag([0.01, 0.0001, 0.04]),
    ref_zero=True,
)
print(result.grid)
print(result.estimate)
print(result.lower, result.upper)
```

## 变量分布与预测设置

`datadist` 接受同长度的命名向量。`limits` 给出各变量的效应范围、调整值、显示范围和观测范围，`values` 记录离散值。数值变量只有两个不同值时用较小值调整，三个时用居中的唯一值调整，其余用中位数。两个或三个不同值时直接用观测范围。默认效应范围取 25% 和 75% 分位数，默认显示范围取两端各 `10 / max(n, 200)` 的分位数。依照 rms 8.2.1 的做法，默认显示分位数由首个具有至少四个唯一值的数值变量的非缺失样本数确定，随后各变量沿用该分位数。两个端点重合时回退到观测范围。 [rms 数值范围](https://cran.r-project.org/web/packages/rms/refman/rms.html#datadist)

文字值推断为分类水平，也可通过 `factor_levels` 明确类别顺序、声明未在数据中出现的水平，或者将数值标签指定为类别。`categorical_adjustment="mode"` 用频数最高的类别，平手时取排在前面的水平。`categorical_adjustment="first"` 始终取首个水平。

`prediction_grid(distribution, vary, points=200)` 返回各变量的设置向量。用 `None` 指定变量的默认取值。分类变量使用全部水平，离散数值变量使用 `values` 中的值，其他连续变量在显示范围内均匀取点。其余变量采用调整值。用 `adjust_to` 可一次覆盖部分调整值，这些覆盖值也用于相对预测的参考设计。笛卡尔积中第一列变化最快。也可将空映射传给 `vary`，取得一个仅含调整值的设置。

## 预拟合设计与推断

`predict` 将设置向量传给调用者提供的设计函数，再使用 `wald_predict` 计算预测。设计函数应输出与系数同顺序的数值列，有截距的模型需要提供截距列。样条及交互项可组合已有 [限制性三次样条基](splines.md) 和分类指示列。已有数值设计矩阵时可直接调用 `wald_predict`。

- 普通预测计算 `X @ beta`，方差为同一设计向量对系数协方差的二次型。`center` 可提供 Cox 模型拟合时的设计列均值，预测和方差均使用中心化设计。
- `ref_zero=True` 先生成调整值对应的参考设计，再使用差值设计 `X - X_ref` 计算估计和方差。参考行的差值和标准误均为零。低层接口用 `reference` 直接提供参考设计。
- 默认用标准正态临界值。提供 `residual_df` 时改用 Student t 临界值，例如 OLS 的残差自由度。可通过 `confidence_level=0` 只返回预测和标准误。
- `transform` 可用单调函数把估计和端点转到所需尺度，例如 `np.exp` 将参考差转换为比值。标准误保留在线性预测尺度。

当前接口覆盖普通分类和数值变量摘要，以及已有系数和协方差下的均值逐点区间。 [rms predictrms](https://cran.r-project.org/web/packages/rms/refman/rms.html#predictrms)

## 本地验证

```bash
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q tests/test_rms_prediction.py
R_LIBS_USER="$PWD/.r-lib" .venv/bin/python validation/rms_prediction/verify.py
```

用 R 4.5.3 和 rms 8.2.1 比较八个案例。三个摘要案例覆盖缺失值、离散变量、分类顺序及并列分位数。五个预测案例使用 R 拟合的系数和协方差，核对 OLS 样条与分组交互、逻辑回归风险概率和参考比值、Cox 中心化线性预测。预测设置、线性预测、标准误及区间均纳入对照，预测值和区间的绝对容差为 `1e-10`。
