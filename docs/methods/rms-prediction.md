# rms 变量分布与预测区间

`rms_prediction` 提供 `datadist` 风格的变量分布摘要，以及从已有系数和协方差计算均值预测、逐点 Wald 区间和同时置信带的接口。可单独生成预测设置，也可将设置传给已有模型的设计函数。 [rms datadist](https://cran.r-project.org/web/packages/rms/refman/rms.html#datadist) [rms Predict](https://cran.r-project.org/web/packages/rms/refman/rms.html#Predict)

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
    confidence_type="simultaneous",
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

`predict` 将设置向量传给调用者提供的设计函数，再使用 `wald_predict` 计算预测。设计函数应输出与系数同顺序的数值列，有截距的模型需要提供截距列。样条及交互项可组合已有 [限制性三次样条基](restricted-cubic-splines.md) 和分类指示列。已有数值设计矩阵时可直接调用 `wald_predict`。

- 普通预测计算 `X @ beta`，方差为同一设计向量对系数协方差的二次型。`center` 可提供 Cox 模型拟合时的设计列均值，预测和方差均使用中心化设计。
- `ref_zero=True` 先生成调整值对应的参考设计，再使用差值设计 `X - X_ref` 计算估计和方差。参考行的差值和标准误均为零。低层接口用 `reference` 直接提供参考设计。
- 默认的 `confidence_type="mean"` 为每个设置计算逐点 Wald 区间。默认用标准正态临界值，提供 `residual_df` 时改用 Student t 临界值，例如 OLS 的残差自由度。可通过 `confidence_level=0` 只返回预测和标准误。
- `confidence_type="simultaneous"` 将一次调用的全部预测行作为同一组，对标准化预测误差的最大绝对值求联合临界值。误差相关性由同一个系数协方差矩阵计算。提供 `residual_df` 时使用有共同尺度的多元 t 分布，否则使用多元正态分布。固定种子的 Sobol 准蒙特卡罗积分近似计算临界值。零标准误行的上下界等于预测值。
- `transform` 可用单调函数把估计和端点转到所需尺度，例如 `np.exp` 将参考差转换为比值。标准误保留在线性预测尺度。

逐点区间和同时置信带的覆盖对象都是均值预测或参考差值。同时置信带的覆盖范围由此次提交的有限预测设置确定。 [rms predictrms](https://cran.r-project.org/web/packages/rms/refman/rms.html#predictrms)

## 本地验证

```bash
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q tests/test_rms_prediction.py
R_LIBS_USER="$PWD/.r-lib" .venv/bin/python validation/rms_prediction/verify.py
```

使用 R 4.5.3、rms 8.2-1、multcomp 1.4-31 和 mvtnorm 1.4-2 比较 12 个案例。三个摘要案例覆盖缺失值、离散变量、分类顺序及并列分位数。九个预测案例使用 R 拟合的系数和协方差，分别验证 OLS 样条与分组交互、逻辑回归风险概率与相对参考设置的优势比、Cox 中心化线性预测。

五个逐点案例与 `rms::Predict` 对照，预测值、标准误和正态区间端点的绝对容差为 `1e-10`，t 区间端点为 `2e-10`。四个同时置信带案例与 `multcomp::glht` 及 `mvtnorm` 积分结果对照，在线性预测尺度上以每行标准误为单位比较端点，绝对容差为 `1.5e-3`。其中三个案例另与 `rms::Predict(conf.type="simultaneous")` 的默认临界值对照，绝对容差为 `6e-3`。rms 8.2-1 为带截距模型计算 `ref.zero` 同时置信带时未构造对应的截距列，因此相对参考设置的优势比案例将截距为零的设计差值直接传入 `glht`。
