# 零膨胀计数模型的预测与模拟

`biomed_parity.count_predictions` 接收已拟合的 Poisson 或 NB2 零膨胀随机截距模型。它提供已观测簇的条件预测、将随机效应置零的预测和按簇生成的模拟。固定效应设计的列顺序须与拟合时相同。 [glmmTMB 预测](https://glmmtmb.github.io/glmmTMB/reference/predict.glmmTMB.html) · [glmmTMB 模拟](https://glmmtmb.github.io/glmmTMB/reference/simulate.glmmTMB.html)

```python
from biomed_parity.count_predictions import predict_count_mixed, simulate_count_mixed

prediction = predict_count_mixed(
    fit, groups=site, count_design=x, inflation_design=z, offset=log_exposure,
)
draws = simulate_count_mixed(
    fit, site, count_design=x, inflation_design=z, offset=log_exposure,
    nsim=500, seed=19,
)
print(prediction.response_mean, prediction.zero_probability)
```

提供 `groups` 时，同名簇使用它在拟合时的随机截距条件众数。省略整个 `groups`，或在个别行填入 `None`，将对应随机截距置零。新簇需设置 `allow_new_levels=True`，此时随机截距也取零。R 默认会警告后接受该组，Python 默认把未见组作为输入错误。NB2 如使用了离散度协变量，预测与模拟都应提供同样编码的 `dispersion_design`。

返回的 `conditional_mean` 是基础计数分布的均值，包含其自身的零计数。混合后的 `response_mean` 为该均值乘以未落入结构零状态的概率。`structural_zero_probability` 只计结构零，`zero_probability` 还计入 Poisson 或 NB2 抽出的零。Poisson 的离散度字段为 1，NB2 的字段为二次型方差的形状参数。当前返回点预测。

每轮模拟从拟合的正态分布为每个簇抽取一次新的随机截距，并依次生成基础计数和结构零。结果形状为「模拟轮数 × 数据行数」。同一轮同一簇共享随机截距；以固定拟合参数为条件。模拟与 R 比较分布矩。 [glmmTMB 模拟规则](https://glmmtmb.github.io/glmmTMB/reference/simulate.glmmTMB.html)

R 4.5.3 和 glmmTMB 1.1.14 的两个方案分别核对 Poisson 与带离散度回归的 NB2，包含暴露量、已知簇、缺失簇标识和新簇。点预测以 `2e-6` 绝对容差和 `4e-6` 相对容差核对。各组使用 R 3,600 次和 Python 4,800 次模拟，逐行均值、零值比例和协方差均在 5 倍联合蒙特卡洛标准误范围内。 [R 验证脚本](../../validation/count_predictions/oracle.R) · [参考结果](../../validation/count_predictions/expected.json)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_count_predictions.py
```
