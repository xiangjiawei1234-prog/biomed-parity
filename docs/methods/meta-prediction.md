# 多层 Meta 分析预测与区间

`biomed_parity.meta_prediction.predict_multilevel_meta` 从 `meta_multilevel` 的拟合结果计算平均效应、置信区间、新研究效应的预测区间及可选的预测均值协方差。输入为已编码的完整设计行，列顺序与拟合系数相同，包含适用的截距。省略 `design` 时使用拟合的原始设计行。 [metafor::predict.rma](https://wviechtb.github.io/metafor/reference/predict.rma.html) · [多层拟合](meta-multilevel.md)

```python
from biomed_parity.meta_prediction import predict_multilevel_meta

# fit 估计了截距及两个调节变量的系数
result = predict_multilevel_meta(fit, [[1, -1, 0], [1, 1, 0]], covariance=True)
print(result.estimate, result.confidence_intervals, result.prediction_intervals)
```

均值置信区间以拟合系数的协方差作线性传播。t 型推断对每行采用其非零设计列中最小的自由度。预测区间将随机效应方差加入均值不确定性，默认面向新研究中的真实效应。提供 `new_sampling_variance` 时，再加入未来观测效应的抽样方差。对于训练行，随机效应方差由各协方差核的对角线与拟合方差分量合成。若训练行合成方差不同，新设计需通过 `heterogeneity` 传入其对应的异质性。也可用该参数覆盖通常的共同异质性。

`prediction="default"` 沿用拟合的正态或 t 校准。`"t"` 使用行自由度，若原拟合使用正态校准，则使用残差自由度。`"riley"` 从这些自由度中扣除估计的方差分量数，最低为 1。`"simple"` 使用正态分位数，仅在预测区间中省略均值估计方差。`adjust=True` 对同批的预测行做 Bonferroni 校正。 [metafor 区间和自由度](https://wviechtb.github.io/metafor/reference/predict.rma.html)

`transform` 可变换点估计和区间端点，例如在对数风险比尺度拟合后使用 `numpy.exp`。变换后，均值标准误记为缺失，`prediction_standard_errors` 仍采用原始线性尺度。当前范围为已实现的多层协方差模型、数值调节变量和模型型推断。

使用 R 4.5.3 和 metafor 5.0-1 核对六组人工案例，包括逐行自由度、Bonferroni 校正、未来观测误差、固定和估计的异质性、变换后端点和预测均值协方差。R 对多层 Knapp–Hartung 拟合标注为实验性。Python 单独核对非单位随机效应核的逐行异质性。 [参考数据](../../validation/meta_prediction/expected.json)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_meta_prediction.py
```
