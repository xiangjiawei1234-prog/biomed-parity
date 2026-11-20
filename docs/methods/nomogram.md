# 临床预测模型的列线图分值

`biomed_parity.nomogram.nomogram` 从已有模型系数与设计函数生成可绘制的列线图数据，包含各条预测变量轴、线性预测轴、一个或多个结局轴以及分值换算。支持分类变量、非线性样条和按交互变量取值拆分的多条轴。 [rms::nomogram](https://search.r-project.org/CRAN/refmans/rms/html/nomogram.html)

```python
import numpy as np
from scipy.special import expit
from biomed_parity.nomogram import NomogramTerm, nomogram
from biomed_parity.rms_prediction import datadist

distribution = datadist({"age": [30, 40, 50, 60], "arm": ["A", "B", "A", "B"]})

def design(grid):
    age = grid["age"]
    arm_b = grid["arm"] == "B"
    return np.column_stack((np.ones(len(age)), age, arm_b, age * arm_b))

scale = nomogram(
    distribution, {"age": NomogramTerm([1, 2, 3], [30, 40, 50, 60], by={"arm": None})},
    design, [-4.1, 0.07, 0.55, -0.015], outcomes={"风险": (expit, [-2, -1, 0, 1])},
)
print(scale.predictor_axes, scale.outcome_axes)
```

每个 `NomogramTerm` 指定构成一个打分项的系数列。设计列从零起算。除截距之外，每一列由一个打分项使用。将相关的主效应和交互项列放入同一个打分项，使用 `by` 为该预测变量生成各个限定水平的轴。名义因子的水平默认来自 `datadist`，连续交互变量需提供显式水平。可通过 `intercept_column=None` 表示无截距模型，通过 `linear_offset` 传入已知的线性偏移。

同一打分项的所有条件轴共用最小效应。将每个效应减去该最小值，再用全模型中跨度最大的打分项确定统一的点数尺度。选择每个打分项中适用的一条轴并合计点数，`to_linear_predictor` 可将总点数换回模型预测尺度。`outcomes` 中的变换接受线性预测轴上的数值，并逐个返回结局轴刻度标签。

可显式提供各变量和线性预测的刻度，以对齐既有 R 图。默认连续轴在已观测范围内均匀取点。R 的 `pretty` 刻度选择、图形布局和列线图的区间仍待迁移。

本地用 R 4.5.3 和 rms 8.2-1 核对两组人工数据，包含限制性三次样条的加性 OLS、名义因子与样条交互的二元 logistic 模型，以及指数与反 logit 结局轴。验证各轴效应和分值、共同零点、比例、结局刻度与合计分值换算。 [R 验证脚本](../../validation/nomogram/oracle.R)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_nomogram.py
```
