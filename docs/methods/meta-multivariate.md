# 多结局 Meta 分析的随机效应协方差

`meta_multivariate` 拟合 `metafor::rma.mv(random = ~ outcome | study)` 的单个多结局随机效应项。每项研究有一个结局随机效应向量，其协方差为待估计的 `G`。研究之间的随机效应独立。同一研究重复报告某个结局时，对应行共享同一随机效应。实际报告的结局可以因研究而异。已知抽样方差或协方差 `V` 由调用方提供。边际协方差在 `V` 上加上与每行研究和结局相匹配的 `G` 元素。 [metafor 多变量模型](https://wviechtb.github.io/metafor/reference/rma.mv.html)

## 接口

```python
import numpy as np
from biomed_parity.meta_multivariate import meta_multivariate

result = meta_multivariate(
    [0.15, 0.35, 0.95, 1.10, 0.30, 0.81, -0.20, 0.22, 0.64, 1.02, 0.08, -0.25],
    np.full(12, 0.04),
    studies=np.repeat(np.arange(6), 2),
    outcomes=["baseline", "followup"] * 6,
    outcome_levels=["baseline", "followup"],
    moderators=np.tile([0, 1], 6),
    structure="UN",
)
print(result.fit.coefficients, result.random_covariance)
```

| `structure` | `G` 的参数 |
| --- | --- |
| `UN` | 各结局各有方差，每对结局的相关可独立变化 |
| `CS` | 共同方差和共同的非对角相关 |
| `HCS` | 各结局各有方差，共同的非对角相关 |
| `ID` | 共同方差，结局随机效应独立 |
| `DIAG` | 各结局各有方差，结局随机效应独立 |
| `AR` | 共同方差，滞后 `d` 的相关为 `ρ**d` |
| `HAR` | 各结局各有方差，滞后 `d` 的相关为 `ρ**d` |

结局默认按首次出现顺序排列。对等间距时点用 `outcome_levels` 指定 `AR` 或 `HAR` 的时点顺序。`UN` 用下三角矩阵保证 `G` 半正定，复合对称结构的相关下限为 `-1/(J-1)`，其中 `J` 为结局数。估计后通过广义最小二乘计算固定效应。`result.fit` 提供系数、系数协方差、Wald 区间、异质性与联合检验、对数似然及边际协方差。`result.random_covariance` 和 `result.random_correlation` 按 `outcome_levels` 返回。方差为零时，对应的相关系数为 NaN。

`method` 接受 `REML` 或 `ML`。`test` 支持正态 Wald、t、Knapp–Hartung 和带下限的调整。`dfs="contain"` 检查每一列设计在研究和研究内结局两级的变化。现有多层预测接口可使用 `result.fit`。新设计的随机效应方差随结局变化时，传入相应 `G` 对角元素作为 `heterogeneity`。以 `reestimate=False` 进行删除诊断时，保持原拟合的 `G`。

## 验证与范围

R 4.5.3、metafor 5.0-1 的八组人工场景覆盖七种结构、两种似然、缺结局、同研究同结局重复汇报、已知非对角 `V`、负自相关以及包含性自由度。对照包括 `G`、系数与系数协方差、置信区间、联合统计量和似然。核对中 `G` 最大绝对偏差低于 `1.4e-6`，对数似然最大绝对偏差低于 `1.5e-10`。单结局特例另与原生随机截距拟合比较。 [对照用例](../../validation/meta_multivariate/cases.json) · [R 参考脚本](../../validation/meta_multivariate/oracle.R) · [参考输出](../../validation/meta_multivariate/expected.json)

相关缺乏识别信息时会报错，包括没有任意结局对在同一研究中共同出现的无结构模型，以及只有偶数滞后的自回归模型。实现采用稠密矩阵。非等间隔时点、空间相关、随机斜率、多项相关随机效应、固定部分协方差参数、协方差参数区间和删除后重估 `G` 仍待迁移。

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 python -m pytest -q -W error tests/test_meta_multivariate.py
```
