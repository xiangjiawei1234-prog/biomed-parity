# 区间删失的 Turnbull 非参数估计

如果定期随访只能确定事件发生在两次检查之间，可以直接对事件区间写出似然。`turnbull` 按开闭端点构造最大交集，并估计这些交集上的概率质量。准确事件时间、左删失、右删失及有限区间可同时使用。 [icenReg::ic_np](https://search.r-project.org/CRAN/refmans/icenReg/html/ic_np.html) · [icenReg 方法论文](https://www.jstatsoft.org/article/view/v081i12)

```python
import numpy as np
from biomed_parity.interval_censoring import turnbull

fit = turnbull(
    [0, 1, 1, 3, 5],
    [1, 1, 2, 4, np.inf],
    weights=[1, 2, 1, 1, 1],
)
print(fit.intervals, fit.mass)
print(fit.survival_bounds([1, 1.5, 4, 7]))
```

| 输入或输出 | 约定 |
| --- | --- |
| `left`、`right` | 每人一个区间，要求 `0 <= left <= right`。零下界可表示左删失，正无穷上界表示右删失，相等的有限端点表示准确时间。 |
| `bounds` | 分别标记两个端点是否闭合。默认 `(False, True)` 对应 `(L, R]`。准确时间始终闭合。 |
| `weights` | 每人一个有限非负权重。零权重行不参与构造或拟合。 |
| `intervals`、`closed`、`mass` | 最大交集的端点、闭合状态和拟合质量，保留估得零质量的交集。 |
| `survival_bounds(times)` | 每个时间上严格生存概率 `P(T > t)` 的上下界。给定拟合质量后，该界反映质量在交集内任意分布产生的不确定性。 |
| `log_likelihood`、`optimality_gap` | 原权重下的对数似然和按总权重归一化的似然最优性上界。 |

每行似然取其观察区间内所有交集的概率质量之和。优化在概率单纯形上进行，用解析导数检查收敛。若约束优化后仍有高于容差的最优性间距，使用活动概率方向上的 Newton 更新，必要时重新纳入可改善似然的交集。允许在同一时间上精确区分闭区间和开区间。

## 对照与范围

R 4.5.3 与 icenReg 2.0.16 的八组人工场景对照最大交集、概率质量及加权对数似然。场景包括混合删失、分数权重、并列时间、全右删失及四种开闭组合。质量最大绝对差小于 `4e-8`，对数似然最大绝对差小于 `5e-12`。R 通过将开端点内移约 `1e-10` 表示开区间。Python 直接记录开闭状态，对距离更小的相邻时间仍保留原始端点。 [R 脚本](../../validation/interval_censoring/oracle.R) · [参考输出](../../validation/interval_censoring/expected.json)

当前是单组非参数拟合。协变量的半参数比例风险及比例优势模型、重采样推断、分组比较和截断数据列入后续工作。

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 python -m pytest -q -W error tests/test_interval_censoring.py
```
