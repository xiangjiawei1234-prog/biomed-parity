# 平滑 ROC 的 Bootstrap 区间

`biomed_parity.roc_smooth_bootstrap` 使用 Bootstrap 对单条平滑二分类 ROC 计算完整或部分 AUC 的分位数区间，也可在固定特异度上计算灵敏度的逐点区间。每个重抽样重新拟合双正态或核密度曲线。区间口径对应 pROC 1.19.1 的 `ci.auc()` 和 `ci.se()`。[pROC AUC 区间手册](https://cran.r-project.org/web/packages/pROC/refman/pROC.html#ci.auc) · [pROC 灵敏度区间手册](https://cran.r-project.org/web/packages/pROC/refman/pROC.html#ci.se)

## 接口

| Python 函数 | 返回字段 |
| --- | --- |
| `bootstrap_smooth_auc_ci(labels, scores, ...)` | `auc` 为原始样本的平滑 AUC，`median` 为重抽样的中位数，`confidence_interval` 为下、上分位点。可用 `partial_auc`、`focus` 和 `standardize` 指定部分 AUC。 |
| `bootstrap_smooth_sensitivity_ci(labels, scores, specificity, ...)` | `specificity` 保留输入顺序，`sensitivity` 为原始样本的估计，`intervals` 每行依次为下分位点、中位数和上分位点。可指定一个数或一维数组。 |

两种结果都提供总抽样次数 `n_resamples` 和被丢弃的次数 `omitted_resamples`。`method`、`direction`、`n`、`bw`、`kernel`、`adjust`、`cut` 和 `percent` 与 [`smooth_roc`](roc-smoothing.md) 一致。`direction="auto"` 只在原始样本上判定一次，之后每个重抽样固定使用该方向。密度带宽规则 `"nrd0"` 和 `"nrd"` 在每个重抽样重新估计。

```python
from biomed_parity.roc_smooth_bootstrap import bootstrap_smooth_auc_ci, bootstrap_smooth_sensitivity_ci

labels = [0, 0, 0, 0, 1, 1, 1, 1]
scores = [0.1, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9]
area = bootstrap_smooth_auc_ci(labels, scores, method="density", random_state=24)
pointwise = bootstrap_smooth_sensitivity_ci(
    labels, scores, specificity=[0.7, 0.8, 0.9], method="density", random_state=24,
)
```

## 重抽样与计算口径

- 默认 `stratified=True`，分别在病例和对照内有放回抽取原人数。`stratified=False` 时在全体中有放回抽取原样本总人数。默认 `n_resamples=2000`，可用 `random_state` 设定 NumPy 种子或生成器。
- `bootstrap_indices` 可提供至少两行、每行与原样本等长的零基整数索引矩阵。行数决定总抽样次数，分层矩阵的每行须保持原有病例和对照人数。通过同一索引矩阵可在 R 与 Python 间对照。
- 使用 7 型分位数，默认 `confidence_level=0.95`。无法取得平滑估计的抽样（例如只有一个类别，或双正态拟合缺乏足够内部点）会被丢弃并产生一条 `RuntimeWarning`。全部抽样无可用估计时，区间和中位数为 `NaN`。
- `percent=True` 时，灵敏度与特异度用 0–100 尺度。AUC 重抽样先换算到 0–1，然后把区间转回百分制，沿用 pROC 的口径。部分 AUC 校正时，重抽样保留低于机会水平的校正值，即使原始样本在默认 `allow_below_chance=False` 下返回 `NaN`。[pROC AUC 区间实现](https://github.com/xrobin/pROC/blob/master/R/ci.auc.R)
- 固定特异度在相邻平滑网格点间线性插值，恰好命中竖直段时采用第一个点。区间逐点计算，各点共享同一批可用重抽样。[pROC 灵敏度区间实现](https://github.com/xrobin/pROC/blob/master/R/ci.se.R)

## 验证与范围

13 个场景包含 8 个 AUC 区间和 5 个灵敏度区间。场景覆盖双正态和核密度、分层和普通抽样、概率及百分制、完整与部分 AUC、机会水平以下的校正、特异度端点与重复输入，以及因单类和完全分离被丢弃的抽样。每个场景用同一 R 种子生成 43 次抽样，手动逐次调用 pROC 并与公开 `ci.auc()` 或 `ci.se()` 交叉核对。3 个场景还加入一条指定的失败抽样，共保存 562 个抽样索引行。[对照用例](../../validation/roc_smooth_bootstrap/cases.json) · [参考脚本](../../validation/roc_smooth_bootstrap/oracle.R) · [R 4.5.3 / pROC 1.19.1 快照](../../validation/roc_smooth_bootstrap/expected.json)

原样本点估计及分位数在 0–1 尺度上的最大绝对偏差为 `7.8e-16`，丢弃抽样数逐场景一致。数值容差为相对 `2e-14`、绝对 `3e-15`，百分制绝对容差乘以 100。

```sh
python -m pytest -q tests/test_roc_smooth_bootstrap.py
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 python -m pytest -q -W error tests/test_roc_smooth_bootstrap.py
R_LIBS_USER="$PWD/.r-lib" Rscript --vanilla validation/roc_smooth_bootstrap/oracle.R validation/roc_smooth_bootstrap/cases.json
```

小样本重抽样可能产生水平的双正态拟合。Python 沿用原平滑模块的确定性端点；pROC 的曲线取点依赖理论零斜率处的浮点舍入，有时会报错或得到不同的点坐标。此时两端的有效抽样计数或区间可能不同。仅有一笔有效抽样时 Python 仍保留结果维度。两端随机数流各自独立。此接口覆盖单条曲线与逐点区间，同时置信带、两条平滑曲线的推断、多分类与其他平滑模型属于独立范围。
