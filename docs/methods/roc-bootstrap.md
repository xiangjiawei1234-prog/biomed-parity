# 完整与部分 ROC AUC 及分层 Bootstrap

用于二分类诊断分数的完整或部分经验 ROC 面积估计、单模型置信区间以及配对比较。Python 运行时依赖 NumPy 和 SciPy。数值已与 pROC 1.19.1 的 `auc`、`ci.auc(method="bootstrap")`、`roc.test(method="bootstrap", paired=TRUE)` 对照。[pROC 参考手册](https://cran.r-project.org/web/packages/pROC/refman/pROC.html)

## 接口

| 函数 | 结果 |
| --- | --- |
| `roc_auc(labels, scores, ...)` | 完整或部分 AUC |
| `bootstrap_auc_ci(labels, scores, ...)` | 原样本 AUC、Bootstrap 中位数、置信区间及有效抽样数 |
| `bootstrap_paired_test(labels, scores_1, scores_2, ...)` | 两个 AUC、差值、Bootstrap 标准误和 D 统计量、p 值及有效抽样数 |

输入是等长一维向量，`0` 代表对照、`1` 代表病例。分数须是有限实数。默认 `direction="<"` 表示高分偏向病例，`">"` 表示低分偏向病例。`"auto"` 按病例和对照的分数中位数判断方向，中位数相等时使用 `"<"`。每条曲线在原样本上确定方向，随后固定于全部重抽样。配对分数须按同一受试者顺序对齐。

```python
from biomed_parity.roc_bootstrap import bootstrap_auc_ci, bootstrap_paired_test

labels = [0, 1, 0, 1, 0, 1, 1]
a = [1, 2, 2, 3, 3, 4, 5]
b = [1, 0, 1, 3, 3, 4, 5]
settings = {
    "partial_auc": (0.35, 0.82),
    "focus": "sensitivity",
    "standardize": True,
    "random_state": 2026,
}
print(bootstrap_auc_ci(labels, a, **settings))
print(bootstrap_paired_test(labels, a, b, **settings))
```

## 面积定义

`partial_auc=None` 计算完整 AUC。部分面积可指定两个端点，端点顺序可以互换。在默认的 `focus="specificity"` 下，以特异度为积分轴并累积灵敏度。用 `focus="sensitivity"` 时，以灵敏度为积分轴并累积特异度。先按分数合并并列的阈值，再在端点处线性插值，用裁切后的梯形求和。默认端点在 [0, 1]。当 `percent=True` 时，端点单位改为 [0, 100]，点估计、标准误及区间均按百分比输出。

`standardize=True` 使用 McClish 校正，将随机对角线与理想曲线的局部面积分别映射到 0.5 和 1。低于随机对角线时，原样本点估计默认为 `NaN` 并发出提示。显式设置 `allow_below_chance=True` 可以返回低于 0.5 的校正值。重抽样曲线保留低于对角线的校正值，与 pROC 一致。宽度为零的原始部分面积为 0，校正面积为 `NaN`。[McClish，1989](https://pubmed.ncbi.nlm.nih.gov/2668680/) [pROC AUC 说明](https://cran.r-project.org/web/packages/pROC/refman/pROC.html#auc)

## Bootstrap 约定

默认在病例和对照内分别有放回抽样 2000 次，并保持两类样本数。可用 `n_resamples` 设置抽样次数，用 `random_state` 指定 NumPy 种子或随机数生成器。若提供 `bootstrap_indices`，每一行须含所有抽取的零基行号，并维持原始病例和对照数。该矩阵的行数决定实际抽样次数，两个模型在每次配对抽样使用同一行。

单模型置信区间由 Bootstrap 分位数给出，默认置信水平 0.95。配对检验用原样本 AUC 差值除以 Bootstrap 差值的样本标准差，再以标准正态分布计算尾概率。支持 `"two.sided"`、`"less"` 和 `"greater"`。两个曲线在原样本与全部抽样的差值均为 0 时，D 统计量设为 0。[pROC Bootstrap 说明](https://cran.r-project.org/web/packages/pROC/refman/pROC.html)

scikit-learn 的 `roc_auc_score(max_fpr=...)` 提供从零假阳性率开始的标准化局部 AUC，pAUC 项目也声明提供两种轴的局部 AUC 与 Bootstrap 比较。本模块的验收范围包含任意内部区间、两种积分轴、给定抽样矩阵和 pROC 配对差值的数值行为。[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) [pAUC](https://pypi.org/project/pauc/0.2.1/)

## 验证范围

41 个场景覆盖两个积分轴、并列分数、正反分数方向、区间内插值、McClish 标准化、百分比单位、原样本低于随机线、空区间和零方差。R 脚本以 pROC 计算 61 次指定分层抽样下的逐次面积，再用相同 R 随机种子核对公开 Bootstrap 接口。Python 读取相同的抽样矩阵。比较容差为相对 `2e-12`、绝对 `2e-14`。[场景](../../validation/roc_bootstrap/scenarios.tsv) [指定抽样](../../validation/roc_bootstrap/draws.tsv) [R 结果](../../validation/roc_bootstrap/expected.tsv)

```sh
python -m pytest -q tests/test_roc_bootstrap.py
BIOMED_R_PARITY=1 python -m pytest -q tests/test_roc_bootstrap.py
R_LIBS_USER=.r-lib Rscript --vanilla validation/roc_bootstrap/generate.R
```

后续范围包括非分层 Bootstrap、独立队列 Bootstrap 比较、阈值及坐标区间、平滑曲线和多分类。完整 AUC 的 DeLong 推断参见独立方法页。[DeLong](roc-delong.md)
