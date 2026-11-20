# Hand–Till 多分类 ROC AUC

`biomed_parity.roc_multiclass.multiclass_auc` 计算多分类 AUC，支持每位受试者一个分数，或每位受试者有一个按类别命名的分数向量。结果保留每个类别对的 AUC 及其二分类分量，用 NumPy 和 SciPy 原生运行。[pROC 多分类手册](https://cran.r-project.org/web/packages/pROC/refman/pROC.html#multiclass.roc) · [pROC 多分类实现](https://github.com/xrobin/pROC/blob/master/R/multiclass.R)

## 计算口径

- **单列分数。** 每个无序类别对只计算一个二分类 AUC。按 `levels` 顺序将该对第一个类别视为对照，第二个视为病例。默认方向为 `"auto"`，对照组中位数不高于病例组时取 `"<"`，否则取 `">"`。也可把方向固定为 `"<"` 或 `">"`。
- **多列分数。** 每个类别对计算两个 AUC。对类别 `i` 的分数，只比较类别 `i` 与类别 `j` 的受试者，并将类别 `i` 视为对照。对类别 `j` 的分数交换两类的角色。默认 `direction=">"` 时，分数越高越符合该列所代表的类别。可用 `"<"` 表示分数越低越符合该类别。两项等权平均得到类别对 AUC。
- **总体。** 对所有类别对 AUC 做等权平均，避免按各类样本数加权。完整 AUC 中两组分数并列时计半个胜对。指定 `partial_auc=(a, b)` 时先计算各二分类部分 AUC，再用相同权重汇总。`focus` 可为 `"specificity"` 或 `"sensitivity"`，`standardize=True` 对每个部分 AUC 分量做 McClish 校正。某分量低于机会水平时，缺省返回 `NaN`，并使总体 AUC 为 `NaN`。`allow_below_chance=True` 可保留下方的校正值。`percent=True` 将 AUC 和部分面积端点按 0–100 尺度解释。[pROC 面积手册](https://cran.r-project.org/web/packages/pROC/refman/pROC.html#auc) · [pROC 面积实现](https://github.com/xrobin/pROC/blob/master/R/auc.R)

## 输入与返回值

```python
from biomed_parity.roc_multiclass import multiclass_auc

labels = ["a", "b", "c", "a", "b", "c"]
single = multiclass_auc(labels, [0.1, 0.3, 0.5, 0.4, 0.6, 0.8])
matrix = multiclass_auc(
    labels,
    [[.80, .15, .05], [.30, .55, .15], [.10, .20, .70],
     [.60, .30, .10], [.35, .45, .20], [.25, .20, .55]],
    score_labels=["a", "b", "c"],
)
print(matrix.auc, matrix.pairs)
```

类别标签可为字符串或有限实数。`levels` 可选择至少两个已观测类别。单列分数默认按类别值排序，给定 `levels` 时保留其顺序。多列分数要求 `score_labels` 为每列提供唯一标签，按列顺序遍历匹配类别。缺少选中类别的评分列或存在未选中的评分列时发出警告，匹配的列至少须有两列。多列分数支持任意实数决策值，概率行合计没有约束。

默认 `na_rm=True` 时移除缺失类别的记录，并在每次二分类比较中独立移除该分数列的 `NaN`。每个参与比较的类别仍须有至少一个有效分数。`na_rm=False` 时缺失标签或比较中出现的缺失分数会报错。评分中的无穷值会报错。

`MulticlassAUCResult` 返回总体 `auc`、`levels` 和 `pairs`。每个 `ClassPairAUC` 包含 `classes`、该对的 `auc`、二分类 `component_aucs` 及 `directions`。多列输入的两个分量依次对应类别对中的两个评分列。单列输入只有一个分量。

## 验证与已知差异

[对照用例](../../validation/roc_multiclass/cases.json)使用 7 组人工数据，共 23 个场景。21 个数值场景比较总体、类别对与二分类分量 AUC，并核验参与类别、类别对顺序与方向。另核验 1 个两侧都拒绝矩阵自动方向的场景和 1 个 pROC 1.19.1 评分列误判场景。覆盖不均衡类别、数值与字符串标签、并列分数、相同中位数、类别子集、缺失分数、缺列与多余列、部分 AUC、McClish 校正和百分制。[R 4.5.3 / pROC 1.19.1 快照](../../validation/roc_multiclass/expected.json) · [参考脚本](../../validation/roc_multiclass/oracle.R)

相对容差为 `2e-14`，绝对容差为 `2e-15`，百分制绝对容差为 `2e-13`。总体与各二分类分量 AUC 在 0–1 尺度上的最大绝对差为 `5.7e-16`。

- pROC 1.19.1 在缺失评分列与匹配评分列数量相同且至少匹配两列时，可能误报完全没有匹配列。本模块发出缺列警告，并对匹配的类别计算 AUC。[pROC 列筛选实现](https://github.com/xrobin/pROC/blob/master/R/multiclass.R#L106-L132)
- 部分 AUC 校正失败时，pROC 可能为同一二分类分量重复警告。本模块每个低于机会水平的分量最多警告一次。
- Python 字符类别按 Unicode 顺序排列。R 默认因子顺序受环境排序规则影响。需要相同的单列类别顺序时可显式传入 `levels`。R 公式、带未观察类别的因子对象、多分类置信区间、曲线平滑和绘图超出此接口范围。

```sh
python -m pytest -q tests/test_roc_multiclass.py
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 python -m pytest -q -W error tests/test_roc_multiclass.py
R_LIBS_USER="$PWD/.r-lib" Rscript --vanilla validation/roc_multiclass/oracle.R validation/roc_multiclass/cases.json
```
