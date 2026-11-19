# 二分类 ROC AUC 的 DeLong 推断

用于诊断研究中估计完整经验 ROC AUC、比较同一组受试者的两项检测，以及比较独立队列的两项检测。实现按 R pROC 1.19.1 的数值行为验证，Python 运行时依赖 NumPy 和 SciPy。[pROC 参考手册](https://cran.r-project.org/web/packages/pROC/refman/pROC.html) [pROC 1.19.1](https://cran.r-project.org/web/packages/pROC/index.html)

## Python 接口

| Python 函数 | 对应 R 调用 | 返回字段 |
| --- | --- | --- |
| `delong_auc(labels, scores)` | `auc`、`var(method="delong")`、`ci.auc(method="delong")` | `auc`、`variance`、`confidence_interval` |
| `delong_paired_test(labels, scores_1, scores_2)` | `roc.test(method="delong", paired=TRUE)`、`var`、`cov` | 两个 AUC、两个方差、协方差、差值、Z 统计量、p 值、差值区间 |
| `delong_unpaired_test(labels_1, scores_1, labels_2, scores_2)` | `roc.test(method="delong", paired=FALSE)`、`var` | 两个 AUC、两个方差、差值、D 统计量、自由度、p 值 |

输入为等长的一维标签和有限实数分数，`0` 代表对照，`1` 代表病例。每个队列至少有 2 个病例和 2 个对照。配对模式要求两个分数向量与标签按受试者逐行对齐。缺失输入须先完成分析所需的数据处理。

三个接口都使用相同的 `direction` 约定。`"<"` 表示分数越高越偏向病例，为默认值。`">"` 表示分数越低越偏向病例。两个比较接口的替代假设为 `"two.sided"`、`"less"`、`"greater"`，并以第一个 AUC 减去第二个 AUC。单 AUC 与配对比较可指定 `confidence_level`，默认 0.95，取值范围为 [0, 1]。配对差值区间始终为双侧。

```python
from biomed_parity.roc import delong_auc, delong_paired_test

labels = [0, 1, 0, 1, 0, 1, 1]
first = [1, 2, 2, 3, 3, 4, 5]
second = [1, 0, 1, 3, 3, 4, 5]
single = delong_auc(labels, first, direction="<")
comparison = delong_paired_test(labels, first, second, direction="<")
print(single.auc, single.confidence_interval)
print(comparison.difference, comparison.p_value, comparison.confidence_interval)
```

结果的 AUC 和差值分别约为 0.833333 和 0.125，配对双侧 p 值约为 0.386476。

## 计算和边界

每对病例和对照的分数比较计为胜 1、平 0.5、负 0。对每个病例和每个对照分别计算跨类别平均，再以两个类别的样本方差和类别人数估计 AUC 方差。排序后查找同分区间，用线性内存和 `O(n log n)` 时间求得计数。配对比较按受试者计算协方差，并直接用两个模型的受试者贡献差估计差值方差。[DeLong 等，1988](https://pubmed.ncbi.nlm.nih.gov/3203132/) [Sun 和 Xu，2014](https://pamixsun.github.io/papers/sun2014fast.pdf)

单 AUC 区间使用正态近似，截断至 [0, 1]。配对差值使用正态参考分布，区间保留原始边界。独立样本的标准误为两个 AUC 方差之和的平方根，以 Satterthwaite 自由度计算 t 分布尾概率。配对且差值方差为零时，统计量根据差值取 0 或带符号的无穷大。独立检验的方差和为零时，自由度及 p 值按 pROC 实测返回 `NaN`。配对差值方差为零且区间水平为 1 时，两侧端点均为 `NaN`。[pROC 参考手册](https://cran.r-project.org/web/packages/pROC/refman/pROC.html)

本页接口覆盖完整、未平滑二分类 AUC 的 DeLong 推断。自动方向、百分比、部分 AUC 及 Bootstrap 对照见另一个方法页。因子等级推断、自动配对和缺失行删除、公式接口、诊断阈值、Venkatraman 检验、多分类和绘图属于后续范围。[部分 AUC 与 Bootstrap](roc-bootstrap.md)

## Python 候选的差异

2026-11-19 核查时，scikit-learn 可以计算 AUC 点估计。MLstatkit 提供配对检验与每个模型自身的 AUC 区间。pAUC 公布了配对与非配对 DeLong 检验，覆盖面最接近本页的完整工作流。[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) [MLstatkit](https://pypi.org/project/MLstatkit/) [pAUC](https://pypi.org/project/pauc/0.2.1/)

本机对 pAUC 0.2.1 的实测包含相同合成分数上的 24 项输出。无并列的配对 p 值一致到浮点精度。以下差异会影响已有 R 工作流的复现。[候选包对照数据](../../validation/roc/pauc-0.2.1.tsv)

| 场景 | 指标 | pROC 1.19.1 | pAUC 0.2.1 |
| --- | --- | ---: | ---: |
| 单曲线，分数并列 | AUC 方差 | 0.026042 | 0.046296 |
| 单曲线，分数并列 | 区间上界 | 1.000000 | 1.255050 |
| 配对，分数并列 | 双侧 p 值 | 0.386476 | 0.288844 |
| 独立队列，样本量不同 | 双侧 p 值 | 0.881320 | 0.902143 |
| 独立队列，第一条曲线完全分离 | 双侧 p 值 | 0.087359 | 0.044171 |

## 本地复核

[场景定义](../../validation/roc/scenarios.tsv) 包含 28 个场景，覆盖并列、方向、类别不平衡、负协方差、同 AUC、极端方差、不同区间水平和单尾概率。Python 数值断言相对容差为 `2e-12`，绝对容差为 `2e-14`。输入均为合成分数。

从仓库根目录运行以下命令。实时 R 对照使用已安装的 pROC，或通过 `R_LIBS_USER` 指定仓库中的 R 库。

```sh
python -m pytest -q tests/test_roc.py
BIOMED_R_PARITY=1 python -m pytest -q tests/test_roc.py
R_LIBS_USER=.r-lib Rscript --vanilla validation/roc/generate.R
```

在独立的 Python 环境安装 `pauc==0.2.1` 后，可运行 `python validation/roc/audit_pauc.py` 重现候选包对照。该环境可以独立于仓库的常规测试和运行环境。
