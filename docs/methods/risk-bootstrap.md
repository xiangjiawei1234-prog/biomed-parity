# 竞争风险的袋外 Bootstrap 验证

`bootstrap_validate_competing_risks` 对应 `riskRegression::Score(split.method="bootcv")`。每次用抽中的受试者重新拟合风险模型，在未抽中的受试者上比较按时点评估的 AUC 和 Brier。全样本估计删失权重，每次将这些权重限制到对应袋外验证集。可使用边际 Kaplan–Meier 或反向 Cox。 [R Score 手册](https://search.r-project.org/CRAN/refmans/riskRegression/html/Score.html) · [竞争风险评分口径](competing-risk-scores.md)

```python
from biomed_parity.risk_bootstrap import bootstrap_validate_competing_risks

result = bootstrap_validate_competing_risks(
    time, status, [30, 90, 180], {"candidate": refit},
    n_resamples=200, seed=10,
)
print(result.auc.estimate, result.auc.lower, result.auc.upper)
```

模型回调使用和 K 折交叉验证相同的接口。训练索引保留重复受试者，袋外索引按原始观测顺序排列，回调返回「袋外受试者 × 时点」预测矩阵。所有模型共享训练和验证分割。`train_indices` 可直接提供固定训练索引矩阵，每行一次重抽样。使用生成索引时，`training_size` 可指定行数。`replace=False` 对应无放回子抽样。迁入 R 的索引时，将其从 R 按时间和结局排序后的行号映射到原输入行。 [回调与行号约定](risk-cross-validation.md)

`auc`、`brier` 及 `integrated_brier` 提供重抽样平均值、样本标准差、线性分位数区间以及「抽样 × 模型 × 时点」逐次结果。积分 Brier 按 R 左阶梯规则计算。两模型差值先在同一次验证集上比较，再汇总。袋外集为空时保留整次缺失记录，某时点缺少有效病例或对照时保留该次 AUC 缺失。区间基于有效抽样。启用 `keep_predictions` 可保留「抽样 × 模型 × 原始受试者 × 时点」矩阵，其中袋内预测记为缺失。

## 本地验证

R 4.5.3、riskRegression 2026.03.11 和 prodlim 2026.03.11 的三个场景包括有放回抽样、无放回子抽样、边际和 Cox 删失模型及两个目标原因。共对照 21 次训练样本，两种分层 Aalen–Johansen 模型，以及每次袋外预测、动态 AUC、Brier、积分 Brier、模型差及重抽样区间。此入口覆盖袋外评分的逐次平均路径。重复 K 折、按受试者或受试者对归并的 `loob`、ROC 坐标及校准图仍列为后续工作。 [R 脚本](../../validation/risk_bootstrap/oracle.R) · [对照数据](../../validation/risk_bootstrap/expected.json)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 python -m pytest -q -W error tests/test_risk_bootstrap.py
```
