# 竞争风险的 K 折交叉验证

`biomed_parity.risk_cross_validation.cross_validate_competing_risks` 在每个训练折重新拟合风险模型，汇总每位受试者的折外预测，再计算随时间变化的 AUC、Brier、积分 Brier 和模型之间的成对差值。对应 `riskRegression::Score(split.method="cvK", B=1)` 的折外汇总路径。删失权重使用全样本估计，支持边际 Kaplan–Meier 和反向 Cox。 [R Score 手册](https://search.r-project.org/CRAN/refmans/riskRegression/html/Score.html) · [竞争风险评分口径](competing-risk-scores.md)

```python
from biomed_parity.risk_cross_validation import cross_validate_competing_risks

def refit(train_rows, validation_rows, horizons):
    model = fit_model(features[train_rows], time[train_rows], status[train_rows])
    return model.predict_risk(features[validation_rows], horizons)

result = cross_validate_competing_risks(
    time, status, [30, 90, 180], {"candidate": refit}, n_folds=5, seed=109,
)
print(result.scores.brier.estimate)
```

回调的两个行索引与输入顺序一致，且训练和验证行互斥。回调返回「验证行 × 预测时点」的累计发生概率矩阵。所有模型共享折分。可明确提供每行的整数折编号；省略时，由 NumPy 种子随机生成规模最多相差一人的折。R 的 `Score` 先按随访时间和结局排序，再分配折编号。迁入 R 的分折时，应将这些编号贴回原始受试者行。

输出的 `predictions` 形状为「模型 × 受试者 × 预测时点」，可核查每位受试者的折外估计。推断沿用已实现的评分影响函数和删失权重规则。其区间基于给定的折外预测，没有把模型重拟合的不确定性另行纳入。目前支持一次 K 折，重复分折与 Bootstrap 验证仍待迁移。 [评分区间与历史 AUC 兼容](competing-risk-scores.md)

本地使用 R 4.5.3、riskRegression 2026.03.11 和 prodlim 核对两组折外评分。各组用两种分层 Aalen–Johansen 模型，覆盖三折边际删失、四折 Cox 删失、两种目标原因和三个预测时点。折外概率最大容差为 `2e-15`，评分、区间和积分 Brier 采用既有 IPCW 容差。选择 `auc_inference="riskregression"` 对照该 R 版本的影响函数。

对不含协变量的 prodlim 预测器，该 R 版本的 `Score` 会把一条按时点排列的预测向量错误铺到多个验证行，两组分别有 84 和 72 个折外概率发生错位。Python 要求回调交付完整矩阵，该路径以逐折直接调用 `predictRisk` 后逐行复用整条预测时间曲线核对。 [R 脚本](../../validation/risk_cross_validation/oracle.R) · [参考值](../../validation/risk_cross_validation/expected.json)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_risk_cross_validation.py
```
