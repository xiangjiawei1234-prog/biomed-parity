# 竞争风险的受试者与受试者对平均评分

`biomed_parity.risk_resampled.score_resampled_competing_risks` 接收同一组验证重抽样上的预测矩阵，按受试者平均 Brier 分数、按受试者对平均时间依赖 AUC。预测通常来自 `bootstrap_validate_competing_risks(..., keep_predictions=True)` 的袋外矩阵，或 `cross_validate_competing_risks` 的重复折外矩阵。对应 `riskRegression::Score(split.method="loob"` 或重复 `"cvK"`，`keep="splitindex"`）的汇总口径。删失权重支持边际 Kaplan–Meier 与反向 Cox。[R Score 手册](https://search.r-project.org/CRAN/refmans/riskRegression/html/Score.html) · [竞争风险评分口径](competing-risk-scores.md)

```python
from biomed_parity.risk_resampled import score_resampled_competing_risks
from biomed_parity.risk_bootstrap import bootstrap_validate_competing_risks

bag = bootstrap_validate_competing_risks(
    time, status, [30, 90, 180], models,
    n_resamples=50, seed=109, keep_predictions=True,
)
scores = score_resampled_competing_risks(
    time, status, [30, 90, 180], bag.predictions, cause=1,
)
print(scores.auc, scores.brier, scores.integrated_brier)
```

## 输入

`risks` 的形状为「重抽样 × 模型 × 受试者 × 时点」；单个模型可省略模型轴，传「重抽样 × 受试者 × 时点」。同一受试者的一次重抽样必须在所有模型和时点上同时为有效预测或同时缺失：有效值写成有限概率，离开验证集的行写 `numpy.nan`。全部缺失的重抽样行不参与该受试者的平均，`subject_validation_counts` 记录每位受试者的有效重抽样次数。`horizons` 须递增且落在随访范围内；状态 0 表示删失，正整数标记事件类型。

`censoring_covariates` 与 `censoring_strata` 的语义与 `score_competing_risks` 相同，对应反向 Cox 删失模型；缺省使用边际反向 Kaplan–Meier。预测概率须位于 [0, 1]。

## 估计量

- **Brier 分数**：每位受试者在其被留出的重抽样上取逆删失概率加权平方误差的平均，再按受试者等权平均。时点前或恰在时点发生目标事件者记为 1，其余（含竞争事件和时点后仍随访者）记为 0。
- **时间依赖 AUC**：对每个目标事件病例与对照对，按预测风险比较大小，同分计半；每次重抽样只使用两个受试者同时被留出的配对，并取逆删失权重加权平均。从未被同时留出的病例-对照对保留在权重分母中，同时计入 `unobserved_pairs`。缺少病例或对照时该时点返回 `NaN`。
- **积分 Brier**：按 riskRegression 左阶梯约定，从 0 开始在每个时点积分再除以时点。
- `auc_method="riskregression"` 复现 riskRegression 2026.03.11 的旧版汇总行为：竞争对照按事件编码顺序取第一/第二类，时点后受试者的权重取首个评估时点的参考权重。默认 `"cause"` 按请求的目标原因组成病例与对照。

结果 `auc`、`brier`、`integrated_brier` 形状为「模型 × 时点」，`subject_validation_counts` 长度为受试者数，`unobserved_pairs` 长度为时点数。

## 本地验证

R 4.5.3、riskRegression 2026.03.11 和 prodlim 2026.03.11 的两个场景使用 42 名受试者与两个分层 Aalen–Johansen 预测模型：

- `loob`：15 次留一重抽样（种子 472，目标原因 1，边际删失），核对留一 Brier 与留对 AUC 的平均，并确认存在从未共同留出的配对；
- 重复四折：3 组固定折编号（种子 681，目标原因 2，反向 Cox 删失），核对重复折外汇总，并确认所有病例-对照对均被共同留出。

R 脚本先由 `Score` 导出实际重抽样/折编号，再逐次重拟合 `prodlim` 并按受试者或受试者对独立计算估计量。Python 重拟合的折外预测与 R 的最大绝对差为 `2e-15`；AUC、Brier 和积分 Brier 采用相对容差 `2e-7`、绝对容差 `2e-8`。另有用例检查缺失配对的计数、AUC 分母与部分验证行的拒绝。[R 脚本](../../validation/risk_resampled/oracle.R) · [参考值](../../validation/risk_resampled/expected.json)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_risk_resampled.py
```

## 已知差异与范围

- riskRegression 的 `Score` 以单张响应表汇总重抽样结果；本入口接收显式的重抽样预测张量，便于核对每次留出。旧版兼容路径通过 `auc_method="riskregression"` 显式选择。
- 缺失配对只计入 `unobserved_pairs`，不提高 AUC 的权重；这与从权重分母中删除缺失配对的做法不同。需要该口径时应检查该计数是否为零。
- 本入口只负责重抽样后的汇总，不生成折分、不重拟合模型，也不计算区间、校准或 ROC 坐标；这些分别由交叉验证、袋外 Bootstrap、竞争风险校准与评分入口提供。
