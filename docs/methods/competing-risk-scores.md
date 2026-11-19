# 竞争风险预测评分

Python 入口为 `biomed_parity.risk_scores.score_competing_risks`。输入每人一条随访记录、递增的评估时点，以及来自外部模型的预测累积发生概率。每个模型的预测矩阵形状为样本数 × 时点数。多个模型使用模型数 × 样本数 × 时点数，患者顺序须一致。状态 0 表示右删失，正整数标记事件类型。 [riskRegression::Score 手册](https://search.r-project.org/CRAN/refmans/riskRegression/html/Score.html)

病例为时点前或恰在时点发生目标事件者。对照包含届时仍无事件、以及已观察到其他事件者。评分按逆删失概率加权。AUC 对同分比较计半，Brier 分数按全样本规模平均。已观察事件在其时间左极限取权重，时点后仍随访者在评估时点取权重。同一时点发生事件与删失时，点估计按先处理事件的逆生存约定。AUC 在缺少病例或对照时返回 NaN。 [AUC 和 Brier 方法背景](https://www.bmj.com/content/377/bmj-2021-069249)

删失权重默认用边际逆 Kaplan–Meier。传入 `censoring_covariates`（样本对齐的协变量列）或 `censoring_strata`（分层标签）时改用反向 Cox 模型：删失指示作结局，分层各自估计基线风险，并列删失按 Efron 处理，权重与影响函数同时计入 Cox 系数的估计方差。 `conservative=True` 可把删失权重视为固定。 [riskRegression::Score 手册](https://search.r-project.org/CRAN/refmans/riskRegression/html/Score.html)

结果包括评分、影响函数标准误、Wald 区间、逐行影响量和积分 Brier 分数，并提供同一数据上较后模型减较前模型的配对比较。`integrated_brier` 按 riskRegression 的左阶梯约定从 0 起在每个时点积分再除以时点，形状为模型数 × 时点数。常规评分区间截断到 [0, 1]，差值区间保留原尺度。

`auc_inference="riskregression"` 复现 R 2026.03.11 的旧版 AUC 影响函数：无论请求哪个目标原因，该版本都按第一类事件作病例、第二类事件作已观察对照。默认 `auc_inference="cause"` 始终按请求的目标原因组成病例。

## 参考对照

八组 R 4.5.3 / riskRegression 2026.03.11 用例覆盖两类删失模型。四组边际 KM 用例覆盖多个预测模型、事件和删失恰在评估时点、同分和并列随访，AUC 点估计、Brier、Brier 影响函数及配对 Brier 推断均通过，三组也通过 AUC 区间、影响函数及配对推断。四组反向 Cox 用例覆盖两个协变量斜率、并列删失、分层和分层无斜率，AUC 与 Brier 的点估计、标准误、区间、逐行影响函数、配对推断以及积分 Brier 均与 R 一致，最大绝对差约 `1e-14`。

目标原因编码为 2 且存在并列的 KM 用例中，R 与 Python 的 AUC 点估计一致，AUC 标准误最大绝对差约 0.255。Python 组成病例和对照时始终依据传入的目标事件编码，任意重新编号事件后结果保持一致。该差异有独立用例，测试不会把它算作数值一致；使用 `auc_inference="riskregression"` 可逐项复现 R 的旧版算法。 [上游 AUC 影响函数实现](https://github.com/tagteam/riskRegression/blob/master/src/getInfluenceFunctionAUC.cpp) · [上游评分推断说明](https://search.r-project.org/CRAN/refmans/riskRegression/html/Score.html)

## 兼容范围

本模块覆盖固定模型的外部验证、边际 KM 与反向 Cox 删失、积分 Brier 和模型配对比较。模型重拟合交叉验证、校准图及成组重抽样推断待扩展。

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest tests/test_risk_scores.py
```
