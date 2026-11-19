# 竞争风险预测评分

Python 入口为 `biomed_parity.risk_scores.score_competing_risks`。输入每人一条随访记录、递增的评估时点，以及来自外部模型的预测累积发生概率。每个模型的预测矩阵形状为样本数 × 时点数。多个模型使用模型数 × 样本数 × 时点数，患者顺序须一致。状态 0 表示右删失，正整数标记事件类型。 [riskRegression::Score 手册](https://search.r-project.org/CRAN/refmans/riskRegression/html/Score.html)

病例为时点前或恰在时点发生目标事件者。对照包含届时仍无事件、以及已观察到其他事件者。评分按逆删失概率加权。AUC 对同分比较计半，Brier 分数按全样本规模平均。删失分布采用边际逆 Kaplan–Meier。已观察事件在其时间左极限取权重，时点后仍随访者在评估时点取权重。同一时点发生事件与删失时，点估计按先处理事件的逆生存约定。AUC 在缺少病例或对照时返回 NaN。 [AUC 和 Brier 方法背景](https://www.bmj.com/content/377/bmj-2021-069249)

结果包括评分、影响函数标准误、Wald 区间和逐行影响量，并提供同一数据上较后模型减较前模型的配对比较。设 `conservative=True` 可将删失权重视为固定。默认计入边际删失估计的方差。常规评分区间截断到 [0, 1]，差值区间保留原尺度。

## 参考对照

四组 R 4.5.3 / riskRegression 2026.03.11 用例覆盖多个预测模型、事件和删失恰在评估时点、同分和并列随访。四组的 AUC 点估计、Brier、Brier 影响函数及配对 Brier 推断均通过，三组也通过 AUC 区间、影响函数及配对推断。

目标原因编码为 2 且存在并列的用例中，R 与 Python 的 AUC 点估计一致，AUC 标准误最大绝对差约 0.255。Python 组成病例和对照时始终依据传入的目标事件编码，任意重新编号事件后结果保持一致。R 版 AUC 推断对不同数据配置仍需逐项核对。该差异有独立用例，测试不会把它算作数值一致。 [上游 AUC 影响函数实现](https://github.com/tagteam/riskRegression/blob/master/src/getInfluenceFunctionAUC.cpp) · [上游评分推断说明](https://search.r-project.org/CRAN/refmans/riskRegression/html/Score.html)

当前实现用于固定模型的外部验证和边际删失。Cox 删失模型、模型重拟合交叉验证、积分 Brier 和校准图待扩展。

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest tests/test_risk_scores.py
```
