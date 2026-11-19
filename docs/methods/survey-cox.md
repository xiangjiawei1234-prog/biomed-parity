# survey 的加权 Cox 回归

Python 入口为 `biomed_parity.survey_cox.svycoxph`。支持右删失、数值设计矩阵、正抽样权重、一阶段分层聚类抽样、有限总体修正，以及独立设置的基线风险模型分层。Efron 和 Breslow 两种并列事件算法可选。 [survey 手册](https://r-survey.r-forge.r-project.org/pkgdown/docs/reference/svycoxph.html) · [带权 Cox 与抽样解释](https://www.mayo.edu/research/documents/biostat-58pdf/doc-10027288)

```python
from biomed_parity.survey_cox import svycoxph

fit = svycoxph(
    time=[1, 2, 3, 4, 5, 6, 7, 8],
    status=[1, 0, 1, 1, 0, 1, 1, 0],
    covariates=[0, 1, 1, 0, 0, 1, 0, 1],
    weights=[1.1, 1.1, 0.8, 0.8, 1.3, 1.3, 1.0, 1.0],
    psu=["a", "a", "b", "b", "c", "c", "d", "d"],
    strata=[1, 1, 1, 1, 2, 2, 2, 2],
)
print(fit.hazard_ratios, fit.covariance)
```

每行一个观测。分类变量应在调用前编码为数值列，设计矩阵无需截距。权重以样本均值归一化，对应 `rescale=TRUE`。估计的 Cox 得分按个体线性化，在每个抽样分层内按 PSU 加总、中心化并作小样本修正。可选 `fpc` 传各行所属抽样分层的第一阶段抽样比例，分层内保持恒定。默认等价于无有限总体修正。孤立 PSU 若无确定性抽样信息会报错。

返回回归系数、模型协方差、设计调整协方差、标准误、风险比、正态 Wald 风险比区间和 p 值、部分对数似然以及设计和残差自由度。R 回放使用 `method="breslow"` 或 `method="efron"` 指定并列事件算法。抽样分层 `strata` 与模型基线风险分层 `model_strata` 可分别传入。

多阶段和两阶段抽样、复制权重、抽样子域、含时变协变量的起止时间记录、惩罚模型及分层预测曲线仍待验证。

本地使用 R 4.5.3、survey 4.5 和 survival 3.8-6，验证五个 72 行人工方案，核对设计和朴素协方差在内的 12 个字段。另验证重排行及统一缩放权重后的结果不变。

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest tests/test_survey_cox.py
```
