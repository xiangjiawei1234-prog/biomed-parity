# REML 随机效应 Meta 分析与异质性区间

将至少两项独立研究的效应量和抽样方差汇总为截距项随机效应模型。Python 实现覆盖 R metafor 的 `rma.uni(method="REML")`、`predict` 与 `confint(type="PL")` 对应的数值输出。合并均值的置信区间、新研究真实效应的预测区间、研究间方差的轮廓似然区间分别返回。[metafor 模型](https://wviechtb.github.io/metafor/reference/rma.uni.html) [metafor 预测](https://wviechtb.github.io/metafor/reference/predict.rma.html) [metafor 置信区间](https://wviechtb.github.io/metafor/reference/confint.rma.html)

## Python 接口

```python
from biomed_parity.meta_analysis import meta_reml

result = meta_reml(
    effects=[0.10, 0.30, 0.45, 0.30, 1.60],
    variances=[0.010, 0.020, 0.031, 0.049, 0.060],
    test="knha",
    prediction="default",
)

print(result.mean_effect, result.confidence_interval)
print(result.prediction_interval)
print(result.tau2, result.tau2_confidence_interval)
```

该示例的合并效应约为 0.527222，95% 置信区间为 [−0.198917, 1.253361]。新研究真实效应的预测区间为 [−1.141053, 2.195497]。研究间方差约为 0.292641，轮廓似然区间为 [0.061893, 2.012526]。

输入是等长的一维有限实数向量。每项抽样方差须严格大于零，参数传方差值。效应估计采用同一分析尺度，例如已经计算好的对数风险比。结果对象还包含标准误、检验统计量、双侧 p 值、I² 和 H² 点估计及其区间。

| 参数 | 行为 |
| --- | --- |
| `test="z"` | 默认值。逆方差标准误和正态参考分布 |
| `test="t"` | 同一标准误，使用自由度为研究数减一的 t 分布 |
| `test="knha"`、`"hksj"` | Knapp–Hartung 调整标准误，使用自由度为研究数减一的 t 分布 |
| `test="adhoc"` | 同一 t 分布，标准误至少为经典逆方差标准误 |
| `prediction="default"` | 预测区间与模型均值推断使用同一临界值 |
| `prediction="riley"` | 预测区间以自由度为研究数减二且至少为 1 的 t 分布计算 |
| `confidence_level=0.95` | 区间水平，以 0 到 1 之间的比例输入 |
| `tau2_upper_limit=None` | 轮廓似然上端搜索界限默认为 100 或估计 τ² 的 10 倍中的较大值。可以显式指定有限界限 |

如果轮廓似然上端超过搜索界限，`tau2_upper_limit_reached` 为真，返回的上界是搜索界限。I² 与 H² 上界由同一 τ² 界限换算。该行为对应 metafor 输出的 `>` 标记。

## 实现

对给定 τ²，令权重为 `1 / (vᵢ + τ²)`，以加权平均计算合并效应。限制似然由各研究总方差的对数和、权重和的对数及加权残差平方和构成。计算中先选定统一数值尺度，在几何网格上定位限制似然得分的变号区间，通过求根精化候选点，再与 τ² 等于零的边界比较。区间通过反演一个自由度的卡方似然比统计量得到。I² 和 H² 使用同一分析的典型抽样方差进行单调变换。轮廓似然区间的覆盖率依赖大样本近似。[metafor 置信区间](https://wviechtb.github.io/metafor/reference/confint.rma.html) [Hardy 和 Thompson，1996](https://doi.org/10.1002/(sici)1097-0258(19960330)15:6%3C619::aid-sim188%3E3.0.co;2-a)

Knapp–Hartung 标准误使用加权残差均方校正。预测方差为该设置下模型均值方差加 τ²。预测目标为独立新研究的真实效应。[metafor 推断约定](https://wviechtb.github.io/metafor/reference/misc-recs.html) [metafor 预测](https://wviechtb.github.io/metafor/reference/predict.rma.html)

首版覆盖正抽样方差的单截距模型。后续范围包括协变量、依赖效应量、多层模型、Q-profile 区间、其他异质性估计方法和森林图。复杂多峰似然的区间覆盖也待单独验证。

## Python 候选

截至 2026-11-19，statsmodels `combine_effects` 提供 Paule–Mandel、DerSimonian–Laird 与 HKSJ 方差选项。PyMARE 当前文档覆盖 REML 与 Knapp–Hartung。PyMetaAnalysis 0.9.0 覆盖 REML、Hartung–Knapp、若干真效应预测规则和 Q-profile 区间，其公开异质性区间接口描述的是 Q-profile。本模块迁移的是 metafor 的轮廓似然区间及指定预测规则。[statsmodels](https://www.statsmodels.org/stable/_modules/statsmodels/stats/meta_analysis.html) [PyMARE](https://pymare.readthedocs.io/en/stable/generated/pymare.estimators.VarianceBasedLikelihoodEstimator.html) [PyMetaAnalysis](https://zhaoboding.github.io/PyMetaAnalysis/reference/api/)

在本机三个对照样例中，PyMetaAnalysis 0.9.0 的合并效应、τ² 点估计和 Knapp–Hartung 均值区间与对应 R 输出接近。文首示例的 Q-profile 区间为 [0.080026, 2.922130]，而 R 轮廓似然区间为 [0.061893, 2.012526]。五项和四项研究的 Riley 预测区间与 R 一致到浮点精度。两项研究时，PyMetaAnalysis 的汇总接口未返回预测区间，metafor 返回按 1 个自由度的 t 分布计算的端点。[候选包对照数据](../../validation/meta_analysis/pymetaanalysis-0.9.0.tsv)

## 本地复核

19 个合成场景按 R 4.5.3 与 metafor 5.0.1 验证。它们覆盖不同抽样精度、τ² 等于零、同效应导致的零标准误、有限搜索上界、两项研究、不同置信水平及默认和 Riley 预测。拟合和区间用 R 的高精度控制参数生成，Python 数值断言为相对容差 `4e-9` 和绝对容差 `8e-12`。[场景定义](../../validation/meta_analysis/scenarios.tsv) [R 参考结果](../../validation/meta_analysis/expected.tsv)

```sh
python -m pytest -q tests/test_meta_analysis.py
BIOMED_R_PARITY=1 python -m pytest -q tests/test_meta_analysis.py
Rscript --vanilla validation/meta_analysis/generate.R
```

在独立的 Python 环境安装 `PyMetaAnalysis==0.9.0` 后，可运行 `python validation/meta_analysis/audit_pymetaanalysis.py` 重现对照。
