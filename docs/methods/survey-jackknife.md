# survey 的 JK1 与 JKn 复制权重构造

`biomed_parity.survey_jackknife.as_svrepdesign` 根据一阶段抽样的原始权重、初级抽样单元（PSU）和抽样层构造删一 PSU 的 jackknife 复制权重。默认按 R `survey::as.svrepdesign(type="auto")` 选择方法。未给 `strata` 时用 JK1，给了 `strata` 时用 JKn。省略 `psu` 时每行是一个独立 PSU。同一 PSU 标签可在不同抽样层重复。 [R 设计转换手册](https://r-survey.r-forge.r-project.org/pkgdown/docs/reference/as.svrepdesign.html)

```python
from biomed_parity.survey_jackknife import as_svrepdesign

design = as_svrepdesign(
    weights=[2.1, 1.8, 2.2, 1.9, 1.5, 2.5],
    strata=["urban", "urban", "urban", "rural", "rural", "rural"],
    psu=[11, 12, 12, 11, 13, 13],
    fpc={"urban": 0.25, "rural": 0.2},
)
fit = design.mean([126, 147, 133, 142, 120, 138])
print(fit.estimate, fit.standard_errors)
```

JK1 在全部 PSU 中逐一删除，JKn 只在该次删除对应的抽样层内调整其余 PSU。复制列按抽样层和层内 PSU 的首次出现顺序排列。`fpc` 可以是统一值、按抽样层首次出现顺序排列的向量、以抽样层标签为键的映射，或每行一个且层内一致的向量。`fpctype="fraction"` 接收已抽 PSU 的比例，`"population"` 接收总体 PSU 数量，`"correction"` 接收方差的有限总体修正因子。完全调查的抽样层保留其原始权重，其自身复制列按 R 默认行为省略。若无复制列可用，转换会报错。 [R jackknife 权重手册](https://r-survey.r-forge.r-project.org/pkgdown/docs/reference/brrweights.html)

JKn 的 `lonely_psu` 指定非确定性单 PSU 层的处理。`"fail"` 默认报错，`"certainty"` 和 `"remove"` 省略该层的复制，`"adjust"` 删除其 PSU 并按 R 的抽样层数量调整其他层，`"average"` 省略该层的复制并按抽样 PSU 总数与保留复制数之比调整整体方差。显式输入零抽样比例的单 PSU 层遵循 R 的特殊规则，保留一个方差乘数为零的复制列。

返回标准的 `SurveyReplicateDesign`。`replicate_weights` 是密集的逐行乘数，`analysis_weights` 将其乘以原始抽样权重。可直接调用 `.mean()`、`.total()` 和 `.subset()`。`mse=True` 用全样本估计为方差中心，默认使用有效复制估计的均值。这里的有限总体修正直接用于复制权重构造，完全调查层仍进入每个保留复制的估计。 [复制权重设计](survey-replicates.md)

范围限于一阶段 JK1 和 JKn 构造。R 4.5.3 和 survey 4.5 的 13 个用例覆盖权重及其顺序、方差缩放、设计自由度、均值与总量的复制估计和协方差，以及两个预期转换错误。 [用例](../../validation/survey_jackknife/cases.json) · [R 对照脚本](../../validation/survey_jackknife/oracle.R) · [对照结果](../../validation/survey_jackknife/expected.json)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest tests/test_survey_jackknife.py
```
