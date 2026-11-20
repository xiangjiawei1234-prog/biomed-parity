# survey 的抽样权重校准

`biomed_parity.survey_calibration.calibrate` 根据已知总体控制量调整抽样权重，覆盖 `survey::calibrate` 的线性校准、广义 raking 和 logit 校准。线性和 raking 均支持无界或有界校准。传入原始抽样权重、数值控制矩阵和按列排列的总体控制量。控制矩阵每行对应一个抽样单位。若需同时校准总体大小，在矩阵中加入一列全 1。分类边际可编码为指示列，连续控制列可直接提供。 [survey 校准手册](https://search.r-project.org/CRAN/refmans/survey/html/calibrate.html)

```python
from biomed_parity.survey_calibration import calibrate

controls = [[1, 0], [1, 1], [1, 0], [1, 1]]
result = calibrate(
    weights=[2.5, 3.2, 1.8, 2.7],
    variables=controls,
    population=[15, 7],
    calfun="logit",
    bounds=(0.7, 2.0),
)
print(result.weights, result.control_totals, result.residuals)
```

校准权重等于原始权重乘以逐行调整因子 `multipliers`。原始权重可为零。`bounds=(lower, upper)` 指定调整因子的下限和上限。每侧均可使用全样本通用的标量，或与观测行对齐的向量。线性和 raking 支持单侧界限，另一侧可传入负无穷或正无穷。`calfun="logit"` 要求各行的两个界限均为有限值，且严格满足 `lower < 1 < upper`。若每个目标总量与原样本对应总量的比值均超过 20，则按 survey 的约定先重新缩放抽样权重，再应用界限，并发出警告。返回的 `multipliers` 包含该缩放因子。 [survey 校准距离](https://search.r-project.org/CRAN/refmans/survey/html/make.calfun.html)

无界线性校准允许带符号的调整因子，并要求非结构零控制列独立。迭代方法可接受线性相关但彼此相容的控制量，例如总体大小和覆盖全部类别的分类边际。`variance` 可按行传入正方差因子，也可传入与控制列对应的系数，由控制矩阵计算逐行因子。长度同时符合两种输入时，按行解释。

`control_totals` 返回校准后的控制量，`residuals` 为实得值减去目标值。`relative_errors` 为绝对残差除以 `1 + abs(population)`，并与 `epsilon` 比较。`rank` 为正权重样本中的有效控制矩阵秩。`iterations` 和 `converged` 记录数值求解结果。可通过 `maxit` 限制牛顿迭代次数。达到迭代上限仍未满足容差时，默认报错，设置 `force=True` 则发出警告并返回当前估计和残差。

## 复制权重和 Taylor 方差

`calibrate_replicates(design, variables, population, ...)` 将全样本权重和每列实际分析所用的复制权重校准至同一组控制量，返回可继续调用 `.mean()`、`.total()` 或 `.subset()` 的复制权重设计。若全样本有界校准成功而个别复制未达到容差，例程会保留该复制的当前权重，并发出带复制编号的警告。无界线性校准重新计算设计自由度，迭代校准保留原设计自由度。后续域估计遵循复制权重设计的自由度规则。 [复制权重设计](survey-replicates.md)

`calibrated_total(values, weights, variables, population, ...)` 返回一个或多个结局总量、设计协方差、标准误和校准诊断。可按行指定 `psu`、`strata` 和取值为 0 到 1 的第一阶段有限总体抽样比例 `fpc`。Taylor 方差先按 survey 的 GREG 投影约定对结局做残差化，再在抽样层内按 PSU 加总并估计协方差。该接口覆盖单阶段抽样。 [survey 方差手册](https://search.r-project.org/CRAN/refmans/survey/html/svyrecvar.html)

当前接口使用数值控制矩阵。绝对权重界限、权重修剪、分阶段与簇内校准及多阶段方差留待扩展。

R 4.5.3 和 survey 4.5 的 15 组人工抽样用例覆盖校准权重、控制总量、标量与逐行界限、不可行控制量、强制返回的未收敛结果、复制协方差及单阶段 Taylor 协方差。 [用例](../../validation/survey_calibration/cases.json) · [R 对照脚本](../../validation/survey_calibration/oracle.R) · [对照结果](../../validation/survey_calibration/expected.json)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest tests/test_survey_calibration.py
```
