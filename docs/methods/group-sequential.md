# gsDesign 组序贯边界、越界概率与条件把握度

## 功能与接口

`efficacy_boundaries(information_fractions, *, alpha=0.025, spending="hsd", parameter=None)` 在标准联合正态模型下，校准单侧上界，用于预设信息量下的期中和最终分析。输入的信息分数应严格递增并在 1 结束，总体单侧 α 的范围为 `(0, 0.5)`。对应 R `gsDesign(test.type=1)` 的上界和零假设越界概率。 [gsDesign 3.11.0 手册](https://cran.r-project.org/web/packages/gsDesign/refman/gsDesign.html#gsDesign)

```python
from biomed_parity.group_sequential import efficacy_boundaries

result = efficacy_boundaries([0.33, 0.67, 1], alpha=0.025, spending="ldof")
print(result.z_boundaries)
print(result.nominal_pvalues)
print(result.null_crossing_probabilities)
```

| 参数或字段 | 含义 |
| --- | --- |
| `spending="hsd"` | Hwang–Shih–DeCani，`parameter` 对应 γ，默认为 −4，允许 [−40, 40]。γ=0 时线性花费。 |
| `spending="ldof"` | Lan–DeMets O'Brien–Fleming 型，`parameter` 对应广义参数 ρ，默认为 1。按照本机 gsDesign 3.11.0，超出 [0.005, 20] 的有限值回退到 1。 |
| `z_boundaries` | 各次分析的疗效 Z 阈值。浮点精度内没有新增 α 花费的期次返回正无穷。 |
| `nominal_pvalues` | 各 Z 阈值的一侧标准正态尾概率。 |
| `cumulative_alpha`、`incremental_alpha` | 至该次分析为止的目标累计 α，以及该次分析的新增 α。 |
| `null_crossing_probabilities` | 在零假设下首次于该分析跨越疗效阈值的计算概率。 |

花费公式和参数以 R 手册为依据。 [Hwang–Shih–DeCani](https://cran.r-project.org/web/packages/gsDesign/refman/gsDesign.html#sfHSD) [Lan–DeMets](https://cran.r-project.org/web/packages/gsDesign/refman/gsDesign.html#sfLDOF)

## 双边界设计与运行特性

| Python 接口 | 对应 R 路径 | 主要返回值 |
| --- | --- | --- |
| `symmetric_boundaries` | `gsDesign(test.type=2)` | 各分析的镜像上下界，零假设下总体双侧错误率为 `2 * alpha`。 |
| `efficacy_futility_boundaries(..., binding=True)` | `gsDesign(test.type=3)` | 单侧疗效上界、约束性无效下界及达到目标把握度的标准化备择效应。 |
| `efficacy_futility_boundaries(..., binding=False)` | `gsDesign(test.type=4)` | 单侧疗效上界、非约束性无效下界及目标标准化备择效应。 |
| `crossing_probabilities` | `gsProbability` | 在给定效应下，各次分析首次越过上界、首次越过下界，以及完成该次分析仍继续的概率。 |
| `conditional_power` | `gsCP` | 给定已观测期中 Z 和假定效应后，未来各次分析的同一组概率。 |

无效下界设计在备择下分配 β，默认 `beta=0.1`、`futility_spending="hsd"`、`futility_parameter=-2`，上界可使用 HSD 或 LDOF。约束性设计按遵守下界时的零假设路径校准 α。非约束性设计按无效下界可被跳过的零假设路径校准 α。两类设计在备择下均按遵守下界的规则校准 β，并让上下界在最终分析重合。 [gsDesign 检验类型说明](https://cran.r-project.org/web/packages/gsDesign/refman/gsDesign.html#gsDesign)

`effect` 是达到最终信息量时 Z 的边际均值，等于 R `theta * sqrt(n.I[k])`。`respect_lower=False` 可在运行特性计算中忽略无效停止。`conditional_power` 使用从 0 开始的 `look` 和已观测的 `z`，计算假定继续试验时的后续越界概率。`upper_total` 是任何未来分析首次越过疗效上界的总概率。对称双侧设计可把上下两类概率相加。 [R 越界概率](https://keaven.github.io/gsDesign/reference/gsProbability.html) [R 条件把握度](https://keaven.github.io/gsDesign/reference/gsCP.html)

```python
from biomed_parity.group_sequential import (
    conditional_power, crossing_probabilities, efficacy_futility_boundaries,
)

fit = efficacy_futility_boundaries([0.33, 0.65, 1], beta=0.15, binding=False)
args = (fit.information_fractions, fit.lower_boundaries, fit.upper_boundaries)
print(crossing_probabilities(*args, respect_lower=False).upper_total)
print(conditional_power(*args, look=0, z=1.2, effect=fit.alternative_effect).upper_total)
```

## 数值方法

零假设下，将不同信息分数的标准化统计量建模为正态增量过程。给定前次 Z 统计量，当前 Z 的条件均值为前次观测值乘以相邻信息分数比的平方根，条件方差为一减该比值。第一次分析通过正态分位数直接确定阈值。后续每一步对尚未越界的前次 Z 密度作 Gauss–Legendre 求积，计算当前首次越界的概率，以 Brent 法求解等于新增 α 的 Z 阈值。求积按相邻增量宽度增加节点。若需要超过 2048 个节点或积分后的继续概率与目标偏差超过 `5e-10`，接口抛出 `ArithmeticError`。 [R 越界概率定义](https://keaven.github.io/gsDesign/reference/gsProbability.html)

双边界使用同一正态转移递推，在上下界之间保留继续试验的密度。非零效应和观测期中 Z 都通过平移、缩放剩余信息量下的正态统计量进入递推。确定无效下界时，另对最终标准化备择效应求解，使按 α 和 β 独立校准的最终边界重合。数值实现允许无停止规则的期次使用无穷界，并从传播网格跳过这类期次。

## 本地验证

```bash
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q tests/test_group_sequential.py
R_LIBS_USER="$PWD/.r-lib" .venv/bin/python validation/group_sequential/verify.py
```

本机 R 4.5.3、`gsDesign` 3.11.0 和 `jsonlite` 生成 8 个固定设计的基准。覆盖等间隔、非等间隔、密集分析、早期极小花费、广义 LDOF 和三种 HSD 参数。普通设计的 Z 阈值绝对容差为 `3e-8`，首次越界概率绝对容差为 `1e-10`。R 基准使用 `r=80`、`tol=1e-10`。

早期极小花费的案例中，R 3.11.0 使用一减正态分布函数，Z 阈值和直接以尾概率计算的结果相差约 `1.07e-5`。此期的对照绝对容差为 `2e-5`，另外独立验证其正态尾概率。密集分析案例的 R 网格收敛较慢，Z 阈值最大差约 `5.90e-6`，对照绝对容差为 `1e-5`。另用自适应一维积分核验其第二次分析的首次越界概率。固定分析、禁用零花费期次和输入错误均有独立测试。

另有 5 个双边界设计基准，覆盖对称 HSD 和 LDOF、约束性和非约束性无效下界。校准对照使用 `r=80` 和 `tol=1e-7`，上下界及标准化备择效应的绝对容差为 `2e-6`。为单独比较概率算法，Python 采用 R 给出的原始边界，与 `gsProbability` 或 `gsCP` 的 25 组结果按绝对容差 `5e-10` 比较。实测最大边界偏差约 `9.84e-7`，最大概率偏差约 `2.83e-10`。纯 Python 测试额外检查累计 α/β、双侧镜像、最终重合及单一剩余分析的正态解析结果。

## 兼容范围

本模块覆盖 `gsDesign` 的 1–4 型中基于 HSD 和 LDOF 花费的边界，以及预设信息量下的越界概率和条件把握度。零假设下花费的无效界、伤害界、端点特异的样本量与日历时间、非恒定疗效、预测把握度、重复置信区间及条件误差仍属于完整临床试验设计的后续迁移范围。 [gsDesign 参考手册](https://cran.r-project.org/web/packages/gsDesign/refman/gsDesign.html) [Python gsdesign](https://pypi.org/project/gsdesign/) [Python sequential-analysis](https://andre-a-no.github.io/sequential-analysis/)
