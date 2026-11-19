# gsDesign 单侧组序贯疗效边界

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

## 数值方法

零假设下，将不同信息分数的标准化统计量建模为正态增量过程。给定前次 Z 统计量，当前 Z 的条件均值为前次观测值乘以相邻信息分数比的平方根，条件方差为一减该比值。第一次分析通过正态分位数直接确定阈值。后续每一步对尚未越界的前次 Z 密度作 Gauss–Legendre 求积，计算当前首次越界的概率，以 Brent 法求解等于新增 α 的 Z 阈值。求积按相邻增量宽度增加节点。若需要超过 2048 个节点或积分后的继续概率与目标偏差超过 `5e-10`，接口抛出 `ArithmeticError`。 [R 越界概率定义](https://keaven.github.io/gsDesign/reference/gsProbability.html)

## 本地验证

```bash
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q tests/test_group_sequential.py
R_LIBS_USER="$PWD/.r-lib" .venv/bin/python validation/group_sequential/verify.py
```

本机 R 4.5.3、`gsDesign` 3.11.0 和 `jsonlite` 生成 8 个固定设计的基准。覆盖等间隔、非等间隔、密集分析、早期极小花费、广义 LDOF 和三种 HSD 参数。普通设计的 Z 阈值绝对容差为 `3e-8`，首次越界概率绝对容差为 `1e-10`。R 基准使用 `r=80`、`tol=1e-10`。

早期极小花费的案例中，R 3.11.0 使用一减正态分布函数，Z 阈值和直接以尾概率计算的结果相差约 `1.07e-5`。此期的对照绝对容差为 `2e-5`，另外独立验证其正态尾概率。密集分析案例的 R 网格收敛较慢，Z 阈值最大差约 `5.90e-6`，对照绝对容差为 `1e-5`。另用自适应一维积分核验其第二次分析的首次越界概率。固定分析、禁用零花费期次和输入错误均有独立测试。

## 兼容范围

本模块提供标准正态模型的单侧疗效上界和零假设概率。双侧与伤害边界、约束性和非约束性无效边界、计算把握度与样本量、日历时间、疗效差异的非恒定模型、重复置信区间及条件误差属于完整临床试验设计的后续迁移范围。R 和 Python 均已有更广的接口可供评估。 [gsDesign 参考手册](https://cran.r-project.org/web/packages/gsDesign/refman/gsDesign.html) [Python gsdesign](https://pypi.org/project/gsdesign/) [Python sequential-analysis](https://andre-a-no.github.io/sequential-analysis/)
