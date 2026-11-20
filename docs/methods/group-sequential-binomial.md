# 精确二项组序贯设计

`group_sequential_binomial` 用累计成功数分析单样本二分类终点。假设每个观察相互独立，且在给定情景下具有相同的成功概率。算法逐期卷积尚未停止的累计计数分布和新增观察的二项分布。 [gsDesign 二项设计手册](https://keaven.github.io/gsd-tech-manual/binomial-exact.html)

```python
from biomed_parity.group_sequential_binomial import (
    binomial_conditional_probabilities,
    binomial_crossing_probabilities,
    binomial_efficacy_boundaries,
)

design = binomial_efficacy_boundaries([25, 50, 80], 0.25, alpha=0.025)
boundaries = (design.sample_sizes, design.lower_boundaries, design.upper_boundaries)
rates = [0.25, 0.4]
operating = binomial_crossing_probabilities(*boundaries, rates)
conditional = binomial_conditional_probabilities(*boundaries, rates, look=0, successes=7)
print(design.upper_boundaries)
print(operating.upper_total, operating.expected_sample_size)
print(conditional.upper_total)
```

| Python 接口 | 计算内容 |
| --- | --- |
| `binomial_crossing_probabilities` | 给定上下界，计算逐期首次越界、继续概率和期望样本量。与 `gsBinomialExact` 对照。 |
| `binomial_conditional_probabilities` | 给定期中累计成功数，计算后续各期的首次越界、继续概率和最终累计样本量的条件期望。 |
| `binomial_harm_probabilities` | 给定下尾疗效界和上尾无效及伤害界，分别计算三类停止概率。与 `gsBinomialExactHarm` 对照。 |
| `binomial_efficacy_boundaries` | 给定零假设成功率和 α 花费，校准单侧整数疗效界并返回目标与实际累计 α。与 `toBinomialExact` 的单侧疗效换算对照。 |
| `binomial_sprt` | 以两种成功率和名义 α、β 计算截断序贯概率比检验的整数界、连续界和运行特性。与 `binomialSPRT` 对照。 |

## 停止规则

样本量为正的严格递增累计整数。若某次分析的累计成功数小于等于下界或大于等于上界，试验在该次分析停止。下界 `-1` 和大于当前样本量的上界表示该期相应边界停用。`respect_lower=False` 在整段计算中停用下界。计算下尾疗效且忽略上尾无效停止时，将上界指定为各期样本量加一。 [gsDesign 单样本二项接口](https://keaven.github.io/gsDesign/reference/gsBinomialExact.html)

`lower`、`upper` 和 `continuing` 以分析期次为行、假设成功率为列。`lower_total` 和 `upper_total` 按列汇总。`continuing` 在最后一期表示到样本量上限仍在两界之间的概率，这部分计入最终样本量。`binomial_conditional_probabilities` 的 `look` 从零开始，输出从选定期次的下一次分析开始。条件计算将给定成功数作为后续观察的起点，原值即使已触及当期边界也可用于推演继续试验的情形。 [gsDesign 条件把握度说明](https://keaven.github.io/gsDesign/reference/gsCPBinomialExact.html)

`binomial_harm_probabilities` 的 `test_futility` 和 `test_harm` 可用单个布尔值控制全部期次，也可逐期设置。双重上尾越界归为伤害。`futility` 和 `harm` 互斥，两者之和等于 `upper`。Python 直接对无效停止的计数区间求和，保留极小概率。gsDesign 3.11.0 从合计上尾概率扣除伤害概率，并将绝对值小于 `1e-14` 的差置零。 [gsDesign 3.11.0 换算源码](https://github.com/keaven/gsDesign/blob/v3.11.0/R/toBinomialExact.R)

## 疗效界与 SPRT

疗效界默认采用上尾检验和 Hwang–Shih–DeCani（HSD）花费，也可用 `direction="lower"` 选择下尾，或用 `spending="ldof"` 选择 Lan–DeMets O’Brien–Fleming 花费。每期选取累计越界概率最接近且在数值精度内满足目标花费的整数界，未用完的花费自动留到后续期次。默认以该期样本量除以最终样本量为花费时间。也可提供与期次数匹配的正的严格递增 `spending_times`。指定末期时间低于 1 时保留部分 α，末期时间达到或超过 1 时使用完整的目标预算。 [组序贯花费函数](group-sequential.md)

截断 SPRT 自 `min_sample_size` 起逐个样本考察整数边界，至 `max_sample_size` 为止。输出省略首次可越界前的期次，整个范围内均无法越界时保留最终样本量。到达上限仍在两界之间的概率作为未作出判定的概率返回。名义 α、β 定义 Wald 似然比界，实际停止概率可从 `operating_characteristics` 读取。gsDesign 3.11.0 的 `gsBinomialExact` 要求至少两次分析，少于两个可越界期次的 SPRT 也会报错。 [gsDesign 3.11.0 二项检验源码](https://github.com/keaven/gsDesign/blob/v3.11.0/R/gsBinomialExact.R)

后续迁移范围包括组合花费下的无效界与伤害界自动校准、时间到事件设计对象的全量转换，以及精确重复及序贯 p 值和区间。

## 本地验证

```bash
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_group_sequential_binomial.py
R_LIBS_USER="$PWD/.r-lib" .venv/bin/python validation/group_sequential_binomial/verify.py
```

用 R 4.5.3 和 gsDesign 3.11.0 核对 10 个合成案例，涵盖约束性和非约束性越界、两个期中条件情景、选择性无效与伤害停止、HSD 和 LDOF 以及独立花费时间、默认和自定义截断 SPRT。条件情景在 R 中通过将剩余样本量和整数界转换为增量后调用 `gsBinomialExact` 验证。样本量和整数界要求完全一致，概率、期望样本量和连续界用绝对容差 `1e-12`。Python 测试还覆盖小样本穷举、极端概率、恰好达到离散花费预算和无可跨越边界时的截断。
