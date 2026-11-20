# 组序贯重复推断

`group_sequential_inference` 在标准联合正态模型下处理已完成分析的 Z 统计量，支持单侧重复 p 值、序贯 p 值及重复 Wald 置信区间。 [gsDesign 分析手册](https://keaven.github.io/gsd-tech-manual/analysis.html)

```python
from biomed_parity.group_sequential import efficacy_boundaries
from biomed_parity.group_sequential_inference import (
    repeated_confidence_intervals,
    repeated_pvalues,
    sequential_pvalue,
)

pvalues = repeated_pvalues(
    [100, 160],
    [1.5, 2.0],
    planned_final_information=220,
    spending="ldof",
)
print(pvalues.repeated)
print(pvalues.cumulative)
print(pvalues.sequential)

design = efficacy_boundaries([0.4, 0.7, 1])
intervals = repeated_confidence_intervals(
    [0.18, 0.24], [0.11, 0.08], design.z_boundaries[:2],
)
print(intervals.lower)
```

| 接口 | 结果 |
| --- | --- |
| `repeated_pvalues(information, z, ...)` | `repeated` 是每次分析达到疗效阈值所需的最小整体 α。校准该期阈值时保留此前分析的花费。`cumulative` 是到各期的最小重复 p 值，`sequential` 是到最新一期的最小值。 |
| `sequential_pvalue(information, z, ...)` | 直接计算任何一次已完成分析达到疗效阈值所需的最小整体 α，对应 `gsDesign::sequentialPValue`。 |
| `repeated_confidence_intervals(estimates, standard_errors, upper_z_boundaries, ...)` | 固定每次分析适用的临界 Z 值，计算各期的置信下限、上限或两侧区间。 |

## 时间、花费及区间

`information` 是已完成分析的累计信息量。一种用法是直接输入信息分数，另一种用法是输入绝对信息量并指定 `planned_final_information`。正态过程的相关系数由实际信息量的比值决定。默认用实际信息量除以计划最终信息量作为花费时间，也可通过 `spending_times` 独立指定正的递增花费时间。最终花费时间达到或超过 1 时用完预算。花费选择 HSD 或 LDOF，与 [原组序贯边界模块](group-sequential.md) 使用同一参数定义。搜索区间默认是 `[1e-5, 0.9999]`。超出时返回最近端点。计算适用于单侧疗效以及非约束性无效或伤害停止规则。 [R 序贯 p 值参考](https://keaven.github.io/gsDesign/reference/sequentiaPValue.html)

反算 p 值时，按候选 α 校准各期的累计花费，再检查观测 Z 是否达到阈值。阈值采用正态增量求积，反算采用 Brent 法并在正态分位数尺度上搜索。序贯 p 值取所有已完成分析的最小值。 [疗效阈值反算说明](https://keaven.github.io/gsd-tech-manual/analysis.html#sequential-p-values)

重复置信下限等于点估计减去「临界 Z × 标准误」。`alternative="less"` 使用反向疗效检验所对应的上限，`alternative="two-sided"` 同时计算两端。要使用对称设计的双侧覆盖率，提供该设计的对称上界。`+inf` 表示该期疗效阈值未启用，相应区间没有有限端点。点估计与标准误应同尺度，且匹配校准边界时采用的正态统计量。 [重复置信区间说明](https://keaven.github.io/gsd-tech-manual/analysis.html#repeated-confidence-intervals)

## 本地验证

```bash
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q tests/test_group_sequential_inference.py
R_LIBS_USER="$PWD/.r-lib" .venv/bin/python validation/group_sequential_inference/verify.py
```

用 R 4.5.3 和 gsDesign 3.11.0 校验 10 个案例，涵盖提前分析、超过原计划的信息量、独立花费时间、前期信号最强、区间截断和单侧及对称区间。重复和序贯 p 值与 R 比较的绝对容差为 `1e-5`，覆盖两者的根求解精度差异。置信区间使用 R 计算 Z 临界值后应用手册的正态端点公式，绝对容差为 `1e-7`。纯 Python 测试另核验首期解析值、尾部 p 值及反算预设边界所得的整体 α。
