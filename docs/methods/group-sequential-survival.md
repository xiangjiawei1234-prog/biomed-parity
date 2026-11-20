# gsDesign 生存试验样本量与分析时间

在比例风险假设下，`group_sequential_survival` 计算两臂生存试验的预期事件数、入组人数、固定分析样本量和组序贯分析时间。接口采用连续期望值，时间单位由使用者选择，同一设计的入组率、事件风险率、脱落风险率和时长应使用一致的单位。 [gsDesign 生存设计参考](https://keaven.github.io/gsDesign/reference/nSurv.html)

```python
from biomed_parity.group_sequential_survival import (
    SurvivalTrial,
    group_sequential_survival_design,
)

trial = SurvivalTrial(
    control_hazards=[0.12, 0.09],
    hazard_durations=[8],
    hazard_ratio=0.70,
    control_dropout=0.01,
    accrual_rates=[5, 15],
    accrual_durations=[4, 16],
    allocation_ratio=1.5,
)
design = group_sequential_survival_design(
    [0.4, 0.7, 1],
    trial=trial,
    minimum_followup=6,
    test_type=4,
)
print(design.event_targets)
print(design.analysis_times)
print(design.counts.total_enrollment)
print(design.upper_boundaries)
```

| Python 接口 | 对应 R 接口 | 结果 |
| --- | --- | --- |
| `SurvivalTrial` | `lambdaC`、`hr`、`hr0`、`eta`、`etaE`、`gamma`、`R`、`S`、`ratio` | 对照事件率、零假设与备择风险比、各臂脱落风险率、总入组率、分段时长和试验组与对照组的入组分配比。 |
| `expected_survival_events(trial, analysis_time)` | `eEvents`，以及共有脱落风险时的 `nEventsIA` | 指定分析时间在备择下各臂、各分层的预计入组人数和事件数。 |
| `survival_event_time(trial, target_events, max_time=...)` | `tEventsIA` | 由绝对目标事件数反求分析时间。可用最终事件数乘以信息分数来指定目标。 |
| `fixed_survival_design` | `nSurv` | 固定分析的预计人数、事件数、调整后的入组计划及功效。 |
| `group_sequential_survival_design` | `gsSurv` | 各期的信息分数、事件目标、分析时间、各臂预计人数及事件数、上下界与标准化备择效应。 |

风险率和入组率矩阵按「分段 × 分层」组织。标量表示各段、各层共用同一风险率或入组率。一维序列逐段给出率，在现有分层间共享。风险分段基于入组后的随访时长，最后一段持续至所有分析时点。入组分段基于试验开始后的日历时长，到计划入组截止日结束。`SurvivalTrialCounts` 的各臂字段在单点预测中以一维分层数组返回，在组序贯设计中以「分析期次 × 分层」矩阵返回。`total_enrollment` 和 `total_events` 对分层及两臂求和。

## 样本量与日历时间

- `solve="accrual_rate"` 使用固定的研究时间和最短随访时间，按同一比例调整所有入组率。`beta=None` 计算现有计划的功效，可用于固定设计和单次分析的组序贯接口。
- `solve="accrual_duration"` 保持入组率和最短随访时间，调整入组截止日。较长的入组期沿用最后一段入组率，较短的入组期在相应时点截断计划。
- `solve="followup_duration"` 保持给定的入组率和入组时长，调整最终分析前的最短随访时间。

固定设计支持 `LachinFoulkes`、`Schoenfeld`、`Freedman` 和 `BernsteinLagakos` 方法。Lachin–Foulkes 与 Bernstein–Lagakos 根据各臂和分层的预计事件数估计零假设与备择下的信息量。Schoenfeld 和 Freedman 根据备择的总事件数及入组分配比缩放样本量，这两种方法要求零假设风险比为 1，Freedman 另限单分层。Lachin–Foulkes 的固定入组率时长求解采用 R 3.11.0 的合并分层信息公式。 [gsDesign 固定设计方法说明](https://keaven.github.io/gsDesign/articles/SurvivalOverview.html)

组序贯设计沿用 [组序贯边界](group-sequential.md) 中 HSD 和 LDOF 花费的 1–4 型检验。用参考固定设计及标准化 Z 效应所需的比例计算最终事件目标，再以预设信息分数计算各次的目标事件数。`fixed_design` 保留参考固定设计。求解组序贯随访时长时，参考固定设计采用组序贯的最终时间，入组率经固定设计校准。对称检验的把握度与 R 一样按备择方向的上界首次越界概率计算。`sided` 控制 `alpha / sided` 的单侧预算。 [gsDesign 检验类型说明](https://keaven.github.io/gsDesign/reference/gsDesign.html)

对每个分析时点，将入组分段转换为随访年龄，并和事件及脱落风险的分段一起确定积分区间。在每一段中积分各臂事件累计发生率乘以入组率。通过 Brent 法从预计累计事件数反求各次分析的时间。两臂始终分别使用各自的脱落风险率。R gsDesign 3.11.0 的 `nEventsIA` 对两臂均采用对照组脱落风险，因此给定不同脱落率时，备择试验组的期中预计事件数及期中分析时间可能与 `gsSurv` 不同。此时按臂调用 R `eEvents` 校验。 [gsDesign 日历事件函数](https://keaven.github.io/gsDesign/reference/nSurv.html)

## 本地验证

```bash
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q tests/test_group_sequential_survival.py
R_LIBS_USER="$PWD/.r-lib" .venv/bin/python validation/group_sequential_survival/verify.py
```

用 R 4.5.3 和 gsDesign 3.11.0 比较 15 个合成案例。覆盖多段入组与随访风险、分层、非等比例分配、各臂独立脱落、不同固定设计方法、四种组序贯检验、三种求解目标和单次分析。固定设计与日历投影使用绝对容差 `1e-8` 和相对容差 `1e-10`。组序贯设计的时长绝对容差为 `3e-5` 个时间单位，其他数值采用绝对容差 `7e-7` 和相对容差 `5e-7`，涵盖两种边界积分算法的差异。纯 Python 测试还检查常风险解析公式、近零事件率和反算的时间。
