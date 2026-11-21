# mstate Markov 假设的 log-rank 检验

`markov_test` 对多状态模型中的一个指定转移计算 Titman–Putter 检验。它逐个考察预先指定的时间点，按该时间点之前所处的状态划分受试者，比较之后所检验转移的风险。接口使用 [msprep](mstate-prep.md) 返回的 `MultiStateRecords`，状态和转移采用从 0 开始的编号。 [Titman–Putter 方法](https://doi.org/10.1093/biostatistics/kxaa030) [mstate 参考](https://hputter.github.io/mstate/reference/MarkovTest.html)

```python
from biomed_parity.mstate_markov import markov_test
from biomed_parity.mstate_prep import msprep

records = msprep(
    [[0, 1, 5], [0, 4, 6], [0, 6, 6]],
    [[0, 1, 1], [0, 1, 1], [0, 0, 1]],
    [(0, 1), (0, 2), (1, 2)],
)
result = markov_test(records, 2, grid=[2, 4, 5], random_state=17)
print(result.z_trace)
print(result.state_pvalues)
print(result.overall_pvalue)
```

| Python 字段 | 对应 R 字段及含义 |
| --- | --- |
| `qualifying_states` | `qualset`，包含转移起始状态及可经一个或多个转移到达该状态的前驱状态。 |
| `n_subjects`、`n_events_after` | `Nsub`、`nobs_grid`，曾处于所检验转移的风险集中的人数及各时间点之后的事件数。 |
| `score_trace`、`z_trace`、`covariance` | 未标准化的 log-rank 得分、`zbar` 及 `est_cov`。记录按「时间点 × 状态」组织，协方差为每个时间点的一张状态矩阵。 |
| `chi_squared_trace` | `obs_chisq_trace`，去掉第一个状态后计算联合二次型。删去该状态后协方差仍奇异的时间点记为零。 |
| `state_statistics`、`overall_statistic` | `orig_stat`、`orig_ch_stat`，分别汇总各状态的 `abs(Z)` 和联合二次型，默认取网格平均。 |
| `bootstrap_z_trace`、`bootstrap_chi_squared_trace` | `n_wb_trace`、`nch_wb_trace`，每次野生自助抽样的统计量轨迹。第一维为抽样次数。 |
| `bootstrap_state_statistics`、`bootstrap_overall_statistics` | 各次抽样的状态统计量与总体统计量。 |
| `state_pvalues`、`overall_pvalue`、`z_quantiles` | `p_stat_wb`、`p_ch_stat_wb` 和 `est_quant`。分别返回严格超过观测统计量的抽样比例及逐点 2.5% 和 97.5% 样本分位数。 |
| `cox_fit` | `coxfit` 的数值结果，包含协变量名称、系数、模型协方差和拟合前后的部分对数似然。 |

## 风险集与抽样

同一受试者、状态和时间区间的多条出边风险记录合成一段状态历程。采用 `start < s <= stop` 识别网格时间 `s` 的分组，并用 `t > s` 的目标转移事件计算得分。每个事件时点按 `start < t <= stop` 构建风险集，并用当前风险集中各分组的比例对事件指示作中心化。并列事件共享风险集比例。零方差状态和因缺少后续事件而无法比较的时间点遵循 R 的记零规则。所检验转移需有至少一个观测事件。 [风险集与抽样方法](https://doi.org/10.1093/biostatistics/kxaa030)

野生自助抽样默认 1000 次，可选择中心化 Poisson 或标准正态乘子。`random_state` 接受整数种子或 NumPy 的生成器。`multipliers` 可提供已生成的「抽样次数 × 事件次数」矩阵，列顺序按受试者 ID 和状态历程的进入时间排序，在结果的 `event_subject_ids`、`event_times` 中可核对。抽样更换目标转移的事件乘子。`n_bootstrap=0` 只计算观测轨迹和统计量。 [R 抽样接口](https://hputter.github.io/mstate/reference/MarkovTest.html)

## Cox 协变量和偏移量

使用 `msprep(..., keep={...})` 保留连续变量、数值哑变量或对数风险尺度的偏移量。然后用 `covariates` 传入一个列名或列名序列，用 `offset` 传入偏移量列名。每位受试者采用首条保留记录中的变量值。协变量列需有限且满秩，可先将分类变量转换为数值哑变量。Cox 模型以 Efron 方法处理并列事件，检验的风险集计算使用 Breslow 增量。抽样和协方差包含估计系数的影响。 [mstate 模型接口](https://hputter.github.io/mstate/reference/MarkovTest.html) [Cox 模型的并列事件处理](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/coxph.html)

`cox_mode="mstate"` 为默认值，复现 mstate 0.3.3 和 survival 3.8.6 的行为。该版本的 `coxph` 在拟合时使用偏移量，mstate 计算轨迹时读取缺失的 `offset` 字段并代入零。mstate 还在多个协变量的协方差计算中按列回收协变量向量，并使用第一个状态的均值处理非对角元。`cox_mode="full"` 在拟合和轨迹中都使用偏移量，并对每个协变量及状态使用各自的风险集均值。 [mstate 检验源码](https://github.com/hputter/mstate/blob/master/R/MarkovTest.R)

```python
adjusted = markov_test(
    records_with_covariates, 2, grid,
    covariates=["age", "treatment_indicator"], offset="log_risk_offset",
    cox_mode="full", random_state=17,
)
print(adjusted.cox_fit.coefficients)
```

## 加权汇总

`state_weights="optimal"` 对各合格状态的绝对 Z 轨迹应用对应列的最优权重，`overall_weights="optimal"` 对联合卡方轨迹应用最优权重。两项可独立设置，并用 `min_time` 指定首个网格区间的左端点，默认为 0。观测轨迹和各次自助抽样采用相同的汇总权重。也可直接传入非负权重向量，每个网格点对应一个值，或为 `state_weights` 传入按合格状态排列的矩阵。权重和为零的汇总统计量及其 p 值返回 `NaN`。

`optimal_weights_matrix(records, transition, grid)` 返回 R 的 `weights` 属性。行对应网格点，列按记录中出现的起始状态编号升序排列。其计算以 `entry <= s < exit` 统计网格点 `s` 上的状态人数，并以 `exit > s` 统计原始记录中之后的目标转移事件，包括零时长历程中的事件。原始状态权重需逐行乘以 `np.diff(grid, prepend=min_time)` 后用于汇总。`optimal_weights_multiple(records, transition, grid, min_time=min_time)` 返回已乘区间宽度的联合权重，每个网格点取所有起始状态的最大平方权重。 [最优权重的 R 源码](https://github.com/hputter/mstate/blob/master/R/MarkovTest.R)

mstate 0.3.3 的 `optimal_weights_matrix` 返回的函数都读取最后一列的区间权重。需要重现这一版本的函数结果时，可将矩阵最后一列乘以区间宽度，作为所有合格状态共用的 `state_weights` 向量。

```python
weighted = markov_test(
    records, 2, [2, 4, 5], state_weights="optimal", overall_weights="optimal",
    min_time=1, random_state=17,
)
```

## 本地验证

```bash
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q tests/test_mstate_markov.py
R_LIBS_USER="$PWD/.r-lib" .venv/bin/python validation/mstate_markov/verify.py
```

R 4.5.3 和 mstate 0.3.3 的三个合成案例覆盖并列事件、延迟入组、竞争转移、间接前驱状态、单一合格状态和奇异协方差。对照从 R 的实际随机流提取有效事件乘子供 Python 重放，同时核验两种乘子分布下的观测及抽样轨迹、协方差、统计量、p 值和分位数。加权对照核验两个最优权重函数的数值、使用各列权重的汇总，以及使用 R 返回函数的汇总，绝对容差为 `1e-12`。五个 Cox 对照覆盖单协变量、仅偏移量、两个协变量加偏移量，并核验 Cox 拟合数值和检验结果，绝对容差为 `1e-8`。`cox_mode="full"` 的对照在 R 运行时修正偏移量提取及协方差中心化。
