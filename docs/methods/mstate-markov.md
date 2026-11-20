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
| `state_statistics`、`overall_statistic` | `orig_stat`、`orig_ch_stat`，分别为各状态 `abs(Z)` 的网格平均及联合二次型的网格平均。 |
| `bootstrap_z_trace`、`bootstrap_chi_squared_trace` | `n_wb_trace`、`nch_wb_trace`，每次野生自助抽样的统计量轨迹。第一维为抽样次数。 |
| `bootstrap_state_statistics`、`bootstrap_overall_statistics` | 各次抽样的状态统计量与总体统计量。 |
| `state_pvalues`、`overall_pvalue`、`z_quantiles` | `p_stat_wb`、`p_ch_stat_wb` 和 `est_quant`。分别返回严格超过观测统计量的抽样比例及逐点 2.5% 和 97.5% 样本分位数。 |

## 风险集与抽样

同一受试者、状态和时间区间的多条出边风险记录合成一段状态历程。采用 `start < s <= stop` 识别网格时间 `s` 的分组，并用 `t > s` 的目标转移事件计算得分。每个事件时点按 `start < t <= stop` 构建风险集，并用当前风险集中各分组的比例对事件指示作中心化。并列事件共享风险集比例。零方差状态和因缺少后续事件而无法比较的时间点遵循 R 的记零规则。所检验转移需有至少一个观测事件。 [风险集与抽样方法](https://doi.org/10.1093/biostatistics/kxaa030)

野生自助抽样默认 1000 次，可选择中心化 Poisson 或标准正态乘子。`random_state` 接受整数种子或 NumPy 的生成器。`multipliers` 可提供已生成的「抽样次数 × 事件次数」矩阵，列顺序按受试者 ID 和状态历程的进入时间排序，在结果的 `event_subject_ids`、`event_times` 中可核对。只需要观测目标转移对应的乘子，因为其他历程的乘子在这种模型下不进入抽样得分。`n_bootstrap=0` 只计算观测轨迹和统计量。当前接口使用未调整的标准 log-rank 权重和默认的轨迹汇总规则。 [R 抽样接口](https://hputter.github.io/mstate/reference/MarkovTest.html)

## 本地验证

```bash
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q tests/test_mstate_markov.py
R_LIBS_USER="$PWD/.r-lib" .venv/bin/python validation/mstate_markov/verify.py
```

R 4.5.3 和 mstate 0.3.3 的三个合成案例覆盖并列事件、延迟入组、竞争转移、间接前驱状态、单一状态和奇异协方差。对照从 R 的实际随机流提取有效事件乘子供 Python 重放，同时核验两种乘子分布下的观测及抽样轨迹、协方差、统计量、p 值和分位数。绝对容差为 `1e-12`。另有手算用例验证并列时间的分组规则、严格超过的 p 值计算及只计算观测轨迹的路径。
