# survival::survdiff 生存曲线比较

## 功能与接口

适用于单事件、右删失数据。多样本接口比较两条或多条生存曲线，可按研究中心等固定分层进行检验。单样本接口使用每位对象在其事件或末次随访时点的预期生存概率。Python 运行依赖 NumPy 和 SciPy。方法来源见 [survival 参考手册](https://cran.r-project.org/web/packages/survival/refman/survival.html#survdiff)。

```python
from biomed_parity.survival import survdiff, survdiff_one_sample

fit = survdiff(
    [1, 2, 3, 3, 4, 5, 6, 7],
    [1, 0, 1, 1, 0, 1, 0, 1],
    ["A", "B", "A", "B", "B", "A", "B", "A"],
    strata=["甲", "甲", "甲", "甲", "乙", "乙", "乙", "乙"],
    rho=1,
)
one = survdiff_one_sample([1, 0, 1], [0.90, 0.65, 0.45])
```

| 接口 | R 对应行为 | 输出 |
| --- | --- | --- |
| `survdiff` | `survdiff(Surv(time, event) ~ group + strata(stratum), rho=...)` | 组与层标签及各自人数、加权观测与期望事件数、协方差矩阵、卡方统计量、自由度、p 值 |
| `survdiff_one_sample` | `survdiff(Surv(time, event) ~ offset(survival_probability), rho=...)` | 总人数、观测与期望事件数、方差、卡方统计量、自由度、p 值 |

- 事件指示 `1` 对应事件，`0` 对应右删失。整型 `1/2` 的 R 数据应先改编码。先按原分析规则过滤和处理缺失行，再传入同长度的一维向量。
- 每个组或分层向量可使用有限数值标签或字符串标签。默认升序。若要对齐 R 因子顺序，传入 `group_order` 或 `strata_order`，完整列出每个实际出现的标签一次。多列分组或分层先编码为组合标签。
- 多层时 `observed` 和 `expected` 为组 × 层矩阵，单层时为组向量。协方差为各层之和，并按组的顺序排列。
- `rho=0` 为 log-rank 检验，`rho=1` 为 Peto–Peto 修正。支持有限实数 `rho`。默认 `timefix=True` 按 R 的 `aeqSurv` 规则合并相邻近似相等时间。
- `survdiff_one_sample` 的概率必须位于闭区间 `[0, 1]`。该方法假定概率已经过外部模型或生命表的计算。概率取端点时可能出现无限期望或不可计算的统计量，结果遵循 R 的数值约定。

## 计算

在每个分层内计算并按随访时间更新风险集。同一时间的事件者和删失者都纳入该时点风险集。令 $Y$ 为当前风险集人数，$D$ 为事件数，$p_g=Y_g/Y$，权重 $w$ 为分层合并样本的左连续 Kaplan–Meier 生存估计的 $\rho$ 次幂。给第 $g$ 组累加 $wD_g$ 个观测事件和 $wDp_g$ 个期望事件。累加的协方差为 $w^2D(Y-D)/(Y-1)$ 乘以 $\mathrm{diag}(p)-pp^\top$。当 $Y=1$ 时该时点的协方差贡献为零。算法参见 [survival 数值实现](https://github.com/therneau/survival/blob/master/src/survdiff2.c)。

汇总各层后，排除期望事件数为零的组并去掉一个冗余对比，解线性方程得到卡方统计量。有效协方差矩阵奇异时抛出 `LinAlgError`。仅一组存在期望事件时返回统计量 0、自由度 0、p 值 1。所有观察均删失时保留 R 的统计量 0 和 p 值 `NaN`，推导出的自由度为 −1。

## 本地对照

```bash
.venv/bin/python -m pytest -q tests/test_survival.py
.venv/bin/python validation/survival/verify.py --random-cases 100
```

R 参考使用本机 R 4.5.3、`survival` 3.8.6 和 `jsonlite`。17 个固定用例覆盖对数秩和非零 `rho`、三组分层、自定义标签顺序、同日删失与多例事件、近似并列、奇异方差、全删失和单样本概率端点。100 个固定随机种子生成的用例进一步覆盖不同大小、组数、分层、权重和近似并列。人数、分组和自由度逐项精确比较。统计量和各矩阵元素使用绝对容差 `5e-13`、相对容差 `1e-11`。p 值用相对容差 `1e-10` 对照。

`survival` 3.8.6 把显式传入的 `timefix` 转发到 `model.frame`，在多行输入上产生长度错误。关闭时间修正的对照用例在 R 包函数的本地副本中将默认值改为 `FALSE`，执行同一函数体中的关闭分支。其他用例使用导出的默认调用。创建或复核固定基准时运行以下命令。

```bash
Rscript --vanilla validation/survival/oracle.R validation/survival/cases.json
```

## 兼容范围

迁移覆盖 `survdiff` 的右删失多样本和单样本统计计算。R 公式、`model.frame`、S3 打印和 `subset` / `na.action` 的表达式处理由上游数据准备替代。R `survdiff` 的输入限制为右删失，计数过程、多状态、区间删失和延迟入组需要其各自的统计方法。寿命表生成的对应方法是 `survival::survexp`，另有独立的迁移工作量。官方资料另列了 [lifelines 的多组加权 log-rank 检验](https://lifelines.readthedocs.io/en/latest/lifelines.statistics.html#lifelines.statistics.multivariate_logrank_test) 和 [statsmodels 的分层两组接口](https://www.statsmodels.org/stable/generated/statsmodels.duration.survfunc.survdiff.html)。
