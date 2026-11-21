# biomed-parity

纯 Python（NumPy + SciPy）实现的生物医学统计方法，每个入口附 R 参考脚本、入库数值快照和可选实时 R 对照。

*Independent Python implementations of selected biomedical statistical methods, with reproducible R reference cases and numerical parity checks.*

[![PyPI](https://img.shields.io/pypi/v/biomed-parity.svg)](https://pypi.org/project/biomed-parity/)
[![CI](https://github.com/xiangjiawei1234-prog/biomed-parity/actions/workflows/ci.yml/badge.svg)](https://github.com/xiangjiawei1234-prog/biomed-parity/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

当前开发仓库跟踪 19 组 R 统计入口。各方法的覆盖、统计差异与本地验证记录见 [迁移目录](catalog/progress.json) 和 [验证报告](docs/validation.md)。

## 覆盖与验收

每行对应一组已建立数值对照的入口。"快照覆盖"是入库 R 参考用例的输入与场景；字段定义、容差和已知差异见对应方法页，完整运行记录见 [docs/validation.md](docs/validation.md)。

| 分类 | Python 接口 | R 参考 | 快照覆盖 |
| --- | --- | --- | --- |
| 生存分析 | `survival.survdiff`、`survdiff_one_sample` | survival 3.8-6 | 17 份固定输入 + 100 组固定种子随机输入；rho、分层、近并列事件、单样本参照 |
| 生存分析 | `rmst.rmst2` | survRM2 1.0-4 | 14 组；未调整与 1–2 个协变量的 RMST、RMTL 及差值/比值推断 |
| 生存分析 | `rmst_curves.rmst_curve_data` | survRM2 1.0-4、survival 3.8-6 | 14 组 Kaplan–Meier 阶梯和限制平均生存时间阴影坐标 |
| 生存分析 | `interval_censoring.turnbull` | icenReg 2.0.16 | 8 组区间删失概率质量，含四种端点组合、准确时间、全右删失及权重 |
| 诊断准确性 | `roc.delong_auc`、`delong_paired_test`、`delong_unpaired_test` | pROC 1.19.1 | 28 个场景；AUC 方差、区间、配对与独立检验 |
| 诊断准确性 | `roc_bootstrap.roc_auc`、`bootstrap_auc_ci`、`bootstrap_paired_test` | pROC 1.19.1 | 41 个完整/部分 AUC 场景；61 次指定分层重抽样与配对检验 |
| 诊断准确性 | `roc_thresholds.roc_coords`、`bootstrap_roc_coords` | pROC 1.19.1 | 47 个坐标场景 + 13 个固定重抽样区间场景；8214 行快照，坐标字段逐位一致 |
| 诊断准确性 | `roc_smoothing.smooth_roc` | pROC 1.19.1、logcondens 2.1.9 | 37 个二正态、核密度及给定密度场景。另有 11 个对数凹场景 |
| 诊断准确性 | `roc_parametric.smooth_parametric_roc` | pROC 1.19.1、MASS 7.3-65 | 17 个场景，含正态、对数正态、指数分布与混合分布族 |
| 诊断准确性 | `roc_smooth_bootstrap.bootstrap_smooth_auc_ci`、`bootstrap_smooth_sensitivity_ci` | pROC 1.19.1、MASS 7.3-65 | 18 个场景与 780 次固定重抽样 |
| 诊断准确性 | `roc_smooth_bootstrap.bootstrap_smooth_paired_test` | pROC 1.19.1 | 6 组配对平滑 AUC，核对 283 次固定重抽样 |
| 诊断准确性 | `roc_smooth_bootstrap.bootstrap_smooth_unpaired_test` | pROC 1.19.1、MASS 7.3-65 | 5 组独立样本的平滑 AUC，核对 215 对固定重抽样 |
| 诊断准确性 | `roc_multiclass.multiclass_auc` | pROC 1.19.1 | 23 个情形，含共享分数、类别分数矩阵与 Hand–Till 成对分量 |
| 临床试验设计 | `group_sequential.efficacy_boundaries`、`symmetric_boundaries`、`efficacy_futility_boundaries`、`crossing_probabilities`、`conditional_power` | gsDesign 3.11.0 | 13 个边界方案（HSD/LDOF、单侧/双侧、约束/非约束无效界）；25 组越界概率与条件把握度 |
| 临床试验设计 | `group_sequential_survival.fixed_survival_design`、`group_sequential_survival_design` | gsDesign 3.11.0 | 15 组固定和组序贯设计，含分层入组、脱落、事件数量和分析日历 |
| 临床试验设计 | `group_sequential_inference.repeated_pvalues`、`repeated_confidence_intervals` | gsDesign 3.11.0 | 10 组正态重复推断，含信息时间调整、单侧和对称区间 |
| 临床试验设计 | `group_sequential_binomial` 的精确越界、整数疗效界和 SPRT | gsDesign 3.11.0 | 10 组单样本二项边界、条件越界和伤害停止 |
| 临床预测 | `calibration.validate_logistic` | rms 8.2-1 | 3 份数据、53 次固定重采样、11 个性能指标 |
| 临床预测 | `calibration.calibrate_logistic` | rms 8.2-1、Hmisc 5.3-1 | 5 组输入、62 次重采样；表观与偏差校正曲线、四种重采样、点态近似区间 |
| 临床预测 | `splines.restricted_cubic_spline_knots`、`fit_restricted_cubic_spline` | Hmisc 5.3-1、rms 8.2-1 | 19 组固定输入 + 540 组自动结点；预测重用与主成分旋转 |
| 临床预测 | `rms_prediction.datadist`、`prediction_grid`、`predict` | rms 8.2-1、multcomp 1.4-31 | 12 组数值与分类默认值、参考归零、点态及同时区间 |
| 临床预测 | `nomogram.nomogram` | rms 8.2-1 | 2 组加性和交互模型的轴刻度与分值 |
| 临床预测 | `rms_logistic.fit_penalized_logistic` | rms 8.2-1 | 13 组二元拟合，含惩罚矩阵、加权、偏移量和仅截距 |
| 竞争风险 | `risk_scores.score_competing_risks` | riskRegression 2026.03.11 | 4 组边际 KM + 5 组反向 Cox；AUC、Brier、影响函数、积分 Brier 与配对比较 |
| 竞争风险 | `risk_cross_validation.cross_validate_competing_risks` | riskRegression 2026.03.11 | 3 折与 4 折校验，核对折外风险及汇总评分 |
| 竞争风险 | `competing_calibration.calibrate_competing_risks`、`jackknife_cumulative_incidence` | riskRegression、prodlim 2026.03.11 | 7 组；分位数、近邻、局部风险与 Jackknife 伪值 |
| 竞争风险 | `competing_risks.cmprsk_timepoints`、`cmprsk_residuals_by_time` | cmprsk、pycmprsk | 82/82 项字段适配对照 |
| 多状态 | `mstate_prep.msprep` | mstate 0.3.3 | 6 组宽表转换；左进入、并列、协变量复写 |
| 多状态 | `mstate_covariates.expand_covariates` | mstate 0.3.3 | 5 组转移特异数值列与因子对比 |
| 多状态 | `mstate_markov.markov_test`、`optimal_weights_matrix`、`optimal_weights_multiple` | mstate 0.3.3 | 3 组 log-rank 轨迹、状态特异和联合权重，以及固定野生 Bootstrap 乘子 |
| 多状态 | `mstate_fit.msfit` | mstate 0.3.3 | 7 组风险拟合；Aalen/Greenwood 协方差，3 组串联到状态概率 |
| 多状态 | `multistate.probtrans` | mstate 0.3.3 | 6 组占用概率；固定起点、固定终点、方差与协方差 |
| 多重插补 | `chained_imputation.mice_impute` | mice 3.19.0 | 连续、二元、多分类、两层高斯与被动插补；按逐值对照、条件概率和分布矩验证 |
| 多重插补 | `multilevel_imputation.impute_two_level_normal` | mice 3.19.0、pan 2.0 | 8 组设计逐项对照 + 6 组随机模型矩对照（每侧 2,200 次抽样，5σ 判定） |
| 多重插补 | `multilevel_binary.impute_two_level_binary` | mice 3.19.0、lme4 2.0-1 | 3 组两层二元拟合、3 种簇抽样策略及链式调用 |
| 多重插补 | `chained_imputation.mice_impute` 的 `mnar.norm`、`mnar.logreg` | mice 3.19.0 | 3 组连续和二元结局的非随机缺失偏移 |
| 多重插补 | `imputation.pool_scalar`、`pool_parameters`、`pool_d1`–`pool_d4` | mice 3.19.0、mitml 0.4-5 | 标量、协方差、Wald、统计量及嵌套模型似然的合并 |
| 抽样调查 | `survey_cox.svycoxph` | survey 4.5 | 5 个方案；加权偏似然、Taylor 方差、区间与设计自由度 |
| 抽样调查 | `survey_twophase.survey_twophase` | survey 4.5 | 10 种设计；两阶段均值与总量、分层、整群与有限总体校正 |
| 抽样调查 | `survey_replicates.svrepdesign` | survey 4.5 | 14 组给定重复权重设计，含均值、总量、协方差和调查域 |
| 抽样调查 | `survey_calibration.calibrate`、`calibrate_replicates`、`calibrated_total` | survey 4.5 | 15 组线性、raking 和 logit 校准，含有限界、重复权重和 Taylor 方差 |
| 抽样调查 | `survey_jackknife.as_svrepdesign` | survey 4.5 | 13 组 JK1/JKn 权重生成、有限总体修正和孤立 PSU |
| 抽样调查 | `survey_glm.svyglm` | survey 4.5 | 7 组重复权重 Gaussian 与二元 quasibinomial 回归 |
| 证据综合 | `meta_analysis.meta_reml` | metafor 5.0-1 | 19 组区间与检验；Knapp–Hartung、预测区间、异质性剖面似然 |
| 证据综合 | `meta_multilevel.meta_multilevel`、`meta_cluster_robust` | metafor 5.0-1、clubSandwich 0.7.0 | 8 个嵌套/交叉方案；嵌套与交叉随机项、已知相关核、CR0–CR2、HTZ |
| 证据综合 | `meta_covariance.vcalc` | metafor 5.0-1 | 11 组相关效应抽样协方差，含重复随访、共用对照和正定修正 |
| 证据综合 | `meta_prediction.predict_multilevel_meta` | metafor 5.0-1 | 6 组模型型置信与预测区间 |
| 证据综合 | `meta_influence.meta_influence`、`meta_hatvalues` | metafor 5.0-1 | 4 组逐效应或逐研究删除、方差重估、残差和杠杆值 |
| 证据综合 | `meta_multivariate.meta_multivariate` | metafor 5.0-1 | 12 组，含八种多结局随机协方差及可估计或固定的额外随机截距 |
| 混合模型 | `zero_inflated.fit_zip_random_intercept`、`zip_random_intercept_loglikelihood` | glmmTMB 1.1.14 | 3 组拟合与给定参数的 Laplace 对数似然 |
| 混合模型 | `zero_inflated.fit_zinb2_random_intercept`、`zinb2_random_intercept_loglikelihood` | glmmTMB 1.1.14 | 2 组 NB2、离散度回归与暴露量 |
| 混合模型 | `count_predictions.predict_count_mixed`、`simulate_count_mixed` | glmmTMB 1.1.14 | 2 组 Poisson/NB2 点预测和模拟分布矩 |
| 表达统计 | `dream.dream_random_intercept` | variancePartition 1.40.2、lmerTest 3.2-1 | 3 组逐特征加权 Gaussian REML 与 Satterthwaite 对比 |
| 表达统计 | `nbinom_lrt.nbinom_lrt` | DESeq2 1.50.2 | 2 个嵌套矩阵设计；14 个可检验特征及两个全零特征 |
| 表达统计 | `vst.local_vst`、`local_vst_grid` | DESeq2 1.50.2、locfit 1.5.9.12 | 5 组给定离散度趋势与归一化因子的变换，含零计数和 R 拟合曲线 |

## 用法

**生存曲线比较**

```python
from biomed_parity.survival import survdiff

result = survdiff(time, event, group, rho=0)          # 可选 strata=分层标签
result.statistic, result.df, result.pvalue
result.observed, result.expected, result.variance       # 分组观测/期望与协方差矩阵
```

**组序贯设计**

```python
from biomed_parity.group_sequential import efficacy_boundaries

boundaries = efficacy_boundaries([0.25, 0.5, 0.75, 1.0], alpha=0.025, spending="hsd")
boundaries.z_boundaries, boundaries.nominal_pvalues, boundaries.cumulative_alpha
```

**竞争风险评分（反向 Cox 删失、积分 Brier）**

```python
import numpy as np
from biomed_parity.risk_scores import score_competing_risks

scores = score_competing_risks(
    time, status, horizons, risks,           # risks: 模型数 × 样本数 × 时点数
    cause=1,
    censoring_covariates=covariates,         # 可选；缺省为边际逆 KM
)
scores.auc.estimate, scores.brier.estimate, scores.integrated_brier
```

**链式插补与参数合并**

```python
import numpy as np
from biomed_parity.chained_imputation import mice_impute
from biomed_parity.imputation import pool_parameters

result = mice_impute(
    data,                                    # 缺失值用 np.nan
    methods=[None, "pmm", "norm"],
    predictor_matrix=...,                    # 行=待插补列，列=预测变量
    m=20, iterations=10, seed=2026,
)
estimates = result.completed[:, :, 0].mean(axis=0)
pooled = pool_parameters(estimates, covariances, complete_df=...)
```

每个方法的完整接口、数据约定和限制见 [docs/methods](docs/methods)。

## 验证方法

- `validation/<方法>/` 保存人工构造输入、`oracle.R` 参考脚本和 JSON/TSV 快照；快照记录生成时的 R 与 R 包版本。
- 默认测试读取入库快照，在只有 NumPy/SciPy 的环境中运行；`BIOMED_R_PARITY=1` 时调用本地 R 逐项实时对照。
- 确定性数值按输出尺度和算法确定容差，例如 survdiff 统计量 `5e-13`、DeLong 与 Bootstrap 面积 `2e-14`、链式插补矩阵 `6e-13`、组序贯越界概率 `5e-10`、多层 meta 参数 `4e-8`。
- 随机算法不比较 R 与 NumPy 的随机数流：Bootstrap 使用固定重抽样矩阵，插补用多组独立抽样的矩与蒙特卡洛误差检验；逐值一致只用于确定性路径。
- 实现与 R 存在有意差异时，用独立用例显式断言并记录在方法页，不把差异算作一致。

```sh
.venv/bin/python -m pytest -q                                          # 默认：只读快照
Rscript --vanilla validation/install_r.R                               # 安装 R 对照依赖到 .r-lib
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error
```

## 已知差异

- riskRegression 2026.03.11 的 AUC 影响函数在非默认目标原因下固定使用第一/第二事件编码。`score_competing_risks(..., auc_inference="riskregression")` 复现该行为；默认 `"cause"` 按请求的目标原因组成病例与对照。
- survey 4.5 在一个两阶段用例中按原始行位置取层样本数，导致总量协方差非对称。Python 使用完整抽样框计数，并验证结果对入样行排列不变。
- 随机插补与 Bootstrap 和 R 的随机数流不同，同种子不保证逐值一致；对照使用固定索引或独立抽样的矩比较。
- 不提供 R 对象结构、公式接口、S3 方法与绘图；这些属于 R 语言层接口，本包只对齐数值行为与统计口径。
- 零膨胀模型的本机 TMB 构建版本警告记录在对应方法页；不影响似然对照。

## 与现成 Python 候选的关系

项目对可复用的第三方实现做了同输入对照，结果记录在 `validation/adoption/` 与各方法页：

- `pycmprsk` 经两个小型数组适配后与 cmprsk 在 82/82 项字段上一致。
- `PyDESeq2` 在 12 样本 × 240 匿名特征上与 R 得到相同的 Wald 显著集合；`edgePython`/`InMoose` 在 4,032 个比较值上通过。
- `pylimma`/`InMoose` 覆盖普通拟合、分块随机相关与质量权重；方阵权重的兼容做法单列。
- `PyMetaAnalysis`、`pimeta` 在所测场景中复现 REML、Knapp–Hartung 与嵌套 CR2 标准误。

## 安装

```sh
python3 -m pip install biomed-parity
```

开发或验证时从源码安装：

```sh
git clone https://github.com/xiangjiawei1234-prog/biomed-parity.git
cd biomed-parity
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

## 项目结构

| 目录 | 内容 |
| --- | --- |
| `src/biomed_parity/` | Python 实现；运行时不调用 R |
| `catalog/` | 方法目录（分类、R 入口、许可、候选）与迁移进展 |
| `docs/methods/` | 每个入口的估计目标、数据约定、结果字段、示例、验证与限制 |
| `docs/research.md`、`docs/development.md` | 调研优先级与验收流程 |
| `validation/<方法>/` | 人工构造输入、R oracle、参考快照与外部候选审计 |
| `tests/` | 默认回归测试与可选实时 R 对照 |

## 范围与限制

- 数值基线对应固定版本的 R 与 R 包，升级上游包后需重新生成快照并复核差异。
- 项目按方法入口逐个验收，未覆盖的路径见 [catalog/progress.json](catalog/progress.json) 与对应方法页。
- 参考输入均为人工构造或匿名数据，不包含真实患者数据。
- 上游 R 包仅作外部数值参考，未复制或翻译其源码；实现依据论文与公开接口描述独立完成，来源记录在 `catalog/methods.json`。
- 研究原型，不构成经监管验证的临床统计软件；用于正式分析前应自行核对结果。

## 贡献

问题反馈与代码贡献通过 Issue 和 Pull Request 提交，流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。发布流程见 [RELEASING.md](RELEASING.md)。

## 许可

[MIT](LICENSE) © 2026 xiangjiawei1234-prog
