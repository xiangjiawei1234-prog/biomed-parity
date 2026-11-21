# 本地验证记录

2026-11-21 在 macOS arm64 本机运行。主要环境为 Python 3.12.7、NumPy 2.1.3、SciPy 1.14.1、pytest 9.2.0 和 R 4.5.3。在 0.3.0 发布提交上开启实时 R 对照并将 Python 警告视为错误，**1,384 项测试全部通过**。默认套件有 1,316 项通过，68 个实时 R 测试按配置跳过。该版本的 wheel 和源码包构建成功。下面列有后续方法的专项验证，参考输入使用人工构造数据。

| 统计入口 | R 参考 | 覆盖 |
| --- | --- | --- |
| [生存曲线比较](methods/survdiff.md) | survival 3.8-6 | 17 份固定输入及 100 组有种子的随机输入。含分层、近并列事件和单样本参照 |
| [ROC DeLong](methods/roc-delong.md) | pROC 1.19.1 | 28 个 AUC、区间、配对及不配对场景 |
| [部分 AUC 与 Bootstrap](methods/roc-bootstrap.md) | pROC 1.19.1 | 41 个完整/部分面积场景，61 次指定分层重抽样，含配对 Bootstrap 检验 |
| [ROC 坐标与阈值区间](methods/roc-thresholds.md) | pROC 1.19.1 | 47 个坐标场景与 13 个固定重抽样区间场景；8214 行快照，坐标字段逐位一致 |
| [平滑 ROC](methods/roc-smoothing.md) | pROC 1.19.1 | 37 个场景，含二正态、七种核密度、给定密度表、完整与部分 AUC |
| [对数凹 ROC](methods/roc-smoothing.md#对数凹密度-roc) | pROC 1.19.1、logcondens 2.1.9 | 11 组，覆盖原估计和高斯卷积、完整与部分 AUC 及常数分数拒绝 |
| [参数分布平滑 ROC](methods/roc-parametric.md) | pROC 1.19.1、MASS 7.3-65 | 17 个场景，含正态、对数正态、指数及混用分布 |
| [平滑 ROC 的 Bootstrap 推断](methods/roc-smooth-bootstrap.md) | pROC 1.19.1、MASS 7.3-65 | 18 个场景和 780 次固定抽样，含 AUC 和指定特异度的灵敏度区间 |
| [配对平滑 ROC 检验](methods/roc-smooth-bootstrap.md#配对平滑-auc-检验) | pROC 1.19.1 | 6 组与 283 次固定重抽样，核对完整与部分 AUC、标准误、检验与遗漏数 |
| [独立样本平滑 ROC 检验](methods/roc-smooth-bootstrap.md#独立样本平滑-auc-检验) | pROC 1.19.1、MASS 7.3-65 | 5 组与 215 对独立重抽样，核对标准误、三种备择和 22 次联合遗漏 |
| [对数凹平滑推断](methods/roc-smooth-bootstrap.md#对数凹密度推断) | pROC 1.19.1、logcondens 2.1.9 | 8 组、354 次固定抽样，含区间、两种样本设计和 38 次联合遗漏 |
| [多分类 AUC](methods/roc-multiclass.md) | pROC 1.19.1 | 23 个场景，核对总体平均、类别对、方向和类别特异评分 |
| [两臂 RMST](methods/rmst2.md) | survRM2 1.0-4 | 14 组未调整和协变量调整输入 |
| [RMST 曲线数据](methods/rmst-curves.md) | survRM2 1.0-4、survival 3.8-6 | 14 组阶梯、阴影坐标及 Kaplan–Meier 点态区间 |
| [组序贯设计](methods/group-sequential.md) | gsDesign 3.11.0 | 13 个边界案例，含对称双侧、约束性和非约束性无效边界、越界概率与条件把握度。极端花费及密集期中分析另核对积分误差 |
| [生存终点样本量与日历](methods/group-sequential-survival.md) | gsDesign 3.11.0 | 15 个固定和组序贯场景，含分层入组、不同脱落率、事件目标和日历反算 |
| [组序贯重复推断](methods/group-sequential-inference.md) | gsDesign 3.11.0 | 10 组重复及序贯 p 值与正态重复区间 |
| [精确二项组序贯](methods/group-sequential-binomial.md) | gsDesign 3.11.0 | 10 组越界、条件概率、伤害、整数疗效界和 SPRT |
| [逻辑回归验证](methods/logistic-validation.md) | rms 8.2-1 | 3 份数据及 R 选出的 53 次固定抽样。核对 11 个性能指标 |
| [二分类校准曲线](methods/logistic-calibration-curve.md) | rms 8.2-1，Hmisc 5.3-1 | 5 组输入和 62 次重采样，含普通 Bootstrap、.632、交叉验证与随机化 |
| [限制性三次样条](methods/restricted-cubic-splines.md) | Hmisc 5.3-1，rms 8.2-1 | 19 组固定输入，并在 540 组随机输入上核对自动选结点 |
| [预测网格与置信带](methods/rms-prediction.md) | rms 8.2-1、multcomp 1.4-31 | 12 组数值与名义因子、样条、参考归零、Cox 定中心、点态与同时区间 |
| [列线图分值](methods/nomogram.md) | rms 8.2-1 | 2 组样条、分类交互及线性和结局轴分值 |
| [二元惩罚似然拟合](methods/rms-logistic.md) | rms 8.2-1 | 13 组系数、惩罚、协方差、有效自由度、加权和偏移量 |
| [惩罚网格选择](methods/rms-penalty.md) | rms 8.2-1 | 4 组共 25 个候选点，核对 AIC/BIC/AICc、所选行及 Gray 协方差 |
| [标量及多参数合并](methods/mi-pooling.md) | mice 3.19.0，mitml 0.4-5 | 11 组标量推断、8 组总协方差及 D1/D2、4 组 D3/D4 嵌套似然 |
| [链式插补](methods/chained-imputation.md) | mice 3.19.0、nnet 7.3-20、MASS 7.3-65 | 3 组确定性链逐值对照、5 组原条件模型分布及 6 组多分类概率 |
| [两层条件插补](methods/multilevel-imputation.md) | mice 3.19.0、pan 2.0 | 8 组设计逐项、6 组独立抽样分布矩，以及两种模型在完整链式入口的均值 |
| [两层二元插补](methods/multilevel-binary.md) | mice 3.19.0、lme4 2.0-1 | 3 组拟合及三种随机效应策略的均值和协方差 |
| [非随机缺失插补](methods/mnar-imputation.md) | mice 3.19.0 | 3 组连续和二元偏移，对照抽样均值及固定种子下偏移的作用 |
| [抽样加权 Cox](methods/survey-cox.md) | survey 4.5 | 5 个方案，核对似然、协方差、区间与设计自由度 |
| [两阶段调查](methods/survey-twophase.md) | survey 4.5 | 10 种设计的均值和总量，含交叉分层、两级整群与普查 |
| [重复权重调查设计](methods/survey-replicates.md) | survey 4.5 | 14 组均值、总量、复制估计、协方差、标准误和自由度 |
| [调查权重校准](methods/survey-calibration.md) | survey 4.5 | 15 组线性、raking 和 logit，有界、重复权重和一阶段校准总量 |
| [调查 jackknife 权重生成](methods/survey-jackknife.md) | survey 4.5 | 13 组 JK1/JKn、有限总体修正、孤立 PSU、均值与总量 |
| [重复权重 GLM](methods/survey-glm.md) | survey 4.5 | 7 组 Gaussian 和二元 quasibinomial，核对 17 个结果字段 |
| [REML Meta 分析](methods/meta-reml.md) | metafor 5.0-1 | 19 个区间和假设检验场景，含 Knapp–Hartung 和异质性剖面似然 |
| [多层 Meta 分析](methods/meta-multilevel.md) | metafor 5.0-1，clubSandwich 0.7.0 | 8 个相关采样及多层场景，含方差剖面区间、CR2 和 HTZ |
| [相关效应的抽样协方差](methods/meta-covariance.md) | metafor 5.0-1 | 11 组含命名相关、重复时点、共用臂、正定修正及显式相关块 |
| [多层模型预测](methods/meta-prediction.md) | metafor 5.0-1 | 6 组平均效应和预测区间，含逐行自由度、Bonferroni 和新观测误差 |
| [多层模型影响诊断](methods/meta-influence.md) | metafor 5.0-1 | 4 组逐行或逐研究删除、Cook 距离、DFBETAS、残差与杠杆值 |
| [多结局随机效应协方差](methods/meta-multivariate.md) | metafor 5.0-1 | 12 组，共八种结构，核对随机协方差、额外随机截距、系数、似然和 Wald 推断 |
| [Royston–Parmar 样条分布](methods/royston-parmar.md) | flexsurv 2.3.2 | 10 组分布与 9 组分位数，含三种链接、两种时间尺度与逐行参数 |
| [区间删失非参数生存](methods/interval-censoring.md) | icenReg 2.0.16 | 8 组开闭端点、最大交集、质量和似然 |
| [区间删失半参数回归](methods/interval-regression.md) | icenReg 2.0.16 | 6 组比例风险和比例优势，核对权重、系数、似然及生存 |
| [区间删失病例 Bootstrap](methods/interval-bootstrap.md) | icenReg 2.0.16 | 2 组 26 次成功重拟合及 1 次失败，核对系数、协方差与 Wald 推断 |
| [竞争风险评分](methods/competing-risk-scores.md) | riskRegression 2026.03.11 | 4 个边际 KM 和 5 个反向 Cox 方案。九组对照兼容模式的 AUC 区间与影响函数，包括两个非默认目标原因案例 |
| [竞争风险交叉验证](methods/risk-cross-validation.md) | riskRegression 2026.03.11 | 3 折与 4 折真实重拟合，核对折外预测、评分和区间 |
| [竞争风险袋外验证](methods/risk-bootstrap.md) | riskRegression 2026.03.11 | 3 组、21 次重拟合，核对袋外预测、评分、差值和 Bootstrap 区间 |
| [竞争风险重抽样平均评分](methods/risk-resampled.md) | riskRegression 和 prodlim 2026.03.11 | 15 次留一与 3 组重复四折，核对留一 Brier、留对 AUC 与积分 Brier |
| [竞争风险校准](methods/competing-risk-calibration.md) | riskRegression 和 prodlim 2026.03.11 | 7 组分位数、近邻、局部风险和逐行伪值对照 |
| [多状态准备](methods/mstate-prep.md) 与 [转移风险](methods/mstate-fit.md) | mstate 0.3.3 | 6 组常规宽表转换和 7 组风险拟合，其中 3 组串联验证最终状态概率与标准误 |
| [转移特异协变量](methods/mstate-covariates.md) | mstate 0.3.3 | 5 组数值、分类或逻辑变量展开，含缺失值与短名 |
| [多状态 Markov 检验](methods/mstate-markov.md) | mstate 0.3.3 | 3 组轨迹、协方差、统计量、野生 Bootstrap 和最优权重 |
| [多状态占用概率](methods/multistate.md) | mstate 0.3.3 | 6 个前向和固定终点场景，含方差及对齐的协方差 |
| [零膨胀随机截距](methods/zip-random-intercept.md) | glmmTMB 1.1.14 | 3 组拟合与给定参数的 Laplace 似然对照。记录本机 TMB 构建版本警告 |
| [零膨胀 NB2 随机截距](methods/zinb2-random-intercept.md) | glmmTMB 1.1.14 | 2 组拟合、离散度回归、给定参数似然与独立混合特例 |
| [零膨胀计数的预测模拟](methods/count-predictions.md) | glmmTMB 1.1.14 | 2 组已知簇、总体及新簇预测，比较零比例、均值和模拟协方差 |
| [带权 Gaussian 随机截距](methods/dream.md) | variancePartition 1.40.2，lmerTest 3.2-1 | 3 组逐特征 REML 和 Satterthwaite 对照 |
| [负二项 LRT](methods/nbinom-lrt.md) | DESeq2 1.50.2 | 2 个嵌套矩阵设计。14 个可检验特征及两个全零特征 |
| [局部方差稳定化变换](methods/local-vst.md) | DESeq2 1.50.2 | 5 组已给定离散度趋势与归一化因子的变换，含零计数和 R 拟合曲线 |

## 现成 Python 候选

- [pycmprsk](../validation/adoption/pycmprsk/README.md) 与 cmprsk 的同一输入对照。两个小型数组适配把 68/82 提升到 82/82 项。
- [pAUC](methods/roc-delong.md) 与 pROC 对照发现并列分数的方差和 p 值差异，独立检验的参考分布也有差异。
- [PyMetaAnalysis](methods/meta-reml.md) 在所测场景中复现 REML 和 Knapp–Hartung。Q-profile 与 R 的剖面似然区间分开记录。
- [pimeta](methods/meta-multilevel.md) 复现所测嵌套、交叉拟合与单系数 CR2。固定方差、已知相关核、方差剖面区间与 HTZ 联合检验另有原生对照。
- [edgePython 与 InMoose](../validation/adoption/inmoose/README.md) 对照 edgeR，有效样本为两个 96 × 8 矩阵。edgePython 0.5.1 在 v4、legacy、固定和自动离散度场景的 4,032 个比较值通过。
- [pylimma、InMoose 与 edgePython](../validation/adoption/limma/README.md) 对照普通拟合、分块随机相关和质量权重。方阵权重的兼容做法单列。
- [PyDESeq2](methods/deseq2-adoption.md) 在 12 样本、240 匿名特征的合成矩阵上与 R 得到相同的两套 Wald 显著集合。报告另含 VST 的差异和 LRT 接口核对。

## 复现

在仓库根目录执行以下几步。R 安装脚本补齐本地参考包并输出版本。参考快照已记录生成时的 R 与包版本。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r validation/requirements-python312.txt
.venv/bin/python -m pip install -e .
Rscript --vanilla validation/install_r.R
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error
```

省略 `BIOMED_R_PARITY=1` 可在仅有 Python 的环境中比较入库 R 快照。对 R 参考包的升级分别运行实时对照，检查异常行为和数据差异再更新参考快照。
