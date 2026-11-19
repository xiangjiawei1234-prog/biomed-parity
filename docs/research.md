# 生物医学 R → Python 替代性调研

核查日期为 2026-11-19。调研覆盖生存、诊断、试验设计、临床建模、流行病学及人体表达统计的 16 组方法。每组从 R 官方接口出发，列出无需运行 R 的 Python 候选、已覆盖的能力及待校验的结果字段。用 `rpy2` 等方式调用 R 可继续运行已有分析，随 Python 发布后仍需 R。评估的检索边界是 CRAN、Bioconductor、维护者手册、PyPI、项目仓库和原作者论文。目录记录查到的替代方案与证据，结论限于所列方法、候选及案例。

详见 [方法目录](../catalog/methods.json)。目录中的 `assessment` 表示调研结论，使用 `compatibility_gap` 标记同类工具与 R 的兼容性事项，`verify_existing` 标记优先实测的现成候选，`workflow_gap` 标记核查范围内缺少完整组合的流程。上游文档声明与本地验证按证据来源各自记载。

## 优先级

| 优先级 | 安排依据 | 方法 |
| --- | --- | --- |
| P0 | 生存终点或诊断比较可直接进入临床研究分析，兼容差异会影响区间、检验或结论。输入和输出能用小样本逐项验证。 | `survival::survdiff`、`survRM2::rmst2`、pROC DeLong |
| P1 | 常见于竞争风险、试验设计、预测模型、抽样、缺失数据与证据综合。优先复用已有覆盖，围绕有实质影响的差异建立对照。 | `cmprsk`、`riskRegression::Score`、`gsDesign`、`rms`、`metafor`、`survey`、`mice` |
| P2 | 需多阶段模型、特定随机效应或多矩阵中间结果。按研究设计与参考案例逐组推进。 | `mstate`、`glmmTMB`、`limma`、`edgeR`、`DESeq2`、`variancePartition` |

P0 首批实现的排序权衡临床应用与短周期可验证性。P1 的试验设计和抽样工具同样可能直接影响研究结论，其多配置验证范围较大。多个 Python 候选已存在时，重现方法的优先级取决于现成候选的实测差异。

## 首批兼容缺口

| 方法 | Python 现状与优先验收的 R 行为 | 官方依据 |
| --- | --- | --- |
| `survival::survdiff` | statsmodels 已有分层及 Fleming–Harrington 权重，scikit-survival 的普通 log-rank 可给出分组统计与协方差，lifelines 提供多种权重。仍需核对 R 的加权 `obs`、`exp`、`var` 同时输出、近并列时点和单样本 `offset`。 | [R 手册](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survdiff.html) · [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.duration.survfunc.survdiff.html) · [scikit-survival](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.compare.compare_survival.html) · [lifelines](https://lifelines.readthedocs.io/en/latest/lifelines.statistics.html) |
| `survRM2::rmst2` | lifelines 求截断曲线面积，也可输出截断生存时间的方差。验收时把截断生存时间的方差和 RMST 估计的抽样方差分开核对。CoxRMSTSimpson 对 Cox 预测曲线做积分。R 的两臂 RMST 差值、RMST/RMTL 比值和 ANCOVA 型协变量调整需同口径验证。 | [R 手册](https://search.r-project.org/CRAN/refmans/survRM2/html/rmst2.html) · [lifelines RMST](https://lifelines.readthedocs.io/en/latest/lifelines.utils.html#lifelines.utils.restricted_mean_survival_time) · [CoxRMSTSimpson](https://pypi.org/project/coxrmstsimpson/) |
| pROC DeLong | scikit-learn 提供 AUC 点估计，MLstatkit 支持配对 DeLong，pauc 同时提供配对、不配对比较和 DeLong 区间。本地并列分数案例中，pauc 0.2.1 与 R 的方差和两类比较 p 值有差异。另需覆盖标签及方向、置信水平、备择、完全分离和差值区间。 | [R 比较方法](https://search.r-project.org/CRAN/refmans/pROC/html/roc.test.html) · [R AUC 区间](https://search.r-project.org/CRAN/refmans/pROC/html/ci.auc.html) · [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) · [MLstatkit](https://github.com/Brritany/MLstatkit) · [pauc](https://pypi.org/project/pauc/0.2.1/) · [本地对照](methods/roc-delong.md) |

## P1 临床与流行病学

| 方法 | Python 现状与待验收范围 | 官方依据 |
| --- | --- | --- |
| `cmprsk::cuminc/crr` | pycmprsk 已有原生移植，声明支持 Gray 检验、时变效应和分组删失方差。comprisk 也提供 Gray 检验、累积发生率和 Fine–Gray。本地 pycmprsk 1.0.0 对照发现跃点取值和并列事件残差有差异，两处适配后 82/82 个字段一致。当前 pycmprsk 要求 Python ≥3.13。 | [R 手册](https://cran.r-project.org/web/packages/cmprsk/refman/cmprsk.html) · [pycmprsk](https://github.com/covertcast/pycmprsk) · [comprisk](https://sunnyadn.github.io/comprisk/reference/) · [本地对照](../validation/adoption/pycmprsk/README.md) |
| `riskRegression::Score` | comprisk 的 `score_cr` 提供竞争风险 IPCW 评分和 Bootstrap。仓库已对照两类删失模型、影响函数、配对比较及校准。R 2026.03.11 对非默认目标原因的历史 AUC 推断可通过显式兼容模式复现。 | [R Score](https://search.r-project.org/CRAN/refmans/riskRegression/html/Score.html) · [comprisk 快速开始](https://sunnyadn.github.io/comprisk/quickstart/) · [本地校准](methods/competing-risk-calibration.md) |
| `gsDesign` | sequential-analysis 提供连续、二分类和生存终点的监测及非绑定无效界值，peekproof 提供 Lan–DeMets 花费。剩余对照集中在 R 的条件效能、分层入组、脱落、事件及日历时间。 | [R 边界与设计](https://keaven.github.io/gsDesign/reference/gsDesign.html) · [R 生存效能](https://keaven.github.io/gsDesign/reference/gsSurvPower.html) · [Python gsdesign](https://pypi.org/project/gsdesign/) · [sequential-analysis](https://pypi.org/project/sequential-analysis/) · [peekproof](https://pypi.org/project/peekproof/) |
| `rms::calibrate/validate/Predict` | lifelines 有固定时点生存校准，scikit-learn 有概率校准。本地已对照 rms 二分类的表观与偏差校正曲线、四种重采样及 Hmisc 的限制性三次样条。参考值、交互、预测与列线图工作流仍需验收。 | [R 校准](https://search.r-project.org/CRAN/refmans/rms/html/calibrate.html) · [R 工作流](https://search.r-project.org/CRAN/refmans/rms/html/zzzrmsOverview.html) · [lifelines 校准](https://lifelines.readthedocs.io/en/stable/lifelines.calibration.html) · [scikit-learn 校准](https://scikit-learn.org/stable/modules/calibration.html) · [样条与结点](methods/restricted-cubic-splines.md) · [本地校准曲线](methods/logistic-calibration-curve.md) |
| `metafor::rma.uni/rma.mv` | pimeta 1.0.0 在所测场景复现嵌套或交叉拟合和单系数 CR2。其被测接口缺少多层方差剖面区间、固定方差、已知相关核和 HTZ 联合检验，这些路径已经原生对照。待核对随机效应相关结构和预测。 | [R 多层模型](https://wviechtb.github.io/metafor/reference/rma.mv.html) · [R 稳健推断](https://wviechtb.github.io/metafor/reference/robust.html) · [pimeta](https://pypi.org/project/pimeta/) · [PyMARE](https://pymare.readthedocs.io/en/latest/api.html) · [PyMetaAnalysis](https://zhaoboding.github.io/PyMetaAnalysis/methods/statistical-methods/) · [本地 REML 实现](methods/meta-reml.md) · [本地多层对照](methods/meta-multilevel.md) |
| `survey::svyglm/twophase/svycoxph` | svy 已提供设计校正 GLM、线性化和复制权重及 GREG 校准，并列出与 R 的少数约定差异。先验证通用设计，再处理二阶段设计和复杂抽样 Cox 推断。 | [R 两阶段设计](https://search.r-project.org/CRAN/refmans/survey/html/twophase.html) · [R 抽样加权 Cox](https://search.r-project.org/CRAN/refmans/survey/html/svycoxph.html) · [svy](https://github.com/samplics-org/svy) |
| `mice::mice/pool` | statsmodels 和 miceforest 有链式插补。本地已迁移数值、二元和被动变量的插补链，支持预测矩阵、访问顺序和单元范围，并对照 Rubin 总协方差、D1/D2 和有限数据自由度。多层、多分类及 D3/D4 仍待验证。 | [R 合并规则](https://amices.org/mice/reference/pool.html) · [R 多层插补](https://amices.org/mice/reference/mice.impute.2l.pan.html) · [statsmodels 插补](https://www.statsmodels.org/stable/imputation.html) · [miceforest](https://miceforest.readthedocs.io/en/latest/ImputationKernel.html) · [链式插补](methods/chained-imputation.md) · [联合推断](methods/mi-pooling.md) |

## P2 多状态、联合模型与人体表达统计

| 方法 | Python 现状与待验收范围 | 官方依据 |
| --- | --- | --- |
| `mstate` | PyMSM 支持拟合与轨迹模拟。本地原生 `msprep → msfit → probtrans` 已对照长表、累计转移风险、协方差、地标时点和最终状态概率。后续验证 Markov 检验和转移别扩展。 | [R 函数索引](https://search.r-project.org/CRAN/refmans/mstate/html/00Index.html) · [PyMSM 多状态模型](https://hrossman.github.io/pymsm/api/multistate_model/) · [本地多状态工作流](methods/mstate-fit.md) |
| `glmmTMB` | statsmodels 有有限的广义混合模型，Bambi 可拟合贝叶斯零膨胀模型。复现目标为条件模型、零膨胀和离散度子模型的联合 Laplace 边际似然，以及相同参数尺度下的推断。 | [R glmmTMB](https://search.r-project.org/CRAN/refmans/glmmTMB/html/glmmTMB.html) · [statsmodels](https://www.statsmodels.org/stable/mixed_glm.html) · [Bambi](https://bambinos.github.io/bambi/notebooks/getting_started.html) |
| `limma` | pylimma 在本地复现普通与分块经验贝叶斯流程和矩形输入的 voom 样本质量权重。正方形输入的权重广播差异可显式复现。InMoose 支持普通流程，分块 GLS 尚缺。 | [Bioconductor](https://bioconductor.org/packages/release/bioc/html/limma.html) · [InMoose limma API](https://inmoose.readthedocs.io/en/stable/limma.html) · [pylimma](https://pylimma.readthedocs.io/en/latest/api.html) · [本地对照](../validation/adoption/limma/README.md) |
| `edgeR` | edgePython 0.5.1 在本地整数与低分数计数中复现 edgeR 4.8.2 的 legacy、v4 固定和自动离散度。InMoose 0.9.1 复现 legacy，对 v4 低计数的 p 值最大绝对差为 0.16948。 | [edgeR v4 论文](https://doi.org/10.1093/nar/gkaf018) · [InMoose edgepy API](https://inmoose.readthedocs.io/en/stable/edgepy.html) · [edgePython](https://pypi.org/project/edgepython/0.5.1/) · [本地对照](../validation/adoption/inmoose/README.md) |
| `DESeq2` | PyDESeq2 0.6.0 的两种 Wald 拟合在所测 5% BH 阈值得到相同发现。该版本缺少 local VST，已验证的后续开发提交可以运行。使用同一离散度曲线时，原生局部变换与 R 的最大绝对差小于 `1.2e-13`。 | [R 用户手册](https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html) · [PyDESeq2](https://pydeseq2.readthedocs.io/en/stable/) · [本地对照](methods/deseq2-adoption.md) · [局部 VST 对照](methods/local-vst.md) |
| `variancePartition::dream` | statsmodels 和 InMoose 提供部分模型组件。对照需覆盖精度加权的逐基因混合模型、小样本自由度和经验贝叶斯的联合输出。 | [dream 示例](https://diseaseneurogenomics.github.io/variancePartition/articles/dream.html) · [statsmodels 混合模型](https://www.statsmodels.org/stable/mixed_linear.html) · [InMoose](https://inmoose.readthedocs.io/en/stable/limma.html) |

方法验收按研究设计和输出字段确定。迁移时以相同统计假设、相同输入条件和逐字段输出为准。若候选测得一致，可采用原生接口或轻量适配。若结果分歧，保留输入、双方版本、预期差异和容差，按归因安排补齐。
