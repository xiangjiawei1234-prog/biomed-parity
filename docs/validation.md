# 本地验证记录

2026-11-19 在 macOS arm64 本机运行。主要环境为 Python 3.12.7、NumPy 2.1.3、SciPy 1.14.1、pytest 9.2.0 和 R 4.5.3。实时 R 套件以警告为错误模式运行了 **566 项测试，全部通过**。默认 Python 套件有 540 项通过，26 个实时 R 测试按配置跳过。Python 3.11.15、NumPy 1.26.4 和 SciPy 1.12.0 也通过了同一套默认测试。参考输入使用人工构造数据。不同统计方法的容差、字段和已知差异见相应文档。

| 统计入口 | R 参考 | 覆盖 |
| --- | --- | --- |
| [生存曲线比较](methods/survdiff.md) | survival 3.8-6 | 17 份固定输入及 100 组有种子的随机输入。含分层、近并列事件和单样本参照 |
| [ROC DeLong](methods/roc-delong.md) | pROC 1.19.1 | 28 个 AUC、区间、配对及不配对场景 |
| [部分 AUC 与 Bootstrap](methods/roc-bootstrap.md) | pROC 1.19.1 | 41 个完整/部分面积场景，61 次指定分层重抽样，含配对 Bootstrap 检验 |
| [两臂 RMST](methods/rmst2.md) | survRM2 1.0-4 | 14 组未调整和协变量调整输入 |
| [组序贯设计](methods/group-sequential.md) | gsDesign 3.11.0 | 13 个边界案例，含对称双侧、约束性和非约束性无效边界、越界概率与条件把握度。极端花费及密集期中分析另核对积分误差 |
| [逻辑回归验证](methods/logistic-validation.md) | rms 8.2-1 | 3 份数据及 R 选出的 53 次固定抽样。核对 11 个性能指标 |
| [二分类校准曲线](methods/logistic-calibration-curve.md) | rms 8.2-1，Hmisc 5.3-1 | 5 组输入和 62 次重采样，含普通 Bootstrap、.632、交叉验证与随机化 |
| [限制性三次样条](methods/restricted-cubic-splines.md) | Hmisc 5.3-1，rms 8.2-1 | 19 组固定输入，并在 540 组随机输入上核对自动选结点 |
| [标量及多参数合并](methods/mi-pooling.md) | mice 3.19.0，mitml 0.4-5 | 11 组标量推断与 8 组总协方差、D1 和 D2 方案 |
| [链式插补](methods/chained-imputation.md) | mice 3.19.0 | 3 组确定性链逐值对照及 5 组随机条件模型分布对照 |
| [抽样加权 Cox](methods/survey-cox.md) | survey 4.5 | 5 个方案，核对似然、协方差、区间与设计自由度 |
| [两阶段调查](methods/survey-twophase.md) | survey 4.5 | 10 种设计的均值和总量，含交叉分层、两级整群与普查 |
| [REML Meta 分析](methods/meta-reml.md) | metafor 5.0-1 | 19 个区间和假设检验场景，含 Knapp–Hartung 和异质性剖面似然 |
| [多层 Meta 分析](methods/meta-multilevel.md) | metafor 5.0-1，clubSandwich 0.7.0 | 8 个相关采样及多层场景，含方差剖面区间、CR2 和 HTZ |
| [竞争风险评分](methods/competing-risk-scores.md) | riskRegression 2026.03.11 | 4 个边际 KM 和 5 个反向 Cox 方案。九组对照兼容模式的 AUC 区间与影响函数，包括两个非默认目标原因案例 |
| [竞争风险校准](methods/competing-risk-calibration.md) | riskRegression 和 prodlim 2026.03.11 | 7 组分位数、近邻、局部风险和逐行伪值对照 |
| [多状态准备](methods/mstate-prep.md) 与 [转移风险](methods/mstate-fit.md) | mstate 0.3.3 | 6 组常规宽表转换和 7 组风险拟合，其中 3 组串联验证最终状态概率与标准误 |
| [多状态占用概率](methods/multistate.md) | mstate 0.3.3 | 6 个前向和固定终点场景，含方差及对齐的协方差 |
| [零膨胀随机截距](methods/zip-random-intercept.md) | glmmTMB 1.1.14 | 3 组拟合与给定参数的 Laplace 似然对照。记录本机 TMB 构建版本警告 |
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
