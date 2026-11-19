# 相关效应量的多层 Meta 回归

`biomed_parity.meta_multilevel.meta_multilevel` 接收效应量及已知抽样协方差，拟合广义最小二乘模型。通过命名随机截距指定研究、终点或中心等随机分组。可以拟合交叉分组，也可以用联合标签表示嵌套分组。该方法的对应 R 入口为 `metafor::rma.mv`。 [metafor 多层模型手册](https://wviechtb.github.io/metafor/reference/rma.mv.html)

## 输入和结果

```python
import numpy as np

from biomed_parity.meta_multilevel import meta_cluster_robust, meta_multilevel

yi = [0.16, 0.32, 0.05, 0.28, 0.54, 0.35, 0.61, 0.76]
vi = np.array([0.04, 0.06, 0.05, 0.05, 0.06, 0.03, 0.04, 0.05])
study = np.repeat(["A", "B", "C", "D"], 2)
dose = [1, 2, 1, 2, 1, 2, 1, 2]

same_study = study[:, None] == study[None, :]
V = 0.25 * np.sqrt(vi[:, None] * vi) * same_study
np.fill_diagonal(V, vi)

fit = meta_multilevel(
    yi, V, moderators=dose,
    random_intercepts={"study": study},
    test="t", dfs="contain", profile=True,
)
robust = meta_cluster_robust(fit, study, correction="CR2")

print(fit.coefficients, fit.inference.confidence_intervals)
print(fit.variance_components, fit.variance_intervals)
print(robust.inference.degrees_of_freedom, robust.omnibus)
```

- 每行对应一个效应量。抽样方差向量表示对角协方差。给定完整矩阵时保留非对角项；抽样协方差可以半正定，最终拟合的边际协方差需要正定。
- `moderators` 可传一列或多列数值设计，默认添加截距。调用前明确类别对照编码。以研究和研究内终点为两层时，传入一个研究标签向量及一个 `(研究, 终点)` 标签向量。交叉项使用独立的标签向量。
- `covariance_components` 可传一个或多个已知的效应量级半正定矩阵，例如按研究标签展开后的已知相关矩阵。乘在每个矩阵上的非负方差系数默认估计，也可以用 `fixed_variances` 固定部分或全部系数。
- `method="REML"` 为默认估计，`method="ML"` 使用完整极大似然。返回固定效应、协方差、边际协方差、残差、对数似然、残余异质性检验和联合检验。默认联合检验涵盖所有非截距系数。可选 `tested_coefficients` 使用从零开始的索引。
- 固定效应支持正态检验、t 检验和 Knapp–Hartung 调整。t 检验可用残差自由度、分组包含规则或显式自由度。包含规则依据随机截距的组内变异判定。具有已知相关核的模型可直接给定自由度。
- `profile=True` 对估计过的方差分量逐一固定候选值，重估其他分量并反解似然比界值。区间是渐近区间；下限可达零。固定方差不出现在剖面区间结果中。 [metafor 区间手册](https://wviechtb.github.io/metafor/reference/confint.rma.html)

`meta_cluster_robust` 支持 CR0、CR1 和 CR2。CR0 和 CR1 按聚类数减去系数数计算自由度。CR2 使用块对角边际协方差作为工作模型，返回 Satterthwaite 单系数检验和近似 Hotelling T² 联合检验。聚类变量需能划分独立组；CR2 会拒绝在组间有实质相关性的边际协方差。 [metafor 稳健推断](https://wviechtb.github.io/metafor/reference/robust.html) · [clubSandwich Meta 分析示例](https://jepusto.github.io/clubSandwich/articles/meta-analysis-with-CRVE.html)

## 本地验证

八个合成方案包括嵌套和交叉随机效应、相关抽样误差、部分固定方差、已知分组相关、边界方差和无截距 GLS。参考环境为 R 4.5.3、metafor 5.0-1、clubSandwich 0.7.0。对照固定效应、模型与边际协方差、似然、异质性、方差区间及稳健推断。交叉方案的最大方差绝对差约为 `7.53e-7`，两端对数似然差约为 `4.1e-11`。使用相对容差 `1.5e-5`，其余方案为 `3e-6`，绝对容差 `4e-8`。另用闭式独立效应模型、等大小聚类和重排行验证。 [共同输入](../../validation/meta_multilevel/cases.json) · [R 快照](../../validation/meta_multilevel/expected.json)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_meta_multilevel.py
```

## Python 候选对照

在同一嵌套和交叉样例中，`pimeta` 1.0.0 已通过固定效应与方差分量对照。嵌套 CR2 标准误与 R 的最大绝对差约为 `1.99e-9`。本次安装的 `rma_mv` 签名缺少固定方差和已知随机项相关矩阵参数。`confint` 仅返回固定效应区间，`profile_rma` 拒绝多层结果对象，`robust` 的返回字段涵盖单系数推断。需方差剖面区间、固定方差、已知相关核或 CR2 联合检验的工作流可使用本实现。 [pimeta 1.0.0](https://pypi.org/project/pimeta/1.0.0/) · [可执行候选审计](../../validation/meta_multilevel/audit_pimeta.py) · [本地审计输出](../../validation/meta_multilevel/pimeta-1.0.0.json)

下一阶段的包级覆盖包括估计随机项间的相关参数、稀疏矩阵与大样本优化、预测区间、约束矩阵和影响诊断。
