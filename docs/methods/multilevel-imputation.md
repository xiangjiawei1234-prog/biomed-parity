# 多层正态条件插补

`biomed_parity.multilevel_imputation.impute_two_level_normal` 为链式插补中的单个连续缺失列提供两层高斯条件模型，对应 mice 的 `mice.impute.2l.pan` 与 `mice.impute.2l.norm`。调用方提供响应、预测变量矩阵和 `types` 编码，得到“抽样次数 × 待插补行”的插补值矩阵，行序与输入数据一致。 [mice 2l.pan](https://amices.org/mice/reference/mice.impute.2l.pan.html) · [mice 2l.norm](https://amices.org/mice/reference/mice.impute.2l.norm.html) · [pan 包](https://cran.r-project.org/package=pan)

```python
import numpy as np
from biomed_parity.multilevel_imputation import TwoLevelOptions, impute_two_level_normal

# 每行一个观测；第 0 列为 cluster，其后为固定/随机预测变量
predictors = np.array([
    [1, -0.8, 0], [1, -0.3, 1], [2, 0.4, 0], [2, 1.1, 1],
    [3, -1.2, 1], [3, -0.1, 0],
])
response = np.array([2.1, np.nan, 3.4, np.nan, 0.8, 1.9])
types = np.array([-2, 2, 1])  # cluster，x 为带总体均值的随机斜率，z 为固定效应

draws = impute_two_level_normal(
    response, predictors, types,
    options=TwoLevelOptions(iterations=100),
    m=5, seed=17,
)
print(draws.shape)  # (5, 2)
```

也可直接在 `mice_impute` 中使用两层模型，让同一链的其他方法读取该列当前的插补值。预测矩阵的一行对应一个目标列，`two_level_options` 以零起点的目标列索引指定配置。 [链式插补](chained-imputation.md)

```python
from biomed_parity.chained_imputation import mice_impute

matrix = np.column_stack((predictors, response))
codes = np.zeros((4, 4), dtype=int)
codes[3, :3] = types
result = mice_impute(
    matrix, methods=[None, None, None, "2l.pan"], predictor_matrix=codes,
    two_level_options={3: TwoLevelOptions(iterations=100)}, m=5, seed=17,
)
```

## 接口与掩码

`impute_two_level_normal(y, predictors, types, *, method="2l.pan", observed=None, where=None, options=TwoLevelOptions(iterations=None, intercept=True, groupcenter_slope=False, ridge=1e-4), m=5, seed=None)`

| 参数 | 含义 |
| --- | --- |
| `y` | 长度为样本量的响应；缺失值用 `numpy.nan`，不允许 `inf` |
| `predictors` | 样本量 × 预测变量矩阵，包含 cluster 列；不允许 `inf` |
| `types` | 与 `predictors` 列一一对应的编码，见下表 |
| `method` | `"2l.pan"` 或 `"2l.norm"` |
| `observed` | 训练行布尔掩码，默认 `numpy.isfinite(y)`；标为 `False` 的行不参与模型拟合 |
| `where` | 需要输出插补值的行布尔掩码，默认 `~observed`；标为 `True` 的已观测行会被重新插补 |
| `options.iterations` | 内层 Gibbs 扫描次数，默认 `2l.pan` 为 500、`2l.norm` 为 100，与 mice 的 `paniter` 与硬编码 `n.iter` 一致 |
| `options.intercept` | 是否加入固定截距（`2l.pan`）或随机截距（`2l.norm`），默认 `True` |
| `options.groupcenter_slope` | 仅 `2l.pan`：编码 3/4 的预测变量是否减去簇均值，默认 `False` |
| `options.ridge` | 仅 `2l.norm`：`mice:::symridge` 式的对角乘子，默认 `1e-4`，与 mice 一致 |
| `m`、`seed` | 独立链数与 NumPy `SeedSequence` 种子；`seed=None` 时不可复现 |

`observed` 为 `False` 的行不提供似然信息，但仍可留在随机效应结构中；`where` 独立指定输出行。`where` 与 `observed` 的交集语义使该入口可以直接实现过度插补和“训练集留出”。返回值只包含 `where` 指定行的插补值，不含完整数据矩阵。

## types 编码与 mice predictorMatrix 的对应

mice 在 `predictorMatrix` 的第 j 行（目标列）用整数编码预测变量的统计角色；本模块接收其中非零项组成的 `types` 向量。两者的取值一一对应。 [mice 预测矩阵](https://amices.org/mice/reference/make.predictorMatrix.html)

| `types` | mice 编码 | 2l.pan | 2l.norm |
| --- | --- | --- | --- |
| `-2` | `-2` | cluster/分类变量，必须恰有一列且完整 | 同左 |
| `0` | `0` | 不使用该列 | 不使用该列 |
| `1` | `1` | 固定效应 | 拒绝（见已知差异） |
| `2` | `2` | 固定效应 + 带总体均值的随机斜率 | 带总体均值的随机斜率 |
| `3` | `3` | 固定效应 + 固定簇均值（`groupcenter_slope` 时原列减去簇均值） | 拒绝 |
| `4` | `4` | 固定效应 + 固定簇均值 + 随机斜率（`groupcenter_slope` 时原列减去簇均值） | 拒绝 |

例如 `types=[-2, 2, 1]` 等价于

```r
pm <- matrix(0, nrow = ncol(data), ncol = ncol(data), dimnames = list(names(data), names(data)))
pm["y", c("cluster", "x", "z")] <- c(-2, 2, 1)
mice(data, m = 5, method = "2l.pan", predictorMatrix = pm, seed = 17)
```

R 的 `intercept=TRUE` 会把截距以编码 2 加入模型（`2l.pan` 中同时作为固定与随机截距，`2l.norm` 中作为随机截距）；本模块的 `options.intercept` 对应这一行为。`groupcenter_slope` 对应 `mice.impute.2l.pan` 的同名参数。簇均值使用全部行中该预测变量的有限值计算，与 `mice:::.mice.impute.2l.groupmean` 的 `mean(..., na.rm = TRUE)` 语义一致。

## 抽样算法

**2l.pan。** 模型为 `y_ij = x_ij'β + z_ij'b_i + e_ij`，`b_i ~ N(0, Ψ)`，标量残差 `e_ij ~ N(0, σ²)`。mice 传给 `pan::pan` 的先验为：`β` 平坦先验、`σ² ~ Inv-χ²(1, 1)`、`Ψ ~ Inv-Wishart(q, I_q)`（`q` 为随机效应维数）。模块使用独立 NumPy 随机数流逐次扫描：`b_i | β, σ², Ψ`、`β | b, σ²`、`σ² | 残差`、`Ψ | b`，最后从 `x'β + z'b + e` 抽取插补值。固定效应列在拟合前做均值/范数重标定，只是平坦先验下的仿射重参数化，不改变预测分布。每条链从同一确定性起点（观测行 OLS、残差方差、`Ψ=I`）出发，默认 500 次扫描（mice `paniter` 默认值）。

**2l.norm。** 模型为 `y_ij = x_ij'b_i + e_ij`，`b_i ~ N(μ, Ψ)`，残差为簇特异方差 `e_ij ~ N(0, σ_i²)`；`σ_i²` 通过共享的 `σ²₀` 与 `θ` 的层次先验收缩。模块与 mice 源码使用相同的更新顺序与分布：每个簇的系数（含总体均值 `μ`）、`Ψ`、簇特异精度、`σ²₀`、`θ`，扫描次数固定为 100（mice 的 `n.iter` 硬编码值），最后一个扫描状态用于生成 `x'b_i + e` 插补值。`ridge` 用于 `mice:::symridge` 式的对称正定修正，默认 `1e-4`。

单列入口使用 `np.random.SeedSequence(seed).spawn(m)` 分配独立随机数流。链式入口在每条链中沿用已有的生成器，并在每轮重新拟合该列。R 和 NumPy 的随机抽样通过分布矩核对。

## 验证范围

验证材料位于 `validation/multilevel_imputation/`：`cases.json` / `expected.json` 记录 R 生成的确定性设计，`distribution_cases.json` / `distribution_expected.json` 记录多抽样矩，`oracle.R` 可从用例重新生成全部参考值。参考环境为 R 4.5.3、mice 3.19.0、pan 2.0。

**确定性子集逐值对照。** 对 8 个配置（`2l.pan` 5 个、`2l.norm` 3 个）逐一调用真实的 mice 核；`2l.pan` 通过对 `pan::pan` 的调用参数做 trace 捕获固定效应设计矩阵（含截距、类型 3/4 的簇均值列与 `groupcenter_slope` 的中心化）以及随机效应设计矩阵，`2l.norm` 按 R 源码的 `x[, type == 2]` 语义取得随机设计。固定效应设计按与 R 相同的仿射列空间对照，随机设计逐列对照。全部配置的最大绝对偏差为 `5.8e-15`（固定效应列空间重构残差）、`5.4e-15`（随机效应列）。另有一个簇数不足的 `2l.norm` 用例，R 与 Python 都报错（Python 提前以 `ValueError` 拒绝），并核对 `2l.pan` 的 `imputed_rows` 行号与 `n_imputed` 计数。

**随机模型多抽样矩对照。** 6 个用例各由 R 生成 2,200 份独立插补（`mice.impute.2l.pan` / `mice.impute.2l.norm` 直接调用，R 侧保持 mice 默认 `paniter=500`、`n.iter=100`），Python 同样生成 2,200 份独立抽样，逐待插补行比较均值、标准差与目标行协方差。判定阈值为 5 倍联合蒙特卡洛标准误（均值与标准差用两实现的抽样标准误合成，协方差用 R 的协方差蒙特卡洛标准误与 Python 的对乘积抽样标准误合成）。为控制测试时长，Python 的 `2l.pan` 用例使用 15 次扫描、`2l.norm` 用例使用 50 次扫描；提高扫描次数的复核（`2l.pan` 100 次、`2l.norm` 100 次）未改变结论。实测最大偏差如下（`|z|` 已按联合标准误归一化）：

| 用例 | 模型 | 目标行 | Python 扫描 | 最大 \|z\|（均值 / 标准差 / 协方差） |
| --- | --- | --- | --- | --- |
| `pan-random-slope` | 2l.pan，类型 2+1 | 8 | 15 | 2.58 / 1.47 / 3.27 |
| `pan-groupcenter-slope` | 2l.pan，类型 3+4，中心化 | 8 | 15 | 2.04 / 1.89 / 1.88 |
| `pan-no-intercept` | 2l.pan，仅随机斜率 | 8 | 15 | 1.45 / 2.95 / 2.97 |
| `pan-random-intercept` | 2l.pan，无随机斜率 | 8 | 15 | 1.71 / 1.82 / 2.16 |
| `pan-masked` | 2l.pan，`observed`/`where` 掩码 | 5 | 15 | 0.95 / 1.04 / 1.69 |
| `norm-random-slope` | 2l.norm，随机截距+斜率 | 7 | 50 | 1.38 / 2.77 / 2.34 |

全部目标行均落在 5σ 联合蒙特卡洛误差内，最大实测 `|z|` 为 3.27。目标行跨簇且各簇后验均值不同，逐行矩对照同时验证了输出行序。

```sh
# 默认回归（读取快照，约 20 秒）
.venv/bin/python -m pytest tests/test_multilevel_imputation.py -q

# 实时 R 对照（重新运行 oracle.R，约 1 分钟）
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest tests/test_multilevel_imputation.py -q -W error
```

## 已知差异与未覆盖范围

- **随机数流。** R 使用 R/pan 的随机数发生器，Python 使用 NumPy 的 PCG64；只能对齐分布，不能逐值对齐随机抽样。
- **2l.pan 内层扫描的参数化。** R 的 `pan` 把 `(σ², β)` 作为一个块更新（`σ²` 先对 `β` 边缘化，自由度 `a + n_obs - p`），Python 按 `β` 已知的条件更新（自由度 `a + n_obs`）；两者是同一后验下的不同 Gibbs 分块，平稳分布一致，但有限扫描次数下的瞬态不同。模块的更新顺序、初值与 R 也不完全相同。验证通过提高扫描次数和多抽样矩对照确认平稳分布一致，未追求逐扫描数值一致。
- **掩码语义。** 真实 mice 的 `mice.impute.2l.pan` 不接受自定义 `where`/`ignore` 矩阵（会以“number of items to replace is not a multiple of replacement length”告警并错位），因此 `observed`/`where` 掩码通过“把训练外响应置为 NA、只比较目标行”的方式与 R 对照；对已观测行过度插补没有 R 对应物，仅验证 Python 自身契约。
- **2l.norm 的编码限制。** mice 的 `2l.norm` 只使用编码 2 的预测变量，遇到编码 1/3/4 会静默忽略；本模块明确以 `ValueError` 拒绝，避免静默丢弃预测变量。簇数要求也不同：mice 在 `n.class - n.rc - 1 <= 0` 时进入数值失败（如 2 簇时 `chol` 报非正定），本模块提前要求 `簇数 > 2 × 随机效应维数`。两者都是报错，但错误类型与触发条件不同。
- **输入完备性。** 目标行预测变量不完整时本模块报错；mice 会静默跳过这些行，令完整数据保留 `NA`。mice 允许单簇 `2l.pan`（完全由先验驱动），本模块要求至少两个簇并直接拒绝。
- **未覆盖。** 多响应变量（`r > 1`）、两层二元或分类响应、MNAR、多列簇 ID、对角或其他随机效应协方差结构、不等簇大小与极端数值边界仍待迁移。验证范围是可观测插补值的分布矩，尚未直接核对 `β`、`σ²`、`Ψ` 等内部后验。模型估计之后的合并见 [多重插补合并](mi-pooling.md)。
