# 经验 ROC 操作点、阈值与 Bootstrap 坐标区间

用于从二分类诊断分数中提取 ROC 操作点（阈值、灵敏度、特异度及其派生指标）、选择 Youden 或 closest.topleft 最优阈值，并用分位数 Bootstrap 给出固定坐标或重选最优阈值的置信区间。Python 运行时依赖 NumPy 和 SciPy。数值已与 pROC 1.19.1 的 `coords()` 和 `ci.coords()` 对照。[pROC 参考手册](https://cran.r-project.org/web/packages/pROC/refman/pROC.html) [pROC 1.19.1](https://cran.r-project.org/web/packages/pROC/index.html)

## 接口

| Python 函数 | 对应 R 调用 | 结果 |
| --- | --- | --- |
| `roc_coords(labels, scores, x="all", *, input="threshold", ret=("threshold","specificity","sensitivity"), direction="<", best_method="youden", best_weights=(1.0,0.5), partial_auc=None, focus="specificity", percent=False)` | `coords(roc, x, input, ret, best.method, best.weights)` | 行数为操作点数、列顺序与 `ret` 一致的二维数组 |
| `bootstrap_roc_coords(labels, scores, x, *, input="threshold", ret=None, direction="<", best_method="youden", best_weights=(1.0,0.5), best_policy="stop", partial_auc=None, focus="specificity", percent=False, stratified=True, confidence_level=0.95, n_resamples=2000, random_state=None, bootstrap_indices=None)` | `ci.coords(roc, x, input, ret, best.method, best.weights, best.policy, conf.level, boot.n, boot.stratified)` | `BootstrapCoordsResult`：原样本坐标 `observed`、分位数区间 `intervals`（点 × 字段 × 下/中/上）、非缺失抽样计数 `counts`、`n_resamples`、`omitted_resamples` |

输入是等长一维向量，`0` 代表对照、`1` 代表病例，分数须为有限实数。默认 `direction="<"` 表示高分偏向病例，`">"` 表示低分偏向病例。`"auto"` 沿用 pROC 的规则：对照中位数不大于病例中位数时取 `"<"`，否则取 `">"`；该规则已在快照中与 pROC 对齐。每条曲线在原样本上确定方向，Bootstrap 全部重抽样沿用该方向。

`ret` 可取 `"all"` 或上述 26 个坐标名的任意有序元组；别名 `"t"`、`"topleft"`、`"npe"`、`"ppe"` 分别映射到 `"threshold"`、`"closest.topleft"`、`"1-npv"`、`"1-ppv"`。`x` 可取 `"all"`、`"best"`、`"local maximas"` 或数值数组；`bootstrap_roc_coords` 只接受数值 `x` 或 `"best"`，与 pROC 一致。`percent=True` 时灵敏度、特异度、准确率、预测值、似然比和最优准则按 0–100 尺度输出，阈值本身不缩放，部分 AUC 端点单位同时改为 0–100。

```python
from biomed_parity.roc_thresholds import bootstrap_roc_coords, roc_coords

labels = [0, 0, 0, 0, 1, 1, 1]
scores = [1, 2, 2, 3, 4, 5, 6]
print(roc_coords(labels, scores, x="all"))
print(roc_coords(labels, scores, x="best", ret=("threshold", "specificity", "sensitivity", "youden")))
print(roc_coords(labels, scores, x=[0.5, 0.8], input="specificity"))
```

## 坐标与阈值口径

先对分数排序并合并并列值，阈值取相邻不同分数的中点；第一行和最后一行的阈值分别为 `-Inf` 和 `Inf`，`direction="<"` 时按阈值升序、`">"` 时按阈值降序排列，与 pROC 返回的行序一致。`x="all"` 返回全部经验操作点，并列分数只产生一个操作点。`x` 为数值时，`input="threshold"` 按 pROC 的 `roc_utils_thr_idx` 定位操作点：分数不小于 `x` 的受试者判为病例，特异度为分数严格小于 `x` 的对照比例，因此等于阈值的受试者计入阈值之上；超出分数范围的阈值分别对应首行或末行，`-Inf`/`Inf` 也可用，`NaN` 被拒绝。

`input` 为单调坐标（`specificity`、`sensitivity`、`tp`、`fn`、`fp`、`tn`、`fpr`、`tnr`、`fnr`、`1-specificity`、`1-sensitivity`、`recall`）时按相邻操作点线性插值。恰好命中经验坐标时返回该点的真实阈值（首末行为 `±Inf`），否则 `threshold` 为 `NaN`；数值超出该坐标的经验范围时抛出 `ValueError`，pROC 同样停止。非单调坐标（`accuracy`、`npv`、`ppv`、`youden`、`closest.topleft` 等）不能作为 `input`。`ret` 中的 26 个字段按 pROC 的定义计算：计数、`accuracy`、`npv`/`ppv`/`fdr`、`lr_pos`/`lr_neg` 等在分母为零时返回 `NaN` 或 `±Inf`，百分制下 `1-` 前缀字段也按当前尺度取补。

`x="best"` 在候选操作点中按准则取全局最大，准则并列时返回全部并列行（pROC 返回多行）。`best_method="youden"` 的准则为 `sensitivity + r * specificity`；`best_method="closest.topleft"` 的准则为 `-((scale - sensitivity)^2 + r * (scale - specificity)^2)`，`scale` 为 1 或 100。权重比 `r=(1-prevalence)/(cost*prevalence)`，由 `best_weights=(cost, prevalence)` 给出，默认 `(1, 0.5)` 对应 `r=1`。若数组同时含 `NaN` 准则（例如命中部分 AUC 区间边界外），`x="best"` 不返回任何行。`ret` 只含核心字段或并列准则字段时，模块按 pROC 的做法复用原始最优化准则（百分制下不除以 100）；同时请求其它派生字段时改用统一尺度的 `closest.topleft`。该差异已在 `best_topleft_percent` 等场景覆盖。

`partial_auc=(a, b)` 指定部分 AUC 的两个端点，端点次序可互换。`focus="specificity"` 时保留特异度落在 `[min(a,b), max(a,b)]` 内的操作点，`focus="sensitivity"` 时改用灵敏度过滤；`x="all"` 还会在端点不在经验网格上时插入边界行，其阈值为 `NaN`。`x="best"` 与 `"local maximas"` 只在过滤后的区间内选点。数值 `x` 不走部分 AUC 过滤，与 pROC 一致。

## Bootstrap 约定

默认在病例和对照内分别有放回抽样，保持两类样本数；`stratified=False` 改为对全部受试者整体重抽样。可用 `n_resamples` 设置抽样次数，用 `random_state` 指定 NumPy 种子或随机数生成器。若提供 `bootstrap_indices`，每一行给出一次抽样的零基行号；分层模式要求每行保持原始类别人数，行数决定实际抽样次数。固定抽样矩阵是唯一可跨 R 与 Python 复现的 Bootstrap 路径，随机数流本身不共享。

数值 `x` 的坐标在每次抽样上按固定输入值计算；`x="best"` 在每次抽样上重新选择最优阈值。重抽样出现多个并列最优时由 `best_policy` 决定：`"stop"` 抛出 `ValueError`，`"omit"` 丢弃该次抽样并发出 `RuntimeWarning`，`"random"` 用 `random_state` 指定的生成器均匀选择一行。pROC 的三种策略语义相同，1.19.1 的 `omit` 实现存在索引缺陷，见下文。

置信区间由 `(1-confidence_level)/2`、`0.5`、`1-(1-confidence_level)/2` 处的 7 型分位数给出，逐字段忽略 `NaN`；`counts` 记录每个字段参与分位数计算的有效抽样数。一次抽样若所有请求字段均为 `NaN` 则计入 `omitted_resamples` 并触发警告，与 pROC 的提示和分位数处理对应。分位数对单点固定输入与多点数组逐点计算。

## pROC 语义与接口差异

Python 返回二维 `ndarray`（字段顺序与 `ret` 一致）；pROC 返回 `data.frame`。两侧对阈值缺失的表示不同：pROC 用 `NA`，Python 用 `NaN`；`-Inf`/`Inf` 在两边的数值坐标中保留。pROC 的 `best.method` 别名 `"topleft"` 在本模块中应写作 `"closest.topleft"`；`best_weights` 在 Python 端额外要求代价为正、患病率位于 `(0, 1)`。Bootstrap 的 `best_policy` 随机选择使用各自随机数流，具体选中哪一行不可跨语言复现，但所选行均满足并列准则。

## 验证范围

[观测数据](../../validation/roc_thresholds/observations.tsv) 为 17 名受试者（10 名对照、7 名病例）的人工分数，含 `score_main`（无并列）、`score_ties`（跨类并列）、`score_tiebest`（Youden 并列最优）、`score_constant`（常数分数）和 `score_perfect`（完全分离）五列。[场景表](../../validation/roc_thresholds/cases.tsv) 共 60 个场景：

- 47 个 `coords` 场景：两种方向、`"auto"`、`x="all"`/`"best"`/`"local maximas"`/数值阈值；`input` 为阈值、特异度、灵敏度、`tp`、`fpr`、`tnr` 及精确命中/插值/区间外；多字段与全字段 `ret`；Youden 与 closest.topleft 及多组权重；`percent=True`；两种 `focus` 的部分 AUC 端点、端点插入与数值输入忽略部分区间；并列分数、常数分数和完全分离曲线。
- 13 个 Bootstrap 场景：12 个固定矩阵下的分层/非分层分位数区间（0.9、0.95、0.99 置信水平、多字段 `ret`、两种方向、`percent=False`、数值阈值/特异度/灵敏度），以及 1 个 `best_policy="stop"` 的并列最优报错场景。5 个固定矩阵各 40 行 × 17 列（`draws_main`、`draws_perfect`、`draws_policy`、`draws_random`、`draws_nonstrat`）。`draws_random` 按 pROC `ci.coords()` 的随机数消耗顺序生成；该场景只请求并列选择不变的 `youden` 字段。

[期望快照](../../validation/roc_thresholds/expected.tsv) 共 8214 行：7889 个坐标值、68 个原样本坐标值、174 个分位数、58 个有效抽样计数、24 个汇总值和 1 个报错标记，均由 [oracle.R](../../validation/roc_thresholds/oracle.R) 从 pROC 生成。脚本自行交叉核验：47 个坐标场景直接调用 `pROC::coords()`；12 个非报错 Bootstrap 场景逐个固定矩阵循环，并与同一 R 随机种子下的 `pROC::ci.coords()` 比较分位数、抽样计数和缺失警告；`stop` 场景确认 pROC 同样报错。`omit` 场景按 pROC 文档语义生成期望，并用打补丁后的 `enforce.best.policy` 交叉核验，原因见下节。

比较容差为相对 `2e-14`、绝对 `2e-15`。实测坐标字段与 pROC 完全一致；唯一偏差出现在 Bootstrap 7 型分位数插值的最后一位，最大相对偏差 `2.0e-16`、最大绝对偏差 `5.6e-17`。

```sh
python -m pytest -q tests/test_roc_thresholds.py
BIOMED_R_PARITY=1 python -m pytest -q tests/test_roc_thresholds.py -W error
R_LIBS_USER=.r-lib Rscript --vanilla validation/roc_thresholds/oracle.R validation/roc_thresholds
```

## 已知差异与未覆盖范围

- **pROC 1.19.1 `best_policy="omit"` 缺陷**：`enforce.best.policy()` 写作 `res[1, drop = FALSE]` 而不是 `res[1, , drop = FALSE]`，返回值是首列而不是首行；`ci.coords()` 随即按错误的元素长度回收抽样结果，区间不可用。本模块实现文档语义（丢弃整次抽样），快照按该语义生成，并在 oracle 中用修正后的辅助函数核验。该缺陷只影响出现并列最优的 `omit` 抽样。
- **`percent=True` 的 Bootstrap 不对齐**：pROC 1.19.1 的 `ci.coords()` 在百分制 ROC 上把重抽样的灵敏度/特异度写成 0–1 尺度，而对象仍标记 `percent=TRUE`，导致数值特异度输入直接越界报错、最优准则混用两种尺度。本模块全程保持百分制，快照因此只覆盖百分制点估计，不包含百分制 Bootstrap。
- **随机选择不可复现**：`best_policy="random"` 在 R 与 Python 各自消耗本地随机数流，选中行号无法跨语言对齐；快照选择对并列选择不变的字段，测试另行验证选择落在并列候选内。
- **非分层单类抽样**：若某次整体重抽样只含一个类别，pROC 保留部分字段的 `NaN`（例如特异度有值而灵敏度为 `NaN`），本模块丢弃该次抽样并计入 `omitted_resamples`。快照矩阵不含单类抽样。
- **平滑 ROC 未覆盖**：仅实现经验 ROC；pROC 的 `smooth=TRUE`/`smooth.roc` 曲线不在本页范围。多分类 ROC、Bootstrap 部分 AUC 区间、`ci.coords` 的 `"all"`/`"local maximas"`（pROC 本身不支持）、随机抽样次数与 pROC 随机数流的一致性也属于未覆盖范围。
- **阈值缺失表示**：pROC 的 `NA` 阈值在数值数组中表示为 `NaN`；`-Inf`/`Inf` 两端阈值保留。该表示差异不影响数值语义。

同项目其它 ROC 方法见 [完整与部分 AUC 及分层 Bootstrap](roc-bootstrap.md) 与 [DeLong 推断](roc-delong.md)。[pROC 论文](https://pubmed.ncbi.nlm.nih.gov/21414208/)
