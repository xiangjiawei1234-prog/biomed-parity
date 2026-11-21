# 平滑 ROC 与面积

`biomed_parity.roc_smoothing.smooth_roc` 从二分类分数计算双正态、核密度或对数凹密度 ROC，并返回完整 AUC 或指定区间的部分 AUC。实现只依赖 NumPy 和 SciPy，数值对照使用 R 4.5.3 和 pROC 1.19.1。[pROC 平滑手册](https://search.r-project.org/CRAN/refmans/pROC/html/smooth.html) · [pROC AUC 手册](https://search.r-project.org/CRAN/refmans/pROC/html/auc.html)

## 接口

```python
from biomed_parity.roc_smoothing import smooth_roc

labels = [0, 0, 0, 0, 1, 1, 1, 1]
scores = [0.1, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9]
binormal = smooth_roc(labels, scores)
density = smooth_roc(
    labels, scores, method="density", direction="auto",
    bw="nrd0", partial_auc=(0.6, 1.0), focus="specificity",
)
```

| 参数 | 口径 |
| --- | --- |
| `labels`, `scores` | 等长一维向量，`0` 为对照，`1` 为病例，分数为有限实数。两组各至少有一名受试者。 |
| `method` | `"binormal"` 为默认方法，也可指定 `"density"`、`"logcondens"` 或 `"logcondens.smooth"`。 |
| `direction` | 默认 `"<"` 表示高分偏向病例，`">"` 表示低分偏向病例。设为 `"auto"` 则复现 `pROC::roc()` 的中位数规则，对照中位数不高于病例时取 `"<"`。 |
| `n` | 默认为 512。双正态和对数凹密度至少需要 2 个网格点，核密度至少需要 1 个。 |
| `bw`, `kernel`, `adjust`, `cut` | 密度参数。`bw` 可取正数、`"nrd0"` 或 `"nrd"`。默认核为 `"gaussian"`，其余支持 `"rectangular"`、`"triangular"`、`"epanechnikov"`、`"biweight"`、`"cosine"`、`"optcosine"`。`adjust` 默认 1，`cut` 默认 3。 |
| `density_controls`、`density_cases` | 为相应组提供按分数升序的等距网格密度，取值有限且非负，两组各有正质量。省略一组时在公共核密度网格上估计该组，表长必须等于 `n`。同时提供两组时由共同表长决定网格点数。 |
| `partial_auc`, `focus` | 两个端点可任意顺序给出。`focus="specificity"` 以特异度为横轴，也可用 `"sensitivity"`。省略 `partial_auc` 就计算完整面积。 |
| `standardize`, `allow_below_chance` | 默认均为 `False`。部分面积可用 McClish 公式校正，校正后机会水平为 0.5。低于机会水平时返回 `NaN` 和 `RuntimeWarning`，设置 `allow_below_chance=True` 可保留该校正值。 |
| `percent` | 默认采用 0–1 尺度。设为 `True` 时，灵敏度、特异度、部分面积的端点及 AUC 采用 0–100 尺度。 |

`SmoothROCResult` 返回按特异度升序排列的 `specificity` 和 `sensitivity`、面积 `auc`、实际采用的 `direction`、`method` 及 `percent` 标志。数组包含计算点和两个锚点，锚点按 `(特异度, 灵敏度)` 表示为 `(0, 1)` 与 `(1, 0)`，百分制时相应缩放。双正态结果提供 `coefficients=(截距, 斜率)`，核密度结果提供调整后的数值 `bandwidth`。两个密度表均给定时，`bandwidth` 为 `None`。曲线点可以包含重复锚点。

## 估计与积分

- **双正态。** 把经验 ROC 的灵敏度和特异度变换为标准正态分位数，并对两者均有限的点，将特异度分位数对灵敏度分位数做带截距的最小二乘拟合。需要至少两个不同的有限灵敏度分位数。以等距灵敏度概率生成曲线。完整 AUC 用 `Φ(截距 / sqrt(1 + 斜率²))`，与 `n` 无关。[pROC 实现](https://github.com/xrobin/pROC/blob/master/R/smooth.R) · [pROC 面积实现](https://github.com/xrobin/pROC/blob/master/R/auc.R)
- **核密度。** 合并两组分数选取共同带宽，再乘以 `adjust`。两组共享等距输出网格，端点是合并分数极值各向外延伸 `cut * bandwidth`。采用 R 4.4 及以上版本的默认坐标约定，以线性分箱、FFT 卷积和插值估计两组密度。灵敏度包含当前网格点的病例质量，特异度使用另一侧的对照质量。[R 核密度手册](https://search.r-project.org/R/refmans/stats/html/density.html) · [pROC 密度坐标实现](https://github.com/xrobin/pROC/blob/master/R/roc.utils.R)
- **曲线积分。** 核密度和对数凹密度的完整 AUC，以及各模型的部分 AUC，都在加锚点后的曲线上用梯形法计算。部分区间边界落在网格点之间时沿曲线线性插值。校正时，机器精度内与对角线的差视为零。零宽区间的部分 AUC 为零，校正值为 `NaN`。[pROC AUC 手册](https://search.r-project.org/CRAN/refmans/pROC/html/auc.html)

## 对照与范围

[人工数据和场景](../../validation/roc_smoothing/cases.json)覆盖 8 组数据和 37 个场景。30 个场景比较完整网格与 AUC，1 个水平双正态场景比较内部点、系数与 AUC，6 个场景验证输入或拟合错误。用例涵盖两种方向与自动方向、并列分数、完全分离、常数分数、单侧及双侧给定密度表、七种核、两种部分面积轴、百分制及校正行为。给定密度表时，自动方向仍由原始分数决定。 [R 参考脚本](../../validation/roc_smoothing/oracle.R) · [版本化快照](../../validation/roc_smoothing/expected.json)

相对容差为 `2e-14`，绝对容差为 `3e-15`（百分制曲线坐标与 AUC 乘以 100）。实测在 0–1 尺度上的最大绝对差为曲线坐标 `1.67e-15`、AUC `6.67e-16`；未缩放的系数和带宽分别为 `2.45e-15` 和 `5.12e-15`。

```sh
python -m pytest -q tests/test_roc_smoothing.py
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 python -m pytest -q -W error tests/test_roc_smoothing.py
R_LIBS_USER="$PWD/.r-lib" Rscript --vanilla validation/roc_smoothing/oracle.R validation/roc_smoothing/cases.json
```

水平双正态拟合采用负斜率趋近零时的端点极限，保持曲线方向。pROC 在无穷分位点上的结果依赖 `lm` 对理论零斜率的浮点舍入，可出现相反的端点顺序或报错。密度方法要求两组在所选输出网格上有正质量，过稀的网格可能无法分辨窄核。

反向评分且对照质量全部位于首个表格单元时，网格坐标和完整 AUC 与 R 对照。以灵敏度为横轴的部分 AUC 采用几何积分，退化曲线的结果可能与 pROC 不同。当前密度表入口仍要求原始二分类分数。函数型密度、`SJ`/`ucv`/`bcv` 带宽及 R 4.4 之前的密度坐标仍待迁移。其他入口详见 [参数分布平滑](roc-parametric.md) · [平滑曲线 Bootstrap](roc-smooth-bootstrap.md) · [多分类 ROC](roc-multiclass.md)。

## 对数凹密度 ROC

`method="logcondens"` 分别拟合对照和病例的单变量对数凹最大似然密度。每组至少需要两个不同分数。拟合将并列分数转成频率权重，在观测分数的支持内使用分段线性对数密度。活动集选择折点，每个折点集合上的拟合使用分段指数积分、带状 Hessian 与阻尼 Newton 法，并沿用 `logcondens` 的 Newton 停止标准。[最大似然与活动集算法](https://arxiv.org/abs/0707.4643)

曲线以 `n` 个等距特异度为网格。给定特异度 `s`，用对照的未卷积密度确定分位点，再用病例分布计算超过该分位点的概率。`method="logcondens.smooth"` 在同一组对照分位点上使用高斯卷积后的病例分布。卷积方差为病例加权样本方差减去拟合密度的方差。pROC 参考实现按不同分数的个数校正加权样本方差。Python 用分段闭式积分计算卷积。[对数凹 ROC 论文](https://arxiv.org/abs/1103.1787) · [logcondens 手册](https://cran.r-project.org/web/packages/logcondens/refman/logcondens.html#logConROC)

为复现 pROC，段内分位数使用对数密度差乘以原分数的区间宽度。因此改变分数单位可能改变曲线。未卷积分布沿用 `logcondens` 在近水平区间的线性近似。pROC 在两种方法中均以原始高分识别病例，结果的 `direction` 字段仍记录原始 ROC 方向，包括 `">"`。网格端点固定为 `(0, 1)` 和 `(1, 0)`，再按 pROC 加入两个锚点。`coefficients` 与 `bandwidth` 均为 `None`。[pROC 平滑实现](https://github.com/xrobin/pROC/blob/v1.19.1/R/smooth.R) · [logcondens 分位数实现](https://github.com/cran/logcondens/blob/2.1.9/R/quantilesLogConDens.r)

11 个合成场景对照 R 4.5.3、pROC 1.19.1 与 logcondens 2.1.9。10 个场景逐点核对坐标与 AUC，1 个验证常数对照组被拒绝。另用 35 组生成数据交叉核对两种方法，0–1 尺度的最大绝对偏差为灵敏度 `2.8e-12`、AUC `4.1e-13`。同一活动集可能因数值舍入和线搜索步长产生小幅差异。[对照用例](../../validation/roc_log_concave/cases.json) · [R 参考脚本](../../validation/roc_log_concave/oracle.R) · [版本化快照](../../validation/roc_log_concave/expected.json) · [对数凹密度 Bootstrap 推断](roc-smooth-bootstrap.md#对数凹密度推断)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 python -m pytest -q -W error tests/test_roc_log_concave.py
```
