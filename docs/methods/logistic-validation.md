# rms::validate.lrm 二分类模型内部验证

## 功能与接口

`validate_logistic(x, y, *, bootstrap_indices=None, n_resamples=None, seed=None, emax_limits=(0, 1), confidence=0.95)` 用普通 bootstrap 量化未惩罚二元 logistic 回归的内部验证偏差。对应 `rms::validate` 对 `lrm` 模型使用 `method="boot"` 与 `bw=FALSE` 的路径。 [rms::validate.lrm](https://search.r-project.org/CRAN/refmans/rms/html/validate.lrm.html) [Bootstrap 验证方法](https://hbiostat.org/rmsc/validate)

```python
from biomed_parity.calibration import validate_logistic

x = [[0]] * 20 + [[1]] * 20 + [[2]] * 20
y = [0] * 15 + [1] * 5 + [0] * 10 + [1] * 10 + [0] * 5 + [1] * 15
fit = validate_logistic(x, y, n_resamples=40, seed=2026)
print(dict(zip(fit.metric_names, fit.corrected)))
```

- `x` 为数值设计矩阵，每行一名对象，由函数内部加截距。将分类变量和样条基等按原模型规范转换为独立的数值列，预先处理缺失值。结局 `y` 使用 0 和 1。
- 可传入 `bootstrap_indices` 精确复核外部的重采样序列。数组形状为重采样次数 × 原始数据行数，采用 Python 从 0 开始的行索引。默认由 NumPy 生成 40 轮样本，可用 `n_resamples` 与 `seed` 指定，自同一个数值种子生成的 R 序列需通过显式索引迁入。
- `emax_limits` 控制最大校准误差的概率区间，步长与 R 相同为 0.0005。`confidence` 控制校正指标的置信水平，传 `None` 可省略区间。
- 输出按指标顺序提供原始值、重采样训练均值、原样本检验均值、平均乐观偏差、校正值、区间上下限和逐指标成功次数。失败的重采样编号见 `failed_replicates`。

## 计算

在原数据上进行一次最大似然拟合。每轮从原始行有放回抽样，重新拟合模型，再分别在抽样数据和原数据上评估。对原数据上的每轮预测，以原始预测对数几率为解释变量拟合 Logistic 校准截距和斜率。用训练值减原样本检验值得到乐观偏差，再从原始指标扣除平均乐观偏差。 [预测能力重采样](https://search.r-project.org/CRAN/refmans/rms/html/predab.resample.html)

| 行名 | 指标 |
| --- | --- |
| `Dxy` | Somers 等级相关，使用平均秩处理并列 |
| `R2` | Nagelkerke 伪 R² |
| `Intercept`、`Slope` | Logistic 校准截距与斜率 |
| `Emax` | 指定概率区间内的最大校准差异 |
| `D`、`U`、`Q` | 基于对数似然的区分、不可靠性和总体质量指标 |
| `B` | 未重新校准预测的 Brier 分数 |
| `g`、`gp` | Logistic 预测值和预测概率的 Gini 平均差 |

置信区间采用本地 `rms` 8.2-1 版实现中训练值减 1.25 倍检验值的离散程度。成功样本数至少 10 时按总均值分为上下两侧，各自计算对总均值的离散程度。成功样本较少时使用普通样本标准差。 [rms::validate.lrm](https://github.com/harrelfe/rms/blob/master/R/validate.lrm.s) [rms::predab.resample](https://github.com/harrelfe/rms/blob/master/R/predab.resample.s)

## 本地验证

```bash
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q tests/test_calibration.py
R_LIBS_USER="$PWD/.r-lib" .venv/bin/python validation/calibration/verify.py
```

本机使用 R 4.5.3、`rms` 8.2-1、`Hmisc` 5.3-1 和 `jsonlite`。基准含 3 个合成数据集和合计 53 轮 bootstrap，覆盖双变量、弱预测时可能出现负校准斜率、分数并列、限制 `Emax` 的概率区间、两种区间计算路径。R 运行公开 `validate(lrm(...))` 接口，并用同一个 R 随机状态导出实际抽样序列供 Python 读入。指标和区间使用绝对容差 `5e-9`、相对容差 `1e-8`。成功次数与指标行名逐项相等。

## 兼容范围

常规路径要求可识别的二元模型。预测变量完全或准完全分离、样本设计矩阵奇异或校准重拟合失败时，Python 记录失败索引并跳过该次重采样。原模型无法拟合或所有次重采样失败时，抛出明确异常。惩罚、权重、按簇和按组采样、逐步筛选、序数模型、交叉验证、.632 bootstrap、外部验证和绘图属于后续范围。Python 已有 logistic 拟合和校准组件，迁移焦点为复现 R 的重拟合规则与结果矩阵。 [statsmodels Logit](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html) [scikit-learn 校准](https://scikit-learn.org/stable/modules/calibration.html)
