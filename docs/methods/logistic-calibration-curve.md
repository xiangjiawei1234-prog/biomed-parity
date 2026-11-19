# 二分类逻辑回归的重采样校准曲线

`biomed_parity.calibration.calibrate_logistic` 对应 `rms::calibrate` 的二分类、无惩罚 `lrm` 路径。输入为数值设计矩阵与 0/1 结局，截距由函数估计。输出包括原模型的逐行概率、表观校准曲线、乐观偏差修正曲线、每个概率点的重采样计数，以及可选的近似点态区间。 [rms 校准手册](https://search.r-project.org/CRAN/refmans/rms/html/calibrate.html)

## 输入和算法

- `predy` 给出评估曲线的概率点。使用默认网格时，需要至少 11 行数据，并在原模型第五小和第五大的预测概率之间取 50 个等距点。
- 默认 LOWESS 在预测概率尺度上拟合，跨度是 `2/3`，稳健迭代次数是 0。可通过 `span`、`iterations` 和 `delta` 配置。可通过 `digits` 在平滑前对概率取小数位，或通过 `smoother` 传入返回一对坐标数组的函数。平滑曲线在 `predy` 上线性插值，超出支持范围的点返回 `NaN`。 [R LOWESS 手册](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/lowess.html)
- `method="boot"` 用 Bootstrap 样本训练并在全数据上测试。`method=".632"` 在每次未抽中个体上测试，用个体未抽中频率对各次结果加权。`method="crossvalidation"` 用 `n_resamples` 指定折数，可通过 `fold_order` 固定划分前的随机排列。Bootstrap 两种方式均接受 `bootstrap_indices`，数组每行给出一次零起点抽样。可用 `seed` 生成 Python 重采样。
- `method="randomization"` 固定训练样本的设计矩阵，打乱结局后拟合，再用原结局测试。通过 `permutation_indices` 可指定重复排列，用于复现随机化对照。
- `training_error` 与 `test_error` 是平滑结局减去 `predy`。普通 Bootstrap 与交叉验证的 `optimism` 来自两者差值。对 .632 则使用 0.632 倍的表观误差减加权测试误差。校准曲线 `corrected = apparent - optimism`。失败拟合的序号放在 `failed_replicates`，每个概率点的 `counts` 记录可成对比较的样本数。
- `lower` 和 `upper` 是校准曲线尺度的点态近似区间，对应 R 结果的 `Lower + predy` 和 `Upper + predy`。使用百分位置信水平时传小数，例如 `confidence=.9`。设置 `confidence=None` 省略区间。该区间允许超出 [0, 1]，并列入样本支持不足时的缺失数值。 [校准区间推导与模拟](https://www.fharrell.com/post/bootcal/)

```python
from biomed_parity.calibration import calibrate_logistic

curve = calibrate_logistic(
    clinical_design, outcomes,
    predy=[0.2, 0.4, 0.6, 0.8],
    n_resamples=100,
    seed=42,
)
print(curve.predy, curve.apparent, curve.corrected, curve.counts)
```

## 对照结果

在本地 R 4.5.3、rms 8.2-1 和 Hmisc 5.3-1 上运行五个场景，共 62 次指定抽样、折划分或结局排列。逐字段比较默认 Bootstrap、稳健 LOWESS 与取整、交叉验证、.632 权重和随机化。点估计、缺失位置、每点比较计数和可用区间均一致，最大数值绝对差约为 `6.57e-11`。rms 8.2-1 的默认校准方法未把独立的 `digits` 参数转交给平滑函数，所以取整场景的 R 参考通过自定义平滑器明确执行 `round`。 [上游实现](https://github.com/harrelfe/rms/blob/master/R/calibrate.default.s)

对三个点恰好局部插值的极小样本，稳健更新可能受浮点残差影响。Python 的所有绝对残差中位数为零时保留已有拟合。本地 R 4.5.3 在一组这样的三点输入中将拟合 `[1, 0, 0]` 更新为 `[0, 0, 0]`。

这里验收的是事先确定设计矩阵后的二分类流程。惩罚模型、抽样内向后选择、按簇重抽样、序数模型和生存曲线校准分别属于后续工作。

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_calibrate_logistic.py
```
