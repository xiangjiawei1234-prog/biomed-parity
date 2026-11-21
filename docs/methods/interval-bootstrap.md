# 区间删失半参数回归的病例 Bootstrap

`bootstrap_interval_regression` 为区间删失比例风险或比例生存优势模型估计系数协方差。每次重抽样重新拟合基线生存与回归系数，`fit` 保留原样本的估计及预测分布。 [icenReg::ic_sp 手册](https://search.r-project.org/CRAN/refmans/icenReg/html/ic_sp.html) · [半参数回归拟合](interval-regression.md)

```python
import numpy as np

from biomed_parity.interval_bootstrap import bootstrap_interval_regression

i = np.arange(1, 41)
x = ((i * 7) % 13 - 6) / 4
elapsed = np.exp(1.1 - 0.36 * x + 0.48 * (np.sin(i * 1.7) + np.cos(i * 0.93)))
left = np.floor(elapsed * 2) / 2
right = left + 0.5
right[left >= 4.5] = np.inf
left[left >= 4.5] = 4.5

result = bootstrap_interval_regression(left, right, x, model="ph", n_resamples=100, seed=42)
print(result.fit.coefficients, result.standard_error)
print(result.lower, result.upper)
```

## 重抽样与推断

- 每次以放回方式抽取 `ceil(sum(weights))` 行，抽取概率与原观测权重成正比。随后用各行被抽到的次数作为重拟合的权重。无权重时每次抽取与原样本等多的行。
- 用 `n_resamples` 和 `seed` 生成重抽样，默认抽样 200 次。也可以传入 `bootstrap_indices` 固定抽样，其中每行是一组 0 基原数据行下标。固定索引可用于以 R 生成的抽样重现系数与协方差。
- `replicates` 依次保存成功的系数向量。协方差除以成功次数减一，`standard_error` 为协方差对角线的平方根。`z`、`pvalue`、`lower` 和 `upper` 为使用原样本系数的正态 Wald 推断，保持 `icenReg` 的无 Bootstrap 偏倚修正口径。`confidence_level` 默认为 0.95。 [icenReg::ic_sp 源码](https://github.com/cran/icenReg/blob/master/R/ic_sp.R)
- 拟合失败或估计非有限时，按反向行顺序对同一似然重试一次。奇异设计及仍无法获得有限拟合的抽样记录在 `failed_replicates`，下标从零开始。成功次数少于两次时仍返回原拟合及抽样记录，协方差和推断数组为 `NaN`。

## 本地验证

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 python -m pytest -q -W error tests/test_interval_bootstrap.py
R_LIBS_USER="$PWD/.r-lib" python validation/interval_bootstrap/verify.py
```

R 4.5.3 和 icenReg 2.0.16 的两个案例覆盖无权重 PH、分数权重 PO、不同开闭端点及奇异抽样。先重现 R 自带 Bootstrap 的行索引，再独立调用 `ic_sp` 重拟合，共核对 26 次成功抽样和 1 次失败。原估计、重拟合系数、协方差及 Wald 统计量的绝对容差为 `3e-5`。 [R 对照脚本](../../validation/interval_bootstrap/oracle.R)

当前范围是独立病例重抽样下的回归系数推断。分组重抽样、基线生存区间及预测曲线置信带另行验证。
