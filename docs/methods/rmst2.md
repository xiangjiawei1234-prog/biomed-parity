# survRM2 的两组限制平均生存时间

Python 入口为 `biomed_parity.rmst.rmst2(time, status, arm, *, tau=None, covariates=None, alpha=.05)`。随访时长使用相同单位。状态 1 表示事件，0 表示右删失。分组 0 为参照，分组 1 为比较组。数组逐行对应且须包含两组。默认截断根据两臂最末时点及删失状态决定，显式截断须处于相同支持范围内。 [R 方法手册](https://search.r-project.org/CRAN/refmans/survRM2/html/rmst2.html)

## 未调整分析

`RMSTResult` 包含两组各自的 RMST、RMTL、标准误和区间，以及组间 RMST 差值、RMST 比值和 RMTL 比值的估计、区间及双侧 p 值。RMST 是 Kaplan–Meier 曲线从 0 到 `tau` 的积分，RMTL 为 `tau - RMST`。抽样方差使用积分后的 Greenwood 估计。比值使用对数尺度 delta 法。当比值涉及零均值，点估计可为 0、无穷或 NaN，区间和 p 值为 NaN。 [方法示例](https://cran.r-project.org/web/packages/survRM2/vignettes/survRM2-vignette3-2.html)

```python
from biomed_parity.rmst import rmst2

result = rmst2(
    time=[0, 2, 4, 4, 7, 1, 3, 5, 7, 8],
    status=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    arm=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    tau=6,
)
print(result.rmst_difference.estimate)  # 1.1333333333333333
```

## 协变量调整

传入一列或多列数值型基线协变量会返回 `AdjustedRMSTResult`。差值模型使用恒等链接，两种比值模型使用对数链接。删失曲线按臂估计。推断包含组内删失过程对夹心方差的贡献，结果中的三个系数表依次为截距、组别和输入的协变量列。该路径要求正随访时间、正截断时间、满秩设计及各臂的可估计对数比值。无有限解、矩阵秩不足或迭代失败时会报错。 [Tian、Zhao 与 Wei 的方法论文](https://doi.org/10.1093/biostatistics/kxt050) · [调整模型和链接函数](https://ageconsearch.umn.edu/record/340147/files/Cronin.pdf)

当前 API 直接返回数值，R 的格式化输出、绘图及完整 `survfit` 对象留在原工作流。缺失记录需在调用前按研究方案处理。

## 验证

R 4.5.3、survRM2 1.0-4 和 survival 3.8-6 生成 14 组人工构造的参考输出。其中 12 组覆盖未调整分析、零时点、行政性删失、并列及近并列随访，另两组逐项比较带一个或两个协变量的模型系数、标准误和区间。使用双精度容差比较，允许回归迭代停止准则导致的末位差异。

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest tests/test_rmst.py
```
