# 相关效应量的抽样协方差

`biomed_parity.meta_covariance.vcalc` 根据效应量的抽样方差、研究分组和依赖结构构造抽样协方差矩阵。不同 `cluster` 或 `subgroup` 的效应互相独立。组内可同时描述同一构念的不同量表、不同结局、重复随访及多臂研究中的共用对照。 [metafor::vcalc](https://wviechtb.github.io/metafor/reference/vcalc.html)

```python
from biomed_parity.meta_covariance import vcalc

V = vcalc(
    vi=[0.09, 0.16, 0.10, 0.12], cluster=[1, 1, 2, 2],
    grp1=["intervention-a", "intervention-b", "intervention-a", "intervention-b"],
    grp2=["control"] * 4,
    w1=[32, 34, 45, 48], w2=[38, 38, 50, 50],
)
```

`V` 的行列顺序与输入效应顺序相同，对角线上是给定的 `vi`。可以传给 `meta_multilevel(..., sampling_covariance=V)`。设置 `sparse=True` 返回 SciPy CSR 矩阵；当前的多层拟合入口使用稠密矩阵。 [多层 Meta 分析](meta-multilevel.md)

| 参数 | 组内依赖 |
| --- | --- |
| `obs`、`type`、`rho` | 指定量表、结局类型与同期抽样相关。两种标签都传入时，`rho=(构念内, 构念间)`。每项可为标量或 `NamedCorrelation(levels, matrix)`。 |
| `grp1`、`grp2`、`w1`、`w2` | 指定对比的两臂与各臂的逆抽样方差权重。权重可由可比的组规模近似。共用臂在同向对比中提供正项，在反向对比中提供负项。 |
| `time1`、`time2`、`phi` | 给定同一臂不同随访时点的自相关基数，用 `phi ** abs(t1 - t2)` 折算。负的 `phi` 要求相关时点的差值为整数。 |
| `rvars` | 直接输入逐研究的相关下三角。每行对应一个效应，每个研究内的行按其在输入中的顺序取值。最大块决定列数，其余单元可以留空。 |

默认检查组内相关矩阵是否正定，并用 `MetaCovarianceWarning` 指出问题块。设置 `nearpd=True` 时，对有问题的块做带 Dykstra 修正的交替投影，并将最小特征值移到正值范围。修正会改变输入相关结构，可结合原始相关假设核查结果。 [Higham 的最近相关矩阵方法](https://doi.org/10.1093/imanum/22.3.329) · [R Matrix::nearPD](https://search.r-project.org/CRAN/refmans/Matrix/html/nearPD.html)

使用 R 4.5.3 和 metafor 5.0-1 核对 11 个合成案例，覆盖独立研究、不同构念、命名相关矩阵、分段时间、共用及反向对比臂、前后变化、显式相关块和正定修正。检验逐项协方差、警告、对角线与稠密和稀疏输出一致性。Python 可保持输入中交错排列的研究行；使用 `rvars` 与 R 对照时采取其手册要求的研究连续排列。 [R 对照用例](../../validation/meta_covariance/cases.json) · [R 生成脚本](../../validation/meta_covariance/oracle.R)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_meta_covariance.py
```
