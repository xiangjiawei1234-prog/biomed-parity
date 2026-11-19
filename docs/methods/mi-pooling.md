# mice 的标量合并推断

`biomed_parity.imputation.pool_scalar(q, u, *, n=inf, k=1, rule="rubin1987", confidence_level=.95, null=0)` 接收至少两份完整数据模型对同一参数的估计 `q` 和对应的方差 `u`。若模型提供标准误，传入其平方。给定完整数据样本量 `n` 与参数数目 `k` 时，完整数据自由度为 `n-k`。省略 `n` 时使用无限完整数据自由度。 [mice 标量接口](https://amices.org/mice/reference/pool.scalar.html) · [Barnard 与 Rubin 的小样本自由度](https://doi.org/10.1093/biomet/86.4.948)

```python
from biomed_parity.imputation import pool_scalar

summary = pool_scalar(
    q=[-1.5457, -1.428],
    u=[0.9723**2, 1.041**2],
    n=25,
    k=2,
)
print(summary.qbar, summary.df, summary.lower, summary.upper)
```

结果包括组内方差、组间方差、总方差、相对方差增加、自由度、缺失信息比例，以及双侧 t 检验和区间。默认 Rubin 规则用于缺失数据多重插补。Reiter 规则用于从完整数据生成的部分合成数据，该规则的缺失信息比例以 NaN 表示。全部组内方差为零时，某些诊断量和 t 推断可能未定义。

本方法从已拟合的标量参数开始。模型适配、插补链、`predictorMatrix`、被动插补、分层插补及合并嵌套模型检验待分别验证。 [mice 完整合并接口](https://amices.org/mice/reference/pool.html)

11 组人工构造用例以 R 4.5.3 和 mice 3.19.0 为参考，逐项比较 Rubin、Reiter 规则、有限及无限自由度、极小组间方差和零方差行为。

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest tests/test_imputation.py
```
