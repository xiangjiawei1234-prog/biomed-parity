# mice 的标量与多参数合并推断

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

完整数据可由 [链式插补](chained-imputation.md) 生成。每份数据使用同一模型、同一列顺序和同一种协方差定义进行拟合。 [mice 完整合并接口](https://amices.org/mice/reference/pool.html)

11 组人工构造用例以 R 4.5.3 和 mice 3.19.0 为参考，逐项比较 Rubin、Reiter 规则、有限及无限自由度、极小组间方差和零方差行为。

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest tests/test_imputation.py
```

## 多参数推断

`pool_parameters(estimates, covariances, complete_df=...)` 接收“插补份数 × 参数数”估计矩阵及每份拟合的完整协方差矩阵。它返回组内、组间及 Rubin 总协方差和每个参数的边际推断。 `pool_d1` 允许给定线性对比矩阵和零假设值，返回联合 F 统计量、分子及分母自由度和 p 值。有限 `complete_df` 使用 Reiter 的多参数自由度修正。 `pool_d2` 可直接合并各份拟合上的非负 Wald 或似然比卡方统计量。D2 在合并变异较大时可给出负 F，对应 p 值为 1。 [mice D1](https://amices.org/mice/reference/D1.html) · [mitml 多参数方法](https://search.r-project.org/CRAN/refmans/mitml/html/testModels.html) · [有限样本自由度](https://doi.org/10.1177/1536867X0900900303)

```python
from biomed_parity.imputation import pool_d1

result = pool_d1(
    estimates=[[1.1, .2], [.8, .3], [1.0, .1], [1.2, .4], [.7, .2], [1.0, .1]],
    covariances=[[[.12, .03], [.03, .08]]] * 6,
    contrast=[[1, -1]],
    null=0,
    complete_df=38,
)
print(result.statistic, result.pvalue)
```

八组多参数用例使用 mitml 0.4-5 对照，覆盖非对角协方差、两组线性约束、有限或无限完整数据自由度、相同统计量和负 D2。公共接口 `testConstraints` 的协方差变换采用数值微分。直接算法与公共接口分别有独立断言。

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest tests/test_pooled_imputation.py
```

## 嵌套模型的似然比检验

`pool_d3` 和 `pool_d4` 接收按插补顺序排列的完整模型和简化模型的最大对数似然。两模型应在每份完成数据中使用相同观测、相同似然定义和最大似然估计。`df` 为新增参数个数。两种方法均返回合并后的 F 统计量、分子和分母自由度、p 值和相对方差增加。相对方差增加为零时使用无限分母自由度。 [Meng 与 Rubin 的 D3](https://doi.org/10.1093/biomet/79.1.103) · [Chan 与 Meng 的 D4](https://doi.org/10.5705/ss.202019.0314)

`pool_d3` 还接收两个模型分别在各自合并参数上逐份求值的对数似然。计算这些受限似然时，mice 3.19.0 使用合并回归系数作为 offset 后重新拟合。新版 mice 对 `lm`、Gaussian、binomial 或 Poisson `glm` 以及 `lmerMod` 使用包含干扰参数的受限似然。数值接口可接收对应目标版本计算出的任一组受限似然。例如高斯最大似然需按目标版本确定是合并残差方差还是在各份数据重新估计。负的 D3 统计量会触发警告。D3 默认保留有符号的相对方差增加，`ariv="positive"` 对应 mitml 的非负处理。 [mice 3.19.0 D3 源码](https://github.com/amices/mice/blob/v3.19.0/R/D3.R) · [mice D3 手册](https://amices.org/mice/reference/D3.html)

`pool_d4` 使用 `mitml::testModels(method="D4")` 的约定。将所有完成数据纵向堆叠后分别拟合两个模型，传入两个堆叠拟合的原始对数似然，函数按插补份数缩放。`ariv="positive"` 将负的相对方差增加截为零。`ariv="robust"` 使用完整模型的单独拟合和堆叠拟合之间的似然差，需用 `full_model_df` 提供完整模型的总参数数，包括估计的干扰参数。 [mitml `testModels` 手册](https://search.r-project.org/CRAN/refmans/mitml/html/testModels.html)

```python
from biomed_parity.imputation import pool_d3, pool_d4

full = [-101.2, -100.7, -100.4]
reduced = [-105.8, -104.9, -106.0]
d3 = pool_d3(full, reduced, [-102.0, -101.4, -101.5], [-106.0, -105.1, -106.2], df=2)
d4 = pool_d4(full, reduced, stacked_full_loglik=-304.7, stacked_reduced_loglik=-317.0, df=2)
print(d3.pvalue, d4.pvalue)
```

Python 调用端负责拟合模型并构造输入似然，数值接口可用于其他能生成可比似然的嵌套模型。R 4.5.3、mice 3.19.0 和 mitml 0.4-5 的对照包含四组人工血压与再入院资料的线性及逻辑回归。测试独立用 Python 计算拟合、受限和堆叠似然。 [验证用例](../../validation/pooled_likelihood/cases.json) · [R 对照脚本](../../validation/pooled_likelihood/oracle.R) · [参考结果](../../validation/pooled_likelihood/expected.json)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest tests/test_pooled_likelihood.py
```
