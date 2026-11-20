# 多状态模型的转移特异协变量

`biomed_parity.mstate_covariates.expand_covariates` 将长表中的协变量按转移展开。每个连续变量或分类对比在每条转移上生成一列。对应该转移的行保留值，其余行取零。源变量缺失时，该变量展开后的列均保留缺失。 [mstate::expand.covs 参考](https://search.r-project.org/CRAN/refmans/mstate/html/expand.covs.msdata.html)

```python
import numpy as np
from biomed_parity.mstate_covariates import expand_covariates
from biomed_parity.mstate_fit import msfit

# records 是 msprep 返回的长表，包含有限的 age 数值
by_transition = expand_covariates(records, "age", append=False)
design = np.column_stack(tuple(by_transition.values()))
fit = msfit(records, covariates=design)
print(dict(zip(by_transition, fit.coefficients, strict=True)))
```

数值协变量默认按连续变量处理。字符串变量默认按字典序排序，布尔变量以 `False` 为参考。通过 `factor_levels={"变量名": [参考水平, ...]}` 固定分类顺序，可包含样本中未出现的水平。列在 `ordered` 中时使用等间距等级的正交多项式对比；其余分类列使用处理对比。数字编码的无序分类变量需通过 `factor_levels` 声明。 [R 对比编码](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/contrast.html)

展开名称采用 R 的一基转移编号。默认使用含水平或对比名称的长名。设置 `long_names=False` 时，多个对比按顺序编号。默认返回保留原始协变量的 `MultiStateRecords`；`append=False` 仅返回按对比和转移排序的字典。输出包含图中声明的每一条边，即使输入长表没有该边的风险记录。拟合前应处理缺失和不可估计的设计列。 [转移风险拟合](mstate-fit.md)

在 R 4.5.3 与 mstate 0.3.3 上核对五组展开矩阵，含数值缺失、三水平因子、五水平有序对比、未出现的水平、逻辑变量、短名及追加模式。另验证三个转移的分层 Cox 联合拟合系数与分别拟合一致。

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_mstate_covariates.py
```
