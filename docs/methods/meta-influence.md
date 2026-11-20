# 多层 Meta 分析的影响诊断

`biomed_parity.meta_influence` 从 `meta_multilevel` 的拟合结果计算 Cook 距离、DFBETAS、逐行或逐研究删除残差及 GLS 杠杆值。可将同一研究的效应量作为一个删除单元。病例标签按输入首次出现的顺序返回，残差与原始效应行对齐。 [metafor 多层影响诊断](https://wviechtb.github.io/metafor/reference/influence.rma.mv.html) · [标准化与删除残差](https://wviechtb.github.io/metafor/reference/residuals.rma.html)

```python
from biomed_parity.meta_influence import meta_hatvalues, meta_influence

study_checks = meta_influence(fit, cluster=study_ids)
print(study_checks.labels, study_checks.cooks_distance)
print(study_checks.studentized_residuals)
print(meta_hatvalues(fit))
```

默认每次删除都重新估计可估的方差分量，并保持原来固定的分量。设置 `reestimate=False` 会沿用拟合的边际协方差，只重新计算固定效应。Cook 距离衡量固定效应变化相对于原模型协方差的大小。`cook_coefficients` 可按零起点的设计列索引指定关注的系数。DFBETAS 使用删除前后的系数差，除以把删除后方差分量应用到全样本设计时得到的系数标准误。

删除残差使用留出单元观测值减去其删除后预测。标准化分母同时纳入留出单元的边际方差和预测均值的不确定性。每个删除单元还返回基于联合预测协方差的卡方统计量，自由度为该单元的效应量数。删除后模型无法估计时，保留其位置并将相关结果记为缺失。`failed_cases` 指出这些单元。杠杆函数默认输出帽子矩阵对角线，`matrix=True` 输出完整矩阵。

R 4.5.3 与 metafor 5.0-1 的四组人工方案覆盖逐效应删除、按研究删除、固定或重新估计的方差分量、Knapp–Hartung 协方差、选择 Cook 系数，以及删除后设计降秩的情形。核对 Cook 距离、DFBETAS、删除残差、卡方统计量、回归系数、方差分量、杠杆值和完整帽子矩阵。当前范围沿用原生多层拟合支持的模型结构。 [验证脚本](../../validation/meta_influence/oracle.R) · [参考结果](../../validation/meta_influence/expected.json)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_meta_influence.py
```
