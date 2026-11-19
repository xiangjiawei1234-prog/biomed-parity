# variancePartition::dream 单随机截距拟合

## 功能与接口

`dream_random_intercept(y, x, group, *, weights=None, contrasts=None)` 对单个特征的已标准化连续丰度拟合含一个分组随机截距的 Gaussian 模型。方差分量用 REML 估计。每个固定效应或指定对比使用 Satterthwaite 自由度和未调节的双侧 t 检验。对应 `variancePartition::dream` 的 `REML=TRUE`、`ddf="Satterthwaite"` 单个随机截距路径。 [dream 参考手册](https://diseaseneurogenomics.github.io/variancePartition/reference/dream-method.html) [dream 论文](https://doi.org/10.1093/bioinformatics/btaa687)

```python
from biomed_parity.dream import dream_random_intercept

result = dream_random_intercept(
    [2.1, 2.7, 4.1, 5.2, 6.0, 6.2, 4.5, 5.1],
    [[1, 0], [1, 1]] * 4,
    ["A", "A", "B", "B", "C", "C", "D", "D"],
    weights=[1.0, 0.8, 1.2, 0.9, 1.1, 1.4, 0.7, 1.0],
    contrasts=[0, 1],
)
print(result.estimate, result.df, result.pvalue)
```

- `y`、`x` 和 `group` 依同一对象顺序排列。`x` 为满秩设计矩阵，按需要显式加入截距及已编码的协变量。标准化和缺失数据处理在拟合前完成。
- `weights` 为已给定的正精度权重，默认全 1。残差方差随权重的倒数变化。
- `contrasts` 为每行一个线性对比的矩阵，也可传一个单独的向量。省略时检验各固定效应系数。
- 返回固定效应系数与协方差、残差方差、随机截距方差，以及所选对比的估计值、标准误、t 值、自由度和 p 值。

## 数值方法

将样本响应、设计矩阵和随机截距设计乘以精度权重的平方根。在固定设计的正交残差空间内，对随机截距的协方差部分做特征分解，利用剖面 REML 目标求解非负方差比。随后通过广义最小二乘求固定效应，并从观察信息矩阵和对比方差的梯度计算 Satterthwaite 自由度。随机截距方差为零时，退化为加权最小二乘与样本数减固定效应参数数目的自由度。 [lmerTest 论文](https://www.jstatsoft.org/article/view/v082i13)

## 本地验证

```bash
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q tests/test_dream.py
R_LIBS_USER="$PWD/.r-lib" .venv/bin/python validation/dream/verify.py
```

R 对照使用 R 4.5.3、Bioconductor 3.22 的 `variancePartition` 1.40.2、`lmerTest` 3.2-1 和 `jsonlite`。3 个合成对数丰度用例涵盖平衡重复测量、非平衡加权设计、多列固定效应和联合线性对比、随机效应方差为零的边界。对比估计、标准误、自由度、t 值与 p 值直接与 `dream` 对照，方差分量及固定效应协方差与 `lmerTest` 对照。p 值使用相对容差 `5e-6`，其他数值使用绝对容差 `1e-8` 和相对容差 `1e-6`。纯 Python 测试还检查整体权重缩放的不变性、秩亏与输入不合法等情况。

## 兼容范围

这一路径接收已经给出的权重和设计矩阵。多随机效应、随机斜率、`voomWithDreamWeights` 权重估计、Kenward–Roger 自由度、经验贝叶斯调节、批量模型整理、基因级错误汇总、`topTable` 与多重检验仍属于完整工作流的后续范围。Python 的 `statsmodels.MixedLM` 提供更广的一般混合模型拟合接口，本方法明确实现精度权重和指定的自由度计算。 [dream 工作流](https://diseaseneurogenomics.github.io/variancePartition/articles/dream.html) [statsmodels MixedLM](https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.html)
