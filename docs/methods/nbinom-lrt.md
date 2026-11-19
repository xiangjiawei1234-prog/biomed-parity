# 固定离散度的负二项似然比检验

`biomed_parity.nbinom_lrt.nbinom_lrt(counts, full, reduced, dispersions, size_factors)` 为 PyDESeq2 结果接上嵌套模型的检验步骤。输入匿名特征 × 样本整数计数、两套样本 × 参数设计矩阵、每个特征的离散度，以及每个样本或特征 × 样本的归一化因子。模型必须满列秩，约简模型的列空间为完整模型的真子空间。 [DESeq2 方法手册](https://github.com/thelovelab/DESeq2/blob/devel/man/nbinomLRT.Rd) · [DESeq2 用户指南](https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html)

实现固定两套模型的离散度和归一化因子，分别最大化带对数链接的负二项似然。自由度是设计矩阵列数之差。返回完整与约简对数似然、两倍似然差、卡方 p 值和对可检验特征进行的 BH 校正。全零特征保留为缺失。PyDESeq2 中用于差异表达的计数矩阵为样本 × 特征，传入此入口前沿两个轴转置。

本地使用 DESeq2 1.50.2 的 `nbinomLRT` 验证两组人工输入。每组为 8 个匿名特征 × 18 个样本。两自由度方案控制批次，一自由度方案使用逐特征归一化。统计量最大绝对差为 `6.31e-8`，BH p 值最大绝对差为 `2.74e-8`。这些用例覆盖常规可估计模型。低均值触及 R 的 `minmu` 下限、估计离散度、Cook 过滤、独立过滤，以及以 glmGamPoi 为后端的检验留作独立工作包。 [PyDESeq2 本地核对](deseq2-adoption.md)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest tests/test_nbinom_lrt.py
```
