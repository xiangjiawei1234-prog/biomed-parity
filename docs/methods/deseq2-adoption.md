# DESeq2 与 PyDESeq2 的采用验证

对人体肿瘤与参考组织的合成表达矩阵，使用 PyDESeq2 0.6.0 和 Bioconductor DESeq2 1.50.2 核对标准 Wald 工作流。PyDESeq2 可以承担本次覆盖的差异表达统计。依赖 R 数值精确复现或依赖似然比检验与局部方差稳定化变换的工作流，仍需单独处理。[PyDESeq2 0.6.0](https://pypi.org/project/pydeseq2/0.6.0/) [DESeq2 参考指南](https://www.bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html)

## 输入与结果

固定随机种子生成 12 个独立样本、240 个匿名特征的负二项计数。两个条件在同一批次内平衡，各模型使用 `~batch + condition`，比较肿瘤与参考条件。两端都采用比值中位数归一化，并关闭 Cook 异常值替换、过滤和独立过滤。采用 Benjamini–Hochberg 校正，显著性阈值为调整后 p 值小于 0.05。一个特征为全零，其余 239 个特征用于推断。

| 对照指标 | 参数化离散度拟合 | 均值离散度拟合 |
| --- | ---: | ---: |
| 归一化因子最大绝对差 | 4.66 × 10⁻¹⁵ | 4.66 × 10⁻¹⁵ |
| 离散度最大绝对差 | 2.44 × 10⁻⁴ | 3.04 × 10⁻³ |
| log₂ 倍数变化最大绝对差 | 1.77 × 10⁻⁵ | 2.51 × 10⁻⁴ |
| 调整后 p 值最大绝对差 | 2.00 × 10⁻⁴ | 3.61 × 10⁻³ |
| 两端完全一致的显著特征数 | 46 | 47 |

逐字段比较涵盖平均归一化计数、离散度、log₂ 倍数变化、标准误、Wald 统计量和两类 p 值。全零特征的推断缺失状态也一致。这些差异和显著特征集合针对本次合成输入，靠近阈值的其他数据仍需核对。[本地对照结果](../../validation/adoption/pydeseq2/results.json)

## 兼容性边界

| 方法 | 实测结果 |
| --- | --- |
| 盲态参数化方差稳定化变换 | 12 × 240 个变换值均有限。绝对差中位数 0.000551，最大值 0.002494 |
| 局部离散度趋势的盲态方差稳定化变换 | R 返回 12 × 240 个变换值。PyDESeq2 的 `vst(fit_type="local")` 报告该拟合类型未经实现 |
| 负二项似然比检验 | R 使用完整模型 `~batch + condition`、约简模型 `~batch`，得到 239 个有限 p 值。PyDESeq2 的 `deseq2` 接口拒绝 `test="LRT"` |

本次 0.6.0 范围集中在固定对比的未收缩效应、离散度和 Wald 推断。Cook 处理、独立过滤、不同归一化、多因子交互及效应收缩待另行对照。后续开发提交的局部变换、已给定离散度曲线的精确复现，以及固定离散度的 LRT 已另外验证。 [局部 VST](local-vst.md) · [负二项 LRT](nbinom-lrt.md)

## 复核

仓库的 Python 核心环境可保持现有依赖。此脚本在独立 Python 环境运行，并使用同一 R 主版本对应的 Bioconductor R 包。在 R 4.5 上运行时，可将 DESeq2 安装到以下仓库内 R 库。该步骤依赖现有的 `BiocManager`。

```sh
mkdir -p .r-lib/deseq2
R_LIBS_USER=.r-lib/deseq2 Rscript --vanilla -e 'BiocManager::install("DESeq2", lib=Sys.getenv("R_LIBS_USER"), ask=FALSE, update=FALSE)'
audit_env="$(mktemp -d)"
python3 -m venv "$audit_env"
"$audit_env/bin/python" -m pip install 'PyDESeq2==0.6.0' 'numpy==2.5.5' 'scipy==1.18.1' 'pandas==3.1.1'
R_LIBS_USER=.r-lib/deseq2 "$audit_env/bin/python" validation/adoption/pydeseq2/audit.py
```

Python 与 R 都在临时目录中读取同一输入。最终报告包含输入散列、版本、配置和差异统计。[Python 验证脚本](../../validation/adoption/pydeseq2/audit.py) [R 对照脚本](../../validation/adoption/pydeseq2/run_r.R)
