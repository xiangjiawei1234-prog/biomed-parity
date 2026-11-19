# limma 线性模型和样本权重对照

2026-11-19 使用 R 4.5.3、limma 3.66.0、statmod 1.5.2、Python 3.12.7 和 NumPy 2.5.5。输入为固定种子生成的人工表达矩阵。线性模型的两套矩阵各有 80 个匿名特征及 36 次观测，分为 12 个重复取样对象。每个对象有三次观测，或在第二套数据中有二至四次观测。

比较两组对比系数、未缩放标准误、残差标准差、残差自由度、后验方差、推断自由度、moderated t、p 值与 BH 调整 p 值。分块模型额外核对逐特征 Fisher z 变换相关系数及综合相关系数。模型比较相对容差为 `1e-6`、绝对容差为 `1e-9`，相关系数比较分别为 `1e-8`、`1e-10`。

| 候选 | 验证范围 | 结果 |
| --- | --- | --- |
| InMoose 0.9.1 | 普通 `lmFit`、单个对比和经验贝叶斯推断 | 1,440/1,440，最大绝对差 `4.66e-14`。分块最小二乘在两套输入上均抛出 `NotImplementedError` |
| pylimma 0.3.0 | 普通线性模型及 `duplicateCorrelation`，随后用 R 给定相关系数或 Python 自估相关系数进行分块拟合和经验贝叶斯推断 | 4,482/4,482，最大绝对差 `7.11e-14` |
| edgePython 0.5.1 | `duplicateCorrelation` 对照 | 162/162。逐特征 Fisher z 变换相关系数最大绝对差 `6.22e-15` |

pylimma 还通过了 `voomWithQualityWeights` 的 80 特征 × 8 样本场景，1,288 个表达值、精度权重和样本权重均匹配，最大绝对差为 `2.62e-13`。在单独的 8 × 8 场景中，原生接口的 64 个转换值匹配，64 个精度权重和 8 个样本权重均超出容差，最大绝对差分别为 `6.36303366` 和 `0.0432060132`。当特征数与样本数相等时，R 3.66.0 将中间向量按特征广播，pylimma 明确按样本广播。使用 `voom` 和 `array_weights` 显式进行两轮拟合，第二轮以 `initial_weights[:, None]` 按特征传入，复现了 R 的全部 136 个值，最大绝对差为 `6.76e-14`。复现脚本保留该分支，供对齐历史结果使用。 [pylimma 已知差异](https://pylimma.readthedocs.io/en/latest/validation/known_differences.html)

该对照覆盖两组设计、完整输入及上述权重场景。缺失值、多因素和稳健经验贝叶斯需另行验证。可在验证过的范围内使用已有 Python 包。 [limma 手册](https://bioconductor.org/packages/release/bioc/manuals/limma/man/limma.pdf) · [pylimma API](https://pylimma.readthedocs.io/en/latest/api.html) · [InMoose API](https://inmoose.readthedocs.io/en/stable/limma.html) · [edgePython](https://pypi.org/project/edgepython/0.5.1/)

## 复现

从仓库根目录运行。命令使用已安装 limma 3.66.0 和 statmod 1.5.2 的 R 环境。

```bash
limma_audit_dir="$(mktemp -d)"
for candidate in inmoose pylimma edgepython; do
    uv venv --python 3.12 "$limma_audit_dir/$candidate"
done
uv pip install --python "$limma_audit_dir/inmoose/bin/python" inmoose==0.9.1 numpy==2.5.5 scipy==1.18.1
uv pip install --python "$limma_audit_dir/pylimma/bin/python" pylimma==0.3.0 numpy==2.5.5 scipy==1.18.1
uv pip install --python "$limma_audit_dir/edgepython/bin/python" edgepython==0.5.1 numpy==2.5.5 scipy==1.18.1
OMP_NUM_THREADS=1 "$limma_audit_dir/inmoose/bin/python" validation/adoption/limma/compare.py inmoose
OMP_NUM_THREADS=1 "$limma_audit_dir/pylimma/bin/python" validation/adoption/limma/compare.py pylimma
OMP_NUM_THREADS=1 "$limma_audit_dir/edgepython/bin/python" validation/adoption/limma/compare.py edgepython
```

脚本在预计一致的输出超出容差时会以非零状态退出，并记录已知行为差异。
