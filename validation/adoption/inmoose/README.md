# edgeR 准似然流程的 Python 候选对照

2026-11-19 在本机使用 R 4.5.3、edgeR 4.8.2、limma 3.66.0、Python 3.12.7 和 NumPy 2.5.5。对比 InMoose 0.9.1 和 edgePython 0.5.1。输入为两套模拟表达矩阵，每套含 96 个匿名特征、8 个样本和两组设计。其中一套为中等与较高整数计数，另一套包含零值、极小计数与四分之一单位分数。两端使用相同的组别设计、文库大小和丰度趋势配置。

逐特征核对 F 统计量、p 值、有效残差自由度、后验准似然离散度、负二项离散度、logFC 和 logCPM。相对容差为 `1e-6`，绝对容差为 `1e-9`。

| Python 候选 | R 对照与场景 | 结果 |
| --- | --- | --- |
| InMoose 0.9.1 | legacy 模式，显式提供负二项离散度 | 两套输入均通过，最大绝对差 `1.94e-13` |
| InMoose 0.9.1 | v4 模式，显式提供负二项离散度 | 两套输入的有效自由度均为 0/96 匹配。低计数分数输入中，p 值仅 2/96 匹配，最大绝对差 `0.169481766` |
| edgePython 0.5.1 | legacy 和 v4，显式提供负二项离散度 | 两套输入均通过，最大绝对差 `2.89e-11` |
| edgePython 0.5.1 | v4 自动估计负二项离散度 | 两套输入均通过，最大绝对差 `1.30e-10` |

edgePython 0.5.1 在此次 4,032 个数值比较中均处于容差内。对这两个准似然用例，edgePython 可从 Python 直接运行，无需仓库新增实现。其他上游功能继续按原工作流验证，包含归一化、阈值检验、稳健经验贝叶斯、多因素设计和全零特征。edgePython 发布说明以 edgeR 5 系列作为主要对照，这里的基准为本机 edgeR 4.8.2。InMoose 的拟合输入使用带列名的 Patsy 设计矩阵和 pandas DataFrame，系数位置使用 Python 的零基索引。 [edgePython 包说明](https://pypi.org/project/edgepython/0.5.1/) · [InMoose QL 接口](https://inmoose.readthedocs.io/en/stable/generated/inmoose.edgepy.glmQLFit.html) · [edgeR v4 方法论文](https://doi.org/10.1093/nar/gkaf018)

## 复现

从仓库根目录运行。使用本机 R 的 edgeR 4.8.2 和 limma 3.66.0。两个候选分开安装，核心项目环境保持独立。

```bash
edger_audit_dir="$(mktemp -d)"
uv venv --python 3.12 "$edger_audit_dir/inmoose"
uv venv --python 3.12 "$edger_audit_dir/edgepython"
uv pip install --python "$edger_audit_dir/inmoose/bin/python" inmoose==0.9.1 numpy==2.5.5 scipy==1.18.1
uv pip install --python "$edger_audit_dir/edgepython/bin/python" edgepython==0.5.1 numpy==2.5.5 scipy==1.18.1
OMP_NUM_THREADS=1 "$edger_audit_dir/inmoose/bin/python" validation/adoption/inmoose/compare.py inmoose
OMP_NUM_THREADS=1 "$edger_audit_dir/edgepython/bin/python" validation/adoption/inmoose/compare.py edgepython
```

脚本打印逐字段的匹配数量和最大绝对差。对 InMoose 的 legacy 对照与 edgePython 的全部对照，超出容差时脚本会以非零状态退出。
