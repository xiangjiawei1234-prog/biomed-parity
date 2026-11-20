# 非随机缺失的链式插补

`biomed_parity.chained_imputation.mice_impute` 支持用 `mnar.norm` 和 `mnar.logreg` 做非随机缺失敏感性分析。每轮使用合格的观测响应拟合条件模型；随后在目标行的预测中施加研究预设的偏移。连续结局的偏移与原结局同单位，二元结局的偏移以对数优势为单位。 [mice NARFCS 手册](https://amices.org/mice/reference/mice.impute.mnar.html) · [Tompsett 等人](https://doi.org/10.1002/sim.7643)

```python
import numpy as np
from biomed_parity.chained_imputation import mice_impute

data = np.array([
    [-1.1, 82], [-0.5, 79], [0.0, np.nan], [0.2, 84], [1.0, 92], [1.3, np.nan],
])
offset = -3 + 2 * data[:, 0]
fit = mice_impute(
    data, methods=[None, "mnar.norm"], mnar_offsets={1: offset},
    m=10, iterations=5, seed=143,
)
```

`mnar_offsets` 以零起点的目标列索引为键，每列可提供常数、与数据行对应的向量，或接收当前链数据并返回每行偏移的函数。固定辅助变量的偏移可以在调用前合成。例如 R 的 `ums="-3+2*x"` 对应 `-3 + 2*x`。当偏移依赖于一列正在插补的协变量时，使用函数读取其当前值。两种方法都支持独立的训练排除和目标单元掩码。偏移作用于 `where` 请求的目标单元，包括显式请求重新插补的已观测单元。 [链式插补](chained-imputation.md)

两种方法沿用已有的贝叶斯正态和带数据增强的 logistic 抽样。偏移由用户指定，数据本身无法识别该偏移。可以在设计矩阵中加入其他变量的原始缺失指示，按分析计划实现较完整的 NARFCS 设定。多重插补的合并推断可继续沿用同一套接口。 [NARFCS 参数与缺失指示](https://pmc.ncbi.nlm.nih.gov/articles/PMC6001532/) · [合并推断](mi-pooling.md)

R 4.5.3 与 mice 3.19.0 的验证包含连续行偏移、二元行偏移和二元常数偏移。每组对比 R 的 1,200 次抽样与 Python 的 1,000 次抽样。逐行均值均在 5 倍联合蒙特卡洛标准误内，同种子下的正态条件抽样差等于给定偏移，二元抽样的变化方向与对数优势偏移一致。另核对链内变化的偏移、重插补及输入验证。 [参考用例](../../validation/mnar_imputation/cases.json) · [R 脚本](../../validation/mnar_imputation/oracle.R)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_mnar_imputation.py
```
