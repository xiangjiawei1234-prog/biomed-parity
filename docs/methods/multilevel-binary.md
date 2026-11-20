# 二元结局的两层链式插补

`biomed_parity.multilevel_binary.impute_two_level_binary` 用 logistic 随机截距模型插补 0/1 结局。也可在 `mice_impute` 中为该列选择 `2l.bin`。预测矩阵用 `-2` 指定完整的簇 ID，用 `1` 指定固定效应预测变量，用 `0` 忽略该列。模型包含固定截距和簇随机截距。随机斜率及多分类响应仍待迁移。 [mice 两层二元手册](https://amices.org/mice/reference/mice.impute.2l.bin.html) · [Audigier 等人](https://doi.org/10.1214/18-STS646)

```python
from biomed_parity.chained_imputation import mice_impute
from biomed_parity.multilevel_binary import TwoLevelBinaryOptions

# 以列 0 为簇编号、列 1 为协变量、列 2 为有部分缺失的二元结局
result = mice_impute(
    data, methods=[None, None, "2l.bin"],
    predictor_matrix=[[0, 0, 0], [0, 0, 0], [-2, 1, 0]],
    two_level_options={2: TwoLevelBinaryOptions(random_effects="laplace")},
    m=10, iterations=5, seed=28,
)
```

模型用簇随机截距的条件众数和曲率计算一阶 Laplace 边际似然，同时拟合固定效应和随机截距方差。每份插补根据拟合系数协方差抽取固定效应，根据随机截距拟合量抽取边际方差。簇内待插补行共享一次簇随机截距抽样，各行随后按自己的预测概率独立抽取 0 或 1。

完全缺失的簇使用零均值的边际随机截距抽样。部分缺失的簇用 `TwoLevelBinaryOptions.random_effects` 选择抽样方式。

| 选项 | 部分缺失簇的随机截距 |
| --- | --- |
| `laplace` | 条件众数为中心，条件曲率方差为抽样方差，也是 Python 默认值 |
| `eb` | 条件众数为中心，用每份插补抽取的边际方差 |
| `marginal` | 零均值，使用边际方差，对应本机 mice 3.19.0 的行为 |

直接调用返回「独立抽样次数 × 目标行数」矩阵，`observed` 和 `where` 分别控制拟合与抽样。链式入口支持与其他方法迭代更新，原始观测响应只在显式指定 `where` 输出时重新插补，并保留为训练输入。 [新版随机效应策略](https://amices.org/mice/articles/random-effects-2l-bin.html)

用 R 4.5.3、lme4 2.0-1 和 mice 3.19.0 核对三个合成场景。包含部分缺失和完全缺失的簇、相关固定效应和方差在零边界的拟合。系数、随机截距方差、条件众数、曲率方差、固定效应协方差和对数似然与 `glmer` 对照。分布对照使用 R 的拟合结果生成三种策略各 4,200 次抽样，Python 各 2,800 次抽样。旧版 `mice.impute.2l.bin` 实际抽样另行核对边际策略。 [R 参考脚本](../../validation/multilevel_binary/oracle.R) · [验证用例](../../validation/multilevel_binary/cases.json)

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_multilevel_binary.py
```
