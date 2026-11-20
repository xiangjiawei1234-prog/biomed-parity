# 多变量链式插补

`biomed_parity.chained_imputation.mice_impute` 对连续、分类和两层聚类数据执行完全条件设定。每条链按指定顺序更新缺失列。返回完整数据（链 × 观测 × 列）、每轮每列插补值的均值和样本方差，以及实际插补单元的布尔矩阵。缺失值使用 `numpy.nan`。 [mice 主接口](https://amices.org/mice/reference/mice.html)

| 方法 | 条件模型 |
| --- | --- |
| `pmm` | 预测均值匹配。支持 0、1、2 三类匹配距离，可设候选供体数，插补值来自观测响应 |
| `norm` | 贝叶斯线性回归。每轮抽样残差尺度和回归系数，然后生成预测残差 |
| `norm.predict` | 线性模型的点预测 |
| `mean` | 合格训练响应的均值 |
| `logreg` | 观测 0/1 响应的 logistic 回归，使用增强样本缓解完全分离，然后抽样回归系数与二元结局 |
| `polyreg` | 无序多分类。拟合基准类别多项 logistic 模型后，按预测概率抽取类别 |
| `polr` | 有序多分类。拟合比例优势模型后，按预测概率抽取类别；拟合失败或仅有两个可训练类别时使用多项模型 |
| `2l.pan`、`2l.norm` | 两层高斯模型，分别支持同质及簇特异残差方差 |
| callable | 被动变量。根据当前矩阵重新计算实际指定的插补单元 |
| `None` 或空字符串 | 跳过该列 |

匹配方法与二元回归依据原作者方法，随机数流采用 NumPy。单元素缺失时，二元增强分支遵循参考方法的启用条件。 [PMM](https://amices.org/mice/reference/mice.impute.pmm.html) · [贝叶斯正态](https://amices.org/mice/reference/mice.impute.norm.html) · [logistic 回归](https://amices.org/mice/reference/mice.impute.logreg.html)

`polyreg` 与 `polr` 使用平衡的伪观测缓解完全分离。经过加权极大似然拟合后，两种方法每轮从拟合概率抽样结局。提供 `category_levels={列索引: 等级列表}` 声明数值标签和等级顺序，`polyreg` 与 `polr` 默认使用观测标签的升序。作为预测变量时，分类列采用处理对比，`polr` 列采用等间距等级的正交多项式对比。只在合格训练行中出现的类别构成响应的抽样支持。只有一个可训练类别时直接填入该类别。类别频次比标签均值更适合检查分类链。 [多项模型](https://amices.org/mice/reference/mice.impute.polyreg.html) · [比例优势模型](https://amices.org/mice/reference/mice.impute.polr.html) · [White 等人的增强方法](https://pmc.ncbi.nlm.nih.gov/articles/PMC3990447/) · [R 对比编码](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/contrast.html)

```python
import numpy as np
from biomed_parity.chained_imputation import mice_impute

data = np.array([
    [0, 12, 1, 24],
    [1, 19, 0, 38],
    [2, np.nan, 1, np.nan],
    [3, 21, 1, 42],
    [4, 16, 0, 32],
    [5, 28, 0, 56],
])

result = mice_impute(
    data,
    methods=[None, "pmm", None, lambda current: 2 * current[:, 1]],
    predictor_matrix=[
        [0, 0, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
    visit_sequence=[1, 3],
    m=5,
    iterations=8,
    donors=3,
    seed=17,
)
print(result.completed[:, 2, 1])
```

预测矩阵的行是待插补列，列是预测变量。`visit_sequence` 以零为起点，允许同一轮多次访问。`where` 可指定插补范围及对已观测单元重新插补。`ignore` 排除训练行，保留这些行的插补输出。链内模型持续使用原始观测值。可提供共享的 `initial` 矩阵，支持固定初值对照；默认从可用的观测响应初始化。被动函数收到只读矩阵，应在目标行返回有限值。按实际缺失分布、变量约束和研究方案选择条件模型。

两层结局的预测矩阵行使用 `-2` 指定唯一完整的簇 ID 列，其余编码按两层条件模型解释。簇 ID 在链中固定。通过 `two_level_options={列索引: TwoLevelOptions(...)}` 指定各两层结局的内层 Gibbs 扫描设置。更新会读取同一链当前的预测变量，对缺失结局、单元补插与拟合留出使用公共链式规则。 [两层模型及示例](multilevel-imputation.md)

当前入口使用数值矩阵和单列条件模型。R 与 Python 的种子分别对应各自可复现的随机链。R 的公式接口、因子字符串、插补块、非随机缺失和诊断仍待迁移。

## 本地对照

- 三组 R 4.5.3 / mice 3.19.0 完整链核对逐格插补、每轮均值及方差，覆盖相互依赖的缺失列、被动计算、重复访问、训练留出和重插补。唯一最近供体的 0 型 PMM 逐值一致。全部确定性矩阵的最大绝对差低于 `6e-13`。
- 五组随机对照分别使用每组 2,200 次独立抽样，涵盖贝叶斯正态、1 型及 2 型 PMM、普通及分离二元回归。逐目标行均值和协方差均落在独立蒙特卡洛误差允许范围内。比较保留随机误差，逐格对照使用确定性场景。
- 六组分类用例上，R `nnet::multinom` 和 `MASS::polr` 的预测概率与 Python 的最大绝对差为 `1.52e-7`，包含完全分离、自定义顺序、单目标和二元回退。四组中进一步各取 R 400 次与 Python 700 次独立抽样，逐类别频率均在 5 倍联合蒙特卡洛标准误范围内。三个等级数的有序对比矩阵与 R 逐项核对。
- 两层链式调用的逐目标行均值分别与 `2l.pan` 和 `2l.norm` 的 R 快照核对。另验证链式更新读取最新插补的协变量、衔接被动变量并保留完整簇 ID。
- 另核对观测值保留、范围掩码、指定初值、共线预测变量、相同种子重放和缺少预测变量时的错误定位。

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error \
    tests/test_chained_imputation.py tests/test_categorical_imputation.py tests/test_multilevel_imputation.py
```
