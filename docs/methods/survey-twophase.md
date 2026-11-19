# survey 两阶段调查中的均值和总量

`survey_twophase` 迁移 `survey::twophase(method = "full")` 的均值、总量、方差和协方差估计。第一阶段可为独立同分布样本，或给定总体容量的有限总体简单随机抽样。有限总体第一阶段可分层。第二阶段从第一阶段样本中抽取，可分层抽元素，也可在各层抽取初级抽样单元，再在各入样初级单元内简单随机抽取元素。所有有限抽样均按无放回处理。 [R twophase 手册](https://search.r-project.org/CRAN/refmans/survey/html/twophase.html) · [两阶段抽样方差](https://r-survey.r-forge.r-project.org/survey/doc/phase1.pdf)

```python
from biomed_parity.survey_twophase import survey_twophase

fit = survey_twophase(
    [4.0, None, 8.0, 6.0, None, 5.0],
    [True, False, True, True, False, True],
    phase2_strata=["A", "A", "A", "B", "B", "B"],
    phase1_population=60,
    statistic="mean",
)
print(fit.estimate, fit.standard_errors)
```

`phase2` 和所有分层及整群标签都按完整的第一阶段抽样框提供。结果中的入样概率和权重依照第二阶段入样行的原始次序排列。指定 `phase2_psu` 后，同一初级抽样单元内的第二阶段元素采用独立于其他初级单元的条件抽样率。同名初级单元按第二阶段的层区分。每个第二阶段层需有入样元素或入样初级单元。

`values` 可传一列或多列数值，完整第一阶段数据允许未入样行缺失。也可仅传第二阶段观测，次序须与选择向量一致。字符串向量自动展开成类别指示列。通过 `categories` 声明类别顺序、保留未出现的水平，或传入数字编码的类别。类别列的均值表示加权比例。

总量使用逆概率权重求和。均值使用归一化权重，方差采用加权中心化数值的一阶线性化。设每阶段的成对系数为 `D = 1 - π_i π_j / π_ij`。总方差的成对系数是 `D1 + D2 - D1 * D2`，第一阶段和第二阶段的贡献分别来自 `D1 * (1 - D2)` 和 `D2`。实现通过分层、整群和交叉分组求和计算这些项，省去构造样本量平方大小的矩阵。结果含估计、标准误、总协方差和两个方差贡献。 [Horvitz–Thompson 两阶段抽样估计](https://faculty.washington.edu/tlumley/old-survey/survcurve.pdf)

已支持的设计使用实际入样单元和抽样框计算条件概率。抽样权重校准、显式概率或不等概率抽样、第一阶段整群、初级单元内再次分层、调查域、缺失入样观测和 R 的 `simple/approx` 方差保留为单独的功能。

## 本地验证

使用 R 4.5.3、survey 4.5 验证 10 种设计的均值和总量，包括交叉分层、单个入样单元、不同设计的普查、类别和两级整群抽样。逐项比较两个阶段的入样概率、权重、点估计、协方差、协方差分量和标准误。可直接对齐以及按下文稳定重排行后的案例均通过，最大绝对差 `7.28e-12`。

在第一阶段不同层的样本数不相等且第二阶段行分散在抽样框时，survey 4.5 在一个用例中基于原始位置取得层样本数，输出非对称总量协方差。其两个非对角项约为 `2940.31` 和 `2919.11`。保持设计和第二阶段相对顺序，将入样行排到抽样框前面后，两项均为 `3014.78`。Python 使用完整抽样框中每个层的实际样本数，结果与稳定重排行后的 R 及独立的逐对入样概率计算一致，行排列也通过了不变性测试。对照文件保留两个 R 输出。

```bash
BIOMED_R_PARITY=1 R_LIBS_USER="$PWD/.r-lib" .venv/bin/python -m pytest -q tests/test_survey_twophase.py
```

[输入用例](../../validation/survey_twophase/cases.json) · [R 对照脚本](../../validation/survey_twophase/oracle.R) · [数值基线](../../validation/survey_twophase/expected.json)
