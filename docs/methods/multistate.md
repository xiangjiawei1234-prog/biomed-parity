# mstate::probtrans 多状态转移概率

## 功能与接口

`probtrans(times, transitions, cumulative_hazards, *, predt=0, direction="forward", n_states=None, cumulative_covariance=None, method="aalen", covariance=False)` 对预估的状态间累积风险作 Aalen–Johansen 乘积积分。给定起始状态和时间，输出各后续时点处于每个状态的概率。也可固定终点，从多个起始时点预测。 [mstate::probtrans 手册](https://search.r-project.org/CRAN/refmans/mstate/html/probtrans.html) [多状态模型教程](https://cloud.r-project.org/web/packages/mstate/vignettes/Tutorial.pdf)

```python
from biomed_parity.multistate import probtrans

fit = probtrans(
    times=[1, 3, 8],
    transitions=[[0, 1], [0, 2], [1, 2]],
    cumulative_hazards=[[0.1, 0.05, 0.05], [0.25, 0.11, 0.15], [0.35, 0.21, 0.40]],
    predt=1,
)
print(fit.times)
print(fit.probabilities[:, 0, :])
```

- `transitions` 的每行是一条零起始编号的有向边，列为来源和目的状态。允许可逆边和吸收状态。用 `n_states` 可保留完全没有邻边的状态。
- 风险矩阵按时间 × 有向边传入。先把各边不同的观测网格对齐到共同时间轴，用各边的前值延续累积风险，首次出现之前填零。若同一时刻有多行累积值，每条边应按非递减顺序排列，程序保留最后一行作为该时刻的整体状态。
- `direction="forward"` 将 `predt` 作为起点，输出 `predt` 和之后的网格时刻。 `direction="fixedhorizon"` 将 `predt` 作为终点，输出从零及此前正时间网格出发的概率。在区间 $(s,t]$ 内应用跳跃，起点排除且终点纳入。空区间返回单位矩阵。
- 累积风险协方差按时间 × 边 × 边传入，对应 R `msfit` 对象的 `varHaz`。提供该数组即可输出标准误，另设 `covariance=True` 可取得完整协方差矩阵。协方差每一行或列的下标为来源编号 + 状态数 × 目的编号，顺序与 R 的列优先向量化相同。
- 对每个返回时点，概率与标准误的两个状态维度依次为来源和目的。R 在某个状态的总离开风险增量超过 1 时会给出警告并继续计算。这里抛出 `ValueError`，避免输出负的占用概率。

## 计算

把相邻累积风险的差写入跳跃矩阵的非对角线。每行的对角线为 1 减去该来源状态到各目的的增量总和。每个不同时间点形成一份矩阵，同一时刻的多条边并入一个矩阵。按时间排列的有序乘积给出区间转移概率。方差计算将累积风险协方差差分，与概率矩阵乘法的递推相结合，按 mstate 0.3.3 中 `aalen` 和 `greenwood` 的分支传播不确定性。 [mstate 数值实现](https://github.com/hputter/mstate/blob/master/R/probtrans.R)

## 本地验证

```bash
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q tests/test_multistate.py
R_LIBS_USER="$PWD/.r-lib" .venv/bin/python validation/multistate/verify.py
```

R 对照使用 R 4.5.3、`mstate` 0.3.3 和 `jsonlite`。6 个固定用例覆盖固定起点、固定终点、预测与跳跃同刻、同刻多条边、可逆路径、孤立状态、Aalen 与 Greenwood 标准误及完整协方差。时间网格逐项相等，数值比较使用绝对容差 `1e-13`、相对容差 `1e-12`。Python 测试另外检查区间端点、空区间及非法风险增量。

R 0.3.3 的固定终点分支把与第 i 行标准误对应的协方差写入数组的第 i+1 个切片。参考脚本按报告的标准误时间轴重新对齐这些切片。Python 返回的概率、标准误与协方差始终共享同一时间索引。

## 兼容范围

当前接口从已估计的累积风险与协方差开始。原始记录到转移风险集的整理、Cox 风险拟合、重采样估计方差、竞争风险回归及绘图是独立步骤。开发版 mstate 另外提供矩阵指数选项，当前路径对照的是 CRAN 0.3.3 的乘积极限。PyMSM 提供多状态拟合和模拟预测，可用于上游风险建模，接口面向个体轨迹。 [CRAN mstate](https://cran.r-project.org/package=mstate) [开发版 probtrans](https://github.com/hputter/mstate/blob/master/R/probtrans.R) [PyMSM 方法](https://hrossman.github.io/pymsm/methods/)
