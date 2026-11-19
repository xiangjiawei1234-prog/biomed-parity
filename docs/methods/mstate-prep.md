# mstate 宽表转风险区间

`biomed_parity.mstate_prep.msprep()` 把每人一行的状态随访宽表转换成计数过程长表。输入时间从统一的时间原点起算。每进入一个有出边的状态，函数为该状态所有出边生成一行风险记录，出边共享进入和离开时点。被选为实际转移的那条出边标记为 1。 [msprep 参考手册](https://hputter.github.io/mstate/reference/msprep.html) · [mstate 方法论文](https://www.jstatsoft.org/article/view/v038i07)

```python
from biomed_parity.mstate_prep import msprep

# 0 为初始状态，1 为中间状态，2 为吸收状态。
long = msprep(
    time=[[None, 2, 6], [None, 4, 7]],
    status=[[None, 1, 1], [None, 1, 0]],
    transitions=[(0, 1), (0, 2), (1, 2)],
    id=["A", "B"],
    keep={"group": ["control", "treatment"]},
)

# 可直接提取用于生存模型的起止时间、事件和分层编号。
start, stop, event, strata = long.t_start, long.t_stop, long.status, long.transition
```

## 输入和返回

- 时间和状态矩阵的行对应受试者，列对应状态。有入边的状态需给出有限非负时点和 0 或 1 的状态指示，图的源状态可以用 `None` 占位。也可传入 `data` 以及长度等于状态数的列名向量。
- `transitions` 接受零基状态的有向边列表，列表顺序决定零基转移编号。也可直接传入 `transition_matrix`，其中 R 风格的一基连续整数代表转移，空位用 `None` 或 `NaN`。两种输入满足相同的无环、无自环约束。
- `start_state` 和 `start_time` 均可按人设置。标量对全体受试者生效，默认状态 0、时间 0。吸收状态没有出边，因此从该状态进入的人不贡献区间。
- `id` 可为一维唯一标识向量，或 `data` 中的列名。默认标识为 1 至样本数。结果依次按 ID、进入时间、出发状态和目标状态排序。`id_order` 可指定分类标识的排序顺序。
- `keep` 可为协变量列映射、向量、矩阵或 `data` 中待保留的列名。各列独立保存于 `covariates`。零基 `subject_index` 可把结果映射回输入表；`transition_matrix` 属性可导出 R 风格的一基矩阵。

## 转移选择

先去掉早于当前状态进入时点的候选时点。若至少一个有效候选记录了事件，采用最早的事件；否则在最早的有效时点删失。若删失记录早于最终选用的事件，发出 `MsprepWarning`，对应本机 R 0.3.3 的行为。最早事件有多个目标时选编号最小的目标并发出警告。恰好在进入时点发生的事件可形成零长度区间。若各出边均无达到进入时点的记录，对照实现使用 `Inf` 作为删失时点。送入回归模型前应按研究计划检查这两类区间。 [msprep 参考手册](https://hputter.github.io/mstate/reference/msprep.html)

## 本地对照

R 4.5.3 和 mstate 0.3.3 生成 8 组测试场景。6 个常规场景的 92 条区间逐字段匹配，覆盖具名数据列、分类 ID 排序、协变量、延迟进入、多源无环图、反向状态编号和并列事件。另两个场景记录历史实现差异。

- 当唯一一人进入中间状态时，R 0.3.3 会漏掉此人的后续区间。Python 保留已知的转移区间，并在独立用例中固定该行为。
- 当全体从吸收状态开始时，R 0.3.3 报错。Python 返回带状态图和正确列形状的空结果。

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_mstate_prep.py
```

可把这一结果直接传入转移风险拟合接口。 [Breslow 转移风险拟合](mstate-fit.md)
