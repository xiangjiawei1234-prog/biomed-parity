# 限制性三次样条设计与预测

`biomed_parity.splines.fit_restricted_cubic_spline` 首先得到结点或读入既有结点，再返回可复用的 `RestrictedCubicSpline`。保存对象后，可用同一组结点、归一化和可选主成分旋转处理新病例。输出为数值设计矩阵，默认与 `rms::rcs` 一样包含线性列。若需要 `Hmisc::rcspline.eval` 的默认列结构，传入 `include_linear=False`。 [Hmisc 样条文档](https://search.r-project.org/CRAN/refmans/Hmisc/html/rcspline.eval.html) · [rms 转换文档](https://search.r-project.org/CRAN/refmans/rms/html/rms.trans.html)

```python
from biomed_parity.splines import fit_restricted_cubic_spline

age_basis = fit_restricted_cubic_spline(age, n_knots=5)
x_train = age_basis.transform(age)
x_predict = age_basis.transform([50, 60, 70])
local_slopes = age_basis.transform([50, 60, 70], derivative=1)
print(age_basis.knots, age_basis.nonlinear_columns)
```

## 结点和基函数

- 传入 `knots` 可运行固定结点分析。结点先去重再升序排列，要求至少三个不同位置。仅估计结点可直接调用 `restricted_cubic_spline_knots`。
- 自动估计使用 R 类型 7 样本分位数。三结点采用 0.10 和 0.90 的外侧分位，四到六结点采用 0.05 和 0.95，七结点及以上采用 0.025 和 0.975。样本较小时采用第五小和第五大观测处理外侧结点，极端并列触发替代结点时保留该端的替代位置。
- `tie_fraction` 默认为 0.05。只有内部并列频率低于阈值时，才检查两端的极端并列频率。触发后把下一个不同值作为外结点，并在剩余样本中计算内部结点。支持稀疏不同取值和密集分位数回退，触发时会发出 `SplineKnotsWarning`，结点不足三个时抛出 `ValueError`。
- `norm=0` 使用原始截断幂基，`norm=1` 按最后两结点间距的立方归一化，`norm=2` 按首尾结点间距的平方归一化。普通基在两端保持线性。按最右结点作多项式展开，避免远尾处近等立方项相减。
- `kind="integral"` 返回积分基，前两列为 (x) 和 (x^2/2)。普通和积分基均接受一阶或二阶导数。默认传播缺失值，指定 `missing_value` 后则先忽略缺失值估计结点，再使用给定值转换缺失行。 [样条公式与建模背景](https://hbiostat.org/rmsc/genreg#restricted-cubic-splines)
- `principal_components=True` 使用训练设计的中心、样本标准差与奇异向量。符号固定为绝对载荷最大的分量取正。复现 R 拟合的已有系数时，应同时使用该 R 拟合的旋转方向。后续转换沿用训练参数。

Patsy 和 scikit-learn 均提供样条设计接口。已有用 Hmisc 或 rms 拟合的系数需配合当时的结点、列顺序及基函数尺度。 [Patsy 样条](https://patsy.readthedocs.io/en/latest/spline-regression.html) · [scikit-learn 样条](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)

## 本地验证

19 个场景对照 R 4.5.3、Hmisc 5.3-1 和 rms 8.2-1。逐字段核对结点、训练矩阵、复用结点后的预测矩阵、积分基的一阶导数及主成分转换。R 与 Python 的主成分按载荷符号对齐。最大数值绝对差约为 `3.27e-12`。另以有限差分检查解析导数，并覆盖远尾、缺失值及保存后恢复的设计对象。

```sh
R_LIBS_USER="$PWD/.r-lib" BIOMED_R_PARITY=1 .venv/bin/python -m pytest -q -W error tests/test_splines.py
```
