# %%
import base
import numpy as np
import pandas as pd

a = 0.5
b = 0.04
x = pd.DataFrame(np.arange(1, 101), columns = ["x1"]).abs()
y = b * x / (a + x) + np.atleast_2d(np.random.randn(100)).T
y.columns = ['y1']


# %%
PR = base.HyperbolicRegression(y, x)
PR
print(PR)
str(PR)
PR.to_latex()
check = 1
