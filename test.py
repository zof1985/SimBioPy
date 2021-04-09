# %%
import kinematics as sbk
import numpy as np

a = sbk._UnitDataFrame(data = {'A': [0, 10, 15, 0.3]}, index = np.linspace(0, 10, 4), dim_unit = "N", time_unit = "s")
b = sbk._UnitDataFrame(data = {'B': [2, 11, 1, 0.1]}, index = np.linspace(0, 10, 4), dim_unit = "N", time_unit = "s")
c = 14.5
d = a + b
e = a + c
f = c + a
check = 1
