# %%
import kinematics as sbk
import numpy as np

a = sbk.ReferenceFrame()
b = sbk.ReferenceFrame(origin=np.array([[1, 0, 0]]), orientation=np.array([[1,0,0], [0, 0,1], [0,1,0]]))
v = np.array([[1.5, 0.3, 0.4]])
k = b.to_local(v)
