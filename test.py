# %%
import kinematics as sbk
import numpy as np

M = sbk.Marker(coords=np.array([[0.5, 0.6, 0.7], [0.4, 0.3, 0.2]]), fs=2.3)
R = sbk.ReferenceFrame(origin=np.array([[1, 0, 0]]), orientation=np.array([[1,0,0], [0, 0,1], [0,1,0]]))
print("M:")
print(M)
print("R:")
print(R)
print("M --> R:")
print(M.to_frame(R))
check = 1
