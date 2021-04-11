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
K = M.change_frame(R).as_vector()
V = M.as_vector()
K = V * np.array([-1, -1, -1]) # 180°
G, M = K.angle_from(V, return_matrix = True)
print(M + K)
print(K + M)
print(M + M.coordinates)
print(M + M.coordinates.values)
print(M + 3.5)
print(3.5 + M)
check = 1
