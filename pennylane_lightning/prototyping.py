import numpy as np
from lightning_qubit_ops import mvp, test

vec = np.zeros(4, dtype="complex")
vec[0] = 1
mat = np.eye(2, dtype="complex")

# res = mvp(mat, vec, [0])

print(test(vec))
# print(res)

# vec = np.array([1, 0, 0, 0, 0, 0, 0, 0])
# mat = np.array([[0, 1], [1, 0]])
#
# np.tensordot(mat, vec, [1])
