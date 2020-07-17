import numpy as np
from lightning_qubit_ops import mvp, test

vec = np.ones(2, dtype="complex")
mat = np.eye(2, dtype="complex")

res = mvp(mat, vec, [0])

print(test(mat, vec, [0]))
print(res)
