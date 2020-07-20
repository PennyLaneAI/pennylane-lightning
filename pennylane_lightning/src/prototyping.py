from lightning_qubit_ops import apply_3q
import numpy as np

qubits = 3
# state = np.zeros(2 ** qubits, dtype="complex")
# state[0] = 1
states = np.eye(8, dtype="complex")
ops = ["Hadamard", "Hadamard"]
wires = [[0], [1], [0, 1]]
params = [[], [], []]

out = np.array([np.real_if_close(apply_3q(states[i], ops, wires, params)) for i in range(1)])
print(out)
