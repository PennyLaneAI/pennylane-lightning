import pennylane as qml
import pennylane.numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
n_qubits = 10
n_layers = 1

dev = qml.device("lightning.kokkos", wires=n_qubits)
# dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, diff_method="adjoint")
def circuit(params):
    for i in range(n_qubits):
        qml.Hadamard(i)
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return np.array([qml.expval(qml.PauliZ(i)) for i in range(n_qubits)])


np.random.seed(10)
params = np.random.rand(n_layers, n_qubits, 3)
params = comm.bcast(params, root=0)
results = circuit(params)
if comm.Get_rank() == 0:
    print(results)

if not MPI.Is_finalized():
    MPI.Finalize()