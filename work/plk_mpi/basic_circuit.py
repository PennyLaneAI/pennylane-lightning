import pennylane as qml
import pennylane.numpy as np

USE_MPI = True
if USE_MPI:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

n_qubits = 5
n_layers = 2

if USE_MPI:
    print("Using lightning.kokkos\n")
    dev = qml.device("lightning.kokkos", wires=n_qubits)
else:
    print("Using lightning.qubit\n")
    dev = qml.device("lightning.qubit", wires=n_qubits)

@qml.qnode(dev, diff_method="adjoint")
def circuit(params):
    for i in range(n_qubits):
        qml.Hadamard(i)
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return np.array([qml.expval(qml.PauliZ(i)) for i in range(n_qubits)])


np.random.seed(10)
params = np.random.rand(n_layers, n_qubits, 3)
if USE_MPI:
    params = comm.bcast(params, root=0)
results = circuit(params)
grad = qml.jacobian(circuit)(params)
if not USE_MPI or comm.Get_rank() == 0:
    print(results)
    print(grad)

if USE_MPI and not MPI.Is_finalized():
    MPI.Finalize()