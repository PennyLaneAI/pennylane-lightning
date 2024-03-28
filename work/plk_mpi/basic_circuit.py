import pennylane as qml
import pennylane.numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD

n_qubits = 30
n_layers = 2

dev = qml.device("lightning.kokkos", wires=n_qubits)
dq = qml.device("default.qubit", wires=n_qubits)


def circuit(params):
    for i in range(n_qubits):
        qml.Hadamard(i)
    # qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))


np.random.seed(10)
params = np.random.rand(n_layers, n_qubits, 3)
params = comm.bcast(params, root=0)

cir_mpi = qml.QNode(circuit, dev, diff_method="adjoint")
res_mpi = cir_mpi(params)
print("done!")
# jac_mpi = qml.jacobian(cir_mpi)(params)
# cir_ref = qml.QNode(circuit, dq, diff_method="adjoint")
# res_ref = cir_ref(params)
# jac_ref = qml.jacobian(cir_ref)(params)
# if comm.Get_rank() == 0:
#     print(np.max(np.abs(res_mpi - res_ref)))
#     print(np.max(np.abs(jac_mpi - jac_ref)))

if not MPI.Is_finalized():
    MPI.Finalize()
