import pennylane as qml
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD


def bench_circuit(n_qubits: int):
    # n_qubits = 25
    n_layers = 1

    dev = qml.device("lightning.kokkos", wires=n_qubits)

    @qml.qnode(dev, diff_method=None)
    def circuit(params):
        for i in range(n_qubits):
            qml.Hadamard(i)
        qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    np.random.seed(10)
    params = np.random.rand(n_layers, n_qubits, 3)
    params = comm.bcast(params, root=0)
    _ = circuit(params)


if __name__ == "__main__":
    import os
    import sys
    import time

    model = sys.argv[1]
    n_qubits = int(sys.argv[2])
    omp_num = int(os.getenv("OMP_NUM_THREADS", -1))
    mpi_size = comm.Get_size()
    # for i in range(20, 30):
    comm.Barrier()
    iter = 0
    t0 = time.time()
    while time.time() - t0 < 1.0:
        iter += 1
        bench_circuit(n_qubits)
    comm.Barrier()
    t1 = (time.time() - t0) / iter
    if comm.Get_rank() == 0:
        datum = f"{mpi_size:12d} {omp_num:12d} {n_qubits:12d} {t1:0.6e}"
        with open("timings_" + model + ".txt", "a") as f:
            f.write(datum + "\n")
        print(datum)
    comm.Barrier()
    if not MPI.Is_finalized():
        MPI.Finalize()
