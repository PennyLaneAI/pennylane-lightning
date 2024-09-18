# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the :mod:`pennylane_lightning.LightningGPU` device (MPI).
"""
import numpy as np
import pennylane as qml
import pytest
from conftest import device_name

# pylint: disable=missing-function-docstring,unnecessary-comprehension,too-many-arguments,wrong-import-order,unused-variable,c-extension-no-member
from mpi4py import MPI

if device_name == "lightning.gpu":
    pytest.skip("LGPU new API in WIP.  Skipping.", allow_module_level=True)

numQubits = 8


def create_random_init_state(numWires, R_DTYPE, seed_value=48):
    np.random.seed(seed_value)
    num_elements = 1 << numWires
    init_state = np.random.rand(num_elements).astype(R_DTYPE) + 1j * np.random.rand(
        num_elements
    ).astype(R_DTYPE)
    scale_sum = np.sqrt(np.sum(np.abs(init_state) ** 2)).astype(R_DTYPE)
    init_state = init_state / scale_sum
    return init_state


def apply_probs_nonparam(tol, operation, GateWires, Wires, C_DTYPE):
    num_wires = numQubits
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    commSize = comm.Get_size()

    dev_cpu = qml.device("lightning.qubit", wires=num_wires, c_dtype=C_DTYPE)
    dev_mpi = qml.device(device_name, wires=num_wires, mpi=True, c_dtype=C_DTYPE)

    state_vector = create_random_init_state(num_wires, dev_mpi.R_DTYPE)
    comm.Bcast(state_vector, root=0)

    def circuit():
        qml.StatePrep(state_vector, wires=range(num_wires))
        operation(wires=GateWires)
        return qml.probs(wires=Wires)

    cpu_qnode = qml.QNode(circuit, dev_cpu)
    probs_cpu = cpu_qnode()

    mpi_qnode = qml.QNode(circuit, dev_mpi)
    local_probs = mpi_qnode()

    recv_counts = comm.gather(len(local_probs), root=0)

    comm.Barrier()

    if rank == 0:
        probs_mpi = np.zeros(1 << len(Wires)).astype(dev_mpi.R_DTYPE)
        displacements = [i for i in range(commSize)]
    else:
        probs_mpi = None
        probs_cpu = None
    comm.Barrier()
    comm.Gatherv(local_probs, [probs_mpi, recv_counts], root=0)

    if rank == 0:
        assert np.allclose(probs_mpi, probs_cpu, atol=tol, rtol=0)
    comm.Barrier()


def apply_probs_param(tol, operation, par, GateWires, Wires, C_DTYPE):
    num_wires = numQubits
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    commSize = comm.Get_size()

    dev_cpu = qml.device("lightning.qubit", wires=num_wires, c_dtype=C_DTYPE)
    dev_mpi = qml.device(device_name, wires=num_wires, mpi=True, c_dtype=C_DTYPE)

    state_vector = create_random_init_state(num_wires, dev_mpi.R_DTYPE)
    comm.Bcast(state_vector, root=0)

    def circuit():
        qml.StatePrep(state_vector, wires=range(num_wires))
        operation(*par, wires=GateWires)
        return qml.probs(wires=Wires)

    cpu_qnode = qml.QNode(circuit, dev_cpu)
    probs_cpu = cpu_qnode()

    mpi_qnode = qml.QNode(circuit, dev_mpi)
    local_probs = mpi_qnode()

    recv_counts = comm.gather(len(local_probs), root=0)

    comm.Barrier()

    if rank == 0:
        probs_mpi = np.zeros(1 << len(Wires)).astype(dev_mpi.R_DTYPE)
    else:
        probs_mpi = None
        probs_cpu = None
    comm.Barrier()

    comm.Gatherv(local_probs, [probs_mpi, recv_counts], root=0)

    if rank == 0:
        assert np.allclose(probs_mpi, probs_cpu, atol=tol, rtol=0)
    comm.Barrier()


class TestProbs:
    """Tests for the probability method."""

    @pytest.mark.parametrize(
        "operation", [qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard, qml.S, qml.T]
    )
    @pytest.mark.parametrize("GateWires", [[0], [numQubits - 1]])
    @pytest.mark.parametrize(
        "Wires",
        [
            [0],
            [1],
            [0, 1],
            [0, 2],
            [0, numQubits - 1],
            [numQubits - 2, numQubits - 1],
            range(numQubits),
        ],
    )
    @pytest.mark.parametrize("C_DTYPE", [np.complex128])
    def test_prob_single_wire_nonparam(self, tol, operation, GateWires, Wires, C_DTYPE):
        apply_probs_nonparam(tol, operation, GateWires, Wires, C_DTYPE)

    @pytest.mark.parametrize("operation", [qml.CNOT, qml.SWAP, qml.CY, qml.CZ])
    @pytest.mark.parametrize(
        "GateWires", [[0, 1], [numQubits - 2, numQubits - 1], [0, numQubits - 1]]
    )
    @pytest.mark.parametrize(
        "Wires",
        [
            [0],
            [1],
            [0, 1],
            [0, 2],
            [0, numQubits - 1],
            [numQubits - 2, numQubits - 1],
            range(numQubits),
        ],
    )
    @pytest.mark.parametrize("C_DTYPE", [np.complex128])
    def test_prob_two_wire_nonparam(self, tol, operation, GateWires, Wires, C_DTYPE):
        apply_probs_nonparam(tol, operation, GateWires, Wires, C_DTYPE)

    @pytest.mark.parametrize("operation", [qml.CSWAP, qml.Toffoli])
    @pytest.mark.parametrize(
        "GateWires",
        [
            [0, 1, 2],
            [numQubits - 3, numQubits - 2, numQubits - 1],
            [0, 1, numQubits - 1],
            [0, numQubits - 2, numQubits - 1],
        ],
    )
    @pytest.mark.parametrize(
        "Wires",
        [
            [0],
            [1],
            [0, 1],
            [0, 2],
            [0, numQubits - 1],
            [numQubits - 2, numQubits - 1],
            range(numQubits),
        ],
    )
    @pytest.mark.parametrize("C_DTYPE", [np.complex128])
    def test_prob_three_wire_nonparam(self, tol, operation, GateWires, Wires, C_DTYPE):
        apply_probs_nonparam(tol, operation, GateWires, Wires, C_DTYPE)

    @pytest.mark.parametrize("operation", [qml.PhaseShift, qml.RX, qml.RY, qml.RZ])
    @pytest.mark.parametrize("par", [[0.1], [0.2], [0.3]])
    @pytest.mark.parametrize("GateWires", [0, numQubits - 1])
    @pytest.mark.parametrize(
        "Wires",
        [
            [0],
            [1],
            [0, 1],
            [0, 2],
            [0, numQubits - 1],
            [numQubits - 2, numQubits - 1],
            range(numQubits),
        ],
    )
    @pytest.mark.parametrize("C_DTYPE", [np.complex128])
    def test_prob_single_wire_param(self, tol, operation, par, GateWires, Wires, C_DTYPE):
        apply_probs_param(tol, operation, par, GateWires, Wires, C_DTYPE)

    @pytest.mark.parametrize("operation", [qml.Rot])
    @pytest.mark.parametrize("par", [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    @pytest.mark.parametrize("GateWires", [0, numQubits - 1])
    @pytest.mark.parametrize(
        "Wires",
        [
            [0],
            [1],
            [0, 1],
            [0, 2],
            [0, numQubits - 1],
            [numQubits - 2, numQubits - 1],
            range(numQubits),
        ],
    )
    @pytest.mark.parametrize("C_DTYPE", [np.complex128])
    def test_prob_single_wire_3param(self, tol, operation, par, GateWires, Wires, C_DTYPE):
        apply_probs_param(tol, operation, par, GateWires, Wires, C_DTYPE)

    @pytest.mark.parametrize("operation", [qml.CRot])
    @pytest.mark.parametrize("par", [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    @pytest.mark.parametrize(
        "GateWires", [[0, numQubits - 1], [0, 1], [numQubits - 2, numQubits - 1]]
    )
    @pytest.mark.parametrize(
        "Wires",
        [
            [0],
            [1],
            [0, 1],
            [0, 2],
            [0, numQubits - 1],
            [numQubits - 2, numQubits - 1],
            range(numQubits),
        ],
    )
    @pytest.mark.parametrize("C_DTYPE", [np.complex128])
    def test_prob_two_wire_3param(self, tol, operation, par, GateWires, Wires, C_DTYPE):
        apply_probs_param(tol, operation, par, GateWires, Wires, C_DTYPE)

    @pytest.mark.parametrize(
        "operation",
        [
            qml.CRX,
            qml.CRY,
            qml.CRZ,
            qml.ControlledPhaseShift,
            qml.SingleExcitation,
            qml.SingleExcitationMinus,
            qml.SingleExcitationPlus,
            qml.IsingXX,
            qml.IsingYY,
            qml.IsingZZ,
        ],
    )
    @pytest.mark.parametrize("par", [[0.1], [0.2], [0.3]])
    @pytest.mark.parametrize(
        "GateWires", [[0, numQubits - 1], [0, 1], [numQubits - 2, numQubits - 1]]
    )
    @pytest.mark.parametrize(
        "Wires",
        [
            [0],
            [1],
            [0, 1],
            [0, 2],
            [0, numQubits - 1],
            [numQubits - 2, numQubits - 1],
            range(numQubits),
        ],
    )
    @pytest.mark.parametrize("C_DTYPE", [np.complex128])
    def test_prob_two_wire_param(self, tol, operation, par, GateWires, Wires, C_DTYPE):
        apply_probs_param(tol, operation, par, GateWires, Wires, C_DTYPE)

    @pytest.mark.parametrize(
        "operation",
        [qml.DoubleExcitation, qml.DoubleExcitationMinus, qml.DoubleExcitationPlus],
    )
    @pytest.mark.parametrize("par", [[0.13], [0.2], [0.3]])
    @pytest.mark.parametrize(
        "GateWires",
        [
            [0, 1, numQubits - 2, numQubits - 1],
            [0, 1, 2, 3],
            [numQubits - 4, numQubits - 3, numQubits - 2, numQubits - 1],
        ],
    )
    @pytest.mark.parametrize(
        "Wires",
        [
            [0],
            [1],
            [0, 1],
            [0, 2],
            [0, numQubits - 1],
            [numQubits - 2, numQubits - 1],
            range(numQubits),
        ],
    )
    @pytest.mark.parametrize("C_DTYPE", [np.complex128])
    def test_prob_four_wire_param(self, tol, operation, par, GateWires, Wires, C_DTYPE):
        apply_probs_param(tol, operation, par, GateWires, Wires, C_DTYPE)
