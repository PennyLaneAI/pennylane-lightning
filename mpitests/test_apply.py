# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane_lightning_gpu.LightningGPU` device (MPI).
"""
# pylint: disable=protected-access,cell-var-from-loop
from mpi4py import MPI
import pytest
import itertools

from conftest import TOL_STOCHASTIC, device_name

import numpy as np
import pennylane as qml

fixture_params = itertools.product(
    [np.complex64, np.complex128],
    [True, False],
)

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


def apply_operation_gates_qnode_param(tol, dev_mpi, operation, par, Wires):
    num_wires = numQubits
    comm = MPI.COMM_WORLD
    commSize = comm.Get_size()
    num_global_wires = commSize.bit_length() - 1
    num_local_wires = num_wires - num_global_wires

    if dev_mpi.R_DTYPE == np.float32:
        c_dtype = np.complex64
    else:
        c_dtype = np.complex128

    expected_output_cpu = np.zeros(1 << num_wires).astype(c_dtype)
    local_state_vector = np.zeros(1 << num_local_wires).astype(c_dtype)
    local_expected_output_cpu = np.zeros(1 << num_local_wires).astype(c_dtype)

    state_vector = create_random_init_state(num_wires, dev_mpi.R_DTYPE)
    comm.Bcast(state_vector, root=0)

    comm.Scatter(state_vector, local_state_vector, root=0)
    dev_cpu = qml.device("lightning.qubit", wires=num_wires, c_dtype=c_dtype)

    def circuit(*params):
        qml.StatePrep(state_vector, wires=range(num_wires))
        operation(*params, wires=Wires)
        return qml.state()

    cpu_qnode = qml.QNode(circuit, dev_cpu)
    expected_output_cpu = cpu_qnode(*par).astype(c_dtype)
    comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

    mpi_qnode = qml.QNode(circuit, dev_mpi)
    local_state_vector = mpi_qnode(*par)

    assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)


def apply_operation_gates_apply_param(tol, dev_mpi, operation, par, Wires):
    num_wires = numQubits
    comm = MPI.COMM_WORLD
    commSize = comm.Get_size()
    num_global_wires = commSize.bit_length() - 1
    num_local_wires = num_wires - num_global_wires

    if dev_mpi.R_DTYPE == np.float32:
        c_dtype = np.complex64
    else:
        c_dtype = np.complex128

    expected_output_cpu = np.zeros(1 << num_wires).astype(c_dtype)
    local_state_vector = np.zeros(1 << num_local_wires).astype(c_dtype)
    local_expected_output_cpu = np.zeros(1 << num_local_wires).astype(c_dtype)

    state_vector = create_random_init_state(num_wires, dev_mpi.R_DTYPE)
    comm.Bcast(state_vector, root=0)

    comm.Scatter(state_vector, local_state_vector, root=0)
    dev_cpu = qml.device("lightning.qubit", wires=num_wires, c_dtype=c_dtype)

    @qml.qnode(dev_cpu)
    def circuit(*params):
        qml.StatePrep(state_vector, wires=range(num_wires))
        operation(*params, wires=Wires)
        return qml.state()

    expected_output_cpu = circuit(*par).astype(c_dtype)
    comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

    dev_mpi.syncH2D(local_state_vector)
    dev_mpi.apply([operation(*par, wires=Wires)])
    dev_mpi.syncD2H(local_state_vector)

    assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)


def apply_operation_gates_qnode_nonparam(tol, dev_mpi, operation, Wires):
    num_wires = numQubits
    comm = MPI.COMM_WORLD
    commSize = comm.Get_size()
    num_global_wires = commSize.bit_length() - 1
    num_local_wires = num_wires - num_global_wires

    if dev_mpi.R_DTYPE == np.float32:
        c_dtype = np.complex64
    else:
        c_dtype = np.complex128

    expected_output_cpu = np.zeros(1 << num_wires).astype(c_dtype)
    local_state_vector = np.zeros(1 << num_local_wires).astype(c_dtype)
    local_expected_output_cpu = np.zeros(1 << num_local_wires).astype(c_dtype)

    state_vector = create_random_init_state(num_wires, dev_mpi.R_DTYPE)
    comm.Bcast(state_vector, root=0)

    comm.Scatter(state_vector, local_state_vector, root=0)
    dev_cpu = qml.device("lightning.qubit", wires=num_wires, c_dtype=c_dtype)

    def circuit():
        qml.StatePrep(state_vector, wires=range(num_wires))
        operation(wires=Wires)
        return qml.state()

    cpu_qnode = qml.QNode(circuit, dev_cpu)
    expected_output_cpu = cpu_qnode().astype(c_dtype)
    comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

    mpi_qnode = qml.QNode(circuit, dev_mpi)
    local_state_vector = mpi_qnode()

    assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)


def apply_operation_gates_apply_nonparam(tol, dev_mpi, operation, Wires):
    num_wires = numQubits
    comm = MPI.COMM_WORLD
    commSize = comm.Get_size()
    num_global_wires = commSize.bit_length() - 1
    num_local_wires = num_wires - num_global_wires

    if dev_mpi.R_DTYPE == np.float32:
        c_dtype = np.complex64
    else:
        c_dtype = np.complex128

    expected_output_cpu = np.zeros(1 << num_wires).astype(c_dtype)
    local_state_vector = np.zeros(1 << num_local_wires).astype(c_dtype)
    local_expected_output_cpu = np.zeros(1 << num_local_wires).astype(c_dtype)

    state_vector = create_random_init_state(num_wires, dev_mpi.R_DTYPE)
    comm.Bcast(state_vector, root=0)

    comm.Scatter(state_vector, local_state_vector, root=0)
    dev_cpu = qml.device("lightning.qubit", wires=num_wires, c_dtype=c_dtype)

    @qml.qnode(dev_cpu)
    def circuit():
        qml.StatePrep(state_vector, wires=range(num_wires))
        operation(wires=Wires)
        return qml.state()

    expected_output_cpu = circuit().astype(c_dtype)
    comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

    dev_mpi.syncH2D(local_state_vector)
    dev_mpi.apply([operation(wires=Wires)])
    dev_mpi.syncD2H(local_state_vector)

    assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)


class TestApply:
    @pytest.fixture(params=fixture_params)
    def dev_mpi(self, request):
        return qml.device(
            device_name,
            wires=numQubits,
            mpi=True,
            c_dtype=request.param[0],
            batch_obs=request.param[1],
        )

    # Parameterized test case for single wire nonparam gates
    @pytest.mark.parametrize(
        "operation", [qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard, qml.S, qml.T]
    )
    @pytest.mark.parametrize("Wires", [0, 1, numQubits - 2, numQubits - 1])
    def test_apply_operation_single_wire_nonparam(self, tol, operation, Wires, dev_mpi):
        apply_operation_gates_qnode_nonparam(tol, dev_mpi, operation, Wires)
        apply_operation_gates_apply_nonparam(tol, dev_mpi, operation, Wires)

    @pytest.mark.parametrize("operation", [qml.CNOT, qml.SWAP, qml.CY, qml.CZ])
    @pytest.mark.parametrize("Wires", [[0, 1], [numQubits - 2, numQubits - 1], [0, numQubits - 1]])
    def test_apply_operation_two_wire_nonparam(self, tol, operation, Wires, dev_mpi):
        apply_operation_gates_qnode_nonparam(tol, dev_mpi, operation, Wires)
        apply_operation_gates_apply_nonparam(tol, dev_mpi, operation, Wires)

    @pytest.mark.parametrize("operation", [qml.CSWAP, qml.Toffoli])
    @pytest.mark.parametrize(
        "Wires",
        [
            [0, 1, 2],
            [numQubits - 3, numQubits - 2, numQubits - 1],
            [0, 1, numQubits - 1],
            [0, numQubits - 2, numQubits - 1],
        ],
    )
    def test_apply_operation_three_wire_nonparam(self, tol, operation, Wires, dev_mpi):
        apply_operation_gates_qnode_nonparam(tol, dev_mpi, operation, Wires)
        apply_operation_gates_apply_nonparam(tol, dev_mpi, operation, Wires)

    @pytest.mark.parametrize("operation", [qml.CSWAP, qml.Toffoli])
    @pytest.mark.parametrize(
        "Wires",
        [
            [0, 1, 2],
            [numQubits - 3, numQubits - 2, numQubits - 1],
            [0, 1, numQubits - 1],
            [0, numQubits - 2, numQubits - 1],
        ],
    )
    def test_apply_operation_three_wire_qnode_nonparam(self, tol, operation, Wires, dev_mpi):
        apply_operation_gates_qnode_nonparam(tol, dev_mpi, operation, Wires)
        apply_operation_gates_apply_nonparam(tol, dev_mpi, operation, Wires)

    @pytest.mark.parametrize("operation", [qml.PhaseShift, qml.RX, qml.RY, qml.RZ])
    @pytest.mark.parametrize("par", [[0.1], [0.2], [0.3]])
    @pytest.mark.parametrize("Wires", [0, numQubits - 1])
    def test_apply_operation_1gatequbit_1param_gate_qnode_param(
        self, tol, operation, par, Wires, dev_mpi
    ):
        apply_operation_gates_qnode_param(tol, dev_mpi, operation, par, Wires)
        apply_operation_gates_apply_param(tol, dev_mpi, operation, par, Wires)

    @pytest.mark.parametrize("operation", [qml.Rot])
    @pytest.mark.parametrize("par", [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    @pytest.mark.parametrize("Wires", [0, numQubits - 1])
    def test_apply_operation_1gatequbit_3param_gate_qnode_param(
        self, tol, operation, par, Wires, dev_mpi
    ):
        apply_operation_gates_qnode_param(tol, dev_mpi, operation, par, Wires)
        apply_operation_gates_apply_param(tol, dev_mpi, operation, par, Wires)

    @pytest.mark.parametrize("operation", [qml.CRot])
    @pytest.mark.parametrize("par", [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    @pytest.mark.parametrize("Wires", [[0, numQubits - 1], [0, 1], [numQubits - 2, numQubits - 1]])
    def test_apply_operation_1gatequbit_3param_gate_qnode_param(
        self, tol, operation, par, Wires, dev_mpi
    ):
        apply_operation_gates_qnode_param(tol, dev_mpi, operation, par, Wires)
        apply_operation_gates_apply_param(tol, dev_mpi, operation, par, Wires)

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
    @pytest.mark.parametrize("Wires", [[0, numQubits - 1], [0, 1], [numQubits - 2, numQubits - 1]])
    def test_apply_operation_2gatequbit_1param_gate_qnode_param(
        self, tol, operation, par, Wires, dev_mpi
    ):
        apply_operation_gates_qnode_param(tol, dev_mpi, operation, par, Wires)
        apply_operation_gates_apply_param(tol, dev_mpi, operation, par, Wires)

    @pytest.mark.parametrize(
        "operation",
        [qml.DoubleExcitation, qml.DoubleExcitationMinus, qml.DoubleExcitationPlus],
    )
    @pytest.mark.parametrize("par", [[0.13], [0.2], [0.3]])
    @pytest.mark.parametrize(
        "Wires",
        [
            [0, 1, numQubits - 2, numQubits - 1],
            [0, 1, 2, 3],
            [numQubits - 4, numQubits - 3, numQubits - 2, numQubits - 1],
        ],
    )
    def test_apply_operation_4gatequbit_1param_gate_qnode_param(
        self, tol, operation, par, Wires, dev_mpi
    ):
        apply_operation_gates_qnode_param(tol, dev_mpi, operation, par, Wires)
        apply_operation_gates_apply_param(tol, dev_mpi, operation, par, Wires)

    # BasisState test
    @pytest.mark.parametrize("operation", [qml.BasisState])
    @pytest.mark.parametrize("index", range(numQubits))
    def test_state_prep(self, tol, operation, index, dev_mpi):
        par = np.zeros(numQubits, dtype=int)
        par[index] = 1
        num_wires = numQubits
        comm = MPI.COMM_WORLD
        commSize = comm.Get_size()
        num_global_wires = commSize.bit_length() - 1
        num_local_wires = num_wires - num_global_wires

        if dev_mpi.R_DTYPE == np.float32:
            c_dtype = np.complex64
        else:
            c_dtype = np.complex128

        state_vector = np.zeros(1 << num_wires).astype(c_dtype)
        expected_output_cpu = np.zeros(1 << num_wires).astype(c_dtype)
        local_state_vector = np.zeros(1 << num_local_wires).astype(c_dtype)
        local_expected_output_cpu = np.zeros(1 << num_local_wires).astype(c_dtype)

        state_vector = create_random_init_state(num_wires, dev_mpi.R_DTYPE)

        comm.Scatter(state_vector, local_state_vector, root=0)
        dev_cpu = qml.device("lightning.qubit", wires=num_wires, c_dtype=c_dtype)

        def circuit():
            operation(par, wires=range(numQubits))
            return qml.state()

        cpu_qnode = qml.QNode(circuit, dev_cpu)
        mpi_qnode = qml.QNode(circuit, dev_mpi)

        expected_output_cpu = cpu_qnode().astype(c_dtype)
        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        local_state_vector = mpi_qnode()

        assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)

    test_qubit_state_prep = [
        (np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [0]),
        (np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [1]),
        (np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [2]),
        (np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [3]),
        (np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [4]),
        (np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), [5]),
        (np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]), [1, 0]),
        (np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]), [0, 1]),
        (np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]), [0, 2]),
        (
            np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]),
            [numQubits - 2, numQubits - 1],
        ),
        (
            np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]),
            [0, numQubits - 1],
        ),
        (
            np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]),
            [0, numQubits - 2],
        ),
    ]

    @pytest.mark.parametrize("par, Wires", test_qubit_state_prep)
    def test_qubit_state_prep(self, tol, par, Wires, dev_mpi):
        num_wires = numQubits
        comm = MPI.COMM_WORLD
        commSize = comm.Get_size()
        num_global_wires = commSize.bit_length() - 1
        num_local_wires = num_wires - num_global_wires

        if dev_mpi.R_DTYPE == np.float32:
            c_dtype = np.complex64
        else:
            c_dtype = np.complex128

        state_vector = np.zeros(1 << num_wires).astype(c_dtype)
        expected_output_cpu = np.zeros(1 << num_wires).astype(c_dtype)
        local_state_vector = np.zeros(1 << num_local_wires).astype(c_dtype)
        local_expected_output_cpu = np.zeros(1 << num_local_wires).astype(c_dtype)

        state_vector = create_random_init_state(num_wires, dev_mpi.R_DTYPE)

        comm.Scatter(state_vector, local_state_vector, root=0)
        dev_cpu = qml.device("lightning.qubit", wires=num_wires, c_dtype=c_dtype)

        def circuit():
            qml.StatePrep(par, wires=Wires)
            return qml.state()

        cpu_qnode = qml.QNode(circuit, dev_cpu)
        mpi_qnode = qml.QNode(circuit, dev_mpi)

        expected_output_cpu = cpu_qnode().astype(c_dtype)
        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        local_state_vector = mpi_qnode()

        assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)

    def test_dev_reset(self, tol, dev_mpi):
        num_wires = numQubits
        comm = MPI.COMM_WORLD
        commSize = comm.Get_size()
        num_global_wires = commSize.bit_length() - 1
        num_local_wires = num_wires - num_global_wires

        if dev_mpi.R_DTYPE == np.float32:
            c_dtype = np.complex64
        else:
            c_dtype = np.complex128

        state_vector = np.zeros(1 << num_wires).astype(c_dtype)
        expected_output_cpu = np.zeros(1 << num_wires).astype(c_dtype)
        local_state_vector = np.zeros(1 << num_local_wires).astype(c_dtype)
        local_expected_output_cpu = np.zeros(1 << num_local_wires).astype(c_dtype)

        state_vector = create_random_init_state(num_wires, dev_mpi.R_DTYPE)

        comm.Scatter(state_vector, local_state_vector, root=0)
        dev_cpu = qml.device("lightning.qubit", wires=num_wires, c_dtype=c_dtype)

        dev_cpu.reset()

        def circuit():
            qml.PauliX(wires=[0])
            qml.PauliX(wires=[0])
            return qml.state()

        cpu_qnode = qml.QNode(circuit, dev_cpu)

        expected_output_cpu = cpu_qnode().astype(c_dtype)
        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        dev_mpi.reset()

        gpumpi_qnode = qml.QNode(circuit, dev_mpi)
        dev_mpi.reset()

        local_state_vector = gpumpi_qnode()
        assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)


class TestSparseHamExpval:
    """Tests sparse hamiltonian expectation values."""

    @pytest.mark.parametrize("C_DTYPE", [np.complex128, np.complex64])
    def test_sparse_hamiltonian_expectation(self, C_DTYPE, tol):
        comm = MPI.COMM_WORLD
        commSize = comm.Get_size()
        num_global_wires = commSize.bit_length() - 1
        num_local_wires = 3 - num_global_wires

        obs = qml.Identity(0) @ qml.PauliX(1) @ qml.PauliY(2)
        obs1 = qml.Identity(1)
        Hmat = qml.Hamiltonian([1.0, 1.0], [obs1, obs]).sparse_matrix()

        state_vector = np.array(
            [
                0.0 + 0.0j,
                0.0 + 0.1j,
                0.1 + 0.1j,
                0.1 + 0.2j,
                0.2 + 0.2j,
                0.2 + 0.3j,
                0.3 + 0.3j,
                0.3 + 0.5j,
            ],
            dtype=C_DTYPE,
        )

        local_state_vector = np.zeros(1 << num_local_wires).astype(C_DTYPE)
        comm.Scatter(state_vector, local_state_vector, root=0)

        dev_gpu = qml.device("lightning.gpu", wires=3, mpi=False, c_dtype=C_DTYPE)
        dev_mpi = qml.device("lightning.gpu", wires=3, mpi=True, c_dtype=C_DTYPE)

        dev_mpi.syncH2D(local_state_vector)
        dev_gpu.syncH2D(state_vector)

        H_sparse = qml.SparseHamiltonian(Hmat, wires=range(3))

        comm.Barrier()

        res = dev_mpi.expval(H_sparse)
        expected = dev_gpu.expval(H_sparse)

        assert np.allclose(res, expected)


class TestExpval:
    """Tests that expectation values are properly calculated or that the proper errors are raised."""

    @pytest.mark.parametrize("C_DTYPE", [np.complex128, np.complex64])
    @pytest.mark.parametrize(
        "operation",
        [
            qml.PauliX,
            qml.PauliY,
            qml.PauliZ,
            qml.Hadamard,
            qml.Identity,
        ],
    )
    @pytest.mark.parametrize("wires", [0, 1, 2, numQubits - 3, numQubits - 2, numQubits - 1])
    def test_expval_single_wire_no_parameters(self, tol, operation, wires, C_DTYPE):
        """Tests that expectation values are properly calculated for single-wire observables without parameters."""
        num_wires = numQubits
        comm = MPI.COMM_WORLD
        commSize = comm.Get_size()
        num_global_wires = commSize.bit_length() - 1
        num_local_wires = num_wires - num_global_wires

        dev_mpi = qml.device("lightning.gpu", wires=numQubits, mpi=True, c_dtype=C_DTYPE)

        state_vector = create_random_init_state(num_wires, dev_mpi.R_DTYPE)
        comm.Bcast(state_vector, root=0)

        local_state_vector = np.zeros(1 << num_local_wires).astype(C_DTYPE)
        comm.Scatter(state_vector, local_state_vector, root=0)
        dev_cpu = qml.device("lightning.qubit", wires=num_wires, c_dtype=C_DTYPE)

        def circuit():
            qml.StatePrep(state_vector, wires=range(num_wires))
            return qml.expval(operation(wires))

        cpu_qnode = qml.QNode(circuit, dev_cpu)
        expected_output_cpu = cpu_qnode()
        comm.Bcast(expected_output_cpu, root=0)

        mpi_qnode = qml.QNode(circuit, dev_mpi)
        expected_output_mpi = mpi_qnode()

        assert np.allclose(expected_output_mpi, expected_output_cpu, atol=tol, rtol=0)

    @pytest.mark.parametrize("C_DTYPE", [np.complex128, np.complex64])
    @pytest.mark.parametrize(
        "obs",
        [
            qml.PauliX(0) @ qml.PauliZ(1),
            qml.PauliX(0) @ qml.PauliZ(numQubits - 1),
            qml.PauliX(numQubits - 2) @ qml.PauliZ(numQubits - 1),
            qml.PauliZ(0) @ qml.PauliZ(1),
            qml.PauliZ(0) @ qml.PauliZ(numQubits - 1),
            qml.PauliZ(numQubits - 2) @ qml.PauliZ(numQubits - 1),
        ],
    )
    def test_expval_multiple_obs(self, obs, request, tol, C_DTYPE):
        """Test expval with Hamiltonian"""
        num_wires = numQubits
        comm = MPI.COMM_WORLD

        dev_cpu = qml.device("lightning.qubit", wires=num_wires, c_dtype=C_DTYPE)
        dev_mpi = qml.device("lightning.gpu", wires=num_wires, mpi=True, c_dtype=C_DTYPE)

        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[num_wires - 1])
            return qml.expval(obs)

        cpu_qnode = qml.QNode(circuit, dev_cpu)
        mpi_qnode = qml.QNode(circuit, dev_mpi)

        assert np.allclose(cpu_qnode(), mpi_qnode(), atol=tol, rtol=0)

    @pytest.mark.parametrize("C_DTYPE", [np.complex128, np.complex64])
    @pytest.mark.parametrize(
        "obs, coeffs",
        [
            ([qml.PauliX(0) @ qml.PauliZ(1)], [0.314]),
            ([qml.PauliX(0) @ qml.PauliZ(numQubits - 1)], [0.314]),
            ([qml.PauliZ(0) @ qml.PauliZ(1)], [0.314]),
            ([qml.PauliZ(0) @ qml.PauliZ(numQubits - 1)], [0.314]),
            (
                [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)],
                [0.314, 0.2],
            ),
            (
                [
                    qml.PauliX(0) @ qml.PauliZ(numQubits - 1),
                    qml.PauliZ(0) @ qml.PauliZ(1),
                ],
                [0.314, 0.2],
            ),
            (
                [
                    qml.PauliX(numQubits - 2) @ qml.PauliZ(numQubits - 1),
                    qml.PauliZ(0) @ qml.PauliZ(1),
                ],
                [0.314, 0.2],
            ),
        ],
    )
    def test_expval_hamiltonian(self, obs, coeffs, tol, C_DTYPE):
        """Test expval with Hamiltonian"""
        num_wires = numQubits
        comm = MPI.COMM_WORLD

        ham = qml.Hamiltonian(coeffs, obs)

        dev_cpu = qml.device("lightning.qubit", wires=num_wires, c_dtype=C_DTYPE)
        dev_mpi = qml.device("lightning.gpu", wires=num_wires, mpi=True, c_dtype=C_DTYPE)

        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[numQubits - 1])
            return qml.expval(ham)

        cpu_qnode = qml.QNode(circuit, dev_cpu)
        mpi_qnode = qml.QNode(circuit, dev_mpi)

        assert np.allclose(cpu_qnode(), mpi_qnode(), atol=tol, rtol=0)

    def test_expval_non_pauli_word_hamiltionian(self, tol):
        comm = MPI.COMM_WORLD
        dev_mpi = qml.device("lightning.gpu", wires=3, mpi=True)
        dev_cpu = qml.device("lightning.qubit", wires=3)

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(0.5 * qml.Hadamard(2))

        cpu_qnode = qml.QNode(circuit, dev_cpu)
        mpi_qnode = qml.QNode(circuit, dev_mpi)

        assert np.allclose(cpu_qnode(), mpi_qnode(), atol=tol, rtol=0)


class TestGenerateSample:
    """Tests that samples are properly calculated."""

    @pytest.mark.parametrize("C_DTYPE", [np.complex128, np.complex64])
    def test_sample_dimensions(self, C_DTYPE):
        """Tests if the samples returned by sample have
        the correct dimensions
        """
        num_wires = numQubits
        comm = MPI.COMM_WORLD

        dev = qml.device("lightning.gpu", wires=num_wires, mpi=True, shots=1000, c_dtype=C_DTYPE)

        dev.apply([qml.RX(1.5708, wires=[0]), qml.RX(1.5708, wires=[1])])

        dev.shots = 10
        dev._wires_measured = {0}
        dev._samples = dev.generate_samples()
        s1 = dev.sample(qml.PauliZ(wires=[0]))
        assert np.array_equal(s1.shape, (10,))

        dev.reset()
        dev.shots = 12
        dev._wires_measured = {1}
        dev._samples = dev.generate_samples()
        s2 = dev.sample(qml.PauliZ(wires=[1]))
        assert np.array_equal(s2.shape, (12,))

        dev.reset()
        dev.shots = 17
        dev._wires_measured = {0, 1}
        dev._samples = dev.generate_samples()
        s3 = dev.sample(qml.PauliX(0) @ qml.PauliZ(1))
        assert np.array_equal(s3.shape, (17,))

    @pytest.mark.parametrize("C_DTYPE", [np.complex128, np.complex64])
    def test_sample_values(self, tol, C_DTYPE):
        """Tests if the samples returned by sample have
        the correct values
        """
        num_wires = numQubits
        comm = MPI.COMM_WORLD

        dev = qml.device("lightning.gpu", wires=num_wires, mpi=True, shots=1000, c_dtype=C_DTYPE)
        dev.reset()
        dev.apply([qml.RX(1.5708, wires=[0])])
        dev._wires_measured = {0}
        dev._samples = dev.generate_samples()

        s1 = dev.sample(qml.PauliZ(0))

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1**2, 1, atol=tol, rtol=0)

    @pytest.mark.parametrize("C_DTYPE", [np.complex128, np.complex64])
    def test_sample_values_qnode(self, tol, C_DTYPE):
        """Tests if the samples returned by sample have
        the correct values
        """
        num_wires = numQubits
        comm = MPI.COMM_WORLD

        dev_mpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, shots=1000, c_dtype=C_DTYPE
        )
        dev_mpi.reset()

        @qml.qnode(dev_mpi)
        def circuit():
            qml.RX(1.5708, wires=0)
            return qml.sample(qml.PauliZ(0))

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(circuit() ** 2, 1, atol=tol, rtol=0)

    @pytest.mark.parametrize("C_DTYPE", [np.complex128, np.complex64])
    def test_multi_samples_return_correlated_results(self, C_DTYPE):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """
        num_wires = 3
        comm = MPI.COMM_WORLD

        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, shots=1000, c_dtype=C_DTYPE
        )

        @qml.qnode(dev_gpumpi)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))

        outcomes = circuit()

        assert np.array_equal(outcomes[0], outcomes[1])

    @pytest.mark.parametrize("C_DTYPE", [np.complex128, np.complex64])
    def test_paulix_pauliy(self, C_DTYPE, tol=TOL_STOCHASTIC):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        num_wires = 3
        comm = MPI.COMM_WORLD

        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, shots=1000, c_dtype=C_DTYPE
        )

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qml.qnode(dev_gpumpi)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.sample(qml.PauliX(wires=[0]) @ qml.PauliY(wires=[2]))

        res = circuit()

        # res should only contain 1 and -1
        assert np.allclose(res**2, 1, atol=tol)

        mean = np.mean(res)
        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)
        assert np.allclose(mean, expected, atol=tol)

        var = np.var(res)
        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16
        assert np.allclose(var, expected, atol=tol)

    @pytest.mark.parametrize("C_DTYPE", [np.complex128, np.complex64])
    def test_pauliz_hadamard(self, C_DTYPE, tol=TOL_STOCHASTIC):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        num_wires = 3
        comm = MPI.COMM_WORLD

        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, shots=1000, c_dtype=C_DTYPE
        )

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qml.qnode(dev_gpumpi)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.sample(
                qml.PauliZ(wires=[0]) @ qml.Hadamard(wires=[1]) @ qml.PauliY(wires=[2])
            )

        res = circuit()

        # s1 should only contain 1 and -1
        assert np.allclose(res**2, 1, atol=tol)

        mean = np.mean(res)
        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)
        assert np.allclose(mean, expected, atol=tol)

        var = np.var(res)
        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4
        assert np.allclose(var, expected, atol=tol)


class TestTensorVar:
    """Test tensor variance measurements."""

    @pytest.mark.parametrize("C_DTYPE", [np.complex128, np.complex64])
    def test_paulix_pauliy(self, C_DTYPE, tol=TOL_STOCHASTIC):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        num_wires = 3
        comm = MPI.COMM_WORLD

        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, shots=1000, c_dtype=C_DTYPE
        )

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qml.qnode(dev_gpumpi)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.var(qml.PauliX(wires=[0]) @ qml.PauliY(wires=[2]))

        res = circuit()

        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16
        assert np.allclose(res, expected, atol=tol)

    @pytest.mark.parametrize("C_DTYPE", [np.complex128, np.complex64])
    def test_pauliz_hadamard(self, C_DTYPE, tol=TOL_STOCHASTIC):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        num_wires = 3
        comm = MPI.COMM_WORLD
        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, shots=1000, c_dtype=C_DTYPE
        )

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qml.qnode(dev_gpumpi)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.var(qml.PauliZ(wires=[0]) @ qml.Hadamard(wires=[1]) @ qml.PauliY(wires=[2]))

        res = circuit()

        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4
        assert np.allclose(res, expected, atol=tol)


def circuit_ansatz(params, wires):
    """Circuit ansatz containing all the parametrized gates"""
    qml.StatePrep(unitary_group.rvs(2**numQubits, random_state=0)[0], wires=wires)
    qml.RX(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.adjoint(qml.RX(params[2], wires=wires[2]))
    qml.RZ(params[0], wires=wires[3])
    qml.CRX(params[3], wires=[wires[3], wires[0]])
    qml.PhaseShift(params[4], wires=wires[2])
    qml.CRY(params[5], wires=[wires[2], wires[1]])
    qml.adjoint(qml.CRZ(params[5], wires=[wires[0], wires[3]]))
    qml.adjoint(qml.PhaseShift(params[6], wires=wires[0]))
    qml.Rot(params[6], params[7], params[8], wires=wires[0])
    qml.adjoint(qml.Rot(params[8], params[8], params[9], wires=wires[1]))
    qml.MultiRZ(params[11], wires=[wires[0], wires[1]])
    qml.CPhase(params[12], wires=[wires[3], wires[2]])
    qml.IsingXX(params[13], wires=[wires[1], wires[0]])
    qml.IsingYY(params[14], wires=[wires[3], wires[2]])
    qml.IsingZZ(params[15], wires=[wires[2], wires[1]])
    qml.SingleExcitation(params[24], wires=[wires[2], wires[0]])
    qml.DoubleExcitation(params[25], wires=[wires[2], wires[0], wires[1], wires[3]])


@pytest.mark.parametrize(
    "returns",
    [
        (qml.PauliX(0),),
        (qml.PauliY(0),),
        (qml.PauliZ(0),),
        (qml.PauliX(1),),
        (qml.PauliY(1),),
        (qml.PauliZ(1),),
        (qml.PauliX(2),),
        (qml.PauliY(2),),
        (qml.PauliZ(2),),
        (qml.PauliX(3),),
        (qml.PauliY(3),),
        (qml.PauliZ(3),),
        (qml.PauliX(0), qml.PauliY(1)),
        (
            qml.PauliZ(0),
            qml.PauliX(1),
            qml.PauliY(2),
        ),
        (
            qml.PauliY(0),
            qml.PauliZ(1),
            qml.PauliY(3),
        ),
        (qml.PauliZ(0) @ qml.PauliY(3),),
        (qml.Hadamard(2),),
        (qml.Hadamard(3) @ qml.PauliZ(2),),
        (qml.PauliX(0) @ qml.PauliY(3),),
        (qml.PauliY(0) @ qml.PauliY(2) @ qml.PauliY(3),),
        (qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2),),
        (0.5 * qml.PauliZ(0) @ qml.PauliZ(2),),
    ],
)
def test_integration(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations"""
    num_wires = numQubits
    comm = MPI.COMM_WORLD
    dev_default = qml.device("lightning.qubit", wires=range(num_wires))
    dev_gpu = qml.device("lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128)

    def circuit(params):
        circuit_ansatz(params, wires=range(num_wires))
        return qml.math.hstack([qml.expval(r) for r in returns])

    n_params = 30
    np.random.seed(1337)
    params = np.random.rand(n_params)

    qnode_gpu = qml.QNode(circuit, dev_gpu, diff_method="parameter-shift")
    qnode_default = qml.QNode(circuit, dev_default, diff_method="parameter-shift")

    def convert_to_array_gpu(params):
        return np.array(qnode_gpu(params))

    def convert_to_array_default(params):
        return np.array(qnode_default(params))

    j_gpu = qml.jacobian(convert_to_array_gpu)(params)
    j_default = qml.jacobian(convert_to_array_default)(params)

    assert np.allclose(j_gpu, j_default, atol=1e-7)


custom_wires = ["alice", 3.14, -1, 0, "bob", "l", "m", "n"]


@pytest.mark.parametrize(
    "returns",
    [
        qml.PauliZ(custom_wires[0]),
        qml.PauliX(custom_wires[2]),
        qml.PauliZ(custom_wires[0]) @ qml.PauliY(custom_wires[3]),
        qml.Hadamard(custom_wires[2]),
        qml.Hadamard(custom_wires[3]) @ qml.PauliZ(custom_wires[2]),
        qml.PauliX(custom_wires[0]) @ qml.PauliY(custom_wires[3]),
        qml.PauliY(custom_wires[0]) @ qml.PauliY(custom_wires[2]) @ qml.PauliY(custom_wires[3]),
    ],
)
def test_integration_custom_wires(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""
    comm = MPI.COMM_WORLD
    dev_lightning = qml.device("lightning.qubit", wires=custom_wires)
    dev_gpu = qml.device("lightning.gpu", wires=custom_wires, mpi=True, c_dtype=np.complex128)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qml.expval(returns), qml.expval(qml.PauliY(custom_wires[1]))

    n_params = 30
    np.random.seed(1337)
    params = np.random.rand(n_params)

    qnode_gpu = qml.QNode(circuit, dev_gpu, diff_method="parameter-shift")
    qnode_lightning = qml.QNode(circuit, dev_lightning, diff_method="parameter-shift")

    def convert_to_array_gpu(params):
        return np.array(qnode_gpu(params))

    def convert_to_array_lightning(params):
        return np.array(qnode_lightning(params))

    j_gpu = qml.jacobian(convert_to_array_gpu)(params)
    j_lightning = qml.jacobian(convert_to_array_lightning)(params)

    assert np.allclose(j_gpu, j_lightning, atol=1e-7)
