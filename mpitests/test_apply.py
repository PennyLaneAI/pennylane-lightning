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
Unit tests for apply on :mod:`pennylane_lightning` MPI-enabled devices.
"""

# pylint: disable=protected-access,cell-var-from-loop,c-extension-no-member,too-many-positional-arguments
import itertools
from functools import partial

import numpy as np
import pennylane as qp
import pytest
from conftest import TOL_STOCHASTIC, device_name, fixture_params
from mpi4py import MPI
from pennylane import numpy as pnp
from scipy.stats import unitary_group

numQubits = 8

# Tuple passed to distributed device ctor
# np.complex for data type and True or False
# for enabling batched_obs.
fixture_params = itertools.product(
    [np.complex64, np.complex128],
    [True, False],
)


def create_random_init_state(numWires, c_dtype, seed=None):
    """Returns a random normalized state of c_dtype with 2**numWires elements."""
    rng = np.random.default_rng(seed)
    r_dtype = np.float64 if c_dtype == np.complex128 else np.float32

    num_elements = 2**numWires
    init_state = rng.random(num_elements).astype(r_dtype) + 1j * rng.random(num_elements).astype(
        r_dtype
    )
    return init_state / np.linalg.norm(init_state)


def apply_operation_gates_qnode_param(tol, dev_mpi, operation, par, wires, seed=None):
    """Wrapper applying a parametric gate with QNode function."""
    num_wires = numQubits
    comm = MPI.COMM_WORLD
    commSize = comm.Get_size()
    num_global_wires = commSize.bit_length() - 1
    num_local_wires = num_wires - num_global_wires

    c_dtype = dev_mpi.c_dtype

    expected_output_cpu = np.zeros(2**num_wires).astype(c_dtype)
    local_state_vector = np.zeros(2**num_local_wires).astype(c_dtype)
    local_expected_output_cpu = np.zeros(2**num_local_wires).astype(c_dtype)
    state_vector = create_random_init_state(num_wires, dev_mpi.c_dtype, seed)
    comm.Bcast(state_vector, root=0)

    comm.Scatter(state_vector, local_state_vector, root=0)
    dev_cpu = qp.device("lightning.qubit", wires=num_wires, c_dtype=c_dtype)

    def circuit(*params):
        qp.StatePrep(state_vector, wires=range(num_wires))
        operation(*params, wires=wires)
        return qp.state()

    cpu_qnode = qp.QNode(circuit, dev_cpu)
    expected_output_cpu = cpu_qnode(*par).astype(c_dtype)
    comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

    mpi_qnode = qp.QNode(circuit, dev_mpi)
    local_state_vector = mpi_qnode(*par)

    assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)


def apply_operation_gates_qnode_nonparam(tol, dev_mpi, operation, wires, seed=None):
    """Wrapper applying a non-parametric gate with QNode function."""
    num_wires = numQubits
    comm = MPI.COMM_WORLD
    commSize = comm.Get_size()
    num_global_wires = commSize.bit_length() - 1
    num_local_wires = num_wires - num_global_wires

    c_dtype = dev_mpi.c_dtype

    expected_output_cpu = np.zeros(2**num_wires).astype(c_dtype)
    local_state_vector = np.zeros(2**num_local_wires).astype(c_dtype)
    local_expected_output_cpu = np.zeros(2**num_local_wires).astype(c_dtype)
    state_vector = create_random_init_state(num_wires, dev_mpi.c_dtype, seed)
    comm.Bcast(state_vector, root=0)

    comm.Scatter(state_vector, local_state_vector, root=0)
    dev_cpu = qp.device("lightning.qubit", wires=num_wires, c_dtype=c_dtype)

    def circuit():
        qp.StatePrep(state_vector, wires=range(num_wires))
        operation(wires=wires)
        return qp.state()

    cpu_qnode = qp.QNode(circuit, dev_cpu)
    expected_output_cpu = cpu_qnode().astype(c_dtype)
    comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

    mpi_qnode = qp.QNode(circuit, dev_mpi)
    local_state_vector = mpi_qnode()

    assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)


class TestApply:  # pylint: disable=missing-function-docstring,too-many-arguments,too-many-positional-arguments
    """Tests whether the device can apply supported quantum gates."""

    @pytest.fixture(params=fixture_params)
    def dev_mpi(self, request):
        return qp.device(
            device_name,
            wires=numQubits,
            mpi=True,
            c_dtype=request.param[0],
            batch_obs=request.param[1],
        )

    # Parameterized test case for single wire nonparam gates
    @pytest.mark.parametrize(
        "operation", [qp.PauliX, qp.PauliY, qp.PauliZ, qp.Hadamard, qp.S, qp.T]
    )
    @pytest.mark.parametrize("wires", [0, 1, numQubits - 2, numQubits - 1])
    def test_apply_operation_single_wire_nonparam(self, tol, operation, wires, dev_mpi, seed):
        apply_operation_gates_qnode_nonparam(tol, dev_mpi, operation, wires, seed)

    @pytest.mark.parametrize("operation", [qp.CNOT, qp.SWAP, qp.CY, qp.CZ])
    @pytest.mark.parametrize("wires", [[0, 1], [numQubits - 2, numQubits - 1], [0, numQubits - 1]])
    def test_apply_operation_two_wire_nonparam(self, tol, operation, wires, dev_mpi, seed):
        apply_operation_gates_qnode_nonparam(tol, dev_mpi, operation, wires, seed)

    @pytest.mark.parametrize("operation", [qp.CSWAP, qp.Toffoli])
    @pytest.mark.parametrize(
        "wires",
        [
            [0, 1, 2],
            [numQubits - 3, numQubits - 2, numQubits - 1],
            [0, 1, numQubits - 1],
            [0, numQubits - 2, numQubits - 1],
        ],
    )
    def test_apply_operation_three_wire_nonparam(self, tol, operation, wires, dev_mpi, seed):
        apply_operation_gates_qnode_nonparam(tol, dev_mpi, operation, wires, seed)

    @pytest.mark.parametrize("operation", [qp.CSWAP, qp.Toffoli])
    @pytest.mark.parametrize(
        "wires",
        [
            [0, 1, 2],
            [numQubits - 3, numQubits - 2, numQubits - 1],
            [0, 1, numQubits - 1],
            [0, numQubits - 2, numQubits - 1],
        ],
    )
    def test_apply_operation_three_wire_qnode_nonparam(self, tol, operation, wires, dev_mpi, seed):
        apply_operation_gates_qnode_nonparam(tol, dev_mpi, operation, wires, seed)

    @pytest.mark.parametrize("operation", [qp.PhaseShift, qp.RX, qp.RY, qp.RZ])
    @pytest.mark.parametrize("par", [[0.1], [0.2], [0.3]])
    @pytest.mark.parametrize("wires", [0, numQubits - 1])
    def test_apply_operation_1gatequbit_1param_gate_qnode_param(
        self, tol, operation, par, wires, dev_mpi, seed
    ):
        apply_operation_gates_qnode_param(tol, dev_mpi, operation, par, wires, seed)

    @pytest.mark.parametrize("operation", [qp.Rot])
    @pytest.mark.parametrize("par", [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    @pytest.mark.parametrize("wires", [0, numQubits - 1])
    def test_apply_operation_1gatequbit_3param_gate_qnode_param(
        self, tol, operation, par, wires, dev_mpi, seed
    ):
        apply_operation_gates_qnode_param(tol, dev_mpi, operation, par, wires, seed)

    @pytest.mark.parametrize("operation", [qp.CRot])
    @pytest.mark.parametrize("par", [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    @pytest.mark.parametrize("wires", [[0, numQubits - 1], [0, 1], [numQubits - 2, numQubits - 1]])
    def test_apply_operation_1gatequbit_3param_cgate_qnode_param(
        self, tol, operation, par, wires, dev_mpi, seed
    ):
        apply_operation_gates_qnode_param(tol, dev_mpi, operation, par, wires, seed)

    @pytest.mark.parametrize(
        "operation",
        [
            qp.CRX,
            qp.CRY,
            qp.CRZ,
            qp.ControlledPhaseShift,
            qp.SingleExcitation,
            qp.SingleExcitationMinus,
            qp.SingleExcitationPlus,
            qp.IsingXX,
            qp.IsingYY,
            qp.IsingZZ,
            qp.PSWAP,
        ],
    )
    @pytest.mark.parametrize("par", [[0.1], [0.2], [0.3]])
    @pytest.mark.parametrize("wires", [[0, numQubits - 1], [0, 1], [numQubits - 2, numQubits - 1]])
    def test_apply_operation_2gatequbit_1param_gate_qnode_param(
        self, tol, operation, par, wires, dev_mpi, seed
    ):
        apply_operation_gates_qnode_param(tol, dev_mpi, operation, par, wires, seed)

    @pytest.mark.parametrize(
        "operation",
        [qp.DoubleExcitation, qp.DoubleExcitationMinus, qp.DoubleExcitationPlus],
    )
    @pytest.mark.parametrize("par", [[0.13], [0.2], [0.3]])
    @pytest.mark.parametrize(
        "wires",
        [
            [0, 1, numQubits - 2, numQubits - 1],
            [0, 1, 2, 3],
            [numQubits - 4, numQubits - 3, numQubits - 2, numQubits - 1],
        ],
    )
    def test_apply_operation_4gatequbit_1param_gate_qnode_param(
        self, tol, operation, par, wires, dev_mpi, seed
    ):
        apply_operation_gates_qnode_param(tol, dev_mpi, operation, par, wires, seed)

    @pytest.mark.parametrize(
        "operation",
        [qp.GlobalPhase],
    )
    @pytest.mark.parametrize("par", [[0.13], [0.2], [0.3]])
    def test_apply_global_phase(self, tol, operation, par, dev_mpi, seed):
        """Test applying the GlobalPhase operation."""
        wires = range(numQubits)

        apply_operation_gates_qnode_param(tol, dev_mpi, operation, par, wires, seed)

    # BasisState test
    @pytest.mark.parametrize("operation", [qp.BasisState])
    @pytest.mark.parametrize("index", range(numQubits))
    def test_state_prep(self, tol, operation, index, dev_mpi, seed):
        par = np.zeros(numQubits, dtype=int)
        par[index] = 1
        num_wires = numQubits
        comm = MPI.COMM_WORLD
        commSize = comm.Get_size()
        num_global_wires = commSize.bit_length() - 1
        num_local_wires = num_wires - num_global_wires

        if dev_mpi.c_dtype == np.float32:
            c_dtype = np.complex64
        else:
            c_dtype = np.complex128

        state_vector = np.zeros(2**num_wires).astype(c_dtype)
        expected_output_cpu = np.zeros(2**num_wires).astype(c_dtype)
        local_state_vector = np.zeros(2**num_local_wires).astype(c_dtype)
        local_expected_output_cpu = np.zeros(2**num_local_wires).astype(c_dtype)

        state_vector = create_random_init_state(num_wires, dev_mpi.c_dtype, seed)

        comm.Scatter(state_vector, local_state_vector, root=0)
        dev_cpu = qp.device("lightning.qubit", wires=num_wires, c_dtype=c_dtype)

        def circuit():
            operation(par, wires=range(numQubits))
            return qp.state()

        cpu_qnode = qp.QNode(circuit, dev_cpu)
        mpi_qnode = qp.QNode(circuit, dev_mpi)

        expected_output_cpu = cpu_qnode().astype(c_dtype)
        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        local_state_vector = mpi_qnode()

        assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "par, wires",
        [
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
        ],
    )
    def test_qubit_state_prep(self, tol, par, wires, dev_mpi, seed):
        num_wires = numQubits
        comm = MPI.COMM_WORLD
        commSize = comm.Get_size()
        num_global_wires = commSize.bit_length() - 1
        num_local_wires = num_wires - num_global_wires

        if dev_mpi.c_dtype == np.float32:
            c_dtype = np.complex64
        else:
            c_dtype = np.complex128

        state_vector = np.zeros(2**num_wires).astype(c_dtype)
        expected_output_cpu = np.zeros(2**num_wires).astype(c_dtype)
        local_state_vector = np.zeros(2**num_local_wires).astype(c_dtype)
        local_expected_output_cpu = np.zeros(2**num_local_wires).astype(c_dtype)

        state_vector = create_random_init_state(num_wires, dev_mpi.c_dtype, seed)

        comm.Scatter(state_vector, local_state_vector, root=0)
        dev_cpu = qp.device("lightning.qubit", wires=num_wires, c_dtype=c_dtype)

        def circuit():
            qp.StatePrep(par, wires=wires)
            return qp.state()

        cpu_qnode = qp.QNode(circuit, dev_cpu)
        mpi_qnode = qp.QNode(circuit, dev_mpi)

        expected_output_cpu = cpu_qnode().astype(c_dtype)
        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        local_state_vector = mpi_qnode()

        assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)

    def test_dev_reset(self, tol, dev_mpi, seed):
        num_wires = numQubits
        comm = MPI.COMM_WORLD
        commSize = comm.Get_size()
        num_global_wires = commSize.bit_length() - 1
        num_local_wires = num_wires - num_global_wires

        if dev_mpi.c_dtype == np.float32:
            c_dtype = np.complex64
        else:
            c_dtype = np.complex128

        state_vector = np.zeros(2**num_wires).astype(c_dtype)
        expected_output_cpu = np.zeros(2**num_wires).astype(c_dtype)
        local_state_vector = np.zeros(2**num_local_wires).astype(c_dtype)
        local_expected_output_cpu = np.zeros(2**num_local_wires).astype(c_dtype)

        state_vector = create_random_init_state(num_wires, dev_mpi.c_dtype, seed)

        comm.Scatter(state_vector, local_state_vector, root=0)
        dev_cpu = qp.device("lightning.qubit", wires=num_wires, c_dtype=c_dtype)

        def circuit():
            qp.PauliX(wires=[0])
            qp.PauliX(wires=[0])
            return qp.state()

        cpu_qnode = qp.QNode(circuit, dev_cpu)

        expected_output_cpu = cpu_qnode().astype(c_dtype)
        comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

        dev_mpi._statevector.reset_state()

        mpi_qnode = qp.QNode(circuit, dev_mpi)
        dev_mpi._statevector.reset_state()

        local_state_vector = mpi_qnode()
        assert np.allclose(local_state_vector, local_expected_output_cpu, atol=tol, rtol=0)


@pytest.mark.skipif(
    device_name == "lightning.kokkos", reason="Sparse Hamiltonian not supported on Kokkos MPI"
)
class TestSparseHamExpval:  # pylint: disable=too-few-public-methods,missing-function-docstring
    """Tests sparse hamiltonian expectation values."""

    @pytest.mark.parametrize("c_dtype", [np.complex128, np.complex64])
    def test_sparse_hamiltonian_expectation(self, c_dtype):
        comm = MPI.COMM_WORLD
        commSize = comm.Get_size()
        num_global_wires = commSize.bit_length() - 1
        num_local_wires = 3 - num_global_wires

        obs = qp.Identity(0) @ qp.PauliX(1) @ qp.PauliY(2)
        obs1 = qp.Identity(1)
        Hmat = qp.Hamiltonian([1.0, 1.0], [obs1, obs]).sparse_matrix()

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
            dtype=c_dtype,
        )

        state_vector /= np.linalg.norm(state_vector)

        local_state_vector = np.zeros(2**num_local_wires).astype(c_dtype)
        comm.Scatter(state_vector, local_state_vector, root=0)

        H_sparse = qp.SparseHamiltonian(Hmat, wires=range(3))

        def circuit():
            qp.StatePrep(state_vector, wires=range(3))
            return qp.expval(H_sparse)

        dev_mpi = qp.device(device_name, wires=3, mpi=False, c_dtype=c_dtype)
        mpi_qnode = qp.QNode(circuit, dev_mpi)
        expected_output_mpi = mpi_qnode()
        comm.Bcast(np.array(expected_output_mpi), root=0)

        dev_mpi = qp.device(device_name, wires=3, mpi=True, c_dtype=c_dtype)
        mpi_qnode = qp.QNode(circuit, dev_mpi)
        expected_output_mpi = mpi_qnode()

        comm.Barrier()

        assert np.allclose(expected_output_mpi, expected_output_mpi)


class TestExpval:
    """Tests that expectation values are properly calculated or that the proper errors are raised."""

    @pytest.mark.parametrize("c_dtype", [np.complex128, np.complex64])
    @pytest.mark.parametrize(
        "operation",
        [
            qp.PauliX,
            qp.PauliY,
            qp.PauliZ,
            qp.Hadamard,
            qp.Identity,
        ],
    )
    @pytest.mark.parametrize("wires", [0, 1, 2, numQubits - 3, numQubits - 2, numQubits - 1])
    def test_expval_single_wire_no_parameters(self, tol, operation, wires, c_dtype, seed):
        """Tests that expectation values are properly calculated for single-wire observables without parameters."""
        num_wires = numQubits
        comm = MPI.COMM_WORLD
        commSize = comm.Get_size()
        num_global_wires = commSize.bit_length() - 1
        num_local_wires = num_wires - num_global_wires

        dev_mpi = qp.device(device_name, wires=numQubits, mpi=True, c_dtype=c_dtype)

        state_vector = create_random_init_state(num_wires, dev_mpi.c_dtype, seed)
        comm.Bcast(state_vector, root=0)

        local_state_vector = np.zeros(2**num_local_wires).astype(c_dtype)
        comm.Scatter(state_vector, local_state_vector, root=0)
        dev_cpu = qp.device("lightning.qubit", wires=num_wires, c_dtype=c_dtype)

        def circuit():
            qp.StatePrep(state_vector, wires=range(num_wires))
            return qp.expval(operation(wires))

        cpu_qnode = qp.QNode(circuit, dev_cpu)
        expected_output_cpu = cpu_qnode()
        comm.Bcast(np.array(expected_output_cpu), root=0)

        mpi_qnode = qp.QNode(circuit, dev_mpi)
        expected_output_mpi = mpi_qnode()

        assert np.allclose(expected_output_mpi, expected_output_cpu, atol=tol, rtol=0)

    @pytest.mark.parametrize("c_dtype", [np.complex128, np.complex64])
    @pytest.mark.parametrize(
        "obs",
        [
            qp.PauliX(0) @ qp.PauliZ(1),
            qp.PauliX(0) @ qp.PauliZ(numQubits - 1),
            qp.PauliX(numQubits - 2) @ qp.PauliZ(numQubits - 1),
            qp.PauliZ(0) @ qp.PauliZ(1),
            qp.PauliZ(0) @ qp.PauliZ(numQubits - 1),
            qp.PauliZ(numQubits - 2) @ qp.PauliZ(numQubits - 1),
        ],
    )
    def test_expval_multiple_obs(self, obs, tol, c_dtype):
        """Test expval with Hamiltonian"""
        num_wires = numQubits

        dev_cpu = qp.device("lightning.qubit", wires=num_wires, c_dtype=c_dtype)
        dev_mpi = qp.device(device_name, wires=num_wires, mpi=True, c_dtype=c_dtype)

        def circuit():
            qp.RX(0.4, wires=[0])
            qp.RY(-0.2, wires=[num_wires - 1])
            return qp.expval(obs)

        cpu_qnode = qp.QNode(circuit, dev_cpu)
        mpi_qnode = qp.QNode(circuit, dev_mpi)

        assert np.allclose(cpu_qnode(), mpi_qnode(), atol=tol, rtol=0)

    @pytest.mark.parametrize("c_dtype", [np.complex128, np.complex64])
    @pytest.mark.parametrize(
        "obs, coeffs",
        [
            ([qp.PauliX(0) @ qp.PauliZ(1)], [0.314]),
            ([qp.PauliX(0) @ qp.PauliZ(numQubits - 1)], [0.314]),
            ([qp.PauliZ(0) @ qp.PauliZ(1)], [0.314]),
            ([qp.PauliZ(0) @ qp.PauliZ(numQubits - 1)], [0.314]),
            (
                [qp.PauliX(0) @ qp.PauliZ(1), qp.PauliZ(0) @ qp.PauliZ(1)],
                [0.314, 0.2],
            ),
            (
                [
                    qp.PauliX(0) @ qp.PauliZ(numQubits - 1),
                    qp.PauliZ(0) @ qp.PauliZ(1),
                ],
                [0.314, 0.2],
            ),
            (
                [
                    qp.PauliX(numQubits - 2) @ qp.PauliZ(numQubits - 1),
                    qp.PauliZ(0) @ qp.PauliZ(1),
                ],
                [0.314, 0.2],
            ),
        ],
    )
    def test_expval_hamiltonian(self, obs, coeffs, tol, c_dtype):
        """Test expval with Hamiltonian"""
        num_wires = numQubits

        ham = qp.Hamiltonian(coeffs, obs)

        dev_cpu = qp.device("lightning.qubit", wires=num_wires, c_dtype=c_dtype)
        dev_mpi = qp.device(device_name, wires=num_wires, mpi=True, c_dtype=c_dtype)

        def circuit():
            qp.RX(0.4, wires=[0])
            qp.RY(-0.2, wires=[numQubits - 1])
            return qp.expval(ham)

        cpu_qnode = qp.QNode(circuit, dev_cpu)
        mpi_qnode = qp.QNode(circuit, dev_mpi)

        assert np.allclose(cpu_qnode(), mpi_qnode(), atol=tol, rtol=0)

    def test_expval_non_pauli_word_hamiltionian(self, tol):
        """Tests expectation values of non-Pauli word Hamiltonians."""
        dev_mpi = qp.device(device_name, wires=3, mpi=True)
        dev_cpu = qp.device("lightning.qubit", wires=3)

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        def circuit():
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            return qp.expval(0.5 * qp.Hadamard(2))

        cpu_qnode = qp.QNode(circuit, dev_cpu)
        mpi_qnode = qp.QNode(circuit, dev_mpi)

        assert np.allclose(cpu_qnode(), mpi_qnode(), atol=tol, rtol=0)


class TestGenerateSample:
    """Tests that samples are properly calculated."""

    @pytest.mark.parametrize("c_dtype", [np.complex128, np.complex64])
    def test_sample_dimensions(self, c_dtype):
        """Tests if the samples returned by sample have
        the correct dimensions
        """
        num_wires = numQubits

        dev = qp.device(device_name, wires=num_wires, mpi=True, c_dtype=c_dtype)

        ops = [qp.RX(1.5708, wires=[0]), qp.RX(1.5708, wires=[1])]

        shots = 10
        obs = qp.PauliZ(wires=[0])
        tape = qp.tape.QuantumScript(ops, [qp.sample(op=obs)], shots=shots)
        s1 = dev.execute(tape)

        assert np.array_equal(s1.shape, (shots,))

        shots = 12
        obs = qp.PauliZ(wires=[1])
        tape = qp.tape.QuantumScript(ops, [qp.sample(op=obs)], shots=shots)
        s2 = dev.execute(tape)

        assert np.array_equal(s2.shape, (shots,))

        shots = 17
        obs = qp.PauliX(0) @ qp.PauliZ(1)
        tape = qp.tape.QuantumScript(ops, [qp.sample(op=obs)], shots=shots)
        s3 = dev.execute(tape)

        assert np.array_equal(s3.shape, (shots,))

    @pytest.mark.parametrize("c_dtype", [np.complex128, np.complex64])
    def test_sample_values(self, tol, c_dtype):
        """Tests if the samples returned by sample have
        the correct values
        """
        num_wires = numQubits

        dev = qp.device(device_name, wires=num_wires, mpi=True, c_dtype=c_dtype)

        shots = qp.measurements.Shots(1000)
        ops = [qp.RX(1.5708, wires=[0])]
        obs = qp.PauliZ(0)
        tape = qp.tape.QuantumScript(ops, [qp.sample(op=obs)], shots=shots)
        s1 = dev.execute(tape)

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1**2, 1, atol=tol, rtol=0)

    @pytest.mark.parametrize("c_dtype", [np.complex128, np.complex64])
    def test_sample_values_qnode(self, tol, c_dtype):
        """Tests if the samples returned by sample have
        the correct values
        """
        num_wires = numQubits

        dev_mpi = qp.device(device_name, wires=num_wires, mpi=True, c_dtype=c_dtype)
        dev_mpi._statevector.reset_state()

        @partial(qp.set_shots, shots=2000)
        @qp.qnode(dev_mpi)
        def circuit():
            qp.RX(1.5708, wires=0)
            return qp.sample(qp.PauliZ(0))

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(circuit() ** 2, 1, atol=tol, rtol=0)

    @pytest.mark.parametrize("c_dtype", [np.complex128, np.complex64])
    def test_multi_samples_return_correlated_results(self, c_dtype):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """
        num_wires = 3

        dev_mpi = qp.device(device_name, wires=num_wires, mpi=True, c_dtype=c_dtype)

        @partial(qp.set_shots, shots=2000)
        @qp.qnode(dev_mpi)
        def circuit():
            qp.Hadamard(0)
            qp.CNOT(wires=[0, 1])
            return qp.sample(qp.PauliZ(0)), qp.sample(qp.PauliZ(1))

        outcomes = circuit()

        assert np.array_equal(outcomes[0], outcomes[1])

    @pytest.mark.local_salt(42)
    @pytest.mark.parametrize("c_dtype", [np.complex128, np.complex64])
    def test_paulix_pauliy(self, c_dtype, tol=TOL_STOCHASTIC):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        num_wires = 3

        dev_mpi = qp.device(device_name, wires=num_wires, mpi=True, c_dtype=c_dtype)

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @partial(qp.set_shots, shots=2000)
        @qp.qnode(dev_mpi)
        def circuit():
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            return qp.sample(qp.PauliX(wires=[0]) @ qp.PauliY(wires=[2]))

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

    @pytest.mark.local_salt(42)
    @pytest.mark.parametrize("c_dtype", [np.complex128, np.complex64])
    def test_pauliz_hadamard(self, c_dtype, tol=TOL_STOCHASTIC):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        num_wires = 3

        dev_mpi = qp.device(device_name, wires=num_wires, mpi=True, c_dtype=c_dtype)

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @partial(qp.set_shots, shots=2000)
        @qp.qnode(dev_mpi)
        def circuit():
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            return qp.sample(qp.PauliZ(wires=[0]) @ qp.Hadamard(wires=[1]) @ qp.PauliY(wires=[2]))

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

    @pytest.mark.parametrize("c_dtype", [np.complex128, np.complex64])
    def test_paulix_pauliy(self, c_dtype, tol=TOL_STOCHASTIC):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        num_wires = 3

        dev_mpi = qp.device(device_name, wires=num_wires, mpi=True, c_dtype=c_dtype)

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @partial(qp.set_shots, shots=2000)
        @qp.qnode(dev_mpi)
        def circuit():
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            return qp.var(qp.PauliX(wires=[0]) @ qp.PauliY(wires=[2]))

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

    @pytest.mark.parametrize("c_dtype", [np.complex128, np.complex64])
    def test_pauliz_hadamard(self, c_dtype, tol=TOL_STOCHASTIC):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        num_wires = 3
        dev_mpi = qp.device(device_name, wires=num_wires, mpi=True, c_dtype=c_dtype)

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @partial(qp.set_shots, shots=2000)
        @qp.qnode(dev_mpi)
        def circuit():
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            return qp.var(qp.PauliZ(wires=[0]) @ qp.Hadamard(wires=[1]) @ qp.PauliY(wires=[2]))

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
    qp.StatePrep(
        unitary_group.rvs(2**numQubits, random_state=0)[0],
        wires=wires,
    )
    qp.RX(params[0], wires=wires[0])
    qp.RY(params[1], wires=wires[1])
    qp.adjoint(qp.RX(params[2], wires=wires[2]))
    qp.RZ(params[0], wires=wires[3])
    qp.CRX(params[3], wires=[wires[3], wires[0]])
    qp.PhaseShift(params[4], wires=wires[2])
    qp.CRY(params[5], wires=[wires[2], wires[1]])
    qp.adjoint(qp.CRZ(params[5], wires=[wires[0], wires[3]]))
    qp.adjoint(qp.PhaseShift(params[6], wires=wires[0]))
    qp.Rot(params[6], params[7], params[8], wires=wires[0])
    qp.adjoint(qp.Rot(params[8], params[8], params[9], wires=wires[1]))
    qp.MultiRZ(params[11], wires=[wires[0], wires[1]])
    qp.CPhase(params[12], wires=[wires[3], wires[2]])
    qp.IsingXX(params[13], wires=[wires[1], wires[0]])
    qp.IsingYY(params[14], wires=[wires[3], wires[2]])
    qp.IsingZZ(params[15], wires=[wires[2], wires[1]])
    qp.PSWAP(params[16], wires=[wires[3], wires[0]])
    qp.SingleExcitation(params[24], wires=[wires[2], wires[0]])
    qp.DoubleExcitation(params[25], wires=[wires[2], wires[0], wires[1], wires[3]])


@pytest.mark.local_salt(42)
@pytest.mark.parametrize(
    "returns",
    [
        (qp.PauliX(0),),
        (qp.PauliY(0),),
        (qp.PauliZ(0),),
        (qp.PauliX(1),),
        (qp.PauliY(1),),
        (qp.PauliZ(1),),
        (qp.PauliX(2),),
        (qp.PauliY(2),),
        (qp.PauliZ(2),),
        (qp.PauliX(3),),
        (qp.PauliY(3),),
        (qp.PauliZ(3),),
        (qp.PauliX(0), qp.PauliY(1)),
        (
            qp.PauliZ(0),
            qp.PauliX(1),
            qp.PauliY(2),
        ),
        (
            qp.PauliY(0),
            qp.PauliZ(1),
            qp.PauliY(3),
        ),
        (qp.PauliZ(0) @ qp.PauliY(3),),
        (qp.Hadamard(2),),
        (qp.Hadamard(3) @ qp.PauliZ(2),),
        (qp.PauliX(0) @ qp.PauliY(3),),
        (qp.PauliY(0) @ qp.PauliY(2) @ qp.PauliY(3),),
        (qp.PauliZ(0) @ qp.PauliZ(1) @ qp.PauliZ(2),),
        (0.5 * qp.PauliZ(0) @ qp.PauliZ(2),),
    ],
)
def test_integration(returns, seed):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations"""
    num_wires = numQubits
    dev_default = qp.device("lightning.qubit", wires=range(num_wires))
    dev_mpi = qp.device(device_name, wires=num_wires, mpi=True, c_dtype=np.complex128)

    def circuit(params):
        circuit_ansatz(params, wires=range(num_wires))
        return qp.math.hstack([qp.expval(r) for r in returns])

    n_params = 30
    rng = pnp.random.default_rng(seed)
    params = rng.random(n_params)

    qnode_mpi = qp.QNode(circuit, dev_mpi, diff_method="parameter-shift")
    qnode_default = qp.QNode(circuit, dev_default, diff_method="parameter-shift")

    def convert_to_array_mpi(params):
        return pnp.array(qnode_mpi(params))

    def convert_to_array_default(params):
        return pnp.array(qnode_default(params))

    j_mpi = qp.jacobian(convert_to_array_mpi)(params)
    j_default = qp.jacobian(convert_to_array_default)(params)

    assert np.allclose(j_mpi, j_default, atol=1e-7)


custom_wires = ["alice", 3.14, -1, 0, "bob", "l", "m", "n"]


@pytest.mark.local_salt(42)
@pytest.mark.parametrize(
    "returns",
    [
        qp.PauliZ(custom_wires[0]),
        qp.PauliX(custom_wires[2]),
        qp.PauliZ(custom_wires[0]) @ qp.PauliY(custom_wires[3]),
        qp.Hadamard(custom_wires[2]),
        qp.Hadamard(custom_wires[3]) @ qp.PauliZ(custom_wires[2]),
        qp.PauliX(custom_wires[0]) @ qp.PauliY(custom_wires[3]),
        qp.PauliY(custom_wires[0]) @ qp.PauliY(custom_wires[2]) @ qp.PauliY(custom_wires[3]),
    ],
)
def test_integration_custom_wires(returns, seed):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""
    dev_lightning = qp.device("lightning.qubit", wires=custom_wires)
    dev_mpi = qp.device(device_name, wires=custom_wires, mpi=True, c_dtype=np.complex128)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qp.expval(returns), qp.expval(qp.PauliY(custom_wires[1]))

    n_params = 30
    rng = pnp.random.default_rng(seed)
    params = rng.random(n_params)

    qnode_mpi = qp.QNode(circuit, dev_mpi, diff_method="parameter-shift")
    qnode_lightning = qp.QNode(circuit, dev_lightning, diff_method="parameter-shift")

    def convert_to_array_mpi(params):
        return pnp.array(qnode_mpi(params))

    def convert_to_array_lightning(params):
        return pnp.array(qnode_lightning(params))

    j_mpi = qp.jacobian(convert_to_array_mpi)(params)
    j_lightning = qp.jacobian(convert_to_array_lightning)(params)

    assert np.allclose(j_mpi, j_lightning, atol=1e-7)
