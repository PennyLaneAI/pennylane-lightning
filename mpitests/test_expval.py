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
Unit tests for the expval method of Lightning devices.
"""
# pylint: disable=protected-access,too-few-public-methods,unused-import,missing-function-docstring,too-many-arguments,c-extension-no-member

import numpy as np
import pennylane as qml
import pytest
from conftest import PHI, THETA, VARPHI, device_name
from mpi4py import MPI

numQubits = 8


def create_random_init_state(numWires, C_DTYPE, seed_value=48):
    """Returns a random initial state of a certain type."""
    np.random.seed(seed_value)

    R_DTYPE = np.float64 if C_DTYPE == np.complex128 else np.float32

    num_elements = 1 << numWires
    init_state = np.random.rand(num_elements).astype(R_DTYPE) + 1j * np.random.rand(
        num_elements
    ).astype(R_DTYPE)

    init_state = init_state / np.linalg.norm(init_state)
    return init_state


def apply_operation_gates_qnode_param(tol, dev_mpi, operation, par, Wires):
    """Wrapper applying a parametric gate with QNode function."""
    num_wires = numQubits
    comm = MPI.COMM_WORLD
    commSize = comm.Get_size()
    num_global_wires = commSize.bit_length() - 1
    num_local_wires = num_wires - num_global_wires

    c_dtype = dev_mpi.c_dtype

    expected_output_cpu = np.zeros(1 << num_wires).astype(c_dtype)
    local_state_vector = np.zeros(1 << num_local_wires).astype(c_dtype)
    local_expected_output_cpu = np.zeros(1 << num_local_wires).astype(c_dtype)

    state_vector = create_random_init_state(num_wires, dev_mpi.c_dtype)
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


def apply_operation_gates_qnode_nonparam(tol, dev_mpi, operation, Wires):
    """Wrapper applying a non-parametric gate with QNode function."""
    num_wires = numQubits
    comm = MPI.COMM_WORLD
    commSize = comm.Get_size()
    num_global_wires = commSize.bit_length() - 1
    num_local_wires = num_wires - num_global_wires

    c_dtype = dev_mpi.c_dtype

    expected_output_cpu = np.zeros(1 << num_wires).astype(c_dtype)
    local_state_vector = np.zeros(1 << num_local_wires).astype(c_dtype)
    local_expected_output_cpu = np.zeros(1 << num_local_wires).astype(c_dtype)

    state_vector = create_random_init_state(num_wires, dev_mpi.c_dtype)
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


@pytest.mark.parametrize("C_DTYPE", [np.complex128, np.complex64])
@pytest.mark.parametrize("batch_obs", [True, False])
class TestExpval:
    """Tests that expectation values are properly calculated or that the proper errors are raised."""

    @pytest.mark.parametrize(
        "operation",
        [
            qml.PauliX,
            qml.PauliY,
            qml.PauliZ,
            qml.Hadamard,
            pytest.param(qml.Identity, marks=pytest.mark.xfail(reason="The Identity gate need a deep review for MPI support")),
        ],
    )
    @pytest.mark.parametrize("wires", [0, 1, 2, numQubits - 3, numQubits - 2, numQubits - 1])
    def test_expval_single_wire_no_parameters(self, tol, operation, wires, C_DTYPE, batch_obs):
        """Tests that expectation values are properly calculated for single-wire observables without parameters."""
        num_wires = numQubits
        comm = MPI.COMM_WORLD
        commSize = comm.Get_size()
        num_global_wires = commSize.bit_length() - 1
        num_local_wires = num_wires - num_global_wires

        dev_mpi = qml.device("lightning.gpu", wires=numQubits, mpi=True, c_dtype=C_DTYPE, batch_obs=batch_obs)

        dev_cpu = qml.device("lightning.qubit", wires=num_wires, c_dtype=C_DTYPE)

        state_vector = create_random_init_state(num_wires, dev_mpi.c_dtype)
        comm.Bcast(state_vector, root=0)

        def circuit():
            qml.StatePrep(state_vector, wires=range(num_wires))
            return qml.expval(operation(0))

        cpu_qnode = qml.QNode(circuit, dev_cpu)
        expected_output_cpu = cpu_qnode()
        comm.Bcast(np.array(expected_output_cpu), root=0)

        mpi_qnode = qml.QNode(circuit, dev_mpi)
        expected_output_mpi = mpi_qnode()

        assert np.allclose(expected_output_mpi, expected_output_cpu, atol=tol, rtol=0)

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
    def test_expval_multiple_obs(self, obs, tol, C_DTYPE, batch_obs):
        """Test expval with Hamiltonian"""
        num_wires = numQubits

        dev_cpu = qml.device("lightning.qubit", wires=num_wires, c_dtype=C_DTYPE)
        dev_mpi = qml.device("lightning.gpu", wires=num_wires, mpi=True, c_dtype=C_DTYPE, batch_obs=batch_obs)

        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[num_wires - 1])
            return qml.expval(obs)

        cpu_qnode = qml.QNode(circuit, dev_cpu)
        mpi_qnode = qml.QNode(circuit, dev_mpi)

        assert np.allclose(cpu_qnode(), mpi_qnode(), atol=tol, rtol=0)

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
    def test_expval_hamiltonian(self, obs, coeffs, tol, C_DTYPE, batch_obs):
        """Test expval with Hamiltonian"""
        num_wires = numQubits

        ham = qml.Hamiltonian(coeffs, obs)

        dev_cpu = qml.device("lightning.qubit", wires=num_wires, c_dtype=C_DTYPE)
        dev_mpi = qml.device("lightning.gpu", wires=num_wires, mpi=True, c_dtype=C_DTYPE, batch_obs=batch_obs)

        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[numQubits - 1])
            return qml.expval(ham)

        cpu_qnode = qml.QNode(circuit, dev_cpu)
        mpi_qnode = qml.QNode(circuit, dev_mpi)

        assert np.allclose(cpu_qnode(), mpi_qnode(), atol=tol, rtol=0)

    def test_expval_non_pauli_word_hamiltionian(self, tol, C_DTYPE, batch_obs):
        """Tests expectation values of non-Pauli word Hamiltonians."""
        dev_mpi = qml.device("lightning.gpu", wires=3, mpi=True, c_dtype=C_DTYPE, batch_obs=batch_obs)
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

    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    @pytest.mark.parametrize("n_wires", range(1, 8))
    def test_hermitian_expectation(self, n_wires, theta, phi, tol, C_DTYPE, batch_obs):
        """Test that Hadamard expectation value is correct"""
        n_qubits = 7
        dev_def = qml.device("default.qubit", wires=n_qubits)
        dev = qml.device(device_name, mpi=True, wires=n_qubits, c_dtype=C_DTYPE, batch_obs=batch_obs)
        comm = MPI.COMM_WORLD

        m = 2**n_wires
        U = np.random.rand(m, m) + 1j * np.random.rand(m, m)
        U = U + np.conj(U.T)
        U = U.astype(dev.c_dtype)
        comm.Bcast(U, root=0)
        obs = qml.Hermitian(U, wires=range(n_wires))

        init_state = np.random.rand(2**n_qubits) + 1j * np.random.rand(2**n_qubits)
        init_state /= np.sqrt(np.dot(np.conj(init_state), init_state))
        init_state = init_state.astype(dev.c_dtype)
        comm.Bcast(init_state, root=0)

        def circuit():
            qml.StatePrep(init_state, wires=range(n_qubits))
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(obs)

        circ = qml.QNode(circuit, dev)
        comm = MPI.COMM_WORLD
        mpisize = comm.Get_size()
        if n_wires > n_qubits - np.log2(mpisize):
            with pytest.raises(
                RuntimeError,
                match="MPI backend does not support Hermitian with number of target wires larger than local wire number",
            ):
                circ()
        else:
            circ_def = qml.QNode(circuit, dev_def)
            assert np.allclose(circ(), circ_def(), tol)


@pytest.mark.parametrize("diff_method", ("parameter-shift", "adjoint"))
class TestExpOperatorArithmetic:
    """Test integration of lightning with SProd, Prod, and Sum."""

    def test_sprod(self, diff_method):
        """Test the `SProd` class with lightning qubit."""

        dev = qml.device(device_name, mpi=True, wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.s_prod(0.5, qml.PauliZ(0)))

        x = qml.numpy.array(0.123, requires_grad=True)
        res = circuit(x)
        assert qml.math.allclose(res, 0.5 * np.cos(x))

        g = qml.grad(circuit)(x)
        expected_grad = -0.5 * np.sin(x)
        assert qml.math.allclose(g, expected_grad)

    def test_prod(self, diff_method):
        """Test the `Prod` class with lightning qubit."""

        dev = qml.device(device_name, mpi=True, wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(1)
            qml.PauliZ(1)
            return qml.expval(qml.prod(qml.PauliZ(0), qml.PauliX(1)))

        x = qml.numpy.array(0.123, requires_grad=True)
        res = circuit(x)
        assert qml.math.allclose(res, -np.cos(x))

        g = qml.grad(circuit)(x)
        expected_grad = np.sin(x)
        assert qml.math.allclose(g, expected_grad)

    def test_sum(self, diff_method):
        """Test the `Sum` class with Lightning."""

        dev = qml.device(device_name, mpi=True, wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return qml.expval(qml.sum(qml.PauliZ(0), qml.PauliX(1)))

        x = qml.numpy.array(-3.21, requires_grad=True)
        y = qml.numpy.array(2.34, requires_grad=True)
        res = circuit(x, y)
        assert qml.math.allclose(res, np.cos(x) + np.sin(y))

        g = qml.grad(circuit)(x, y)
        expected = (-np.sin(x), np.cos(y))
        assert qml.math.allclose(g, expected)

    def test_integration(self, diff_method):
        """Test a Combination of `Sum`, `SProd`, and `Prod`."""

        obs = qml.sum(
            qml.s_prod(2.3, qml.PauliZ(0)),
            -0.5 * qml.prod(qml.PauliY(0), qml.PauliZ(1)),
        )

        dev = qml.device(device_name, mpi=True, wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return qml.expval(obs)

        x = qml.numpy.array(0.654, requires_grad=True)
        y = qml.numpy.array(-0.634, requires_grad=True)

        res = circuit(x, y)
        expected = 2.3 * np.cos(x) + 0.5 * np.sin(x) * np.cos(y)
        assert qml.math.allclose(res, expected)

        g = qml.grad(circuit)(x, y)
        expected = (
            -2.3 * np.sin(x) + 0.5 * np.cos(y) * np.cos(x),
            -0.5 * np.sin(x) * np.sin(y),
        )
        assert qml.math.allclose(g, expected)


@pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorExpval:
    """Test tensor expectation values"""

    @pytest.mark.parametrize(
        "obs,expected",
        [
            (qml.PauliX(0) @ qml.PauliY(2), "PXPY"),
            (qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2), "PZIPZ"),
            (qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2), "PZHPY"),
        ],
    )
    def test_tensor(self, theta, phi, varphi, obs, expected, tol):
        """Test that a tensor product involving PauliX and PauliY works
        correctly"""
        dev = qml.device(device_name, mpi=True, wires=3)

        def circuit():
            qml.RX(theta, wires=[0]),
            qml.RX(phi, wires=[1]),
            qml.RX(varphi, wires=[2]),
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=[1, 2]),
            return qml.expval(obs)

        mpi_qnode = qml.QNode(circuit, dev)
        res = mpi_qnode()

        if expected == "PXPY":
            expected_val = np.sin(theta) * np.sin(phi) * np.sin(varphi)
        elif expected == "PZIPZ":
            expected_val = np.cos(varphi) * np.cos(phi)
        elif expected == "PZHPY":
            expected_val = -(
                np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)
            ) / np.sqrt(2)

        assert np.allclose(res, expected_val, atol=tol)
