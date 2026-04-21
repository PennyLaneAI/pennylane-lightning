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
Unit tests for expval on :mod:`pennylane_lightning` MPI-enabled devices.
"""

# pylint: disable=protected-access,too-few-public-methods,unused-import,missing-function-docstring,too-many-arguments,too-many-positional-arguments,c-extension-no-member

import numpy as np
import pennylane as qp
import pytest
from conftest import PHI, THETA, VARPHI, device_name
from mpi4py import MPI

numQubits = 8


def create_random_init_state(numWires, c_dtype, seed=None):
    """Returns a random normalized state of c_dtype with 2**numWires elements."""
    rng = np.random.default_rng(seed)
    r_dtype = np.float64 if c_dtype == np.complex128 else np.float32

    num_elements = 2**numWires
    init_state = rng.random(num_elements).astype(r_dtype) + 1j * rng.random(num_elements).astype(
        r_dtype
    )
    return init_state / np.linalg.norm(init_state)


@pytest.mark.parametrize("c_dtype", [np.complex128, np.complex64])
@pytest.mark.parametrize("batch_obs", [True, False])
class TestExpval:
    """Tests that expectation values are properly calculated or that the proper errors are raised."""

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
    @pytest.mark.parametrize("wires", [0, 1, 2, numQubits - 2, numQubits - 1])
    def test_expval_single_wire_no_parameters(
        self, tol, operation, wires, c_dtype, batch_obs, seed
    ):
        """Tests that expectation values are properly calculated for single-wire observables without parameters."""
        num_wires = numQubits
        comm = MPI.COMM_WORLD

        dev_mpi = qp.device(
            device_name, wires=numQubits, mpi=True, c_dtype=c_dtype, batch_obs=batch_obs
        )

        dev_cpu = qp.device("lightning.qubit", wires=num_wires, c_dtype=c_dtype)

        state_vector = create_random_init_state(num_wires, dev_mpi.c_dtype, seed)
        comm.Bcast(state_vector, root=0)

        def circuit():
            qp.StatePrep(state_vector, wires=range(num_wires))
            return qp.expval(operation(wires))

        cpu_qnode = qp.QNode(circuit, dev_cpu)
        expected_output_cpu = cpu_qnode()
        comm.Bcast(np.array(expected_output_cpu), root=0)

        mpi_qnode = qp.QNode(circuit, dev_mpi)
        expected_output_mpi = mpi_qnode()

        assert np.allclose(expected_output_mpi, expected_output_cpu, atol=tol, rtol=0)

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
    def test_expval_multiple_obs(self, obs, tol, c_dtype, batch_obs):
        """Test expval with Hamiltonian"""
        num_wires = numQubits

        dev_cpu = qp.device("lightning.qubit", wires=num_wires, c_dtype=c_dtype)
        dev_mpi = qp.device(
            device_name, wires=num_wires, mpi=True, c_dtype=c_dtype, batch_obs=batch_obs
        )

        def circuit():
            qp.RX(0.4, wires=[0])
            qp.RY(-0.2, wires=[num_wires - 1])
            return qp.expval(obs)

        cpu_qnode = qp.QNode(circuit, dev_cpu)
        mpi_qnode = qp.QNode(circuit, dev_mpi)

        assert np.allclose(cpu_qnode(), mpi_qnode(), atol=tol, rtol=0)

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
    def test_expval_hamiltonian(self, obs, coeffs, tol, c_dtype, batch_obs):
        """Test expval with Hamiltonian"""
        num_wires = numQubits

        ham = qp.Hamiltonian(coeffs, obs)

        dev_cpu = qp.device("lightning.qubit", wires=num_wires, c_dtype=c_dtype)
        dev_mpi = qp.device(
            device_name, wires=num_wires, mpi=True, c_dtype=c_dtype, batch_obs=batch_obs
        )

        def circuit():
            qp.RX(0.4, wires=[0])
            qp.RY(-0.2, wires=[numQubits - 1])
            return qp.expval(ham)

        cpu_qnode = qp.QNode(circuit, dev_cpu)
        mpi_qnode = qp.QNode(circuit, dev_mpi)

        assert np.allclose(cpu_qnode(), mpi_qnode(), atol=tol, rtol=0)

    def test_expval_non_pauli_word_hamiltonian(self, tol, c_dtype, batch_obs):
        """Tests expectation values of non-Pauli word Hamiltonians."""
        dev_mpi = qp.device(device_name, wires=3, mpi=True, c_dtype=c_dtype, batch_obs=batch_obs)
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

    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    @pytest.mark.parametrize("n_wires", range(1, numQubits))
    def test_hermitian_expectation(self, n_wires, theta, phi, tol, c_dtype, batch_obs, seed):
        """Test that Hadamard expectation value is correct"""
        n_qubits = numQubits - 1
        dev_def = qp.device("default.qubit", wires=n_qubits)
        dev = qp.device(
            device_name, mpi=True, wires=n_qubits, c_dtype=c_dtype, batch_obs=batch_obs
        )
        comm = MPI.COMM_WORLD

        rng = np.random.default_rng(seed)
        m = 2**n_wires
        U = rng.random((m, m)) + 1j * rng.random((m, m))
        U = U + np.conj(U.T)
        U = U.astype(dev.c_dtype)
        comm.Bcast(U, root=0)
        obs = qp.Hermitian(U, wires=range(n_wires))

        init_state = rng.random(2**n_qubits) + 1j * rng.random(2**n_qubits)
        init_state = init_state / np.linalg.norm(init_state)
        init_state = init_state.astype(dev.c_dtype)
        comm.Bcast(init_state, root=0)

        def circuit():
            qp.StatePrep(init_state, wires=range(n_qubits))
            qp.RY(theta, wires=[0])
            qp.RY(phi, wires=[1])
            qp.CNOT(wires=[0, 1])
            return qp.expval(obs)

        circ = qp.QNode(circuit, dev)
        comm = MPI.COMM_WORLD
        mpisize = comm.Get_size()
        if n_wires > n_qubits - np.log2(mpisize):
            with pytest.raises(
                RuntimeError,
                match="MPI backend does not support Hermitian with number of target wires larger than local wire number",
            ):
                circ()
        else:
            circ_def = qp.QNode(circuit, dev_def)
            assert np.allclose(circ(), circ_def(), tol)


@pytest.mark.parametrize("diff_method", ("parameter-shift", "adjoint"))
class TestExpOperatorArithmetic:
    """Test integration of lightning with SProd, Prod, and Sum."""

    def test_sprod(self, diff_method):
        """Test the `SProd` class with lightning qubit."""

        dev = qp.device(device_name, mpi=True, wires=2)

        @qp.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qp.RX(x, wires=0)
            return qp.expval(qp.s_prod(0.5, qp.PauliZ(0)))

        x = qp.numpy.array(0.123, requires_grad=True)
        res = circuit(x)
        assert qp.math.allclose(res, 0.5 * np.cos(x))

        g = qp.grad(circuit)(x)
        expected_grad = -0.5 * np.sin(x)
        assert qp.math.allclose(g, expected_grad)

    def test_prod(self, diff_method):
        """Test the `Prod` class with lightning qubit."""

        dev = qp.device(device_name, mpi=True, wires=2)

        @qp.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qp.RX(x, wires=0)
            qp.Hadamard(1)
            qp.PauliZ(1)
            return qp.expval(qp.prod(qp.PauliZ(0), qp.PauliX(1)))

        x = qp.numpy.array(0.123, requires_grad=True)
        res = circuit(x)
        assert qp.math.allclose(res, -np.cos(x))

        g = qp.grad(circuit)(x)
        expected_grad = np.sin(x)
        assert qp.math.allclose(g, expected_grad)

    def test_sum(self, diff_method):
        """Test the `Sum` class with Lightning."""

        dev = qp.device(device_name, mpi=True, wires=2)

        @qp.qnode(dev, diff_method=diff_method)
        def circuit(x, y):
            qp.RX(x, wires=0)
            qp.RY(y, wires=1)
            return qp.expval(qp.sum(qp.PauliZ(0), qp.PauliX(1)))

        x = qp.numpy.array(-3.21, requires_grad=True)
        y = qp.numpy.array(2.34, requires_grad=True)
        res = circuit(x, y)
        assert qp.math.allclose(res, np.cos(x) + np.sin(y))

        g = qp.grad(circuit)(x, y)
        expected = (-np.sin(x), np.cos(y))
        assert qp.math.allclose(g, expected)

    def test_integration(self, diff_method):
        """Test a Combination of `Sum`, `SProd`, and `Prod`."""

        obs = qp.sum(
            qp.s_prod(2.3, qp.PauliZ(0)),
            -0.5 * qp.prod(qp.PauliY(0), qp.PauliZ(1)),
        )

        dev = qp.device(device_name, mpi=True, wires=2)

        @qp.qnode(dev, diff_method=diff_method)
        def circuit(x, y):
            qp.RX(x, wires=0)
            qp.RY(y, wires=1)
            return qp.expval(obs)

        x = qp.numpy.array(0.654, requires_grad=True)
        y = qp.numpy.array(-0.634, requires_grad=True)

        res = circuit(x, y)
        expected = 2.3 * np.cos(x) + 0.5 * np.sin(x) * np.cos(y)
        assert qp.math.allclose(res, expected)

        g = qp.grad(circuit)(x, y)
        expected = (
            -2.3 * np.sin(x) + 0.5 * np.cos(y) * np.cos(x),
            -0.5 * np.sin(x) * np.sin(y),
        )
        assert qp.math.allclose(g, expected)


@pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorExpval:
    """Test tensor expectation values"""

    @pytest.mark.parametrize(
        "obs,expected",
        [
            (qp.PauliX(0) @ qp.PauliY(2), "PXPY"),
            (qp.PauliZ(0) @ qp.Identity(1) @ qp.PauliZ(2), "PZIPZ"),
            (qp.PauliZ(0) @ qp.Hadamard(1) @ qp.PauliY(2), "PZHPY"),
        ],
    )
    def test_tensor(self, theta, phi, varphi, obs, expected, tol):
        """Test that a tensor product involving PauliX and PauliY works
        correctly"""
        dev = qp.device(device_name, mpi=True, wires=3)

        def circuit():
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            return qp.expval(obs)

        mpi_qnode = qp.QNode(circuit, dev)
        res = mpi_qnode()

        if expected == "PXPY":
            expected_val = np.sin(theta) * np.sin(phi) * np.sin(varphi)
        elif expected == "PZIPZ":
            expected_val = np.cos(varphi) * np.cos(phi)
        elif expected == "PZHPY":
            expected_val = -(
                np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)
            ) / np.sqrt(2)
        else:
            expected_val = 0

        assert np.allclose(res, expected_val, atol=tol)
