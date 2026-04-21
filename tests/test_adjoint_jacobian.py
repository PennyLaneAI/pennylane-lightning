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
Tests for ``adjoint_jacobian`` method on Lightning devices.
"""

import itertools
import math

import pennylane as qp
import pytest
from conftest import LightningDevice as ld
from conftest import device_name, get_random_matrix, get_random_normalized_state
from pennylane import QNode
from pennylane import numpy as np
from pennylane import qchem, qnode
from pennylane.exceptions import QuantumFunctionError
from scipy.stats import unitary_group

I, X, Y, Z = (
    np.eye(2),
    qp.X.compute_matrix(),
    qp.Y.compute_matrix(),
    qp.Z.compute_matrix(),
)

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

if device_name == "lightning.tensor":
    pytest.skip("lightning.tensor doesn't support adjoint jacobian.", allow_module_level=True)

kokkos_args = [None]
if device_name == "lightning.kokkos":
    from pennylane_lightning.lightning_kokkos_ops import InitializationSettings

    kokkos_args += [InitializationSettings().set_num_threads(2)]

fixture_params = list(
    itertools.product(
        [np.complex64, np.complex128], kokkos_args, [None, 3]  # c_dtype x kokkos_args x wires
    )
)


def Rx(theta):
    r"""One-qubit rotation about the x axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}`
    """
    return math.cos(theta / 2) * I + 1j * math.sin(-theta / 2) * X


def Ry(theta):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    return math.cos(theta / 2) * I + 1j * math.sin(-theta / 2) * Y


def Rz(theta):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    return math.cos(theta / 2) * I + 1j * math.sin(-theta / 2) * Z


def get_tolerance_and_stepsize(device, step_size=False):
    """Helper function to get tolerance and finite diff step size for
    different device dtypes"""
    tol = 1e-3 if device.c_dtype == np.complex64 else 1e-7
    h = tol if step_size else None
    return tol, h


class TestAdjointJacobian:
    """Tests for the adjoint_jacobian method"""

    @pytest.fixture(params=fixture_params)
    def dev(self, request):
        params = request.param
        if device_name == "lightning.kokkos":
            return qp.device(device_name, wires=3, c_dtype=params[0], kokkos_args=params[1])
        return qp.device(device_name, wires=params[2], c_dtype=params[0])

    def test_not_expval(self, dev):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""

        with qp.tape.QuantumTape() as tape:
            qp.RX(0.1, wires=0)
            qp.var(qp.PauliZ(0))

        method = dev.compute_derivatives

        with pytest.raises(QuantumFunctionError, match="Adjoint differentiation method does not"):
            method(tape)

        with qp.tape.QuantumTape() as tape:
            qp.RX(0.1, wires=0)
            qp.state()

        with pytest.raises(
            QuantumFunctionError,
            match="Adjoint differentiation method does not support measurement StateMP.",
        ):
            method(tape)

    def test_finite_shots_error(self):
        """Tests warning raised when finite shots specified"""

        dev = qp.device(device_name, wires=1)

        tape = qp.tape.QuantumScript([], [qp.expval(qp.PauliZ(0))], shots=1)

        with pytest.raises(
            QuantumFunctionError,
            match="Requested adjoint differentiation to be computed with finite shots.",
        ):
            dev.compute_derivatives(tape)

    def test_empty_measurements(self, dev):
        """Tests if an empty array is returned when the measurements of the tape is empty."""

        with qp.tape.QuantumTape() as tape:
            qp.RX(0.4, wires=[0])

        method = dev.compute_derivatives
        jac = method(tape)
        assert len(jac) == 0

    def test_unsupported_op(self, dev):
        """Test if a QuantumFunctionError is raised for an unsupported operation, i.e.,
        multi-parameter operations that are not qp.Rot"""

        with qp.tape.QuantumTape() as tape:
            qp.CRot(0.1, 0.2, 0.3, wires=[0, 1])
            qp.expval(qp.PauliZ(0))

        with pytest.raises(
            RuntimeError,
            match="The operation is not supported using the adjoint differentiation method",
        ):
            dev.compute_derivatives(tape)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qp.RX, qp.RY, qp.RZ])
    def test_pauli_rotation_gradient(self, G, theta, dev):
        """Tests that the automatic gradients of Pauli rotations are correct."""
        random_state = np.array(
            [0.43593284 - 0.02945156j, 0.40812291 + 0.80158023j], requires_grad=False
        )

        tape = qp.tape.QuantumScript(
            [qp.StatePrep(random_state, 0), G(theta, 0)], [qp.expval(qp.PauliZ(0))]
        )

        tape.trainable_params = {1}

        method = dev.compute_derivatives
        calculated_val = method(tape)

        tol, _ = get_tolerance_and_stepsize(dev)

        # compare to finite differences
        tapes, fn = qp.gradients.param_shift(tape)
        numeric_val = fn(qp.execute(tapes, dev, None))
        assert np.allclose(calculated_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    def test_Rot_gradient(self, theta, dev):
        """Tests that the device gradient of an arbitrary Euler-angle-parameterized gate is
        correct."""
        params = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qp.tape.QuantumTape() as tape:
            qp.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            qp.Rot(*params, wires=[0])
            qp.expval(qp.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        method = dev.compute_derivatives
        calculated_val = method(tape)

        tol, _ = get_tolerance_and_stepsize(dev)

        # compare to finite differences
        tapes, fn = qp.gradients.param_shift(tape)
        numeric_val = fn(qp.execute(tapes, dev, None))
        assert np.allclose(calculated_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
    @pytest.mark.parametrize("par", [-np.pi / 7, np.pi / 5, 2 * np.pi / 3])
    def test_phaseshift_gradient(self, n_qubits, par, tol):
        """Test that the gradient of the phaseshift gate matches the exact analytic formula."""
        par = np.array(par)
        dev = qp.device(device_name, wires=n_qubits)
        init_state = np.zeros(2**n_qubits)
        init_state[-2::] = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], requires_grad=False)

        with qp.tape.QuantumTape() as tape:
            qp.StatePrep(init_state, wires=range(n_qubits))
            qp.ctrl(qp.PhaseShift(par, wires=n_qubits - 1), range(0, n_qubits - 1))
            qp.expval(qp.PauliY(n_qubits - 1))

        tape.trainable_params = {1}

        exact = np.cos(par)
        method = dev.compute_derivatives
        grad_A = method(tape)

        # different methods must agree
        assert np.allclose(grad_A, exact, atol=tol, rtol=0)

    @pytest.mark.parametrize("par", [1, -2, 1.623, -0.051, 0])  # integers, floats, zero
    def test_ry_gradient(self, par, tol, dev):
        """Test that the gradient of the RY gate matches the exact analytic formula."""
        with qp.tape.QuantumTape() as tape:
            qp.RY(par, wires=[0])
            qp.expval(qp.PauliX(0))

        tape.trainable_params = {0}

        # gradients
        exact = np.cos(par)
        method = dev.compute_derivatives
        grad_A = method(tape)

        # different methods must agree
        assert np.allclose(grad_A, exact, atol=tol, rtol=0)

    def test_rx_gradient(self, tol, dev):
        """Test that the gradient of the RX gate matches the known formula."""
        a = 0.7418

        with qp.tape.QuantumTape() as tape:
            qp.RX(a, wires=0)
            qp.expval(qp.PauliZ(0))

        # circuit jacobians
        method = dev.compute_derivatives
        dev_jacobian = method(tape)
        expected_jacobian = -np.sin(a)
        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    def test_multiple_rx_gradient_pauliz(self, tol, dev):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result."""
        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        with qp.tape.QuantumTape() as tape:
            qp.RX(params[0], wires=0)
            qp.RX(params[1], wires=1)
            qp.RX(params[2], wires=2)

            for idx in range(3):
                qp.expval(qp.PauliZ(idx))

        # circuit jacobians
        method = dev.compute_derivatives
        dev_jacobian = method(tape)
        expected_jacobian = -np.diag(np.sin(params))
        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    def test_multiple_rx_gradient_hermitian(self, tol, dev):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result
        with Hermitian observable
        """
        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        with qp.tape.QuantumTape() as tape:
            qp.RX(params[0], wires=0)
            qp.RX(params[1], wires=1)
            qp.RX(params[2], wires=2)

            for idx in range(3):
                qp.expval(qp.Hermitian([[1, 0], [0, -1]], wires=[idx]))

        tape.trainable_params = {0, 1, 2}
        # circuit jacobians
        method = dev.compute_derivatives
        dev_jacobian = method(tape)
        expected_jacobian = -np.diag(np.sin(params))

        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    qubit_ops = [getattr(qp, name) for name in qp.ops._qubit__ops__]
    ops = {qp.RX, qp.RY, qp.RZ, qp.PhaseShift, qp.CRX, qp.CRY, qp.CRZ, qp.Rot}

    def test_multiple_rx_gradient_expval_hermitian(self, tol, dev):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result
        with Hermitian observable
        """
        params = np.array([np.pi / 3, np.pi / 4, np.pi / 5])

        with qp.tape.QuantumTape() as tape:
            qp.RX(params[0], wires=0)
            qp.RX(params[1], wires=1)
            qp.RX(params[2], wires=2)

            qp.expval(
                qp.Hermitian(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], wires=[0, 2]
                )
            )

        tape.trainable_params = {0, 1, 2}
        method = dev.compute_derivatives
        dev_jacobian = method(tape)
        expected_jacobian = np.array(
            [-np.sin(params[0]) * np.cos(params[2]), 0, -np.cos(params[0]) * np.sin(params[2])]
        )

        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    qubit_ops = [getattr(qp, name) for name in qp.ops._qubit__ops__]
    ops = {qp.RX, qp.RY, qp.RZ, qp.PhaseShift, qp.CRX, qp.CRY, qp.CRZ, qp.Rot}

    def test_multiple_rx_gradient_expval_hamiltonian(self, tol, dev):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result
        with Hermitian observable
        """
        params = np.array([np.pi / 3, np.pi / 4, np.pi / 5])

        ham = qp.Hamiltonian(
            [1.0, 0.3, 0.3, 0.4],
            [
                qp.PauliX(0) @ qp.PauliX(1),
                qp.PauliZ(0),
                qp.PauliZ(1),
                qp.Hermitian(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], wires=[0, 2]
                ),
            ],
        )

        with qp.tape.QuantumTape() as tape:
            qp.RX(params[0], wires=0)
            qp.RX(params[1], wires=1)
            qp.RX(params[2], wires=2)

            qp.expval(ham)

        tape.trainable_params = {0, 1, 2}
        method = dev.compute_derivatives
        dev_jacobian = method(tape)
        expected_jacobian = (
            0.3 * np.array([-np.sin(params[0]), 0, 0])
            + 0.3 * np.array([0, -np.sin(params[1]), 0])
            + 0.4
            * np.array(
                [-np.sin(params[0]) * np.cos(params[2]), 0, -np.cos(params[0]) * np.sin(params[2])]
            )
        )
        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    qubit_ops = [getattr(qp, name) for name in qp.ops._qubit__ops__]
    ops = {qp.RX, qp.RY, qp.RZ, qp.PhaseShift, qp.CRX, qp.CRY, qp.CRZ, qp.Rot}

    @pytest.mark.parametrize("obs", [qp.PauliX, qp.PauliY])
    @pytest.mark.parametrize(
        "op",
        [
            qp.RX(0.4, wires=0),
            qp.RY(0.6, wires=0),
            qp.RZ(0.8, wires=0),
            qp.CRX(1.0, wires=[0, 1]),
            qp.CRY(2.0, wires=[0, 1]),
            qp.CRZ(3.0, wires=[0, 1]),
            qp.Rot(0.2, -0.1, 0.2, wires=0),
        ],
    )
    def test_gradients_pauliz(self, op, obs, dev):
        """Tests that the gradients of circuits match between the finite difference and device
        methods."""
        # op.num_wires and op.num_params must be initialized a priori
        with qp.tape.QuantumTape() as tape:
            qp.Hadamard(wires=0)
            qp.RX(0.543, wires=0)
            qp.CNOT(wires=[0, 1])

            op

            qp.Rot(1.3, -2.3, 0.5, wires=[0])
            qp.RZ(-0.5, wires=0)
            qp.adjoint(qp.RY(0.5, wires=1), lazy=False)
            qp.CNOT(wires=[0, 1])

            qp.expval(obs(wires=0))
            qp.expval(qp.PauliZ(wires=1))

        tape.trainable_params = set(range(1, 1 + op.num_params))

        tol, _ = get_tolerance_and_stepsize(dev)

        grad_F = (lambda t, fn: fn(qp.execute(t, dev, None)))(*qp.gradients.param_shift(tape))
        method = dev.compute_derivatives
        grad_D = method(tape)

        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "op",
        [
            qp.RX(0.4, wires=0),
            qp.RY(0.6, wires=0),
            qp.RZ(0.8, wires=0),
            qp.CRX(1.0, wires=[0, 1]),
            qp.CRY(2.0, wires=[0, 1]),
            qp.CRZ(3.0, wires=[0, 1]),
            qp.Rot(0.2, -0.1, 0.2, wires=0),
        ],
    )
    def test_gradients_hermitian(self, op, dev):
        """Tests that the gradients of circuits match between the finite difference and device
        methods."""
        # op.num_wires and op.num_params must be initialized a priori
        with qp.tape.QuantumTape() as tape:
            qp.Hadamard(wires=0)
            qp.RX(0.543, wires=0)
            qp.CNOT(wires=[0, 1])

            op.queue()

            qp.Rot(1.3, -2.3, 0.5, wires=[0])
            qp.RZ(-0.5, wires=0)
            qp.adjoint(qp.RY(0.5, wires=1), lazy=False)
            qp.CNOT(wires=[0, 1])

            qp.expval(
                qp.Hermitian(
                    [[0, 0, 1, 1], [0, 1, 2, 1], [1, 2, 1, 0], [1, 1, 0, 0]], wires=[0, 1]
                )
            )

        tape.trainable_params = set(range(1, 1 + op.num_params))

        tol, _ = get_tolerance_and_stepsize(dev)

        grad_F = (lambda t, fn: fn(qp.execute(t, dev, None)))(*qp.gradients.param_shift(tape))
        method = dev.compute_derivatives
        grad_D = method(tape)

        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    def test_gradient_gate_with_multiple_parameters_pauliz(self, dev):
        """Tests that gates with multiple free parameters yield correct gradients."""
        x, y, z = [0.5, 0.3, -0.7]

        tape = qp.tape.QuantumScript(
            [qp.RX(0.4, wires=[0]), qp.Rot(x, y, z, wires=[0]), qp.RY(-0.2, wires=[0])],
            [qp.expval(qp.PauliZ(0))],
        )

        tape.trainable_params = {1, 2, 3}

        tol, _ = get_tolerance_and_stepsize(dev)

        method = dev.compute_derivatives
        grad_D = method(tape)
        tapes, fn = qp.gradients.param_shift(tape)
        grad_F = fn(qp.execute(tapes, dev, None))

        # gradient has the correct shape and every element is nonzero
        assert len(grad_D) == 3
        assert all(isinstance(v, np.ndarray) for v in grad_D)
        assert np.count_nonzero(grad_D) == 3
        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    def test_gradient_gate_with_multiple_parameters_hermitian(self, dev):
        """Tests that gates with multiple free parameters yield correct gradients."""
        x, y, z = [0.5, 0.3, -0.7]

        tape = qp.tape.QuantumScript(
            [qp.RX(0.4, wires=[0]), qp.Rot(x, y, z, wires=[0]), qp.RY(-0.2, wires=[0])],
            [qp.expval(qp.Hermitian([[0, 1], [1, 1]], wires=0))],
        )

        tape.trainable_params = {1, 2, 3}

        tol, _ = get_tolerance_and_stepsize(dev)

        method = dev.compute_derivatives
        grad_D = method(tape)
        tapes, fn = qp.gradients.param_shift(tape)
        grad_F = fn(qp.execute(tapes, dev, None))

        # gradient has the correct shape and every element is nonzero
        assert len(grad_D) == 3
        assert all(isinstance(v, np.ndarray) for v in grad_D)
        assert np.count_nonzero(grad_D) == 3
        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    def test_gradient_gate_with_multiple_parameters_hamiltonian(self, dev):
        """Tests that gates with multiple free parameters yield correct gradients."""
        x, y, z = [0.5, 0.3, -0.7]

        ham = qp.Hamiltonian(
            [1.0, 0.3, 0.3], [qp.PauliX(0) @ qp.PauliX(1), qp.PauliZ(0), qp.PauliZ(1)]
        )

        tape = qp.tape.QuantumScript(
            [qp.RX(0.4, wires=[0]), qp.Rot(x, y, z, wires=[0]), qp.RY(-0.2, wires=[0])],
            [qp.expval(ham)],
        )

        tape.trainable_params = {1, 2, 3}

        tol, _ = get_tolerance_and_stepsize(dev)

        method = dev.compute_derivatives
        grad_D = method(tape)
        tapes, fn = qp.gradients.param_shift(tape)
        grad_F = fn(qp.execute(tapes, dev, None))

        # gradient has the correct shape and every element is nonzero
        assert len(grad_D) == 3
        assert all(isinstance(v, np.ndarray) for v in grad_D)
        assert np.count_nonzero(grad_D) == 3
        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    @pytest.mark.skipif(
        device_name == "lightning.gpu",
        reason="Adjoint differentiation does not support State measurements.",
    )
    def test_state_return_type(self, dev):
        """Tests raise an exception when the return type is State"""
        with qp.tape.QuantumTape() as tape:
            qp.RX(0.4, wires=[0])
            qp.state()

        tape.trainable_params = {0}
        method = dev.compute_derivatives

        with pytest.raises(
            QuantumFunctionError,
            match="Adjoint differentiation method does not support measurement StateMP.",
        ):
            method(tape)


class TestAdjointJacobianQNode:
    """Test QNode integration with the adjoint_jacobian method"""

    @pytest.fixture(params=fixture_params)
    def dev(self, request):
        params = request.param
        if device_name == "lightning.kokkos":
            return qp.device(
                device_name, wires=params[2], c_dtype=params[0], kokkos_args=params[1]
            )
        return qp.device(device_name, wires=params[2], c_dtype=params[0])

    def test_qnode(self, mocker, dev):
        """Test that specifying diff_method allows the adjoint method to be selected"""
        args = np.array([0.54, 0.1, 0.5], requires_grad=True)

        def circuit(x, y, z):
            qp.Hadamard(wires=0)
            qp.RX(0.543, wires=0)
            qp.CNOT(wires=[0, 1])

            qp.Rot(x, y, z, wires=0)

            qp.Rot(1.3, -2.3, 0.5, wires=[0])
            qp.RZ(-0.5, wires=0)
            qp.RY(0.5, wires=1)
            qp.CNOT(wires=[0, 1])

            return qp.expval(qp.PauliX(0) @ qp.PauliZ(1))

        qnode1 = QNode(circuit, dev, diff_method="adjoint")
        spy = mocker.spy(dev, "execute_and_compute_derivatives")
        tol, h = get_tolerance_and_stepsize(dev, step_size=True)

        grad_fn = qp.grad(qnode1)
        grad_A = grad_fn(*args)

        spy.assert_called()

        qnode2 = QNode(circuit, dev, diff_method="finite-diff", gradient_kwargs={"h": h})
        grad_fn = qp.grad(qnode2)
        grad_F = grad_fn(*args)

        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "operation",
        [
            qp.PhaseShift,
            qp.RX,
            qp.RY,
            qp.RZ,
            qp.IsingXX,
            qp.IsingXY,
            qp.IsingYY,
            qp.IsingZZ,
            qp.CRX,
            qp.CRY,
            qp.CRZ,
            qp.ControlledPhaseShift,
            qp.PSWAP,
            qp.SingleExcitation,
            qp.SingleExcitationMinus,
            qp.SingleExcitationPlus,
            qp.DoubleExcitation,
            qp.DoubleExcitationMinus,
            qp.DoubleExcitationPlus,
            qp.MultiRZ,
            qp.GlobalPhase,
        ],
    )
    @pytest.mark.parametrize("n_qubits", range(2, 6))
    @pytest.mark.parametrize("par", [-np.pi / 7, np.pi / 5, 2 * np.pi / 3])
    def test_gate_jacobian(self, par, n_qubits, operation, tol):
        """Test that the jacobian of the controlled gate matches backprop."""
        par = np.array([0.1234, par, 0.5678])
        dev = qp.device(device_name, wires=n_qubits)
        dqu = qp.device("default.qubit", wires=n_qubits)
        init_state = get_random_normalized_state(2**n_qubits)
        init_state = np.array(init_state, requires_grad=False)

        num_wires = max(operation.num_wires, 1) if operation.num_wires else 1
        if num_wires > n_qubits:
            return

        for w in range(0, n_qubits - num_wires):

            def circuit(p):
                qp.StatePrep(init_state, wires=range(n_qubits))
                qp.RX(p[0], 0)
                if operation is qp.GlobalPhase:
                    operation(p[1], wires=range(n_qubits))
                else:
                    operation(p[1], wires=range(w, w + num_wires))
                qp.RY(p[2], 0)
                return np.array([qp.expval(qp.PauliY(i)) for i in range(n_qubits)])

            circ_ad = qp.QNode(circuit, dev, diff_method="adjoint")
            circ_bp = qp.QNode(circuit, dqu, diff_method="backprop")
            jac_ad = np.array(qp.jacobian(circ_ad)(par))
            jac_bp = np.array(qp.jacobian(circ_bp)(par))

            # different methods must agree
            assert jac_ad.size == n_qubits * 3
            assert np.allclose(jac_ad.shape, [n_qubits, 3])
            assert np.allclose(jac_ad.shape, jac_bp.shape)
            assert np.allclose(jac_ad, jac_bp, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "operation",
        [
            qp.PhaseShift,
            qp.RX,
            qp.RY,
            qp.RZ,
            qp.IsingXX,
            qp.IsingXY,
            qp.IsingYY,
            qp.IsingZZ,
            qp.CRX,
            qp.CRY,
            qp.CRZ,
            qp.ControlledPhaseShift,
            qp.PSWAP,
            qp.SingleExcitation,
            qp.SingleExcitationMinus,
            qp.SingleExcitationPlus,
            qp.DoubleExcitation,
            qp.DoubleExcitationMinus,
            qp.DoubleExcitationPlus,
            qp.MultiRZ,
            qp.GlobalPhase,
        ],
    )
    @pytest.mark.parametrize("n_qubits", range(2, 6))
    @pytest.mark.parametrize("par", [-np.pi / 7, np.pi / 5, 2 * np.pi / 3])
    def test_inverse_jacobian(self, par, n_qubits, operation, tol):
        """Test that the jacobian of the controlled gate matches backprop."""
        par = np.array([0.1234, par, 0.5678])
        dev = qp.device(device_name, wires=n_qubits)
        dqu = qp.device("default.qubit", wires=n_qubits)
        init_state = get_random_normalized_state(2**n_qubits)
        init_state = np.array(init_state, requires_grad=False)

        num_wires = max(operation.num_wires, 1) if operation.num_wires else 1
        if num_wires > n_qubits:
            return

        for w in range(0, n_qubits - num_wires):

            def circuit(p):
                qp.StatePrep(init_state, wires=range(n_qubits))
                qp.RX(p[0], 0)
                if operation is qp.GlobalPhase:
                    qp.adjoint(operation(p[1], wires=range(n_qubits)))
                else:
                    qp.adjoint(operation(p[1], wires=range(w, w + num_wires)))
                qp.RY(p[2], 0)
                return np.array([qp.expval(qp.PauliY(i)) for i in range(n_qubits)])

            circ_ad = qp.QNode(circuit, dev, diff_method="adjoint")
            circ_bp = qp.QNode(circuit, dqu, diff_method="backprop")
            jac_ad = np.array(qp.jacobian(circ_ad)(par))
            jac_bp = np.array(qp.jacobian(circ_bp)(par))

            # different methods must agree
            assert jac_ad.size == n_qubits * 3
            assert np.allclose(jac_ad.shape, [n_qubits, 3])
            assert np.allclose(jac_ad.shape, jac_bp.shape)
            assert np.allclose(jac_ad, jac_bp, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "operation",
        [
            qp.PhaseShift,
            qp.RX,
            qp.RY,
            qp.RZ,
            qp.Rot,
            qp.IsingXX,
            qp.IsingXY,
            qp.IsingYY,
            qp.IsingZZ,
            qp.PSWAP,
            qp.SingleExcitation,
            qp.SingleExcitationMinus,
            qp.SingleExcitationPlus,
            qp.DoubleExcitation,
            qp.DoubleExcitationMinus,
            qp.DoubleExcitationPlus,
            qp.MultiRZ,
            qp.GlobalPhase,
        ],
    )
    @pytest.mark.parametrize("control_value", [False, True])
    @pytest.mark.parametrize("n_qubits", range(2, 6))
    @pytest.mark.parametrize("par", [-np.pi / 7, np.pi / 5, 2 * np.pi / 3])
    def test_controlled_jacobian(self, par, n_qubits, control_value, operation, tol):
        """Test that the jacobian of the controlled gate matches the parameter-shift formula."""
        par = np.array([0.1234, par, 0.5678])
        dev = qp.device(device_name, wires=n_qubits)
        dqu = qp.device("default.qubit", wires=n_qubits)
        init_state = get_random_normalized_state(2**n_qubits)
        init_state = np.array(init_state, requires_grad=False)
        num_wires = max(operation.num_wires, 1) if operation.num_wires else 1
        if num_wires > n_qubits:
            return

        for n_controls in range(0, n_qubits - num_wires):
            control_wires = range(n_controls, n_qubits - num_wires)

            def circuit(p):
                qp.StatePrep(init_state, wires=range(n_qubits))
                if operation.num_params == 3:
                    qp.ctrl(
                        operation(*p, wires=range(n_qubits - num_wires, n_qubits)),
                        control_wires,
                        control_values=[
                            control_value or bool(i % 2) for i, _ in enumerate(control_wires)
                        ],
                    )
                else:
                    qp.RX(p[0], 0)
                    qp.ctrl(
                        operation(p[1], wires=range(n_qubits - num_wires, n_qubits)),
                        control_wires,
                        control_values=[
                            control_value or bool(i % 2) for i, _ in enumerate(control_wires)
                        ],
                    )
                    qp.RY(p[2], 0)
                return np.array([qp.expval(qp.PauliY(i)) for i in range(n_qubits)])

            circ_ad = qp.QNode(circuit, dev, diff_method="adjoint")
            circ_bp = qp.QNode(circuit, dqu, diff_method="backprop")
            jac_ad = np.array(qp.jacobian(circ_ad)(par))
            jac_bp = np.array(qp.jacobian(circ_bp)(par))

            # different methods must agree
            assert jac_ad.size == n_qubits * 3
            assert np.allclose(jac_ad.shape, [n_qubits, 3])
            assert np.allclose(jac_ad.shape, jac_bp.shape)
            assert np.allclose(jac_ad, jac_bp, atol=tol, rtol=0)

    thetas = np.linspace(-2 * np.pi, 2 * np.pi, 8)

    @pytest.mark.parametrize("reused_p", thetas**3 / 19)
    @pytest.mark.parametrize("other_p", thetas**2 / 1)
    def test_fanout_multiple_params(self, reused_p, other_p, tol, mocker, dev):
        """Tests that the correct gradient is computed for qnodes which
        use the same parameter in multiple gates."""

        def expZ(state):
            return np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2

        extra_param = np.array(0.31, requires_grad=False)

        @qnode(dev, diff_method="adjoint")
        def cost(p1, p2):
            qp.RX(extra_param, wires=[0])
            qp.RY(p1, wires=[0])
            qp.RZ(p2, wires=[0])
            qp.RX(p1, wires=[0])
            return qp.expval(qp.PauliZ(0))

        zero_state = np.array([1.0, 0.0])
        cost(reused_p, other_p)

        spy = mocker.spy(dev, "execute_and_compute_derivatives")

        # analytic gradient
        grad_fn = qp.grad(cost)
        grad_D = grad_fn(reused_p, other_p)

        spy.assert_called_once()

        # manual gradient
        grad_true0 = (
            expZ(
                Rx(reused_p) @ Rz(other_p) @ Ry(reused_p + np.pi / 2) @ Rx(extra_param) @ zero_state
            )
            - expZ(
                Rx(reused_p) @ Rz(other_p) @ Ry(reused_p - np.pi / 2) @ Rx(extra_param) @ zero_state
            )
        ) / 2
        grad_true1 = (
            expZ(
                Rx(reused_p + np.pi / 2) @ Rz(other_p) @ Ry(reused_p) @ Rx(extra_param) @ zero_state
            )
            - expZ(
                Rx(reused_p - np.pi / 2) @ Rz(other_p) @ Ry(reused_p) @ Rx(extra_param) @ zero_state
            )
        ) / 2
        expected = grad_true0 + grad_true1  # product rule

        assert np.allclose(grad_D[0], expected, atol=tol, rtol=0)

    def test_gradient_repeated_gate_parameters(self, mocker, dev):
        """Tests that repeated use of a free parameter in a multi-parameter gate yields correct
        gradients."""
        params = np.array([0.8, 1.3], requires_grad=True)

        def circuit(params):
            qp.RX(np.array(np.pi / 4, requires_grad=False), wires=[0])
            qp.Rot(params[1], params[0], 2 * params[0], wires=[0])
            return qp.expval(qp.PauliX(0))

        spy_analytic = mocker.spy(dev, "execute_and_compute_derivatives")

        tol, h = get_tolerance_and_stepsize(dev, step_size=True)

        cost = QNode(circuit, dev, diff_method="finite-diff", gradient_kwargs={"h": h})

        grad_fn = qp.grad(cost)
        grad_F = grad_fn(params)

        spy_analytic.assert_not_called()

        cost = QNode(circuit, dev, diff_method="adjoint")
        grad_fn = qp.grad(cost)
        grad_D = grad_fn(params)

        spy_analytic.assert_called_once()

        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    def test_interface_torch(self, dev):
        """Test if gradients agree between the adjoint and finite-diff methods when using the
        Torch interface"""

        torch = pytest.importorskip("torch")

        def f(params1, params2):
            qp.RX(0.4, wires=[0])
            qp.RZ(params1 * torch.sqrt(params2), wires=[0])
            qp.RY(torch.cos(params2), wires=[0])
            return qp.expval(qp.PauliZ(0))

        params1 = torch.tensor(0.3, requires_grad=True)
        params2 = torch.tensor(0.4, requires_grad=True)

        tol, h = get_tolerance_and_stepsize(dev, step_size=True)

        qnode1 = QNode(f, dev, interface="torch", diff_method="adjoint")
        qnode2 = QNode(
            f, dev, interface="torch", diff_method="finite-diff", gradient_kwargs={"h": h}
        )

        res1 = qnode1(params1, params2)
        res1.backward()

        grad_adjoint = params1.grad, params2.grad

        res2 = qnode2(params1, params2)
        res2.backward()

        grad_fd = params1.grad, params2.grad

        assert np.allclose(grad_adjoint, grad_fd)

    def test_interface_jax(self, dev):
        """Test if the gradients agree between adjoint and finite-difference methods in the
        jax interface"""

        jax = pytest.importorskip("jax")

        # Determine the correct dtype for JAX arrays based on device configuration
        if dev.c_dtype == np.complex64:
            r_dtype = np.float32
            x64_enabled = False
        else:
            r_dtype = np.float64
            x64_enabled = True

        # Save original JAX configuration and set it appropriately
        original_x64 = jax.config.jax_enable_x64
        try:
            jax.config.update("jax_enable_x64", x64_enabled)

            def circuit(params1, params2):
                qp.RX(0.4, wires=[0])
                qp.RZ(params1 * jax.numpy.sqrt(params2), wires=[0])
                qp.RY(jax.numpy.cos(params2), wires=[0])
                return qp.expval(qp.PauliZ(0))

            params1 = jax.numpy.array(0.3, r_dtype)
            params2 = jax.numpy.array(0.4, r_dtype)
            tol, h = get_tolerance_and_stepsize(dev, step_size=True)

            qnode_adjoint = QNode(circuit, dev, interface="jax", diff_method="adjoint")
            qnode_fd = QNode(
                circuit, dev, interface="jax", diff_method="finite-diff", gradient_kwargs={"h": h}
            )

            grad_adjoint = jax.grad(qnode_adjoint)(params1, params2)
            grad_fd = jax.grad(qnode_fd)(params1, params2)

            assert np.allclose(grad_adjoint, grad_fd, atol=tol)
        finally:
            jax.config.update("jax_enable_x64", original_x64)

    def test_torch_amplitude_embedding(self, dev):
        """Test that the adjoint Jacobian works with AmplitudeEmbedding in a TorchLayer"""

        torch = pytest.importorskip("torch")

        # Define a simple circuit with AmplitudeEmbedding and BasicEntanglerLayers
        n_qubits = 3
        qp.AmplitudeEmbedding.ndim_params = (1,)

        @qp.qnode(dev)
        def qnode(inputs, weights):
            qp.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
            qp.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qp.expval(qp.PauliZ(wires=i)) for i in range(n_qubits)]

        n_layers = 1
        weight_shapes = {"weights": (n_layers, n_qubits)}

        qlayer = qp.qnn.TorchLayer(qnode, weight_shapes)
        clayer_1 = torch.nn.Linear(2, 8)
        xs_test = torch.tensor([[-0.7380, 0.3596], [-0.1870, 0.9423]])

        t = clayer_1(xs_test)
        qlayer(t)


def circuit_ansatz(params, wires):
    """Circuit ansatz containing all the parametrized gates"""
    qp.StatePrep(unitary_group.rvs(2**4, random_state=0)[0], wires=wires)
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
    qp.PauliRot(params[12], "XXYZ", wires=[wires[0], wires[1], wires[2], wires[3]])
    qp.CPhase(params[12], wires=[wires[3], wires[2]])
    qp.IsingXX(params[13], wires=[wires[1], wires[0]])
    qp.IsingXY(params[14], wires=[wires[3], wires[2]])
    qp.IsingYY(params[14], wires=[wires[3], wires[2]])
    qp.IsingZZ(params[14], wires=[wires[2], wires[1]])
    qp.U1(params[15], wires=wires[0])
    qp.U2(params[16], params[17], wires=wires[0])
    qp.U3(params[18], params[19], params[20], wires=wires[1])
    qp.adjoint(qp.CRot(params[21], params[22], params[23], wires=[wires[1], wires[2]]))
    qp.SingleExcitation(params[24], wires=[wires[2], wires[0]])
    qp.DoubleExcitation(params[25], wires=[wires[2], wires[0], wires[1], wires[3]])
    qp.SingleExcitationPlus(params[26], wires=[wires[0], wires[2]])
    qp.SingleExcitationMinus(params[27], wires=[wires[0], wires[2]])
    qp.DoubleExcitationPlus(params[27], wires=[wires[2], wires[0], wires[1], wires[3]])
    qp.DoubleExcitationMinus(params[27], wires=[wires[2], wires[0], wires[1], wires[3]])
    qp.RX(params[28], wires=wires[0])
    qp.RX(params[29], wires=wires[1])


def test_tape_qchem(tol):
    """Tests the circuit Ansatz with a QChem Hamiltonian produces correct results"""

    H, qubits = qp.qchem.molecular_hamiltonian(
        ["H", "H"], np.array([0.0, 0.1, 0.0, 0.0, -0.1, 0.0])
    )

    def circuit(params):
        circuit_ansatz(params, wires=range(4))
        return qp.expval(H)

    params = np.linspace(0, 29, 30) * 0.111

    dev_ld = qp.device(device_name, wires=qubits)
    dev_dq = qp.device("default.qubit", wires=qubits)

    circuit_ld = qp.QNode(circuit, dev_ld, diff_method="adjoint")
    circuit_dq = qp.QNode(circuit, dev_dq, diff_method="parameter-shift")
    res = qp.grad(circuit_ld)(params)
    ref = qp.grad(circuit_dq)(params)
    assert np.allclose(res, ref, tol)


def test_tape_qchem_sparse(tol):
    """Tests the circuit Ansatz with a QChem Hamiltonian produces correct results"""

    H, qubits = qp.qchem.molecular_hamiltonian(
        ["H", "H"], np.array([0.0, 0.1, 0.0, 0.0, -0.1, 0.0])
    )

    H_sparse = H.sparse_matrix(range(4))

    def circuit_sparse(params):
        circuit_ansatz(params, wires=range(4))
        return qp.expval(qp.SparseHamiltonian(H_sparse, wires=range(4)))

    def circuit(params):
        circuit_ansatz(params, wires=range(4))
        return qp.expval(H)

    params = np.linspace(0, 29, 30) * 0.111

    dev_ld = qp.device(device_name, wires=qubits)
    dev_dq = qp.device("default.qubit", wires=qubits)

    circuit_ld = qp.QNode(circuit_sparse, dev_ld, diff_method="adjoint")
    circuit_dq = qp.QNode(circuit, dev_dq, diff_method="parameter-shift")

    assert np.allclose(qp.grad(circuit_ld)(params), qp.grad(circuit_dq)(params), tol)


custom_wires = ["alice", 3.14, -1, 0]


@pytest.mark.local_salt(42)
@pytest.mark.parametrize(
    "returns",
    [
        qp.SparseHamiltonian(
            qp.Hamiltonian(
                [0.1],
                [qp.PauliX(wires=custom_wires[0]) @ qp.PauliZ(wires=custom_wires[1])],
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
        qp.SparseHamiltonian(
            qp.Hamiltonian(
                [2.0],
                [qp.PauliX(wires=custom_wires[2]) @ qp.PauliZ(wires=custom_wires[0])],
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
        qp.SparseHamiltonian(
            qp.Hamiltonian(
                [1.1],
                [qp.PauliX(wires=custom_wires[0]) @ qp.PauliZ(wires=custom_wires[2])],
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
    ],
)
def test_adjoint_SparseHamiltonian(returns, seed):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    dev = qp.device(device_name, wires=custom_wires)
    dev_default = qp.device("default.qubit", wires=custom_wires)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qp.expval(returns)

    n_params = 30
    rng = np.random.default_rng(seed)
    params = rng.random(n_params)

    qnode = qp.QNode(circuit, dev, diff_method="adjoint")
    qnode_default = qp.QNode(circuit, dev_default, diff_method="parameter-shift")

    j_device = qp.jacobian(qnode)(params)
    j_default = qp.jacobian(qnode_default)(params)

    assert np.allclose(j_device, j_default)


@pytest.mark.parametrize(
    "returns",
    [
        qp.PauliZ(0),
        qp.PauliX(2),
        qp.PauliZ(0) @ qp.PauliY(3),
        qp.Hadamard(2),
        qp.Hadamard(3) @ qp.PauliZ(2),
        qp.PauliX(0) @ qp.PauliY(3),
        qp.PauliY(0) @ qp.PauliY(2) @ qp.PauliY(3),
        qp.Hermitian(
            np.kron(qp.PauliY.compute_matrix(), qp.PauliZ.compute_matrix()), wires=[3, 2]
        ),
        qp.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=0),
        qp.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=0) @ qp.PauliZ(2),
    ],
)
def test_integration(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations"""
    dev_def = qp.device("default.qubit", wires=range(4))
    dev_lightning = qp.device(device_name, wires=range(4))

    def circuit(params):
        circuit_ansatz(params, wires=range(4))
        return np.array([qp.expval(returns), qp.expval(qp.PauliY(1))])

    n_params = 30
    params = np.linspace(0, 10, n_params)

    qnode_def = qp.QNode(circuit, dev_def)
    qnode_lightning = qp.QNode(circuit, dev_lightning, diff_method="adjoint")

    j_def = qp.jacobian(qnode_def)(params)
    j_lightning = qp.jacobian(qnode_lightning)(params)

    assert np.allclose(j_def, j_lightning)


def test_integration_chunk_observables():
    """Integration tests that compare to default.qubit for a large circuit with multiple expectation values. Expvals are generated in parallelized chunks."""
    num_qubits = 4

    dev_def = qp.device("default.qubit", wires=range(num_qubits))
    dev_lightning = qp.device(device_name, wires=range(num_qubits))
    dev_lightning_batched = qp.device(device_name, wires=range(num_qubits), batch_obs=True)

    def circuit(params):
        circuit_ansatz(params, wires=range(num_qubits))
        return np.array(
            [qp.expval(qp.PauliZ(i)) for i in range(num_qubits)]
            + [
                qp.expval(
                    qp.Hamiltonian(
                        np.arange(1, num_qubits + 1),
                        [
                            qp.PauliZ(i % num_qubits) @ qp.PauliY((i + 1) % num_qubits)
                            for i in range(num_qubits)
                        ],
                    )
                )
            ]
            + [qp.expval(qp.PauliY(i)) for i in range(num_qubits)]
        )

    n_params = 30
    params = np.linspace(0, 10, n_params)

    qnode_def = qp.QNode(circuit, dev_def)
    qnode_lightning = qp.QNode(circuit, dev_lightning, diff_method="adjoint")
    qnode_lightning_batched = qp.QNode(circuit, dev_lightning_batched, diff_method="adjoint")

    j_def = qp.jacobian(qnode_def)(params)
    j_lightning = qp.jacobian(qnode_lightning)(params)
    j_lightning_batched = qp.jacobian(qnode_lightning_batched)(params)

    assert np.allclose(j_def, j_lightning)
    assert np.allclose(j_def, j_lightning_batched)


@pytest.mark.parametrize(
    "returns",
    [
        qp.PauliZ(custom_wires[0]),
        qp.PauliX(custom_wires[2]),
        qp.PauliZ(custom_wires[0]) @ qp.PauliY(custom_wires[3]),
        qp.Hadamard(custom_wires[2]),
        qp.Hadamard(custom_wires[3]) @ qp.PauliZ(custom_wires[2]),
        # qp.Projector([0, 1], wires=[custom_wires[0], custom_wires[2]]) @ qp.Hadamard(custom_wires[3])
        # qp.Projector([0, 0], wires=[custom_wires[2], custom_wires[0]])
        qp.PauliX(custom_wires[0]) @ qp.PauliY(custom_wires[3]),
        qp.PauliY(custom_wires[0]) @ qp.PauliY(custom_wires[2]) @ qp.PauliY(custom_wires[3]),
        qp.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=custom_wires[0]),
        qp.Hermitian(
            np.kron(qp.PauliY.compute_matrix(), qp.PauliZ.compute_matrix()),
            wires=[custom_wires[3], custom_wires[2]],
        ),
        qp.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=custom_wires[0])
        @ qp.PauliZ(custom_wires[2]),
    ],
)
def test_integration_custom_wires(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""
    dev_def = qp.device("default.qubit", wires=custom_wires)
    dev_lightning = qp.device(device_name, wires=custom_wires)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return np.array([qp.expval(returns), qp.expval(qp.PauliY(custom_wires[1]))])

    n_params = 30
    params = np.linspace(0, 10, n_params)

    qnode_def = qp.QNode(circuit, dev_def)
    qnode_lightning = qp.QNode(circuit, dev_lightning, diff_method="adjoint")

    j_def = qp.jacobian(qnode_def)(params)
    j_lightning = qp.jacobian(qnode_lightning)(params)

    assert np.allclose(j_def, j_lightning)


@pytest.mark.local_salt(42)
@pytest.mark.skipif(
    device_name not in ("lightning.qubit", "lightning.gpu"),
    reason="Tests only for lightning.qubit and lightning.gpu",
)
@pytest.mark.parametrize(
    "returns",
    [
        (qp.PauliZ(custom_wires[0]),),
        (qp.PauliZ(custom_wires[0]), qp.PauliZ(custom_wires[1])),
        (qp.PauliZ(custom_wires[0]), qp.PauliZ(custom_wires[1]), qp.PauliZ(custom_wires[3])),
        (
            qp.PauliZ(custom_wires[0]),
            qp.PauliZ(custom_wires[1]),
            qp.PauliZ(custom_wires[3]),
            qp.PauliZ(custom_wires[2]),
        ),
        (
            qp.PauliZ(custom_wires[0]) @ qp.PauliY(custom_wires[3]),
            qp.PauliZ(custom_wires[1]) @ qp.PauliY(custom_wires[2]),
        ),
        (qp.PauliZ(custom_wires[0]) @ qp.PauliY(custom_wires[3]), qp.PauliZ(custom_wires[1])),
    ],
)
def test_integration_custom_wires_batching(returns, seed):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    dev_def = qp.device("default.qubit", wires=custom_wires)
    dev_gpu = qp.device(device_name, wires=custom_wires, batch_obs=True)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return [qp.expval(r) for r in returns] + [qp.expval(qp.PauliY(custom_wires[1]))]

    n_params = 30
    rng = np.random.default_rng(seed)
    params = rng.random(n_params)

    qnode_gpu = qp.QNode(circuit, dev_gpu, diff_method="adjoint")
    qnode_def = qp.QNode(circuit, dev_def)

    def convert_to_array_gpu(params):
        return np.hstack(qnode_gpu(params))

    def convert_to_array_def(params):
        return np.hstack(qnode_def(params))

    j_gpu = qp.jacobian(convert_to_array_gpu)(params)
    j_def = qp.jacobian(convert_to_array_def)(params)

    assert np.allclose(j_gpu, j_def, atol=1e-7)


@pytest.mark.local_salt(42)
@pytest.mark.skipif(
    device_name not in ("lightning.qubit", "lightning.gpu"),
    reason="Tests only for lightning.qubit and lightning.gpu",
)
@pytest.mark.parametrize(
    "returns",
    [
        (0.5 * qp.PauliZ(custom_wires[0]),),
        (0.5 * qp.PauliZ(custom_wires[0]), qp.PauliZ(custom_wires[1])),
        (
            qp.PauliZ(custom_wires[0]),
            0.5 * qp.PauliZ(custom_wires[1]),
            qp.PauliZ(custom_wires[3]),
        ),
        (
            qp.PauliZ(custom_wires[0]),
            qp.PauliZ(custom_wires[1]),
            qp.PauliZ(custom_wires[3]),
            0.5 * qp.PauliZ(custom_wires[2]),
        ),
        (
            qp.PauliZ(custom_wires[0]) @ qp.PauliY(custom_wires[3]),
            0.5 * qp.PauliZ(custom_wires[1]) @ qp.PauliY(custom_wires[2]),
        ),
        (
            qp.PauliZ(custom_wires[0]) @ qp.PauliY(custom_wires[3]),
            0.5 * qp.PauliZ(custom_wires[1]),
        ),
        (
            0.0 * qp.PauliZ(custom_wires[0]) @ qp.PauliZ(custom_wires[1]),
            1.0 * qp.Identity(10),
            1.2 * qp.PauliZ(custom_wires[2]) @ qp.PauliZ(custom_wires[3]),
        ),
    ],
)
def test_batching_H(returns, seed):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    dev_cpu = qp.device("default.qubit", wires=custom_wires + [10, 72])
    dev_gpu = qp.device(device_name, wires=custom_wires + [10, 72], batch_obs=True)
    dev_gpu_default = qp.device(device_name, wires=custom_wires + [10, 72], batch_obs=False)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qp.math.hstack([qp.expval(r) for r in returns])

    n_params = 30
    rng = np.random.default_rng(seed)
    params = rng.random(n_params)

    qnode_cpu = qp.QNode(circuit, dev_cpu, diff_method="parameter-shift")
    qnode_gpu = qp.QNode(circuit, dev_gpu, diff_method="adjoint")
    qnode_gpu_default = qp.QNode(circuit, dev_gpu_default, diff_method="adjoint")

    def convert_to_array_cpu(params):
        return np.hstack(qnode_cpu(params))

    def convert_to_array_gpu(params):
        return np.hstack(qnode_gpu(params))

    def convert_to_array_gpu_default(params):
        return np.hstack(qnode_gpu_default(params))

    i_cpu = qnode_cpu(params)
    i_gpu = qnode_gpu(params)
    i_gpu_default = qnode_gpu_default(params)

    assert np.allclose(i_cpu, i_gpu)
    assert np.allclose(i_gpu, i_gpu_default)

    j_cpu = qp.jacobian(qnode_cpu)(params)
    j_gpu = qp.jacobian(qnode_gpu)(params)
    j_gpu_default = qp.jacobian(qnode_gpu_default)(params)

    assert np.allclose(j_cpu, j_gpu)
    assert np.allclose(j_gpu, j_gpu_default)


@pytest.fixture(scope="session")
def create_xyz_file(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tmp")
    file = directory / "h2.xyz"
    file.write_text("""2\nH2, Unoptimized\nH  1.0 0.0 0.0\nH -1.0 0.0 0.0""")
    yield file


@pytest.mark.parametrize("batches", [False, True, 1, 2, 3, 4])
def test_integration_H2_Hamiltonian(create_xyz_file, batches, seed):
    _ = pytest.importorskip("openfermionpyscf")
    n_electrons = 2
    np.random.seed(seed)

    str_path = create_xyz_file
    symbols, coordinates = qp.qchem.read_structure(str(str_path), outpath=str(str_path.parent))

    H, qubits = qp.qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        method="pyscf",
        active_electrons=n_electrons,
        name="h2",
        outpath=str(str_path.parent),
        load_data=True,
    )
    hf_state = qp.qchem.hf_state(n_electrons, qubits)
    _, doubles = qp.qchem.excitations(n_electrons, qubits)

    # Choose different batching supports here
    dev = qp.device(device_name, wires=qubits, batch_obs=batches)
    dev_comp = qp.device("default.qubit", wires=qubits)

    @qp.qnode(dev, diff_method="adjoint")
    def circuit(params, excitations):
        qp.BasisState(hf_state, wires=H.wires)
        for i, excitation in enumerate(excitations):
            if len(excitation) == 4:
                qp.DoubleExcitation(params[i], wires=excitation)
            else:
                qp.SingleExcitation(params[i], wires=excitation)
        return qp.expval(H)

    @qp.qnode(dev_comp, diff_method="parameter-shift")
    def circuit_compare(params, excitations):
        qp.BasisState(hf_state, wires=H.wires)

        for i, excitation in enumerate(excitations):
            if len(excitation) == 4:
                qp.DoubleExcitation(params[i], wires=excitation)
            else:
                qp.SingleExcitation(params[i], wires=excitation)
        return qp.expval(H)

    jac_func = qp.jacobian(circuit)
    jac_func_comp = qp.jacobian(circuit_compare)

    params = qp.numpy.array([0.0] * len(doubles), requires_grad=True)
    jacs = jac_func(params, excitations=doubles)
    jacs_comp = jac_func_comp(params, excitations=doubles)

    assert np.allclose(jacs, jacs_comp)


@pytest.mark.local_salt(42)
@pytest.mark.parametrize("n_targets", range(1, 6))
def test_qubit_unitary(n_targets, seed):
    """Tests that ``qp.QubitUnitary`` can be included in circuits differentiated with the adjoint method."""
    n_wires = 6
    dev = qp.device(device_name, wires=n_wires)
    dev_def = qp.device("default.qubit", wires=n_wires)

    init_state = get_random_normalized_state(2**n_wires)
    U = get_random_matrix(2**n_targets)
    U, _ = np.linalg.qr(U)
    U = np.array(U, requires_grad=False)

    obs = qp.prod(*(qp.PauliZ(i) for i in range(n_wires)))

    rng = np.random.default_rng(seed)
    par = 2 * np.pi * rng.random(n_wires)

    def circuit(x):
        qp.StatePrep(init_state, wires=range(n_wires))
        for i in range(n_wires // 2):
            qp.RY(x[i], wires=i)
        qp.QubitUnitary(U, wires=range(n_targets))
        for i in range(n_wires // 2, n_wires):
            qp.RY(x[i], wires=i)
        return qp.expval(obs)

    circ = qp.QNode(circuit, dev, diff_method="adjoint")
    circ_ps = qp.QNode(circuit, dev, diff_method="parameter-shift")
    circ_def = qp.QNode(circuit, dev_def, diff_method="adjoint")
    jac = qp.jacobian(circ)(par)
    jac_ps = qp.jacobian(circ_ps)(par)
    jac_def = qp.jacobian(circ_def)(par)

    assert jac.size == n_wires
    assert not np.allclose(jac, 0.0)
    assert np.allclose(jac, jac_ps)
    assert np.allclose(jac, jac_def)


@pytest.mark.local_salt(42)
@pytest.mark.parametrize("n_targets", [1, 2])
def test_diff_qubit_unitary(n_targets, seed):
    """Tests that ``qp.QubitUnitary`` can be differentiated with the adjoint method."""
    n_wires = 6
    dev = qp.device(device_name, wires=n_wires)
    dev_def = qp.device("default.qubit", wires=n_wires)
    _, h = get_tolerance_and_stepsize(dev, step_size=True)

    init_state = get_random_normalized_state(2**n_wires)
    init_state = np.array(init_state, requires_grad=False)
    U = get_random_matrix(2**n_targets)
    U, _ = np.linalg.qr(U)
    U = np.array(U, requires_grad=False)

    obs = qp.prod(*(qp.PauliZ(i) for i in range(n_wires)))

    rng = np.random.default_rng(seed)
    par = 2 * np.pi * rng.random(n_wires)

    def circuit(x, u_mat):
        qp.StatePrep(init_state, wires=range(n_wires))
        for i in range(n_wires // 2):
            qp.RY(x[i], wires=i)
        qp.QubitUnitary(u_mat, wires=range(n_targets))
        for i in range(n_wires // 2, n_wires):
            qp.RY(x[i], wires=i)
        return qp.expval(obs)

    circ = qp.QNode(circuit, dev, diff_method="adjoint")
    circ_def = qp.QNode(circuit, dev_def, diff_method="adjoint")
    circ_fd = qp.QNode(circuit, dev, diff_method="finite-diff", gradient_kwargs={"h": h})
    circ_ps = qp.QNode(circuit, dev, diff_method="parameter-shift")
    jacs = qp.jacobian(circ)(par, U)
    jacs_def = qp.jacobian(circ_def)(par, U)
    jacs_fd = qp.jacobian(circ_fd)(par, U)
    jacs_ps = qp.jacobian(circ_ps)(par, U)

    for jac, jac_def, jac_fd, jac_ps in zip(jacs, jacs_def, jacs_fd, jacs_ps):
        assert not np.allclose(jac, 0.0)
        assert np.allclose(jac, jac_fd)
        assert np.allclose(jac, jac_ps)
        assert np.allclose(jac, jac_def)
