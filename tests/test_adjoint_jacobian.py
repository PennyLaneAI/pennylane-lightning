# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Tests for the ``adjoint_jacobian`` method of LightningQubit.
"""
import math
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane import QNode, qnode
from scipy.stats import unitary_group
from pennylane_lightning import LightningQubit as lq

I, X, Y, Z = (
    np.eye(2),
    qml.PauliX.compute_matrix(),
    qml.PauliY.compute_matrix(),
    qml.PauliZ.compute_matrix(),
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


class TestAdjointJacobian:
    """Tests for the adjoint_jacobian method"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return qml.device("lightning.qubit", wires=3, c_dtype=request.param)

    def test_not_expval(self, dev):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.1, wires=0)
            qml.var(qml.PauliZ(0))

        with pytest.raises(
            qml.QuantumFunctionError, match="Adjoint differentiation method does not"
        ):
            dev.adjoint_jacobian(tape)

    def test_finite_shots_warns(self):
        """Tests warning raised when finite shots specified"""

        dev = qml.device("lightning.qubit", wires=1, shots=1)

        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        with pytest.warns(
            UserWarning, match="Requested adjoint differentiation to be computed with finite shots."
        ):
            dev.adjoint_jacobian(tape)

    def test_empty_measurements(self, tol, dev):
        """Tests if an empty array is returned when the measurements of the tape is empty."""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])

        jac = dev.adjoint_jacobian(tape)
        assert len(jac) == 0

    @pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_unsupported_op(self, dev):
        """Test if a QuantumFunctionError is raised for an unsupported operation, i.e.,
        multi-parameter operations that are not qml.Rot"""

        with qml.tape.QuantumTape() as tape:
            qml.CRot(0.1, 0.2, 0.3, wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with pytest.raises(
            qml.QuantumFunctionError, match="The CRot operation is not supported using the"
        ):
            dev.adjoint_jacobian(tape)

    @pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_proj_unsupported(self, dev):
        """Test if a QuantumFunctionError is raised for a Projector observable"""
        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0, 1], wires=[0, 1]))

        with pytest.raises(
            qml.QuantumFunctionError, match="differentiation method does not support the Projector"
        ):
            dev.adjoint_jacobian(tape)

        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0], wires=[0]) @ qml.PauliZ(0))

        with pytest.raises(
            qml.QuantumFunctionError, match="differentiation method does not support the Projector"
        ):
            dev.adjoint_jacobian(tape)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ])
    def test_pauli_rotation_gradient(self, G, theta, dev):
        """Tests that the automatic gradients of Pauli rotations are correct."""

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0]) / np.sqrt(2), wires=0)
            G(theta, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1}

        calculated_val = dev.adjoint_jacobian(tape)

        h = 2e-3 if dev.R_DTYPE == np.float32 else 1e-7
        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape, h=h)
        numeric_val = fn(qml.execute(tapes, dev, None))
        assert np.allclose(calculated_val, numeric_val[0][2], atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    def test_Rot_gradient(self, theta, dev):
        """Tests that the device gradient of an arbitrary Euler-angle-parameterized gate is
        correct."""

        params = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0]) / np.sqrt(2), wires=0)
            qml.Rot(*params, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        calculated_val = dev.adjoint_jacobian(tape)

        h = 2e-3 if dev.R_DTYPE == np.float32 else 1e-7
        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape, h=h)
        numeric_val = fn(qml.execute(tapes, dev, None))
        assert np.allclose(calculated_val, numeric_val[0][2:], atol=tol, rtol=0)

    @pytest.mark.parametrize("par", [1, -2, 1.623, -0.051, 0])  # integers, floats, zero
    def test_ry_gradient(self, par, tol, dev):
        """Test that the gradient of the RY gate matches the exact analytic formula."""

        with qml.tape.QuantumTape() as tape:
            qml.RY(par, wires=[0])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {0}

        # gradients
        exact = np.cos(par)
        grad_A = dev.adjoint_jacobian(tape)

        # different methods must agree
        assert np.allclose(grad_A, exact, atol=tol, rtol=0)

    def test_rx_gradient(self, tol, dev):
        """Test that the gradient of the RX gate matches the known formula."""
        a = 0.7418

        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.expval(qml.PauliZ(0))

        # circuit jacobians
        dev_jacobian = dev.adjoint_jacobian(tape)
        expected_jacobian = -np.sin(a)
        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    def test_multiple_rx_gradient_pauliz(self, tol, dev):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result."""
        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        with qml.tape.QuantumTape() as tape:
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[2], wires=2)

            for idx in range(3):
                qml.expval(qml.PauliZ(idx))

        # circuit jacobians
        dev_jacobian = dev.adjoint_jacobian(tape)
        expected_jacobian = -np.diag(np.sin(params))
        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    def test_multiple_rx_gradient_hermitian(self, tol, dev):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result
        with Hermitian observable
        """
        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        with qml.tape.QuantumTape() as tape:
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[2], wires=2)

            for idx in range(3):
                qml.expval(qml.Hermitian([[1, 0], [0, -1]], wires=[idx]))

        tape.trainable_params = {0, 1, 2}
        # circuit jacobians
        dev_jacobian = dev.adjoint_jacobian(tape)
        expected_jacobian = -np.diag(np.sin(params))

        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    qubit_ops = [getattr(qml, name) for name in qml.ops._qubit__ops__]
    ops = {qml.RX, qml.RY, qml.RZ, qml.PhaseShift, qml.CRX, qml.CRY, qml.CRZ, qml.Rot}

    def test_multiple_rx_gradient_expval_hermitian(self, tol, dev):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result
        with Hermitian observable
        """
        params = np.array([np.pi / 3, np.pi / 4, np.pi / 5])

        with qml.tape.QuantumTape() as tape:
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[2], wires=2)

            qml.expval(
                qml.Hermitian(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], wires=[0, 2]
                )
            )

        tape.trainable_params = {0, 1, 2}
        dev_jacobian = dev.adjoint_jacobian(tape)
        expected_jacobian = np.array(
            [-np.sin(params[0]) * np.cos(params[2]), 0, -np.cos(params[0]) * np.sin(params[2])]
        )

        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    qubit_ops = [getattr(qml, name) for name in qml.ops._qubit__ops__]
    ops = {qml.RX, qml.RY, qml.RZ, qml.PhaseShift, qml.CRX, qml.CRY, qml.CRZ, qml.Rot}

    @pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_multiple_rx_gradient_expval_hamiltonian(self, tol, dev):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result
        with Hermitian observable
        """
        params = np.array([np.pi / 3, np.pi / 4, np.pi / 5])

        ham = qml.Hamiltonian(
            [1.0, 0.3, 0.3, 0.4],
            [
                qml.PauliX(0) @ qml.PauliX(1),
                qml.PauliZ(0),
                qml.PauliZ(1),
                qml.Hermitian(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], wires=[0, 2]
                ),
            ],
        )

        with qml.tape.QuantumTape() as tape:
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[2], wires=2)

            qml.expval(ham)

        tape.trainable_params = {0, 1, 2}
        dev_jacobian = dev.adjoint_jacobian(tape)
        expected_jacobian = (
            0.3 * np.array([-np.sin(params[0]), 0, 0])
            + 0.3 * np.array([0, -np.sin(params[1]), 0])
            + 0.4
            * np.array(
                [-np.sin(params[0]) * np.cos(params[2]), 0, -np.cos(params[0]) * np.sin(params[2])]
            )
        )

        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    qubit_ops = [getattr(qml, name) for name in qml.ops._qubit__ops__]
    ops = {qml.RX, qml.RY, qml.RZ, qml.PhaseShift, qml.CRX, qml.CRY, qml.CRZ, qml.Rot}

    @pytest.mark.parametrize("obs", [qml.PauliX, qml.PauliY])
    @pytest.mark.parametrize(
        "op",
        [
            qml.RX(0.4, wires=0),
            qml.RY(0.6, wires=0),
            qml.RZ(0.8, wires=0),
            qml.CRX(1.0, wires=[0, 1]),
            qml.CRY(2.0, wires=[0, 1]),
            qml.CRZ(3.0, wires=[0, 1]),
            qml.Rot(0.2, -0.1, 0.2, wires=0),
        ],
    )
    def test_gradients_pauliz(self, op, obs, dev):
        """Tests that the gradients of circuits match between the finite difference and device
        methods."""

        # op.num_wires and op.num_params must be initialized a priori
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.RX(0.543, wires=0)
            qml.CNOT(wires=[0, 1])

            op

            qml.Rot(1.3, -2.3, 0.5, wires=[0])
            qml.RZ(-0.5, wires=0)
            qml.RY(0.5, wires=1).inv()
            qml.CNOT(wires=[0, 1])

            qml.expval(obs(wires=0))
            qml.expval(qml.PauliZ(wires=1))

        tape.trainable_params = set(range(1, 1 + op.num_params))

        h = 2e-3 if dev.R_DTYPE == np.float32 else 1e-7
        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        grad_F = (lambda t, fn: fn(qml.execute(t, dev, None)))(
            *qml.gradients.finite_diff(tape, h=h)
        )
        grad_D = dev.adjoint_jacobian(tape)

        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "op",
        [
            qml.RX(0.4, wires=0),
            qml.RY(0.6, wires=0),
            qml.RZ(0.8, wires=0),
            qml.CRX(1.0, wires=[0, 1]),
            qml.CRY(2.0, wires=[0, 1]),
            qml.CRZ(3.0, wires=[0, 1]),
            qml.Rot(0.2, -0.1, 0.2, wires=0),
        ],
    )
    def test_gradients_hermitian(self, op, dev):
        """Tests that the gradients of circuits match between the finite difference and device
        methods."""

        # op.num_wires and op.num_params must be initialized a priori
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.RX(0.543, wires=0)
            qml.CNOT(wires=[0, 1])

            op.queue()

            qml.Rot(1.3, -2.3, 0.5, wires=[0])
            qml.RZ(-0.5, wires=0)
            qml.RY(0.5, wires=1).inv()
            qml.CNOT(wires=[0, 1])

            qml.expval(
                qml.Hermitian(
                    [[0, 0, 1, 1], [0, 1, 2, 1], [1, 2, 1, 0], [1, 1, 0, 0]], wires=[0, 1]
                )
            )

        tape.trainable_params = set(range(1, 1 + op.num_params))

        h = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7
        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        grad_F = (lambda t, fn: fn(qml.execute(t, dev, None)))(
            *qml.gradients.finite_diff(tape, h=h)
        )
        grad_D = dev.adjoint_jacobian(tape)

        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    def test_gradient_gate_with_multiple_parameters_pauliz(self, dev):
        """Tests that gates with multiple free parameters yield correct gradients."""

        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        h = 2e-3 if dev.R_DTYPE == np.float32 else 1e-7
        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        grad_D = dev.adjoint_jacobian(tape)
        tapes, fn = qml.gradients.finite_diff(tape, h=h)
        grad_F = fn(qml.execute(tapes, dev, None))

        # gradient has the correct shape and every element is nonzero
        assert grad_D.shape == (1, 3)
        assert np.count_nonzero(grad_D) == 3
        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    def test_gradient_gate_with_multiple_parameters_hermitian(self, dev):
        """Tests that gates with multiple free parameters yield correct gradients."""
        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.Hermitian([[0, 1], [1, 1]], wires=0))

        tape.trainable_params = {1, 2, 3}

        h = 2e-3 if dev.R_DTYPE == np.float32 else 1e-7
        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        grad_D = dev.adjoint_jacobian(tape)
        tapes, fn = qml.gradients.finite_diff(tape, h=h)
        grad_F = fn(qml.execute(tapes, dev, None))

        # gradient has the correct shape and every element is nonzero
        assert grad_D.shape == (1, 3)
        assert np.count_nonzero(grad_D) == 3
        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    @pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_gradient_gate_with_multiple_parameters_hamiltonian(self, dev):
        """Tests that gates with multiple free parameters yield correct gradients."""
        x, y, z = [0.5, 0.3, -0.7]

        ham = qml.Hamiltonian(
            [1.0, 0.3, 0.3], [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0), qml.PauliZ(1)]
        )

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(ham)

        tape.trainable_params = {1, 2, 3}

        h = 2e-3 if dev.R_DTYPE == np.float32 else 1e-7
        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        grad_D = dev.adjoint_jacobian(tape)
        tapes, fn = qml.gradients.finite_diff(tape, h=h)
        grad_F = fn(qml.execute(tapes, dev, None))

        # gradient has the correct shape and every element is nonzero
        assert grad_D.shape == (1, 3)
        assert np.count_nonzero(grad_D) == 3
        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    def test_use_device_state(self, tol, dev):
        """Tests that when using the device state, the correct answer is still returned."""

        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        dM1 = dev.adjoint_jacobian(tape)

        qml.execute([tape], dev, None)
        dM2 = dev.adjoint_jacobian(tape, use_device_state=True)

        assert np.allclose(dM1, dM2, atol=tol, rtol=0)

    def test_provide_starting_state(self, tol, dev):
        """Tests provides correct answer when provided starting state."""
        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        dM1 = dev.adjoint_jacobian(tape)

        qml.execute([tape], dev, None)
        dM2 = dev.adjoint_jacobian(tape, starting_state=dev._pre_rotated_state)

        assert np.allclose(dM1, dM2, atol=tol, rtol=0)

    @pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_provide_wrong_starting_state(self, dev):
        """Tests raise an exception when provided starting state mismatches."""
        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The number of qubits of starting_state must be the same as",
        ):
            dev.adjoint_jacobian(tape, starting_state=np.ones(7))

    @pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_state_return_type(self, dev):
        """Tests raise an exception when the return type is State"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.state()

        tape.trainable_params = {0}

        with pytest.raises(
            qml.QuantumFunctionError, match="This method does not support statevector return type."
        ):
            dev.adjoint_jacobian(tape)


class TestAdjointJacobianQNode:
    """Test QNode integration with the adjoint_jacobian method"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return qml.device("lightning.qubit", wires=2, c_dtype=request.param)

    def test_finite_shots_warning(self):
        """Tests that a warning is raised when computing the adjoint diff on a device with finite shots"""

        dev = qml.device("lightning.qubit", wires=1, shots=1)

        with pytest.warns(
            UserWarning, match="Requested adjoint differentiation to be computed with finite shots."
        ):

            @qml.qnode(dev, diff_method="adjoint")
            def circ(x):
                qml.RX(x, wires=0)
                return qml.expval(qml.PauliZ(0))

        with pytest.warns(
            UserWarning, match="Requested adjoint differentiation to be computed with finite shots."
        ):
            qml.grad(circ)(0.1)

    @pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_qnode(self, mocker, dev):
        """Test that specifying diff_method allows the adjoint method to be selected"""
        args = np.array([0.54, 0.1, 0.5], requires_grad=True)

        def circuit(x, y, z):
            qml.Hadamard(wires=0)
            qml.RX(0.543, wires=0)
            qml.CNOT(wires=[0, 1])

            qml.Rot(x, y, z, wires=0)

            qml.Rot(1.3, -2.3, 0.5, wires=[0])
            qml.RZ(-0.5, wires=0)
            qml.RY(0.5, wires=1)
            qml.CNOT(wires=[0, 1])

            return qml.expval(qml.PauliX(0) @ qml.PauliZ(1))

        qnode1 = QNode(circuit, dev, diff_method="adjoint")
        spy = mocker.spy(dev, "adjoint_jacobian")

        grad_fn = qml.grad(qnode1)
        grad_A = grad_fn(*args)

        spy.assert_called()

        h = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7
        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        qnode2 = QNode(circuit, dev, diff_method="finite-diff", h=h)
        grad_fn = qml.grad(qnode2)
        grad_F = grad_fn(*args)

        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

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
            qml.RX(extra_param, wires=[0])
            qml.RY(p1, wires=[0])
            qml.RZ(p2, wires=[0])
            qml.RX(p1, wires=[0])
            return qml.expval(qml.PauliZ(0))

        zero_state = np.array([1.0, 0.0])
        cost(reused_p, other_p)

        spy = mocker.spy(dev, "adjoint_jacobian")

        # analytic gradient
        grad_fn = qml.grad(cost)
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

    @pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_gradient_repeated_gate_parameters(self, mocker, dev):
        """Tests that repeated use of a free parameter in a multi-parameter gate yields correct
        gradients."""
        params = np.array([0.8, 1.3], requires_grad=True)

        def circuit(params):
            qml.RX(np.array(np.pi / 4, requires_grad=False), wires=[0])
            qml.Rot(params[1], params[0], 2 * params[0], wires=[0])
            return qml.expval(qml.PauliX(0))

        spy_analytic = mocker.spy(dev, "adjoint_jacobian")

        h = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7
        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        cost = QNode(circuit, dev, diff_method="finite-diff", h=h)

        grad_fn = qml.grad(cost)
        grad_F = grad_fn(params)

        spy_analytic.assert_not_called()

        cost = QNode(circuit, dev, diff_method="adjoint")
        grad_fn = qml.grad(cost)
        grad_D = grad_fn(params)

        spy_analytic.assert_called_once()

        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    def test_interface_tf(self, dev):
        """Test if gradients agree between the adjoint and finite-diff methods when using the
        TensorFlow interface"""

        tf = pytest.importorskip("tensorflow")

        def f(params1, params2):
            qml.RX(0.4, wires=[0])
            qml.RZ(params1 * tf.sqrt(params2), wires=[0])
            qml.RY(tf.cos(params2), wires=[0])
            return qml.expval(qml.PauliZ(0))

        if dev.R_DTYPE == np.float32:
            tf_r_dtype = tf.float32
        else:
            tf_r_dtype = tf.float64

        params1 = tf.Variable(0.3, dtype=tf_r_dtype)
        params2 = tf.Variable(0.4, dtype=tf_r_dtype)

        h = 2e-3 if dev.R_DTYPE == np.float32 else 1e-7
        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        qnode1 = QNode(f, dev, interface="tf", diff_method="adjoint")
        qnode2 = QNode(f, dev, interface="tf", diff_method="finite-diff", h=h)

        with tf.GradientTape() as tape:
            res1 = qnode1(params1, params2)

        g1 = tape.gradient(res1, [params1, params2])

        with tf.GradientTape() as tape:
            res2 = qnode2(params1, params2)

        g2 = tape.gradient(res2, [params1, params2])

        assert np.allclose(g1, g2, atol=tol)

    def test_interface_torch(self, dev):
        """Test if gradients agree between the adjoint and finite-diff methods when using the
        Torch interface"""

        torch = pytest.importorskip("torch")

        def f(params1, params2):
            qml.RX(0.4, wires=[0])
            qml.RZ(params1 * torch.sqrt(params2), wires=[0])
            qml.RY(torch.cos(params2), wires=[0])
            return qml.expval(qml.PauliZ(0))

        params1 = torch.tensor(0.3, requires_grad=True)
        params2 = torch.tensor(0.4, requires_grad=True)

        h = 2e-3 if dev.R_DTYPE == np.float32 else 1e-7
        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        qnode1 = QNode(f, dev, interface="torch", diff_method="adjoint")
        qnode2 = QNode(f, dev, interface="torch", diff_method="finite-diff", h=h)

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

        def f(params1, params2):
            qml.RX(0.4, wires=[0])
            qml.RZ(params1 * jax.numpy.sqrt(params2), wires=[0])
            qml.RY(jax.numpy.cos(params2), wires=[0])
            return qml.expval(qml.PauliZ(0))

        params1 = jax.numpy.array(0.3)
        params2 = jax.numpy.array(0.4)

        h = 2e-3 if dev.R_DTYPE == np.float32 else 1e-7
        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        qnode_adjoint = QNode(f, dev, interface="jax", diff_method="adjoint")
        qnode_fd = QNode(f, dev, interface="jax", diff_method="finite-diff", h=h)

        grad_adjoint = jax.grad(qnode_adjoint)(params1, params2)
        grad_fd = jax.grad(qnode_fd)(params1, params2)

        assert np.allclose(grad_adjoint, grad_fd, atol=tol)


@pytest.mark.parametrize(
    "r_dtype,c_dtype", [[np.float32, np.complex64], [np.float64, np.complex128]]
)
def test_qchem_expvalcost_correct(r_dtype, c_dtype):
    """EvpvalCost with qchem Hamiltonian work corectly"""
    from pennylane import qchem

    symbols = ["Li", "H"]
    geometry = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.969280527])
    H, qubits = qchem.molecular_hamiltonian(
        symbols, geometry, active_electrons=2, active_orbitals=5
    )
    active_electrons = 2
    hf_state = qchem.hf_state(active_electrons, qubits)

    def circuit_1(params, wires):
        qml.BasisState(hf_state, wires=wires)
        qml.RX(params[0], wires=0)
        qml.RY(params[0], wires=1)
        qml.RZ(params[0], wires=2)
        qml.Hadamard(wires=1)

    diff_method = "adjoint"
    dev_lig = qml.device("lightning.qubit", wires=qubits, c_dtype=c_dtype)
    cost_fn_lig = qml.ExpvalCost(circuit_1, H, dev_lig, optimize=False, diff_method=diff_method)
    circuit_gradient_lig = qml.grad(cost_fn_lig, argnum=0)
    params = np.array([0.123], requires_grad=True)
    grads_lig = circuit_gradient_lig(params)

    dev_def = qml.device("default.qubit", wires=qubits)
    cost_fn_def = qml.ExpvalCost(circuit_1, H, dev_def, optimize=False, diff_method=diff_method)
    circuit_gradient_def = qml.grad(cost_fn_def, argnum=0)
    params = np.array([0.123], requires_grad=True)
    grads_def = circuit_gradient_def(params)

    assert np.allclose(grads_lig, grads_def)


def circuit_ansatz(params, wires):
    """Circuit ansatz containing all the parametrized gates"""
    qml.QubitStateVector(unitary_group.rvs(2**4, random_state=0)[0], wires=wires)
    qml.RX(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.RX(params[2], wires=wires[2]).inv()
    qml.RZ(params[0], wires=wires[3])
    qml.CRX(params[3], wires=[wires[3], wires[0]])
    qml.PhaseShift(params[4], wires=wires[2])
    qml.CRY(params[5], wires=[wires[2], wires[1]])
    qml.CRZ(params[5], wires=[wires[0], wires[3]]).inv()
    qml.PhaseShift(params[6], wires=wires[0]).inv()
    qml.Rot(params[6], params[7], params[8], wires=wires[0])
    # #     qml.Rot(params[8], params[8], params[9], wires=wires[1]).inv()
    qml.MultiRZ(params[11], wires=[wires[0], wires[1]])
    # #     qml.PauliRot(params[12], "XXYZ", wires=[wires[0], wires[1], wires[2], wires[3]])
    qml.CPhase(params[12], wires=[wires[3], wires[2]])
    qml.IsingXX(params[13], wires=[wires[1], wires[0]])
    qml.IsingXY(params[14], wires=[wires[3], wires[2]])
    qml.IsingYY(params[14], wires=[wires[3], wires[2]])
    qml.IsingZZ(params[14], wires=[wires[2], wires[1]])
    qml.U1(params[15], wires=wires[0])
    qml.U2(params[16], params[17], wires=wires[0])
    qml.U3(params[18], params[19], params[20], wires=wires[1])
    # #     qml.CRot(params[21], params[22], params[23], wires=[wires[1], wires[2]]).inv()  # expected tofail
    qml.SingleExcitation(params[24], wires=[wires[2], wires[0]])
    qml.DoubleExcitation(params[25], wires=[wires[2], wires[0], wires[1], wires[3]])
    qml.SingleExcitationPlus(params[26], wires=[wires[0], wires[2]])
    qml.SingleExcitationMinus(params[27], wires=[wires[0], wires[2]])
    qml.DoubleExcitationPlus(params[27], wires=[wires[2], wires[0], wires[1], wires[3]])
    qml.DoubleExcitationMinus(params[27], wires=[wires[2], wires[0], wires[1], wires[3]])
    qml.RX(params[28], wires=wires[0])
    qml.RX(params[29], wires=wires[1])


@pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
def test__tape_qchem(tol):
    """The circit Ansatz with a QChem Hamiltonian produces correct results"""

    H, qubits = qml.qchem.molecular_hamiltonian(
        ["H", "H"], np.array([0.0, 0.1, 0.0, 0.0, -0.1, 0.0])
    )

    def circuit(params):
        circuit_ansatz(params, wires=range(4))
        return qml.expval(H)

    params = np.arange(30) * 0.111

    dev_lq = qml.device("lightning.qubit", wires=4)
    dev_dq = qml.device("default.qubit", wires=4)

    circuit_lq = qml.QNode(circuit, dev_lq, diff_method="adjoint")
    circuit_dq = qml.QNode(circuit, dev_lq, diff_method="parameter-shift")

    assert np.allclose(qml.grad(circuit_lq)(params), qml.grad(circuit_dq)(params), tol)


@pytest.mark.parametrize(
    "returns",
    [
        qml.PauliZ(0),
        qml.PauliX(2),
        qml.PauliZ(0) @ qml.PauliY(3),
        qml.Hadamard(2),
        qml.Hadamard(3) @ qml.PauliZ(2),
        # qml.Projector([0, 1], wires=[0, 2]) @ qml.Hadamard(3)
        # qml.Projector([0, 0], wires=[2, 0])
        qml.PauliX(0) @ qml.PauliY(3),
        qml.PauliY(0) @ qml.PauliY(2) @ qml.PauliY(3),
        qml.Hermitian(
            np.kron(qml.PauliY.compute_matrix(), qml.PauliZ.compute_matrix()), wires=[3, 2]
        ),
        qml.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=0),
        qml.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=0) @ qml.PauliZ(2),
    ],
)
def test_integration(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations"""
    dev_def = qml.device("default.qubit", wires=range(4))
    dev_lightning = qml.device("lightning.qubit", wires=range(4))

    def circuit(params):
        circuit_ansatz(params, wires=range(4))
        return qml.expval(returns), qml.expval(qml.PauliY(1))

    n_params = 30
    params = np.linspace(0, 10, n_params)

    qnode_def = qml.QNode(circuit, dev_def)
    qnode_lightning = qml.QNode(circuit, dev_lightning, diff_method="adjoint")

    j_def = qml.jacobian(qnode_def)(params)
    j_lightning = qml.jacobian(qnode_lightning)(params)

    assert np.allclose(j_def, j_lightning)


@pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
def test_integration_chunk_observables():
    """Integration tests that compare to default.qubit for a large circuit with multiple expectation values. Expvals are generated in parallelized chunks."""
    dev_def = qml.device("default.qubit", wires=range(4))
    dev_lightning = qml.device("lightning.qubit", wires=range(4))
    dev_lightning_batched = qml.device("lightning.qubit", wires=range(4), batch_obs=True)

    def circuit(params):
        circuit_ansatz(params, wires=range(4))
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    n_params = 30
    params = np.linspace(0, 10, n_params)

    qnode_def = qml.QNode(circuit, dev_def)
    qnode_lightning = qml.QNode(circuit, dev_lightning, diff_method="adjoint")
    qnode_lightning_batched = qml.QNode(circuit, dev_lightning_batched, diff_method="adjoint")

    j_def = qml.jacobian(qnode_def)(params)
    j_lightning = qml.jacobian(qnode_lightning)(params)
    j_lightning_batched = qml.jacobian(qnode_lightning_batched)(params)

    assert np.allclose(j_def, j_lightning)
    assert np.allclose(j_def, j_lightning_batched)


custom_wires = ["alice", 3.14, -1, 0]


@pytest.mark.parametrize(
    "returns",
    [
        qml.PauliZ(custom_wires[0]),
        qml.PauliX(custom_wires[2]),
        qml.PauliZ(custom_wires[0]) @ qml.PauliY(custom_wires[3]),
        qml.Hadamard(custom_wires[2]),
        qml.Hadamard(custom_wires[3]) @ qml.PauliZ(custom_wires[2]),
        # qml.Projector([0, 1], wires=[custom_wires[0], custom_wires[2]]) @ qml.Hadamard(custom_wires[3])
        # qml.Projector([0, 0], wires=[custom_wires[2], custom_wires[0]])
        qml.PauliX(custom_wires[0]) @ qml.PauliY(custom_wires[3]),
        qml.PauliY(custom_wires[0]) @ qml.PauliY(custom_wires[2]) @ qml.PauliY(custom_wires[3]),
        qml.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=custom_wires[0]),
        qml.Hermitian(
            np.kron(qml.PauliY.compute_matrix(), qml.PauliZ.compute_matrix()),
            wires=[custom_wires[3], custom_wires[2]],
        ),
        qml.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=custom_wires[0])
        @ qml.PauliZ(custom_wires[2]),
    ],
)
def test_integration_custom_wires(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""
    dev_def = qml.device("default.qubit", wires=custom_wires)
    dev_lightning = qml.device("lightning.qubit", wires=custom_wires)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qml.expval(returns), qml.expval(qml.PauliY(custom_wires[1]))

    n_params = 30
    params = np.linspace(0, 10, n_params)

    qnode_def = qml.QNode(circuit, dev_def)
    qnode_lightning = qml.QNode(circuit, dev_lightning, diff_method="adjoint")

    j_def = qml.jacobian(qnode_def)(params)
    j_lightning = qml.jacobian(qnode_lightning)(params)

    assert np.allclose(j_def, j_lightning)
