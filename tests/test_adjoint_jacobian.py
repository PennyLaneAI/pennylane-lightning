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

import pennylane as qml
import pytest
from conftest import LightningDevice as ld
from conftest import LightningException, device_name
from pennylane import QNode
from pennylane import numpy as np
from pennylane import qchem, qnode
from scipy.stats import unitary_group

I, X, Y, Z = (
    np.eye(2),
    qml.X.compute_matrix(),
    qml.Y.compute_matrix(),
    qml.Z.compute_matrix(),
)

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

if device_name == "lightning.tensor":
    pytest.skip("lightning.tensor doesn't support adjoint jacobian.", allow_module_level=True)

kokkos_args = [None]
if device_name == "lightning.kokkos":
    from pennylane_lightning.lightning_kokkos_ops import InitializationSettings

    kokkos_args += [InitializationSettings().set_num_threads(2)]

fixture_params = itertools.product(
    [np.complex64, np.complex128],
    kokkos_args,
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
    tol = 1e-3 if device.dtype == np.complex64 else 1e-7
    h = tol if step_size else None
    return tol, h


class TestAdjointJacobian:
    """Tests for the adjoint_jacobian method"""

    @staticmethod
    def get_derivatives_method(device):
        return device.compute_derivatives if device._new_API else device.adjoint_jacobian

    @pytest.fixture(params=fixture_params)
    def dev(self, request):
        params = request.param
        if device_name == "lightning.kokkos":
            return qml.device(device_name, wires=3, c_dtype=params[0], kokkos_args=params[1])
        return qml.device(device_name, wires=3, c_dtype=params[0])

    def test_not_expval(self, dev):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.1, wires=0)
            qml.var(qml.PauliZ(0))

        method = self.get_derivatives_method(dev)

        with pytest.raises(
            qml.QuantumFunctionError, match="Adjoint differentiation method does not"
        ):
            method(tape)

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.1, wires=0)
            qml.state()

        if dev._new_API:
            message = "Adjoint differentiation method does not support measurement StateMP."
        elif device_name == "lightning.gpu":
            message = "Adjoint differentiation does not support State measurements."

        with pytest.raises(
            qml.QuantumFunctionError,
            match=message,
        ):
            method(tape)

    @pytest.mark.skipif(ld._new_API, reason="Requires old API")
    def test_finite_shots_warns(self):
        """Tests warning raised when finite shots specified"""

        dev = qml.device(device_name, wires=1, shots=1)

        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        with pytest.warns(
            UserWarning, match="Requested adjoint differentiation to be computed with finite shots."
        ):
            dev.adjoint_jacobian(tape)

    @pytest.mark.skipif(not ld._new_API, reason="Requires new API")
    def test_finite_shots_error(self):
        """Tests warning raised when finite shots specified"""

        dev = qml.device(device_name, wires=1)

        tape = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))], shots=1)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Requested adjoint differentiation to be computed with finite shots.",
        ):
            dev.compute_derivatives(tape)

    def test_empty_measurements(self, dev):
        """Tests if an empty array is returned when the measurements of the tape is empty."""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])

        method = self.get_derivatives_method(dev)
        jac = method(tape)
        assert len(jac) == 0

    def test_unsupported_op(self, dev):
        """Test if a QuantumFunctionError is raised for an unsupported operation, i.e.,
        multi-parameter operations that are not qml.Rot"""

        with qml.tape.QuantumTape() as tape:
            qml.CRot(0.1, 0.2, 0.3, wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        if dev._new_API:
            with pytest.raises(
                LightningException,
                match="The operation is not supported using the adjoint differentiation method",
            ):
                dev.compute_derivatives(tape)
        else:
            with pytest.raises(
                qml.QuantumFunctionError, match="The CRot operation is not supported using the"
            ):
                dev.adjoint_jacobian(tape)

    @pytest.mark.usefixtures("use_legacy_and_new_opmath")
    @pytest.mark.skipif(ld._new_API, reason="Old API required")
    def test_proj_unsupported(self, dev):
        """Test if a QuantumFunctionError is raised for a Projector observable"""
        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0, 1], wires=[0, 1]))

        method = self.get_derivatives_method(dev)

        with pytest.raises(
            qml.QuantumFunctionError, match="differentiation method does not support the Projector"
        ):
            method(tape)

        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0], wires=[0]) @ qml.PauliZ(1))

        with pytest.raises(
            qml.QuantumFunctionError, match="differentiation method does not support the Projector"
        ):
            method(tape)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ])
    @pytest.mark.parametrize("stateprep", [qml.QubitStateVector, qml.StatePrep])
    def test_pauli_rotation_gradient(self, stateprep, G, theta, dev):
        """Tests that the automatic gradients of Pauli rotations are correct."""
        random_state = np.array(
            [0.43593284 - 0.02945156j, 0.40812291 + 0.80158023j], requires_grad=False
        )

        tape = qml.tape.QuantumScript(
            [stateprep(random_state, 0), G(theta, 0)], [qml.expval(qml.PauliZ(0))]
        )

        tape.trainable_params = {1}

        method = self.get_derivatives_method(dev)
        calculated_val = method(tape)

        tol, _ = get_tolerance_and_stepsize(dev)

        # compare to finite differences
        tapes, fn = qml.gradients.param_shift(tape)
        numeric_val = fn(qml.execute(tapes, dev, None))
        assert np.allclose(calculated_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("stateprep", [qml.QubitStateVector, qml.StatePrep])
    def test_Rot_gradient(self, stateprep, theta, dev):
        """Tests that the device gradient of an arbitrary Euler-angle-parameterized gate is
        correct."""
        params = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.tape.QuantumTape() as tape:
            stateprep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            qml.Rot(*params, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        method = self.get_derivatives_method(dev)
        calculated_val = method(tape)

        tol, _ = get_tolerance_and_stepsize(dev)

        # compare to finite differences
        tapes, fn = qml.gradients.param_shift(tape)
        numeric_val = fn(qml.execute(tapes, dev, None))
        assert np.allclose(calculated_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.skipif(
        device_name != "lightning.qubit",
        reason="N-controlled operations only implemented in lightning.qubit.",
    )
    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
    @pytest.mark.parametrize("par", [-np.pi / 7, np.pi / 5, 2 * np.pi / 3])
    def test_phaseshift_gradient(self, n_qubits, par, tol):
        """Test that the gradient of the phaseshift gate matches the exact analytic formula."""
        par = np.array(par)
        dev = qml.device(device_name, wires=n_qubits)
        init_state = np.zeros(2**n_qubits)
        init_state[-2::] = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], requires_grad=False)

        with qml.tape.QuantumTape() as tape:
            qml.StatePrep(init_state, wires=range(n_qubits))
            qml.ctrl(qml.PhaseShift(par, wires=n_qubits - 1), range(0, n_qubits - 1))
            qml.expval(qml.PauliY(n_qubits - 1))

        tape.trainable_params = {1}

        exact = np.cos(par)
        method = self.get_derivatives_method(dev)
        grad_A = method(tape)

        # different methods must agree
        assert np.allclose(grad_A, exact, atol=tol, rtol=0)

    @pytest.mark.parametrize("par", [1, -2, 1.623, -0.051, 0])  # integers, floats, zero
    def test_ry_gradient(self, par, tol, dev):
        """Test that the gradient of the RY gate matches the exact analytic formula."""
        with qml.tape.QuantumTape() as tape:
            qml.RY(par, wires=[0])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {0}

        # gradients
        exact = np.cos(par)
        method = self.get_derivatives_method(dev)
        grad_A = method(tape)

        # different methods must agree
        assert np.allclose(grad_A, exact, atol=tol, rtol=0)

    def test_rx_gradient(self, tol, dev):
        """Test that the gradient of the RX gate matches the known formula."""
        a = 0.7418

        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.expval(qml.PauliZ(0))

        # circuit jacobians
        method = self.get_derivatives_method(dev)
        dev_jacobian = method(tape)
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
        method = self.get_derivatives_method(dev)
        dev_jacobian = method(tape)
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
        method = self.get_derivatives_method(dev)
        dev_jacobian = method(tape)
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
        method = self.get_derivatives_method(dev)
        dev_jacobian = method(tape)
        expected_jacobian = np.array(
            [-np.sin(params[0]) * np.cos(params[2]), 0, -np.cos(params[0]) * np.sin(params[2])]
        )

        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    qubit_ops = [getattr(qml, name) for name in qml.ops._qubit__ops__]
    ops = {qml.RX, qml.RY, qml.RZ, qml.PhaseShift, qml.CRX, qml.CRY, qml.CRZ, qml.Rot}

    @pytest.mark.usefixtures("use_legacy_and_new_opmath")
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
        method = self.get_derivatives_method(dev)
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
            qml.adjoint(qml.RY(0.5, wires=1), lazy=False)
            qml.CNOT(wires=[0, 1])

            qml.expval(obs(wires=0))
            qml.expval(qml.PauliZ(wires=1))

        tape.trainable_params = set(range(1, 1 + op.num_params))

        tol, _ = get_tolerance_and_stepsize(dev)

        grad_F = (lambda t, fn: fn(qml.execute(t, dev, None)))(*qml.gradients.param_shift(tape))
        method = self.get_derivatives_method(dev)
        grad_D = method(tape)

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
            qml.adjoint(qml.RY(0.5, wires=1), lazy=False)
            qml.CNOT(wires=[0, 1])

            qml.expval(
                qml.Hermitian(
                    [[0, 0, 1, 1], [0, 1, 2, 1], [1, 2, 1, 0], [1, 1, 0, 0]], wires=[0, 1]
                )
            )

        tape.trainable_params = set(range(1, 1 + op.num_params))

        tol, _ = get_tolerance_and_stepsize(dev)

        grad_F = (lambda t, fn: fn(qml.execute(t, dev, None)))(*qml.gradients.param_shift(tape))
        method = self.get_derivatives_method(dev)
        grad_D = method(tape)

        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    def test_gradient_gate_with_multiple_parameters_pauliz(self, dev):
        """Tests that gates with multiple free parameters yield correct gradients."""
        x, y, z = [0.5, 0.3, -0.7]

        tape = qml.tape.QuantumScript(
            [qml.RX(0.4, wires=[0]), qml.Rot(x, y, z, wires=[0]), qml.RY(-0.2, wires=[0])],
            [qml.expval(qml.PauliZ(0))],
        )

        tape.trainable_params = {1, 2, 3}

        tol, _ = get_tolerance_and_stepsize(dev)

        method = self.get_derivatives_method(dev)
        grad_D = method(tape)
        tapes, fn = qml.gradients.param_shift(tape)
        grad_F = fn(qml.execute(tapes, dev, None))

        # gradient has the correct shape and every element is nonzero
        assert len(grad_D) == 3
        assert all(isinstance(v, np.ndarray) for v in grad_D)
        assert np.count_nonzero(grad_D) == 3
        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    def test_gradient_gate_with_multiple_parameters_hermitian(self, dev):
        """Tests that gates with multiple free parameters yield correct gradients."""
        x, y, z = [0.5, 0.3, -0.7]

        tape = qml.tape.QuantumScript(
            [qml.RX(0.4, wires=[0]), qml.Rot(x, y, z, wires=[0]), qml.RY(-0.2, wires=[0])],
            [qml.expval(qml.Hermitian([[0, 1], [1, 1]], wires=0))],
        )

        tape.trainable_params = {1, 2, 3}

        tol, _ = get_tolerance_and_stepsize(dev)

        method = self.get_derivatives_method(dev)
        grad_D = method(tape)
        tapes, fn = qml.gradients.param_shift(tape)
        grad_F = fn(qml.execute(tapes, dev, None))

        # gradient has the correct shape and every element is nonzero
        assert len(grad_D) == 3
        assert all(isinstance(v, np.ndarray) for v in grad_D)
        assert np.count_nonzero(grad_D) == 3
        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    @pytest.mark.usefixtures("use_legacy_and_new_opmath")
    def test_gradient_gate_with_multiple_parameters_hamiltonian(self, dev):
        """Tests that gates with multiple free parameters yield correct gradients."""
        x, y, z = [0.5, 0.3, -0.7]

        ham = qml.Hamiltonian(
            [1.0, 0.3, 0.3], [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0), qml.PauliZ(1)]
        )

        tape = qml.tape.QuantumScript(
            [qml.RX(0.4, wires=[0]), qml.Rot(x, y, z, wires=[0]), qml.RY(-0.2, wires=[0])],
            [qml.expval(ham)],
        )

        tape.trainable_params = {1, 2, 3}

        tol, _ = get_tolerance_and_stepsize(dev)

        method = self.get_derivatives_method(dev)
        grad_D = method(tape)
        tapes, fn = qml.gradients.param_shift(tape)
        grad_F = fn(qml.execute(tapes, dev, None))

        # gradient has the correct shape and every element is nonzero
        assert len(grad_D) == 3
        assert all(isinstance(v, np.ndarray) for v in grad_D)
        assert np.count_nonzero(grad_D) == 3
        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    @pytest.mark.skipif(ld._new_API, reason="Old API required")
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

    @pytest.mark.skipif(ld._new_API, reason="Old API required")
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

        if device_name in ["lightning.kokkos", "lightning.qubit"]:
            qml.execute([tape], dev, None)
            dM2 = dev.adjoint_jacobian(tape, starting_state=dev.state_vector)

            assert np.allclose(dM1, dM2, atol=tol, rtol=0)
        else:
            state_vector = dev.state
            qml.execute([tape], dev, None)
            dM2 = dev.adjoint_jacobian(tape, starting_state=state_vector)
            assert np.allclose(dM1, dM2, atol=tol, rtol=0)

    @pytest.mark.skipif(ld._new_API, reason="Old API required")
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

    @pytest.mark.skipif(
        device_name == "lightning.kokkos" or device_name == "lightning.gpu",
        reason="Adjoint differentiation does not support State measurements.",
    )
    def test_state_return_type(self, dev):
        """Tests raise an exception when the return type is State"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.state()

        tape.trainable_params = {0}
        method = self.get_derivatives_method(dev)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Adjoint differentiation method does not support measurement StateMP.",
        ):
            method(tape)


class TestAdjointJacobianQNode:
    """Test QNode integration with the adjoint_jacobian method"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return qml.device(device_name, wires=2, c_dtype=request.param)

    @pytest.mark.skipif(ld._new_API, reason="Old API required")
    def test_finite_shots_error(self):
        """Tests that an error is raised when computing the adjoint diff on a device with finite shots"""

        dev = qml.device(device_name, wires=1, shots=1)

        with pytest.raises(
            qml.QuantumFunctionError, match="does not support adjoint with requested circuit."
        ):

            @qml.qnode(dev, diff_method="adjoint")
            def circ(x):
                qml.RX(x, wires=0)
                return qml.expval(qml.PauliZ(0))

            qml.grad(circ)(0.1)

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
        spy = (
            mocker.spy(dev, "execute_and_compute_derivatives")
            if ld._new_API
            else mocker.spy(dev.target_device, "adjoint_jacobian")
        )
        tol, h = get_tolerance_and_stepsize(dev, step_size=True)

        grad_fn = qml.grad(qnode1)
        grad_A = grad_fn(*args)

        spy.assert_called()

        qnode2 = QNode(circuit, dev, diff_method="finite-diff", h=h)
        grad_fn = qml.grad(qnode2)
        grad_F = grad_fn(*args)

        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "operation",
        [
            qml.PhaseShift,
            qml.RX,
            qml.RY,
            qml.RZ,
            qml.IsingXX,
            qml.IsingXY,
            qml.IsingYY,
            qml.IsingZZ,
            qml.CRX,
            qml.CRY,
            qml.CRZ,
            qml.ControlledPhaseShift,
            qml.SingleExcitation,
            qml.SingleExcitationMinus,
            qml.SingleExcitationPlus,
            qml.DoubleExcitation,
            qml.DoubleExcitationMinus,
            qml.DoubleExcitationPlus,
            qml.MultiRZ,
            qml.GlobalPhase,
        ],
    )
    @pytest.mark.parametrize("n_qubits", range(2, 6))
    @pytest.mark.parametrize("par", [-np.pi / 7, np.pi / 5, 2 * np.pi / 3])
    def test_gate_jacobian(self, par, n_qubits, operation, tol):
        """Test that the jacobian of the controlled gate matches backprop."""
        par = np.array([0.1234, par, 0.5678])
        dev = qml.device(device_name, wires=n_qubits)
        dqu = qml.device("default.qubit", wires=n_qubits)
        np.random.seed(1337)
        init_state = np.random.rand(2**n_qubits) + 1.0j * np.random.rand(2**n_qubits)
        init_state /= np.linalg.norm(init_state)
        init_state = np.array(init_state, requires_grad=False)

        num_wires = max(operation.num_wires, 1)
        if num_wires > n_qubits:
            return

        for w in range(0, n_qubits - num_wires):

            def circuit(p):
                qml.StatePrep(init_state, wires=range(n_qubits))
                qml.RX(p[0], 0)
                if operation is qml.GlobalPhase:
                    operation(p[1], wires=range(n_qubits))
                else:
                    operation(p[1], wires=range(w, w + num_wires))
                qml.RY(p[2], 0)
                return np.array([qml.expval(qml.PauliY(i)) for i in range(n_qubits)])

            circ_ad = qml.QNode(circuit, dev, diff_method="adjoint")
            circ_bp = qml.QNode(circuit, dqu, diff_method="backprop")
            jac_ad = np.array(qml.jacobian(circ_ad)(par))
            jac_bp = np.array(qml.jacobian(circ_bp)(par))

            # different methods must agree
            assert jac_ad.size == n_qubits * 3
            assert np.allclose(jac_ad.shape, [n_qubits, 3])
            assert np.allclose(jac_ad.shape, jac_bp.shape)
            assert np.allclose(jac_ad, jac_bp, atol=tol, rtol=0)

    @pytest.mark.skipif(
        device_name != "lightning.qubit",
        reason="N-controlled operations only implemented in lightning.qubit.",
    )
    @pytest.mark.parametrize(
        "operation",
        [
            qml.PhaseShift,
            qml.RX,
            qml.RY,
            qml.RZ,
            qml.Rot,
            qml.IsingXX,
            qml.IsingXY,
            qml.IsingYY,
            qml.IsingZZ,
            qml.SingleExcitation,
            qml.SingleExcitationMinus,
            qml.SingleExcitationPlus,
            qml.DoubleExcitation,
            qml.DoubleExcitationMinus,
            qml.DoubleExcitationPlus,
            qml.MultiRZ,
            qml.GlobalPhase,
        ],
    )
    @pytest.mark.parametrize("control_value", [False, True])
    @pytest.mark.parametrize("n_qubits", range(2, 6))
    @pytest.mark.parametrize("par", [-np.pi / 7, np.pi / 5, 2 * np.pi / 3])
    def test_controlled_jacobian(self, par, n_qubits, control_value, operation, tol):
        """Test that the jacobian of the controlled gate matches the parameter-shift formula."""
        par = np.array([0.1234, par, 0.5678])
        dev = qml.device("lightning.qubit", wires=n_qubits)
        dqu = qml.device("default.qubit", wires=n_qubits)
        np.random.seed(1337)
        init_state = np.random.rand(2**n_qubits) + 1.0j * np.random.rand(2**n_qubits)
        init_state /= np.linalg.norm(init_state)
        init_state = np.array(init_state, requires_grad=False)
        num_wires = max(operation.num_wires, 1)
        if num_wires > n_qubits:
            return

        for n_controls in range(0, n_qubits - num_wires):
            control_wires = range(n_controls, n_qubits - num_wires)

            def circuit(p):
                qml.StatePrep(init_state, wires=range(n_qubits))
                if operation.num_params == 3:
                    qml.ctrl(
                        operation(*p, wires=range(n_qubits - num_wires, n_qubits)),
                        control_wires,
                        control_values=[
                            control_value or bool(i % 2) for i, _ in enumerate(control_wires)
                        ],
                    )
                else:
                    qml.RX(p[0], 0)
                    qml.ctrl(
                        operation(p[1], wires=range(n_qubits - num_wires, n_qubits)),
                        control_wires,
                        control_values=[
                            control_value or bool(i % 2) for i, _ in enumerate(control_wires)
                        ],
                    )
                    qml.RY(p[2], 0)
                return np.array([qml.expval(qml.PauliY(i)) for i in range(n_qubits)])

            circ_ad = qml.QNode(circuit, dev, diff_method="adjoint")
            circ_bp = qml.QNode(circuit, dqu, diff_method="backprop")
            jac_ad = np.array(qml.jacobian(circ_ad)(par))
            jac_bp = np.array(qml.jacobian(circ_bp)(par))

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
            qml.RX(extra_param, wires=[0])
            qml.RY(p1, wires=[0])
            qml.RZ(p2, wires=[0])
            qml.RX(p1, wires=[0])
            return qml.expval(qml.PauliZ(0))

        zero_state = np.array([1.0, 0.0])
        cost(reused_p, other_p)

        if ld._new_API:
            spy = mocker.spy(dev, "execute_and_compute_derivatives")
        else:
            spy = mocker.spy(dev.target_device, "adjoint_jacobian")

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

    def test_gradient_repeated_gate_parameters(self, mocker, dev):
        """Tests that repeated use of a free parameter in a multi-parameter gate yields correct
        gradients."""
        params = np.array([0.8, 1.3], requires_grad=True)

        def circuit(params):
            qml.RX(np.array(np.pi / 4, requires_grad=False), wires=[0])
            qml.Rot(params[1], params[0], 2 * params[0], wires=[0])
            return qml.expval(qml.PauliX(0))

        spy_analytic = (
            mocker.spy(dev, "execute_and_compute_derivatives")
            if ld._new_API
            else mocker.spy(dev.target_device, "adjoint_jacobian")
        )
        tol, h = get_tolerance_and_stepsize(dev, step_size=True)

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

        tf_r_dtype = tf.float32 if dev.dtype == np.complex64 else tf.float64
        tol, h = get_tolerance_and_stepsize(dev, step_size=True)

        params1 = tf.Variable(0.3, dtype=tf_r_dtype)
        params2 = tf.Variable(0.4, dtype=tf_r_dtype)

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

        tol, h = get_tolerance_and_stepsize(dev, step_size=True)

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
        dtype = np.float32 if dev.dtype == np.complex64 else np.float64

        if dtype == np.float64:
            from jax import config

            config.update("jax_enable_x64", True)

        def f(params1, params2):
            qml.RX(0.4, wires=[0])
            qml.RZ(params1 * jax.numpy.sqrt(params2), wires=[0])
            qml.RY(jax.numpy.cos(params2), wires=[0])
            return qml.expval(qml.PauliZ(0))

        params1 = jax.numpy.array(0.3, dtype)
        params2 = jax.numpy.array(0.4, dtype)
        tol, h = get_tolerance_and_stepsize(dev, step_size=True)

        qnode_adjoint = QNode(f, dev, interface="jax", diff_method="adjoint")
        qnode_fd = QNode(f, dev, interface="jax", diff_method="finite-diff", h=h)

        grad_adjoint = jax.grad(qnode_adjoint)(params1, params2)
        grad_fd = jax.grad(qnode_fd)(params1, params2)

        assert np.allclose(grad_adjoint, grad_fd, atol=tol)


def circuit_ansatz(params, wires):
    """Circuit ansatz containing all the parametrized gates"""
    qml.QubitStateVector(unitary_group.rvs(2**4, random_state=0)[0], wires=wires)
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
    qml.PauliRot(params[12], "XXYZ", wires=[wires[0], wires[1], wires[2], wires[3]])
    qml.CPhase(params[12], wires=[wires[3], wires[2]])
    qml.IsingXX(params[13], wires=[wires[1], wires[0]])
    qml.IsingXY(params[14], wires=[wires[3], wires[2]])
    qml.IsingYY(params[14], wires=[wires[3], wires[2]])
    qml.IsingZZ(params[14], wires=[wires[2], wires[1]])
    qml.U1(params[15], wires=wires[0])
    qml.U2(params[16], params[17], wires=wires[0])
    qml.U3(params[18], params[19], params[20], wires=wires[1])
    qml.adjoint(qml.CRot(params[21], params[22], params[23], wires=[wires[1], wires[2]]))
    qml.SingleExcitation(params[24], wires=[wires[2], wires[0]])
    qml.DoubleExcitation(params[25], wires=[wires[2], wires[0], wires[1], wires[3]])
    qml.SingleExcitationPlus(params[26], wires=[wires[0], wires[2]])
    qml.SingleExcitationMinus(params[27], wires=[wires[0], wires[2]])
    qml.DoubleExcitationPlus(params[27], wires=[wires[2], wires[0], wires[1], wires[3]])
    qml.DoubleExcitationMinus(params[27], wires=[wires[2], wires[0], wires[1], wires[3]])
    qml.RX(params[28], wires=wires[0])
    qml.RX(params[29], wires=wires[1])


@pytest.mark.usefixtures("use_legacy_and_new_opmath")
def test_tape_qchem(tol):
    """Tests the circuit Ansatz with a QChem Hamiltonian produces correct results"""

    H, qubits = qml.qchem.molecular_hamiltonian(
        ["H", "H"], np.array([0.0, 0.1, 0.0, 0.0, -0.1, 0.0])
    )

    def circuit(params):
        circuit_ansatz(params, wires=range(4))
        return qml.expval(H)

    params = np.linspace(0, 29, 30) * 0.111

    dev_ld = qml.device(device_name, wires=qubits)
    dev_dq = qml.device("default.qubit", wires=qubits)

    circuit_ld = qml.QNode(circuit, dev_ld, diff_method="adjoint")
    circuit_dq = qml.QNode(circuit, dev_dq, diff_method="parameter-shift")
    res = qml.grad(circuit_ld)(params)
    ref = qml.grad(circuit_dq)(params)
    assert np.allclose(res, ref, tol)


@pytest.mark.usefixtures("use_legacy_and_new_opmath")
def test_tape_qchem_sparse(tol):
    """Tests the circuit Ansatz with a QChem Hamiltonian produces correct results"""

    H, qubits = qml.qchem.molecular_hamiltonian(
        ["H", "H"], np.array([0.0, 0.1, 0.0, 0.0, -0.1, 0.0])
    )

    H_sparse = H.sparse_matrix(range(4))

    def circuit_sparse(params):
        circuit_ansatz(params, wires=range(4))
        return qml.expval(qml.SparseHamiltonian(H_sparse, wires=range(4)))

    def circuit(params):
        circuit_ansatz(params, wires=range(4))
        return qml.expval(H)

    params = np.linspace(0, 29, 30) * 0.111

    dev_ld = qml.device(device_name, wires=qubits)
    dev_dq = qml.device("default.qubit", wires=qubits)

    circuit_ld = qml.QNode(circuit_sparse, dev_ld, diff_method="adjoint")
    circuit_dq = qml.QNode(circuit, dev_dq, diff_method="parameter-shift")

    assert np.allclose(qml.grad(circuit_ld)(params), qml.grad(circuit_dq)(params), tol)


custom_wires = ["alice", 3.14, -1, 0]


@pytest.mark.parametrize(
    "returns",
    [
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [0.1],
                [qml.PauliX(wires=custom_wires[0]) @ qml.PauliZ(wires=custom_wires[1])],
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [2.0],
                [qml.PauliX(wires=custom_wires[2]) @ qml.PauliZ(wires=custom_wires[0])],
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [1.1],
                [qml.PauliX(wires=custom_wires[0]) @ qml.PauliZ(wires=custom_wires[2])],
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
    ],
)
def test_adjoint_SparseHamiltonian(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    dev = qml.device(device_name, wires=custom_wires)
    dev_default = qml.device("default.qubit", wires=custom_wires)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qml.expval(returns)

    n_params = 30
    np.random.seed(1337)
    params = np.random.rand(n_params)

    qnode = qml.QNode(circuit, dev, diff_method="adjoint")
    qnode_default = qml.QNode(circuit, dev_default, diff_method="parameter-shift")

    j_device = qml.jacobian(qnode)(params)
    j_default = qml.jacobian(qnode_default)(params)

    assert np.allclose(j_device, j_default)


@pytest.mark.parametrize(
    "returns",
    [
        qml.PauliZ(0),
        qml.PauliX(2),
        qml.PauliZ(0) @ qml.PauliY(3),
        qml.Hadamard(2),
        qml.Hadamard(3) @ qml.PauliZ(2),
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
    dev_lightning = qml.device(device_name, wires=range(4))

    def circuit(params):
        circuit_ansatz(params, wires=range(4))
        return np.array([qml.expval(returns), qml.expval(qml.PauliY(1))])

    n_params = 30
    params = np.linspace(0, 10, n_params)

    qnode_def = qml.QNode(circuit, dev_def)
    qnode_lightning = qml.QNode(circuit, dev_lightning, diff_method="adjoint")

    j_def = qml.jacobian(qnode_def)(params)
    j_lightning = qml.jacobian(qnode_lightning)(params)

    assert np.allclose(j_def, j_lightning)


def test_integration_chunk_observables():
    """Integration tests that compare to default.qubit for a large circuit with multiple expectation values. Expvals are generated in parallelized chunks."""
    num_qubits = 4

    dev_def = qml.device("default.qubit", wires=range(num_qubits))
    dev_lightning = qml.device(device_name, wires=range(num_qubits))
    dev_lightning_batched = qml.device(device_name, wires=range(num_qubits), batch_obs=True)

    def circuit(params):
        circuit_ansatz(params, wires=range(num_qubits))
        return np.array(
            [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
            + [
                qml.expval(
                    qml.Hamiltonian(
                        np.arange(1, num_qubits + 1),
                        [
                            qml.PauliZ(i % num_qubits) @ qml.PauliY((i + 1) % num_qubits)
                            for i in range(num_qubits)
                        ],
                    )
                )
            ]
            + [qml.expval(qml.PauliY(i)) for i in range(num_qubits)]
        )

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
    dev_lightning = qml.device(device_name, wires=custom_wires)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return np.array([qml.expval(returns), qml.expval(qml.PauliY(custom_wires[1]))])

    n_params = 30
    params = np.linspace(0, 10, n_params)

    qnode_def = qml.QNode(circuit, dev_def)
    qnode_lightning = qml.QNode(circuit, dev_lightning, diff_method="adjoint")

    j_def = qml.jacobian(qnode_def)(params)
    j_lightning = qml.jacobian(qnode_lightning)(params)

    assert np.allclose(j_def, j_lightning)


@pytest.mark.skipif(
    device_name not in ("lightning.qubit", "lightning.gpu"),
    reason="Tests only for lightning.qubit and lightning.gpu",
)
@pytest.mark.parametrize(
    "returns",
    [
        (qml.PauliZ(custom_wires[0]),),
        (qml.PauliZ(custom_wires[0]), qml.PauliZ(custom_wires[1])),
        (qml.PauliZ(custom_wires[0]), qml.PauliZ(custom_wires[1]), qml.PauliZ(custom_wires[3])),
        (
            qml.PauliZ(custom_wires[0]),
            qml.PauliZ(custom_wires[1]),
            qml.PauliZ(custom_wires[3]),
            qml.PauliZ(custom_wires[2]),
        ),
        (
            qml.PauliZ(custom_wires[0]) @ qml.PauliY(custom_wires[3]),
            qml.PauliZ(custom_wires[1]) @ qml.PauliY(custom_wires[2]),
        ),
        (qml.PauliZ(custom_wires[0]) @ qml.PauliY(custom_wires[3]), qml.PauliZ(custom_wires[1])),
    ],
)
def test_integration_custom_wires_batching(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    dev_def = qml.device("default.qubit", wires=custom_wires)
    dev_gpu = qml.device(device_name, wires=custom_wires, batch_obs=True)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return [qml.expval(r) for r in returns] + [qml.expval(qml.PauliY(custom_wires[1]))]

    n_params = 30
    np.random.seed(1337)
    params = np.random.rand(n_params)

    qnode_gpu = qml.QNode(circuit, dev_gpu, diff_method="adjoint")
    qnode_def = qml.QNode(circuit, dev_def)

    def convert_to_array_gpu(params):
        return np.hstack(qnode_gpu(params))

    def convert_to_array_def(params):
        return np.hstack(qnode_def(params))

    j_gpu = qml.jacobian(convert_to_array_gpu)(params)
    j_def = qml.jacobian(convert_to_array_def)(params)

    assert np.allclose(j_gpu, j_def, atol=1e-7)


@pytest.mark.skipif(
    device_name not in ("lightning.qubit", "lightning.gpu"),
    reason="Tests only for lightning.qubit and lightning.gpu",
)
@pytest.mark.parametrize(
    "returns",
    [
        (0.5 * qml.PauliZ(custom_wires[0]),),
        (0.5 * qml.PauliZ(custom_wires[0]), qml.PauliZ(custom_wires[1])),
        (
            qml.PauliZ(custom_wires[0]),
            0.5 * qml.PauliZ(custom_wires[1]),
            qml.PauliZ(custom_wires[3]),
        ),
        (
            qml.PauliZ(custom_wires[0]),
            qml.PauliZ(custom_wires[1]),
            qml.PauliZ(custom_wires[3]),
            0.5 * qml.PauliZ(custom_wires[2]),
        ),
        (
            qml.PauliZ(custom_wires[0]) @ qml.PauliY(custom_wires[3]),
            0.5 * qml.PauliZ(custom_wires[1]) @ qml.PauliY(custom_wires[2]),
        ),
        (
            qml.PauliZ(custom_wires[0]) @ qml.PauliY(custom_wires[3]),
            0.5 * qml.PauliZ(custom_wires[1]),
        ),
        (
            0.0 * qml.PauliZ(custom_wires[0]) @ qml.PauliZ(custom_wires[1]),
            1.0 * qml.Identity(10),
            1.2 * qml.PauliZ(custom_wires[2]) @ qml.PauliZ(custom_wires[3]),
        ),
    ],
)
def test_batching_H(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    dev_cpu = qml.device("default.qubit", wires=custom_wires + [10, 72])
    dev_gpu = qml.device(device_name, wires=custom_wires + [10, 72], batch_obs=True)
    dev_gpu_default = qml.device(device_name, wires=custom_wires + [10, 72], batch_obs=False)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qml.math.hstack([qml.expval(r) for r in returns])

    n_params = 30
    np.random.seed(1337)
    params = np.random.rand(n_params)

    qnode_cpu = qml.QNode(circuit, dev_cpu, diff_method="parameter-shift")
    qnode_gpu = qml.QNode(circuit, dev_gpu, diff_method="adjoint")
    qnode_gpu_default = qml.QNode(circuit, dev_gpu_default, diff_method="adjoint")

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

    j_cpu = qml.jacobian(qnode_cpu)(params)
    j_gpu = qml.jacobian(qnode_gpu)(params)
    j_gpu_default = qml.jacobian(qnode_gpu_default)(params)

    assert np.allclose(j_cpu, j_gpu)
    assert np.allclose(j_gpu, j_gpu_default)


@pytest.fixture(scope="session")
def create_xyz_file(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tmp")
    file = directory / "h2.xyz"
    file.write_text("""2\nH2, Unoptimized\nH  1.0 0.0 0.0\nH -1.0 0.0 0.0""")
    yield file


@pytest.mark.parametrize("batches", [False, True, 1, 2, 3, 4])
def test_integration_H2_Hamiltonian(create_xyz_file, batches):
    _ = pytest.importorskip("openfermionpyscf")
    n_electrons = 2
    np.random.seed(1337)

    str_path = create_xyz_file
    symbols, coordinates = qml.qchem.read_structure(str(str_path), outpath=str(str_path.parent))

    H, qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        method="pyscf",
        active_electrons=n_electrons,
        name="h2",
        outpath=str(str_path.parent),
        load_data=True,
    )
    hf_state = qml.qchem.hf_state(n_electrons, qubits)
    _, doubles = qml.qchem.excitations(n_electrons, qubits)

    # Choose different batching supports here
    dev = qml.device(device_name, wires=qubits, batch_obs=batches)
    dev_comp = qml.device("default.qubit", wires=qubits)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(params, excitations):
        qml.BasisState(hf_state, wires=H.wires)
        for i, excitation in enumerate(excitations):
            if len(excitation) == 4:
                qml.DoubleExcitation(params[i], wires=excitation)
            else:
                qml.SingleExcitation(params[i], wires=excitation)
        return qml.expval(H)

    @qml.qnode(dev_comp, diff_method="parameter-shift")
    def circuit_compare(params, excitations):
        qml.BasisState(hf_state, wires=H.wires)

        for i, excitation in enumerate(excitations):
            if len(excitation) == 4:
                qml.DoubleExcitation(params[i], wires=excitation)
            else:
                qml.SingleExcitation(params[i], wires=excitation)
        return qml.expval(H)

    jac_func = qml.jacobian(circuit)
    jac_func_comp = qml.jacobian(circuit_compare)

    params = qml.numpy.array([0.0] * len(doubles), requires_grad=True)
    jacs = jac_func(params, excitations=doubles)
    jacs_comp = jac_func_comp(params, excitations=doubles)

    assert np.allclose(jacs, jacs_comp)


@pytest.mark.parametrize("n_targets", range(1, 6))
def test_qubit_unitary(n_targets):
    """Tests that ``qml.QubitUnitary`` can be included in circuits differentiated with the adjoint method."""
    n_wires = 6
    dev = qml.device(device_name, wires=n_wires)
    dev_def = qml.device("default.qubit", wires=n_wires)

    np.random.seed(1337)
    init_state = np.random.rand(2**n_wires) + 1j * np.random.rand(2**n_wires)
    init_state /= np.linalg.norm(init_state)
    init_state = np.array(init_state, requires_grad=False)
    U = np.random.rand(2**n_targets, 2**n_targets) + 1j * np.random.rand(2**n_targets, 2**n_targets)
    U, _ = np.linalg.qr(U)
    U = np.array(U, requires_grad=False)

    obs = qml.operation.Tensor(*(qml.PauliZ(i) for i in range(n_wires)))

    par = 2 * np.pi * np.random.rand(n_wires)

    def circuit(x):
        qml.StatePrep(init_state, wires=range(n_wires))
        for i in range(n_wires // 2):
            qml.RY(x[i], wires=i)
        qml.QubitUnitary(U, wires=range(n_targets))
        for i in range(n_wires // 2, n_wires):
            qml.RY(x[i], wires=i)
        return qml.expval(obs)

    circ = qml.QNode(circuit, dev, diff_method="adjoint")
    circ_ps = qml.QNode(circuit, dev, diff_method="parameter-shift")
    circ_def = qml.QNode(circuit, dev_def, diff_method="adjoint")
    jac = qml.jacobian(circ)(par)
    jac_ps = qml.jacobian(circ_ps)(par)
    jac_def = qml.jacobian(circ_def)(par)

    assert jac.size == n_wires
    assert not np.allclose(jac, 0.0)
    assert np.allclose(jac, jac_ps)
    assert np.allclose(jac, jac_def)


@pytest.mark.parametrize("n_targets", [1, 2])
def test_diff_qubit_unitary(n_targets):
    """Tests that ``qml.QubitUnitary`` can be differentiated with the adjoint method."""
    n_wires = 6
    dev = qml.device(device_name, wires=n_wires)
    dev_def = qml.device("default.qubit", wires=n_wires)
    _, h = get_tolerance_and_stepsize(dev, step_size=True)

    np.random.seed(1337)
    init_state = np.random.rand(2**n_wires) + 1j * np.random.rand(2**n_wires)
    init_state /= np.linalg.norm(init_state)
    init_state = np.array(init_state, requires_grad=False)
    U = np.random.rand(2**n_targets, 2**n_targets) + 1j * np.random.rand(2**n_targets, 2**n_targets)
    U, _ = np.linalg.qr(U)
    U = np.array(U, requires_grad=False)

    obs = qml.operation.Tensor(*(qml.PauliZ(i) for i in range(n_wires)))

    par = 2 * np.pi * np.random.rand(n_wires)

    def circuit(x, u_mat):
        qml.StatePrep(init_state, wires=range(n_wires))
        for i in range(n_wires // 2):
            qml.RY(x[i], wires=i)
        qml.QubitUnitary(u_mat, wires=range(n_targets))
        for i in range(n_wires // 2, n_wires):
            qml.RY(x[i], wires=i)
        return qml.expval(obs)

    circ = qml.QNode(circuit, dev, diff_method="adjoint")
    circ_def = qml.QNode(circuit, dev_def, diff_method="adjoint")
    circ_fd = qml.QNode(circuit, dev, diff_method="finite-diff", h=h)
    circ_ps = qml.QNode(circuit, dev, diff_method="parameter-shift")
    jacs = qml.jacobian(circ)(par, U)
    jacs_def = qml.jacobian(circ_def)(par, U)
    jacs_fd = qml.jacobian(circ_fd)(par, U)
    jacs_ps = qml.jacobian(circ_ps)(par, U)

    for jac, jac_def, jac_fd, jac_ps in zip(jacs, jacs_def, jacs_fd, jacs_ps):
        assert not np.allclose(jac, 0.0)
        assert np.allclose(jac, jac_fd)
        assert np.allclose(jac, jac_ps)
        assert np.allclose(jac, jac_def)
