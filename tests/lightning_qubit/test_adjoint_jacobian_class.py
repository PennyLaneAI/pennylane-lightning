# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Unit tests for the adjoint Jacobian and VJP methods.
"""
import math

import pennylane as qml
import pytest
from conftest import (  # tested device
    LightningAdjointJacobian,
    LightningDevice,
    LightningStateVector,
    device_name,
)
from pennylane import numpy as np
from pennylane.tape import QuantumScript
from scipy.stats import unitary_group

if not LightningDevice._new_API:
    pytest.skip(
        "Exclusive tests for new API backends LightningAdjointJacobian class. Skipping.",
        allow_module_level=True,
    )

if device_name == "lightning.gpu":
    pytest.skip("LGPU new API in WIP.  Skipping.", allow_module_level=True)

if device_name == "lightning.tensor":
    pytest.skip("Skipping tests for the LightningTensor class.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


I, X, Y, Z = (
    np.eye(2),
    qml.PauliX.compute_matrix(),
    qml.PauliY.compute_matrix(),
    qml.PauliZ.compute_matrix(),
)

kokkos_args = [None]
if device_name == "lightning.kokkos":
    from pennylane_lightning.lightning_kokkos_ops import InitializationSettings

    kokkos_args += [InitializationSettings().set_num_threads(2)]


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


def test_initialization(lightning_sv):
    """Tests for the initialization of the LightningAdjointJacobian class."""
    statevector = lightning_sv(num_wires=5)
    aj = LightningAdjointJacobian(statevector)

    assert aj.qubit_state is statevector
    assert aj.state is statevector.state_vector
    assert aj.dtype == statevector.dtype


class TestAdjointJacobian:
    """Tests for the adjoint Jacobian functionality"""

    def test_finite_shots_error(self, lightning_sv):
        """Tests error raised when finite shots specified"""
        tape = qml.tape.QuantumTape(measurements=[qml.expval(qml.PauliZ(0))], shots=1)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Requested adjoint differentiation to be computed with finite shots.",
        ):
            self.calculate_jacobian(lightning_sv(num_wires=1), tape)

    @staticmethod
    def calculate_jacobian(statevector, tape):
        statevector = statevector.get_final_state(tape)

        return LightningAdjointJacobian(statevector).calculate_jacobian(tape)

    def test_not_supported_state(self, lightning_sv):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        supported"""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.1, wires=0)
            qml.state()

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Adjoint differentiation method does not support measurement StateMP.",
        ):
            self.calculate_jacobian(lightning_sv(num_wires=3), tape)

    def test_empty_measurements(self, lightning_sv):
        """Tests if an empty array is returned when the measurements of the tape is empty."""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])

        jac = self.calculate_jacobian(lightning_sv(num_wires=3), tape)
        assert len(jac) == 0

    def test_empty_trainable_params(self, lightning_sv):
        """Tests if an empty array is returned when the number trainable params is zero."""

        with qml.tape.QuantumTape() as tape:
            qml.X(wires=[0])
            qml.expval(qml.PauliZ(0))

        jac = self.calculate_jacobian(lightning_sv(num_wires=3), tape)
        assert len(jac) == 0

    def test_not_expectation_return_type(self, lightning_sv):
        """Tests if an empty array is returned when the number trainable params is zero."""

        with qml.tape.QuantumTape() as tape:
            qml.X(wires=[0])
            qml.RX(0.4, wires=[0])
            qml.var(qml.PauliZ(1))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Adjoint differentiation method does not support expectation return type "
            "mixed with other return types",
        ):
            self.calculate_jacobian(lightning_sv(num_wires=1), tape)

    @pytest.mark.skipif(
        device_name != "lightning.qubit",
        reason="N-controlled operations only implemented in lightning.qubit.",
    )
    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
    @pytest.mark.parametrize("par", [-np.pi / 7, np.pi / 5, 2 * np.pi / 3])
    def test_phaseshift_gradient(self, n_qubits, par, tol, lightning_sv):
        """Test that the gradient of the phaseshift gate matches the exact analytic formula."""
        par = np.array(par)

        init_state = np.zeros(2**n_qubits)
        init_state[-2::] = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], requires_grad=False)

        with qml.tape.QuantumTape() as tape:
            qml.StatePrep(init_state, wires=range(n_qubits))
            qml.ctrl(qml.PhaseShift(par, wires=n_qubits - 1), range(0, n_qubits - 1))
            qml.expval(qml.PauliY(n_qubits - 1))

        tape.trainable_params = {1}

        expected = np.cos(par)
        result = self.calculate_jacobian(lightning_sv(num_wires=n_qubits), tape)

        assert np.allclose(expected, result, atol=tol, rtol=0)

    @pytest.mark.parametrize("par", [1, -2, 1.623, -0.051, 0])  # integers, floats, zero
    def test_ry_gradient(self, par, tol, lightning_sv):
        """Test that the gradient of the RY gate matches the exact analytic formula."""
        with qml.tape.QuantumTape() as tape:
            qml.RY(par, wires=[0])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {0}

        expected = np.cos(par)
        result = self.calculate_jacobian(lightning_sv(num_wires=3), tape)

        assert np.allclose(expected, result, atol=tol, rtol=0)

    def test_rx_gradient(self, tol, lightning_sv):
        """Test that the gradient of the RX gate matches the known formula."""
        par = 0.7418

        with qml.tape.QuantumTape() as tape:
            qml.RX(par, wires=0)
            qml.expval(qml.PauliZ(0))

        expected = -np.sin(par)
        result = self.calculate_jacobian(lightning_sv(num_wires=3), tape)

        assert np.allclose(expected, result, atol=tol, rtol=0)

    def test_multiple_rx_gradient_pauliz(self, tol, lightning_sv):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result."""
        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        with qml.tape.QuantumTape() as tape:
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[2], wires=2)

            for idx in range(3):
                qml.expval(qml.PauliZ(idx))

        expected = -np.diag(np.sin(params))
        result = self.calculate_jacobian(lightning_sv(num_wires=3), tape)

        assert np.allclose(expected, result, atol=tol, rtol=0)

    def test_multiple_rx_gradient_hermitian(self, tol, lightning_sv):
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

        expected = -np.diag(np.sin(params))
        result = self.calculate_jacobian(lightning_sv(num_wires=3), tape)

        assert np.allclose(expected, result, atol=tol, rtol=0)

    def test_multiple_rx_gradient_expval_hermitian(self, tol, lightning_sv):
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

        expected = np.array(
            [-np.sin(params[0]) * np.cos(params[2]), 0, -np.cos(params[0]) * np.sin(params[2])]
        )
        result = self.calculate_jacobian(lightning_sv(num_wires=3), tape)

        assert np.allclose(expected, result, atol=tol, rtol=0)

    @pytest.mark.usefixtures("use_legacy_and_new_opmath")
    def test_multiple_rx_gradient_expval_hamiltonian(self, tol, lightning_sv):
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

        expected = (
            0.3 * np.array([-np.sin(params[0]), 0, 0])
            + 0.3 * np.array([0, -np.sin(params[1]), 0])
            + 0.4
            * np.array(
                [-np.sin(params[0]) * np.cos(params[2]), 0, -np.cos(params[0]) * np.sin(params[2])]
            )
        )
        result = self.calculate_jacobian(lightning_sv(num_wires=3), tape)

        assert np.allclose(expected, result, atol=tol, rtol=0)


def simple_circuit_ansatz(params, wires):
    """Circuit ansatz containing a large circuit"""
    return [
        qml.QubitStateVector(unitary_group.rvs(2**4, random_state=0)[0], wires=wires),
        qml.RX(params[0], wires=wires[0]),
        qml.RY(params[1], wires=wires[1]),
        qml.RZ(params[2], wires=wires[3]),
        qml.CRX(params[3], wires=[wires[3], wires[0]]),
        qml.CRY(params[4], wires=[wires[2], wires[1]]),
        qml.CRZ(params[5], wires=[wires[2], wires[1]]),
        qml.PhaseShift(params[6], wires=wires[2]),
        qml.MultiRZ(params[7], wires=[wires[0], wires[1]]),
        qml.IsingXX(params[8], wires=[wires[1], wires[0]]),
        qml.IsingXY(params[9], wires=[wires[3], wires[2]]),
        qml.IsingYY(params[10], wires=[wires[3], wires[2]]),
        qml.IsingZZ(params[11], wires=[wires[2], wires[1]]),
    ]


# fmt: off
expected_jac_simple_circuit_ansatz = np.array([[
        -9.77334961e-03, -1.30657957e-01, -1.66427588e-17,
         1.66379059e-01, -3.56645181e-02, -1.95066583e-01,
        -1.65685685e-16,  4.85722573e-17, -3.56748062e-01,
        -1.73472348e-17,  2.94902991e-17, -2.44233119e-18],
       [-5.55111512e-17,  2.22411150e-02, -3.47974591e-17,
         6.93889390e-18,  1.50214879e-01, -1.86416270e-01,
        -1.29381272e-16,  0.00000000e+00,  8.63611865e-02,
        -2.42861287e-17,  5.55111512e-17,  1.90053083e-17]])
# fmt: on


def test_large_circuit(tol, lightning_sv):
    """Test the adjoint Jacobian pipeline for a large circuit with multiple expectation values.
    When batch_obs=True, expvals are generated in parallelized chunks.
    Expected results were calculated with default.qubit"""
    statevector = lightning_sv(num_wires=4)

    n_params = 12
    params = np.linspace(0, 10, n_params)
    tape = QuantumScript(
        simple_circuit_ansatz(params, wires=range(4)), [qml.expval(qml.PauliZ(i)) for i in range(2)]
    )

    tape.trainable_params = set(range(1, n_params + 1))

    statevector = statevector.get_final_state(tape)

    jac = LightningAdjointJacobian(statevector).calculate_jacobian(tape)
    batched_jac = LightningAdjointJacobian(statevector, batch_obs=True).calculate_jacobian(tape)

    assert np.allclose(expected_jac_simple_circuit_ansatz, jac, atol=tol)
    assert np.allclose(expected_jac_simple_circuit_ansatz, batched_jac, atol=tol)


class TestVectorJacobianProduct:
    """Tests for the `vjp` functionality"""

    @staticmethod
    def calculate_jacobian(statevector, tape):
        statevector = statevector.get_final_state(tape)

        return LightningAdjointJacobian(statevector).calculate_jacobian(tape)

    @staticmethod
    def calculate_vjp(statevector, tape, vector):
        statevector = statevector.get_final_state(tape)

        return LightningAdjointJacobian(statevector).calculate_vjp(tape, vector)

    @pytest.mark.usefixtures("use_legacy_and_new_opmath")
    def test_multiple_measurements(self, tol, lightning_sv):
        """Tests provides correct answer when provided multiple measurements."""
        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliX(0))
            qml.expval(qml.PauliY(1))
            qml.expval(qml.PauliZ(1))

        dy = np.array([1.0, 2.0, 3.0])
        tape1.trainable_params = {1, 2, 3}

        with qml.tape.QuantumTape() as tape2:
            ham = qml.Hamiltonian(dy, [qml.PauliX(0), qml.PauliY(1), qml.PauliY(1)])
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(ham)

        tape2.trainable_params = {1, 2, 3}

        statevector = lightning_sv(num_wires=2)
        result_vjp = self.calculate_vjp(statevector, tape1, dy)

        statevector.reset_state()

        result_jac = self.calculate_jacobian(statevector, tape2)

        assert np.allclose(result_vjp, result_jac, atol=tol, rtol=0)

    def test_wrong_dy_expval(self, lightning_sv):
        """Tests raise an exception when dy is incorrect"""
        statevector = lightning_sv(num_wires=2)

        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliX(0))
            qml.expval(qml.PauliY(1))
            qml.expval(qml.PauliZ(1))

        dy1 = np.array([1.0, 2.0])
        tape1.trainable_params = {1, 2, 3}

        with pytest.raises(
            ValueError, match="Number of observables in the tape must be the same as"
        ):
            self.calculate_vjp(statevector, tape1, dy1)

        dy2 = np.array([1.0 + 3.0j, 0.3 + 2.0j, 0.5 + 0.1j])
        with pytest.raises(
            ValueError, match="The vjp method only works with a real-valued grad_vec"
        ):
            self.calculate_vjp(statevector, tape1, dy2)

    def test_finite_shots_error(self, lightning_sv):
        """Tests error raised when finite shots specified"""

        statevector = lightning_sv(num_wires=2)

        tape = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))], shots=1)
        dy = np.array([1.0])

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Requested adjoint differentiation to be computed with finite shots.",
        ):
            self.calculate_vjp(statevector, tape, dy)

    def test_hermitian_expectation(self, tol, lightning_sv):
        statevector = lightning_sv(num_wires=2)

        obs = np.array([[1, 0], [0, -1]], dtype=statevector.dtype, requires_grad=False)
        dy = np.array([0.8])

        for x in np.linspace(-2 * math.pi, 2 * math.pi, 7):
            with qml.tape.QuantumTape() as tape:
                qml.RY(x, wires=(0,))
                qml.expval(qml.Hermitian(obs, wires=(0,)))
            tape.trainable_params = {0}

            statevector.reset_state()
            vjp = self.calculate_vjp(statevector, tape, dy)

            assert np.allclose(vjp, -0.8 * np.sin(x), atol=tol)

    def test_hermitian_tensor_expectation(self, tol, lightning_sv):
        statevector = lightning_sv(num_wires=2)

        obs = np.array([[1, 0], [0, -1]], dtype=statevector.dtype, requires_grad=False)
        dy = np.array([0.8])

        for x in np.linspace(-2 * math.pi, 2 * math.pi, 7):
            with qml.tape.QuantumTape() as tape:
                qml.RY(x, wires=(0,))
                qml.expval(qml.Hermitian(obs, wires=(0,)) @ qml.PauliZ(wires=1))
            tape.trainable_params = {0}

            statevector.reset_state()
            vjp = self.calculate_vjp(statevector, tape, dy)

            assert np.allclose(vjp, -0.8 * np.sin(x), atol=tol)

    def test_state_measurement_not_supported(self, lightning_sv):
        """Tests raise an exception when dy is incorrect"""
        statevector = lightning_sv(num_wires=2)

        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.state()

        tape.trainable_params = {1, 2, 3}

        dy = np.ones(3, dtype=statevector.dtype)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Adjoint differentiation does not support State measurements.",
        ):
            self.calculate_vjp(statevector, tape, dy)

    def test_no_trainable_parameters(self, lightning_sv):
        """A tape with no trainable parameters will simply return None"""
        statevector = lightning_sv(num_wires=2)
        x = 0.4

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {}
        dy = np.array([1.0])
        vjp = self.calculate_vjp(statevector, tape, dy)

        assert len(vjp) == 0

    def test_zero_dy(self, lightning_sv):
        """A zero dy vector will return no tapes and a zero matrix"""
        statevector = lightning_sv(num_wires=2)
        x = 0.4
        y = 0.6

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.RX(y, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {0, 1}
        dy = np.array([0.0])

        vjp = self.calculate_vjp(statevector, tape, dy)

        assert np.all(vjp == np.zeros([len(tape.trainable_params)]))

    def test_empty_dy(self, tol, lightning_sv):
        """A zero dy vector will return no tapes and a zero matrix"""
        statevector = lightning_sv(num_wires=2)
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape.trainable_params = {0, 1}
        dy = np.array(1.0)

        vjp = self.calculate_vjp(statevector, tape, dy)

        expected = np.array([-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)])
        assert np.allclose(vjp, expected, atol=tol, rtol=0)

    def test_single_expectation_value(self, tol, lightning_sv):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        statevector = lightning_sv(num_wires=2)
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape.trainable_params = {0, 1}
        dy = np.array([1.0])

        vjp = self.calculate_vjp(statevector, tape, dy)

        expected = np.array([-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)])
        assert np.allclose(vjp, expected, atol=tol, rtol=0)

    def test_multiple_expectation_values(self, tol, lightning_sv):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        statevector = lightning_sv(num_wires=2)
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        tape.trainable_params = {0, 1}
        dy = np.array([1.0, 2.0])

        vjp = self.calculate_vjp(statevector, tape, dy)

        expected = np.array([-np.sin(x), 2 * np.cos(y)])
        assert np.allclose(vjp, expected, atol=tol, rtol=0)
