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
Tests for LightningQubit's compute_derivatives (adjoint jacobian method).
"""
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.devices import ExecutionConfig, DefaultQubit
from pennylane_lightning.core._preprocess import preprocess, validate_and_expand_adjoint

from pennylane.devices.qubit.preprocess import preprocess as preprocess_default
from pennylane_lightning.lightning_qubit import LightningQubit


if not LightningQubit._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

AdjointConfig = ExecutionConfig(use_device_gradient=True, gradient_method="adjoint")


class TestAdjointJacobianSupport:
    """Tests adjoint jacobian support"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return LightningQubit(c_dtype=request.param)

    def test_CRot_unsupported(self, dev):
        """Tests if a QuantumFunctionError is raised for an unsupported operation, i.e.,
        multi-parameter operations that are not qml.Rot"""

        with qml.tape.QuantumTape() as tape:
            qml.CRot(0.1, 0.2, 0.3, wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with pytest.raises(
            qml.QuantumFunctionError, match="The CRot operation is not supported using the"
        ):
            dev.supports_derivatives(AdjointConfig, circuit=tape)

    def test_Projector_unsupported(self, dev):
        """Tests if a QuantumFunctionError is raised for a Projector observable"""
        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0, 1], wires=[0, 1]))

        with pytest.raises(
            qml.QuantumFunctionError, match="differentiation method does not support the Projector"
        ):
            dev.supports_derivatives(AdjointConfig, circuit=tape)

        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0], wires=[0]) @ qml.PauliZ(0))

        with pytest.raises(
            qml.QuantumFunctionError, match="differentiation method does not support the Projector"
        ):
            dev.supports_derivatives(AdjointConfig, circuit=tape)

    def test_state_return_type_unsupported(self, dev):
        """Tests if a DeviceError is raised when the return type is State"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.state()

        tape.trainable_params = {0}

        with pytest.raises(
            qml.QuantumFunctionError, match="This method does not support state vector return type"
        ):
            dev.supports_derivatives(AdjointConfig, circuit=tape)


class TestAdjointJacobianComputeDerivatives:
    """Tests for the adjoint_jacobian method"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return LightningQubit(c_dtype=request.param)

    @staticmethod
    def process_and_compute_derivatives(dev, tape, trainable_params=None):
        if trainable_params != None:
            tape.trainable_params = trainable_params
        program, _ = preprocess(AdjointConfig)
        res_tapes, _ = program([tape])
        tape = res_tapes[0]
        if trainable_params != None:
            tape.trainable_params = trainable_params
        results = dev.compute_derivatives(tape, AdjointConfig)
        return results

    @staticmethod
    def calculate_reference(tape, trainable_params=None):
        dev = DefaultQubit(max_workers=1)
        if trainable_params != None:
            tape.trainable_params = trainable_params
        program, _ = preprocess_default(AdjointConfig)
        res_tapes, _ = program([tape])
        tape = res_tapes[0]
        if trainable_params != None:
            tape.trainable_params = trainable_params
        results = dev.compute_derivatives(tape, AdjointConfig)
        return results

    def test_empty_measurements(self, dev):
        """Tests if an empty array is returned when the measurements of the tape is empty."""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])

        calculated_val = self.process_and_compute_derivatives(dev, tape)
        assert len(calculated_val) == 0

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ])
    def test_pauli_gradient(self, G, theta, dev):
        """Tests gradient for Pauli gates."""

        random_state = np.array(
            [0.43593284 - 0.02945156j, 0.40812291 + 0.80158023j], requires_grad=False
        )

        tape = qml.tape.QuantumScript(
            [G(theta, 0)], [qml.expval(qml.PauliZ(0))], [qml.QubitStateVector(random_state, 0)]
        )
        trainable_params = {1}

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape, trainable_params)
        reference_val = self.calculate_reference(tape, trainable_params)

        tol = 1e-6 if dev.C_DTYPE == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    def test_Rot_gradient(self, theta, dev):
        """Tests an arbitrary Euler-angle-parameterized gate."""

        params = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            qml.Rot(*params, wires=[0])
            qml.expval(qml.PauliZ(0))

        trainable_params = {1, 2, 3}

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape, trainable_params)
        reference_val = self.calculate_reference(tape, trainable_params)

        tol = 1e-6 if dev.C_DTYPE == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("par", [1, -2, 1.623, -0.051, 0])  # integers, floats, zero
    def test_ry_gradient(self, par, tol, dev):
        """Tests that the gradient of the RY gate matches the exact analytic formula."""

        with qml.tape.QuantumTape() as tape:
            qml.RY(par, wires=[0])
            qml.expval(qml.PauliX(0))

        trainable_params = {0}

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape, trainable_params)
        reference_val = np.cos(par)

        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("par", [1, -2, 1.623, -0.051, 0])
    def test_rx_gradient(self, par, tol, dev):
        """Tests that the gradient of the RX gate matches the known formula."""

        with qml.tape.QuantumTape() as tape:
            qml.RX(par, wires=0)
            qml.expval(qml.PauliZ(0))

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape)
        reference_val = -np.sin(par)

        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

    def test_multiple_rx_gradient_PauliZ(self, tol, dev):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result."""
        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        with qml.tape.QuantumTape() as tape:
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[2], wires=2)

            for idx in range(3):
                qml.expval(qml.PauliZ(idx))

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape)
        reference_val = -np.diag(np.sin(params))

        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

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

        trainable_params = {0, 1, 2}

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape, trainable_params)
        reference_val = -np.diag(np.sin(params))
        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

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

        trainable_params = {0, 1, 2}

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape, trainable_params)
        reference_val = np.array(
            [-np.sin(params[0]) * np.cos(params[2]), 0, -np.cos(params[0]) * np.sin(params[2])]
        )

        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

    qubit_ops = [getattr(qml, name) for name in qml.ops._qubit__ops__]
    ops = {qml.RX, qml.RY, qml.RZ, qml.PhaseShift, qml.CRX, qml.CRY, qml.CRZ, qml.Rot}

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

        trainable_params = {0, 1, 2}

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape, trainable_params)
        reference_val = (
            0.3 * np.array([-np.sin(params[0]), 0, 0])
            + 0.3 * np.array([0, -np.sin(params[1]), 0])
            + 0.4
            * np.array(
                [-np.sin(params[0]) * np.cos(params[2]), 0, -np.cos(params[0]) * np.sin(params[2])]
            )
        )

        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

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
    def test_gradients_PauliZ(self, op, obs, dev):
        """Tests that the gradients of circuits match a reference method."""

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

        trainable_params = set(range(1, 1 + op.num_params))

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape, trainable_params)
        reference_val = self.calculate_reference(tape, trainable_params)

        tol = 1e-6 if dev.C_DTYPE == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

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
        """Tests that the gradients of circuits match a reference method."""

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.RX(0.543, wires=0)
            qml.CNOT(wires=[0, 1])
            op
            qml.Rot(1.3, -2.3, 0.5, wires=[0])
            qml.RZ(-0.5, wires=0)
            qml.adjoint(qml.RY(0.5, wires=1), lazy=False)
            qml.CNOT(wires=[0, 1])

            qml.expval(
                qml.Hermitian(
                    [[0, 0, 1, 1], [0, 1, 2, 1], [1, 2, 1, 0], [1, 1, 0, 0]], wires=[0, 1]
                )
            )

        trainable_params = set(range(1, 1 + op.num_params))

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape, trainable_params)
        reference_val = self.calculate_reference(tape, trainable_params)

        tol = 1e-6 if dev.C_DTYPE == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

    def test_gradient_gate_with_identity(self, dev):
        """Tests gates with multiple free parameters and Identity."""

        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.Identity(0))

        trainable_params = {1, 2, 3}

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape, trainable_params)
        reference_val = self.calculate_reference(tape, trainable_params)

        tol = 1e-6 if dev.C_DTYPE == np.complex64 else 1e-7

        # the different methods agree
        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

    def test_gradient_gate_with_multi_wire_identity(self, dev):
        """Tests gates with multiple free parameters and multi-wire Identity."""

        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[1])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.Identity([0, 1]))

        trainable_params = {1, 2, 3}

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape, trainable_params)
        reference_val = self.calculate_reference(tape, trainable_params)

        tol = 1e-6 if dev.C_DTYPE == np.complex64 else 1e-7

        # the different methods agree
        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

    def test_gradient_gate_with_multiple_parameters_PauliZ(self, dev):
        """Tests gates with multiple free parameters and PauliZ."""

        x, y, z = [0.5, 0.3, -0.7]

        tape = qml.tape.QuantumScript(
            [qml.RX(0.4, wires=[0]), qml.Rot(x, y, z, wires=[0]), qml.RY(-0.2, wires=[0])],
            [qml.expval(qml.PauliZ(0))],
        )

        trainable_params = {1, 2, 3}

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape, trainable_params)
        reference_val = self.calculate_reference(tape, trainable_params)

        tol = 1e-6 if dev.C_DTYPE == np.complex64 else 1e-7

        # gradient has the correct shape and every element is nonzero
        assert len(calculated_val) == 3
        assert np.count_nonzero(calculated_val) == 3
        # the different methods agree
        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

    def test_gradient_gate_with_multiple_parameters_hermitian(self, dev):
        """Tests gates with multiple free parameters and an Hermitian operator."""
        x, y, z = [0.5, 0.3, -0.7]

        tape = qml.tape.QuantumScript(
            [qml.RX(0.4, wires=[0]), qml.Rot(x, y, z, wires=[0]), qml.RY(-0.2, wires=[0])],
            [qml.expval(qml.Hermitian([[0, 1], [1, 1]], wires=0))],
        )

        trainable_params = {1, 2, 3}

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape, trainable_params)
        reference_val = self.calculate_reference(tape, trainable_params)

        tol = 1e-6 if dev.C_DTYPE == np.complex64 else 1e-7

        # gradient has the correct shape and every element is nonzero
        assert len(calculated_val) == 3
        assert np.count_nonzero(calculated_val) == 3
        # the different methods agree
        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

    def test_gradient_gate_with_multiple_parameters_hamiltonian(self, dev):
        """Tests gates with multiple free parameters and a Hamiltonian."""
        x, y, z = [0.5, 0.3, -0.7]

        ham = qml.Hamiltonian(
            [1.0, 0.3, 0.3], [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0), qml.PauliZ(1)]
        )

        tape = qml.tape.QuantumScript(
            [qml.RX(0.4, wires=[0]), qml.Rot(x, y, z, wires=[0]), qml.RY(-0.2, wires=[0])],
            [qml.expval(ham)],
        )

        trainable_params = {1, 2, 3}

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape, trainable_params)
        reference_val = (np.array(-0.00764278), np.array(-0.09487448), np.array(0.03287789))

        tol = 1e-6 if dev.C_DTYPE == np.complex64 else 1e-7

        # gradient has the correct shape and every element is nonzero
        assert len(calculated_val) == 3
        assert np.count_nonzero(calculated_val) == 3
        # the different methods agree
        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)


@pytest.mark.skipif(not LightningQubit._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
class TestAdjointDifferentiation:
    """Tests adjoint differentiation integration with LightningQubit."""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return LightningQubit(c_dtype=request.param)

    def test_derivatives_single_circuit(self, dev):
        """Tests derivatives with a single circuit."""
        x = np.array(np.pi / 7)
        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        qs, _ = validate_and_expand_adjoint(qs)
        qs = qs[0]
        expected_grad = -qml.math.sin(x)
        actual_grad = dev.compute_derivatives(qs, AdjointConfig)
        assert isinstance(actual_grad, np.ndarray)
        assert actual_grad.shape == ()  # pylint: disable=no-member
        assert np.isclose(actual_grad, expected_grad)

        expected_val = qml.math.cos(x)
        actual_val, actual_grad = dev.execute_and_compute_derivatives(qs, AdjointConfig)
        assert np.isclose(actual_val, expected_val)
        assert np.isclose(actual_grad, expected_grad)

    def test_derivatives_list_with_single_circuit(self, dev):
        """Tests a basic example with a batch containing a single circuit."""
        x = np.array(np.pi / 7)
        qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        qs, _ = validate_and_expand_adjoint(qs)
        qs = qs[0]
        expected_grad = -qml.math.sin(x)
        actual_grad = dev.compute_derivatives([qs], AdjointConfig)
        assert isinstance(actual_grad, tuple)
        assert isinstance(actual_grad[0], np.ndarray)
        assert np.isclose(actual_grad[0], expected_grad)

        expected_val = qml.math.cos(x)
        actual_val, actual_grad = dev.execute_and_compute_derivatives([qs], AdjointConfig)
        assert np.isclose(expected_val, actual_val[0])
        assert np.isclose(expected_grad, actual_grad[0])

    def test_derivatives_many_tapes_many_results(self, dev):
        """Tests a basic example with a batch of circuits of varying return shapes."""
        x = np.array(np.pi / 7)
        single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        multi_meas = qml.tape.QuantumScript(
            [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
        )
        expected_grad = (-qml.math.sin(x), (qml.math.cos(x), -qml.math.sin(x)))
        actual_grad = dev.compute_derivatives([single_meas, multi_meas], AdjointConfig)
        assert np.isclose(actual_grad[0], expected_grad[0])
        assert isinstance(actual_grad[1], tuple)
        assert qml.math.allclose(actual_grad[1], expected_grad[1])

    def test_derivatives_integration(self, dev):
        """Tests the expected workflow done by a calling method."""
        x = np.array(np.pi / 7)
        expected_grad = (-qml.math.sin(x), (qml.math.cos(x), -qml.math.sin(x)))
        single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        multi_meas = qml.tape.QuantumScript(
            [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
        )

        program, new_ec = dev.preprocess(AdjointConfig)
        circuits, _ = program([single_meas, multi_meas])
        actual_grad = dev.compute_derivatives(circuits, AdjointConfig)

        assert new_ec.use_device_gradient
        assert new_ec.grad_on_execution

        assert np.isclose(actual_grad[0], expected_grad[0])
        assert isinstance(actual_grad[1], tuple)
        assert qml.math.allclose(actual_grad[1], expected_grad[1])

    # def test_jvps_single_circuit(self, max_workers):
    #     """Tests jvps with a single circuit."""
    #     dev = DefaultQubit(max_workers=max_workers)
    #     x = np.array(np.pi / 7)
    #     tangent = (0.456,)

    #     qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])

    #     qs, _ = validate_and_expand_adjoint(qs)
    #     qs = qs[0]

    #     expected_grad = -qml.math.sin(x) * tangent[0]
    #     actual_grad = dev.compute_jvp(qs, tangent, self.ec)
    #     assert isinstance(actual_grad, np.ndarray)
    #     assert actual_grad.shape == ()  # pylint: disable=no-member
    #     assert np.isclose(actual_grad, expected_grad)

    #     expected_val = qml.math.cos(x)
    #     actual_val, actual_grad = dev.execute_and_compute_jvp(qs, tangent, self.ec)
    #     assert np.isclose(actual_val, expected_val)
    #     assert np.isclose(actual_grad, expected_grad)

    # def test_jvps_list_with_single_circuit(self, max_workers):
    #     """Tests a basic example with a batch containing a single circuit."""
    #     dev = DefaultQubit(max_workers=max_workers)
    #     x = np.array(np.pi / 7)
    #     tangent = (0.456,)

    #     qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])

    #     qs, _ = validate_and_expand_adjoint(qs)
    #     qs = qs[0]

    #     expected_grad = -qml.math.sin(x) * tangent[0]
    #     actual_grad = dev.compute_jvp([qs], [tangent], self.ec)
    #     assert isinstance(actual_grad, tuple)
    #     assert isinstance(actual_grad[0], np.ndarray)
    #     assert np.isclose(actual_grad[0], expected_grad)

    #     expected_val = qml.math.cos(x)
    #     actual_val, actual_grad = dev.execute_and_compute_jvp([qs], [tangent], self.ec)
    #     assert np.isclose(expected_val, actual_val[0])
    #     assert np.isclose(expected_grad, actual_grad[0])

    # def test_jvps_many_tapes_many_results(self, max_workers):
    #     """Tests a basic example with a batch of circuits of varying return shapes."""
    #     dev = DefaultQubit(max_workers=max_workers)
    #     x = np.array(np.pi / 7)
    #     single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
    #     multi_meas = qml.tape.QuantumScript(
    #         [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
    #     )
    #     tangents = [(0.456,), (0.789,)]

    #     expected_grad = (
    #         -qml.math.sin(x) * tangents[0][0],
    #         (qml.math.cos(x) * tangents[1][0], -qml.math.sin(x) * tangents[1][0]),
    #     )
    #     actual_grad = dev.compute_jvp([single_meas, multi_meas], tangents, self.ec)
    #     assert np.isclose(actual_grad[0], expected_grad[0])
    #     assert isinstance(actual_grad[1], tuple)
    #     assert qml.math.allclose(actual_grad[1], expected_grad[1])

    #     expected_val = (qml.math.cos(x), (qml.math.sin(x), qml.math.cos(x)))
    #     actual_val, actual_grad = dev.execute_and_compute_jvp(
    #         [single_meas, multi_meas], tangents, self.ec
    #     )
    #     assert np.isclose(actual_val[0], expected_val[0])
    #     assert qml.math.allclose(actual_val[1], expected_val[1])
    #     assert np.isclose(actual_grad[0], expected_grad[0])
    #     assert qml.math.allclose(actual_grad[1], expected_grad[1])

    # def test_jvps_integration(self, max_workers):
    #     """Tests the expected workflow done by a calling method."""
    #     dev = DefaultQubit(max_workers=max_workers)
    #     x = np.array(np.pi / 7)

    #     single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
    #     multi_meas = qml.tape.QuantumScript(
    #         [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
    #     )
    #     tangents = [(0.456,), (0.789,)]
    #     circuits = [single_meas, multi_meas]
    #     program, new_ec = dev.preprocess(self.ec)
    #     circuits, _ = program(circuits)
    #     actual_grad = dev.compute_jvp(circuits, tangents, self.ec)
    #     expected_grad = (
    #         -qml.math.sin(x) * tangents[0][0],
    #         (qml.math.cos(x) * tangents[1][0], -qml.math.sin(x) * tangents[1][0]),
    #     )

    #     assert new_ec.use_device_gradient
    #     assert new_ec.grad_on_execution

    #     assert np.isclose(actual_grad[0], expected_grad[0])
    #     assert isinstance(actual_grad[1], tuple)
    #     assert qml.math.allclose(actual_grad[1], expected_grad[1])

    # def test_vjps_single_circuit(self, max_workers):
    #     """Tests vjps with a single circuit."""
    #     dev = DefaultQubit(max_workers=max_workers)
    #     x = np.array(np.pi / 7)
    #     cotangent = (0.456,)

    #     qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
    #     qs, _ = validate_and_expand_adjoint(qs)
    #     qs = qs[0]

    #     expected_grad = -qml.math.sin(x) * cotangent[0]
    #     actual_grad = dev.compute_vjp(qs, cotangent, self.ec)
    #     assert isinstance(actual_grad, np.ndarray)
    #     assert actual_grad.shape == ()  # pylint: disable=no-member
    #     assert np.isclose(actual_grad, expected_grad)

    #     expected_val = qml.math.cos(x)
    #     actual_val, actual_grad = dev.execute_and_compute_vjp(qs, cotangent, self.ec)
    #     assert np.isclose(actual_val, expected_val)
    #     assert np.isclose(actual_grad, expected_grad)

    # def test_vjps_list_with_single_circuit(self, max_workers):
    #     """Tests a basic example with a batch containing a single circuit."""
    #     dev = DefaultQubit(max_workers=max_workers)
    #     x = np.array(np.pi / 7)
    #     cotangent = (0.456,)

    #     qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
    #     qs, _ = validate_and_expand_adjoint(qs)
    #     qs = qs[0]

    #     expected_grad = -qml.math.sin(x) * cotangent[0]
    #     actual_grad = dev.compute_vjp([qs], [cotangent], self.ec)
    #     assert isinstance(actual_grad, tuple)
    #     assert isinstance(actual_grad[0], np.ndarray)
    #     assert np.isclose(actual_grad[0], expected_grad)

    #     expected_val = qml.math.cos(x)
    #     actual_val, actual_grad = dev.execute_and_compute_vjp([qs], [cotangent], self.ec)
    #     assert np.isclose(expected_val, actual_val[0])
    #     assert np.isclose(expected_grad, actual_grad[0])

    # def test_vjps_many_tapes_many_results(self, max_workers):
    #     """Tests a basic example with a batch of circuits of varying return shapes."""
    #     dev = DefaultQubit(max_workers=max_workers)
    #     x = np.array(np.pi / 7)
    #     single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
    #     multi_meas = qml.tape.QuantumScript(
    #         [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
    #     )
    #     cotangents = [(0.456,), (0.789, 0.123)]

    #     expected_grad = (
    #         -qml.math.sin(x) * cotangents[0][0],
    #         qml.math.cos(x) * cotangents[1][0] - qml.math.sin(x) * cotangents[1][1],
    #     )
    #     actual_grad = dev.compute_vjp([single_meas, multi_meas], cotangents, self.ec)
    #     assert np.isclose(actual_grad[0], expected_grad[0])
    #     assert np.isclose(actual_grad[1], expected_grad[1])

    #     expected_val = (qml.math.cos(x), (qml.math.sin(x), qml.math.cos(x)))
    #     actual_val, actual_grad = dev.execute_and_compute_vjp(
    #         [single_meas, multi_meas], cotangents, self.ec
    #     )
    #     assert np.isclose(actual_val[0], expected_val[0])
    #     assert qml.math.allclose(actual_val[1], expected_val[1])
    #     assert np.isclose(actual_grad[0], expected_grad[0])
    #     assert np.isclose(actual_grad[1], expected_grad[1])

    # def test_vjps_integration(self, max_workers):
    #     """Tests the expected workflow done by a calling method."""
    #     dev = DefaultQubit(max_workers=max_workers)
    #     x = np.array(np.pi / 7)

    #     single_meas = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
    #     multi_meas = qml.tape.QuantumScript(
    #         [qml.RY(x, 0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]
    #     )
    #     cotangents = [(0.456,), (0.789, 0.123)]
    #     circuits = [single_meas, multi_meas]
    #     program, new_ec = dev.preprocess(self.ec)
    #     circuits, _ = program(circuits)

    #     actual_grad = dev.compute_vjp(circuits, cotangents, self.ec)
    #     expected_grad = (
    #         -qml.math.sin(x) * cotangents[0][0],
    #         qml.math.cos(x) * cotangents[1][0] - qml.math.sin(x) * cotangents[1][1],
    #     )

    #     assert new_ec.use_device_gradient
    #     assert new_ec.grad_on_execution

    #     assert np.isclose(actual_grad[0], expected_grad[0])
    #     assert np.isclose(actual_grad[1], expected_grad[1])


class TestOperatorArithmeticComputeDerivatives:
    """Tests integration with SProd, Prod, and Sum."""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return LightningQubit(c_dtype=request.param)

    @staticmethod
    def process_and_compute_derivatives(dev, tape, trainable_params=None):
        if trainable_params != None:
            tape.trainable_params = trainable_params
        program, _ = preprocess(AdjointConfig)
        res_tapes, _ = program([tape])
        tape = res_tapes[0]
        if trainable_params != None:
            tape.trainable_params = trainable_params
        results = dev.compute_derivatives(tape, AdjointConfig)
        return results

    def test_s_prod(self, dev, tol):
        """Tests the `SProd` class."""

        x = np.array(0.123)

        tape = qml.tape.QuantumScript(
            [qml.RX(x, wires=[0])],
            [qml.expval(qml.s_prod(0.5, qml.PauliZ(0)))],
        )
        trainable_params = {0}

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape, trainable_params)
        reference_val = -0.5 * np.sin(x)

        tol = 1e-5 if dev.C_DTYPE == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_prod(self, dev, tol):
        """Tests the `Prod` class."""

        x = np.array(0.123)

        tape = qml.tape.QuantumScript(
            [qml.RX(x, wires=[0]), qml.Hadamard(wires=[1]), qml.PauliZ(wires=[1])],
            [qml.expval(qml.prod(qml.PauliZ(0), qml.PauliX(1)))],
        )
        trainable_params = {0}

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape, trainable_params)
        reference_val = np.sin(x)

        tol = 1e-5 if dev.C_DTYPE == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_sum(self, dev, tol):
        """Tests the `Sum` class."""

        x, y = [-3.21, 2.34]

        tape = qml.tape.QuantumScript(
            [qml.RX(x, wires=[0]), qml.RY(y, wires=[1])],
            [qml.expval(qml.sum(qml.PauliZ(0), qml.PauliX(1)))],
        )
        trainable_params = {0, 1}

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape, trainable_params)
        reference_val = np.array([-np.sin(x), np.cos(y)])

        tol = 1e-5 if dev.C_DTYPE == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_integration(self, dev, tol):
        """Tests a Combination of `Sum`, `SProd`, and `Prod`."""

        x, y = [0.654, -0.634]

        obs = qml.sum(qml.s_prod(2.3, qml.PauliZ(0)), -0.5 * qml.prod(qml.PauliY(0), qml.PauliZ(1)))

        tape = qml.tape.QuantumScript(
            [qml.RX(x, wires=[0]), qml.RY(y, wires=[1])],
            [qml.expval(obs)],
        )
        trainable_params = {0, 1}

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, tape, trainable_params)
        reference_val = np.array(
            [-2.3 * np.sin(x) + 0.5 * np.cos(y) * np.cos(x), -0.5 * np.sin(x) * np.sin(y)]
        )

        tol = 1e-5 if dev.C_DTYPE == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)


class TestBasicCircuit:
    """Tests a basic circuit with one RX gate and two simple expectation values, without preprocessing."""

    @staticmethod
    def process_and_compute_derivatives(dev, tape):
        results = dev.compute_derivatives(tape, AdjointConfig)
        return results

    @staticmethod
    def calculate_reference(tape):
        dev = DefaultQubit(max_workers=1)
        results = dev.compute_derivatives(tape, AdjointConfig)
        return results

    def test_basic_circuit_numpy(self, tol):
        """Test execution with a basic circuit."""
        phi = np.array(0.397)
        qs = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
        )

        dev = LightningQubit()

        # gradients
        calculated_val = self.process_and_compute_derivatives(dev, qs)
        reference_val = self.calculate_reference(qs)

        assert len(calculated_val) == 2
        assert np.allclose(calculated_val, reference_val, tol)
