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
Tests for the ``vjp`` method.
"""
import itertools
import math

import pennylane as qml
import pytest
from conftest import LightningDevice as ld
from conftest import device_name
from pennylane import numpy as np
from pennylane.exceptions import QuantumFunctionError

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

if device_name == "lightning.tensor":
    pytest.skip("lightning.tensor doesn't support vjp.", allow_module_level=True)


class TestVectorJacobianProduct:
    """Tests for the `vjp` function"""

    fixture_params = itertools.product([np.complex64, np.complex128], [None, 2])

    @pytest.fixture(params=fixture_params)
    def dev(self, request):
        return qml.device(device_name, wires=request.param[1], c_dtype=request.param[0])

    def test_multiple_measurements(self, tol, dev):
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

        vjp = dev.compute_vjp(tape1, dy)
        jac = dev.compute_derivatives(tape2)

        assert np.allclose(vjp, jac, atol=tol, rtol=0)

    def test_wrong_dy_expval(self, dev):
        """Tests raise an exception when dy is incorrect"""
        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliX(0))
            qml.expval(qml.PauliY(1))
            qml.expval(qml.PauliZ(1))

        dy1 = np.array([1.0, 2.0])
        dy2 = np.array([1.0 + 3.0j, 0.3 + 2.0j, 0.5 + 0.1j])
        tape1.trainable_params = {1, 2, 3}

        with pytest.raises(
            ValueError, match="Number of observables in the tape must be the same as"
        ):
            dev.compute_vjp(tape1, dy1)

        with pytest.raises(
            ValueError, match="The vjp method only works with a real-valued grad_vec"
        ):
            dev.compute_vjp(tape1, dy2)

    def test_not_expval(self, dev):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.1, wires=0)
            qml.var(qml.PauliZ(0))

        dy = np.array([1.0])

        with pytest.raises(QuantumFunctionError, match="Adjoint differentiation method does not"):
            dev.compute_vjp(tape, dy)

    def test_finite_shots_error(self):
        """Test that an error is raised when finite shots specified"""
        dev = qml.device(device_name, wires=1)

        tape = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))], shots=1)
        dy = np.array([1.0])

        with pytest.raises(
            QuantumFunctionError,
            match="Requested adjoint differentiation to be computed with finite shots.",
        ):
            dev.compute_vjp(tape, dy)

    def test_unsupported_op(self, dev):
        """Test if a QuantumFunctionError is raised for an unsupported operation, i.e.,
        multi-parameter operations that are not qml.Rot"""

        with qml.tape.QuantumTape() as tape:
            qml.CRot(0.1, 0.2, 0.3, wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        dy = np.array([1.0])

        with pytest.raises(
            RuntimeError,
            match="The operation is not supported using the",
        ):
            dev.compute_vjp(tape, dy)

    def test_hermitian_expectation(self, dev, tol):
        obs = np.array([[1, 0], [0, -1]], dtype=dev.c_dtype, requires_grad=False)
        dy = np.array([0.8])

        for x in np.linspace(-2 * math.pi, 2 * math.pi, 7):
            with qml.tape.QuantumTape() as tape:
                qml.RY(x, wires=(0,))
                qml.expval(qml.Hermitian(obs, wires=(0,)))

            tape.trainable_params = {0}
            vjp = dev.compute_vjp(tape, dy)
            assert np.allclose(vjp, -0.8 * np.sin(x), atol=tol)

    def test_hermitian_tensor_expectation(self, dev, tol):
        obs = np.array([[1, 0], [0, -1]], dtype=dev.c_dtype, requires_grad=False)
        dy = np.array([0.8])

        for x in np.linspace(-2 * math.pi, 2 * math.pi, 7):
            with qml.tape.QuantumTape() as tape:
                qml.RY(x, wires=(0,))
                qml.expval(qml.Hermitian(obs, wires=(0,)) @ qml.PauliZ(wires=1))

            tape.trainable_params = {0}
            vjp = dev.compute_vjp(tape, dy)
            assert np.allclose(vjp, -0.8 * np.sin(x), atol=tol)

    def test_no_trainable_parameters(self, dev):
        """A tape with no trainable parameters will simply return None"""
        x = 0.4

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {}
        dy = np.array([1.0])

        vjp = dev.compute_vjp(tape, dy)

        assert len(vjp) == 0

    def test_zero_dy(self, dev):
        """A zero dy vector will return no tapes and a zero matrix"""
        x = 0.4
        y = 0.6

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.RX(y, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {0, 1}
        dy = np.array([0.0])

        vjp = dev.compute_vjp(tape, dy)

        assert np.all(vjp == np.zeros([len(tape.trainable_params)]))

    def test_single_expectation_value(self, tol, dev):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape.trainable_params = {0, 1}
        dy = np.array([1.0])

        vjp = dev.compute_vjp(tape, dy)

        expected = np.array([-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)])
        assert np.allclose(vjp, expected, atol=tol, rtol=0)

    def test_multiple_expectation_values(self, tol, dev):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
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

        vjp = dev.compute_vjp(tape, dy)

        expected = np.array([-np.sin(x), 2 * np.cos(y)])
        assert np.allclose(vjp, expected, atol=tol, rtol=0)

    def test_prob_expectation_values(self, dev):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0, 1])

        tape.trainable_params = {0, 1}
        dy = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(
            QuantumFunctionError,
            match="Adjoint differentiation method does not support",
        ):
            dev.compute_vjp(tape, dy)


class TestBatchVectorJacobianProduct:
    """Tests for the batch_vjp function"""

    fixture_params = itertools.product([np.complex64, np.complex128], [None, 2])

    @pytest.fixture(params=fixture_params)
    def dev(self, request):
        return qml.device(device_name, wires=request.param[1], c_dtype=request.param[0])

    def test_one_tape_no_trainable_parameters_1(self, dev):
        """A tape with no trainable parameters will simply return None"""
        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape1.trainable_params = {}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        dys = [np.array([1.0]), np.array([1.0])]

        vjps = dev.compute_vjp(tapes, dys)

        assert len(vjps[0]) == 0
        assert vjps[1] is not None

    def test_all_tapes_no_trainable_parameters_2(self, dev):
        """If all tapes have no trainable parameters all outputs will be None"""
        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape1.trainable_params = set()
        tape2.trainable_params = set()

        tapes = [tape1, tape2]
        dys = [np.array([1.0]), np.array([1.0])]

        vjps = dev.compute_vjp(tapes, dys)

        assert len(vjps[0]) == 0
        assert len(vjps[1]) == 0

    def test_zero_dy(self, dev):
        """A zero dy vector will return no tapes and a zero matrix"""
        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape1.trainable_params = {0}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        dys = [np.array([0.0]), np.array([1.0])]

        vjps = dev.compute_vjp(tapes, dys)

        assert np.allclose(vjps[0], 0)
