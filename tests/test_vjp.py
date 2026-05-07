# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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

import pennylane as qp
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
        return qp.device(device_name, wires=request.param[1], c_dtype=request.param[0])

    def test_multiple_measurements(self, tol, dev):
        """Tests provides correct answer when provided multiple measurements."""
        x, y, z = [0.5, 0.3, -0.7]

        with qp.tape.QuantumTape() as tape1:
            qp.RX(0.4, wires=[0])
            qp.Rot(x, y, z, wires=[0])
            qp.RY(-0.2, wires=[0])
            qp.expval(qp.PauliX(0))
            qp.expval(qp.PauliY(1))
            qp.expval(qp.PauliZ(1))

        dy = np.array([1.0, 2.0, 3.0])
        tape1.trainable_params = {1, 2, 3}

        with qp.tape.QuantumTape() as tape2:
            ham = qp.Hamiltonian(dy, [qp.PauliX(0), qp.PauliY(1), qp.PauliY(1)])
            qp.RX(0.4, wires=[0])
            qp.Rot(x, y, z, wires=[0])
            qp.RY(-0.2, wires=[0])
            qp.expval(ham)

        tape2.trainable_params = {1, 2, 3}

        vjp = dev.compute_vjp(tape1, dy)
        jac = dev.compute_derivatives(tape2)

        assert np.allclose(vjp, jac, atol=tol, rtol=0)

    def test_wrong_dy_expval(self, dev):
        """Tests raise an exception when dy is incorrect"""
        x, y, z = [0.5, 0.3, -0.7]

        with qp.tape.QuantumTape() as tape1:
            qp.RX(0.4, wires=[0])
            qp.Rot(x, y, z, wires=[0])
            qp.RY(-0.2, wires=[0])
            qp.expval(qp.PauliX(0))
            qp.expval(qp.PauliY(1))
            qp.expval(qp.PauliZ(1))

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
        with qp.tape.QuantumTape() as tape:
            qp.RX(0.1, wires=0)
            qp.var(qp.PauliZ(0))

        dy = np.array([1.0])

        with pytest.raises(QuantumFunctionError, match="Adjoint differentiation method does not"):
            dev.compute_vjp(tape, dy)

    def test_finite_shots_error(self):
        """Test that an error is raised when finite shots specified"""
        dev = qp.device(device_name, wires=1)

        tape = qp.tape.QuantumScript([], [qp.expval(qp.Z(0))], shots=1)
        dy = np.array([1.0])

        with pytest.raises(
            QuantumFunctionError,
            match="Requested adjoint differentiation to be computed with finite shots.",
        ):
            dev.compute_vjp(tape, dy)

    def test_unsupported_op(self, dev):
        """Test if a QuantumFunctionError is raised for an unsupported operation, i.e.,
        multi-parameter operations that are not qp.Rot"""

        with qp.tape.QuantumTape() as tape:
            qp.CRot(0.1, 0.2, 0.3, wires=[0, 1])
            qp.expval(qp.PauliZ(0))

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
            with qp.tape.QuantumTape() as tape:
                qp.RY(x, wires=(0,))
                qp.expval(qp.Hermitian(obs, wires=(0,)))

            tape.trainable_params = {0}
            vjp = dev.compute_vjp(tape, dy)
            assert np.allclose(vjp, -0.8 * np.sin(x), atol=tol)

    def test_hermitian_tensor_expectation(self, dev, tol):
        obs = np.array([[1, 0], [0, -1]], dtype=dev.c_dtype, requires_grad=False)
        dy = np.array([0.8])

        for x in np.linspace(-2 * math.pi, 2 * math.pi, 7):
            with qp.tape.QuantumTape() as tape:
                qp.RY(x, wires=(0,))
                qp.expval(qp.Hermitian(obs, wires=(0,)) @ qp.PauliZ(wires=1))

            tape.trainable_params = {0}
            vjp = dev.compute_vjp(tape, dy)
            assert np.allclose(vjp, -0.8 * np.sin(x), atol=tol)

    def test_no_trainable_parameters(self, dev):
        """A tape with no trainable parameters will simply return None"""
        x = 0.4

        with qp.tape.QuantumTape() as tape:
            qp.RX(x, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))

        tape.trainable_params = {}
        dy = np.array([1.0])

        vjp = dev.compute_vjp(tape, dy)

        assert len(vjp) == 0

    def test_zero_dy(self, dev):
        """A zero dy vector will return no tapes and a zero matrix"""
        x = 0.4
        y = 0.6

        with qp.tape.QuantumTape() as tape:
            qp.RX(x, wires=0)
            qp.RX(y, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))

        tape.trainable_params = {0, 1}
        dy = np.array([0.0])

        vjp = dev.compute_vjp(tape, dy)

        assert np.all(vjp == np.zeros([len(tape.trainable_params)]))

    def test_single_expectation_value(self, tol, dev):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        x = 0.543
        y = -0.654

        with qp.tape.QuantumTape() as tape:
            qp.RX(x, wires=[0])
            qp.RY(y, wires=[1])
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0) @ qp.PauliX(1))

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

        with qp.tape.QuantumTape() as tape:
            qp.RX(x, wires=[0])
            qp.RY(y, wires=[1])
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))
            qp.expval(qp.PauliX(1))

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

        with qp.tape.QuantumTape() as tape:
            qp.RX(x, wires=[0])
            qp.RY(y, wires=[1])
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))
            qp.probs(wires=[0, 1])

        tape.trainable_params = {0, 1}
        dy = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(
            QuantumFunctionError,
            match="Adjoint differentiation method does not support",
        ):
            dev.compute_vjp(tape, dy)

    def test_device_vjp(self, seed, tol, dev):
        """Tests that the device vjp method works as expected"""
        rng = np.random.default_rng(seed)
        params = rng.random(3)

        def circuit(params):
            qp.RX(params[0], wires=[0])
            qp.RY(params[1], wires=[1])
            qp.RZ(params[2], wires=[0])
            qp.CNOT(wires=[0, 1])
            return [qp.expval(qp.PauliZ(0)), qp.expval(qp.PauliX(1))]

        def make_loss_fxn(qnode):
            def loss(params):
                expval_by_class = qnode(params)
                return sum(expval_by_class)  # Should be single dimensional

            return loss

        comparison_dev = qp.device("default.qubit")

        expected_circuit = qp.qnode(comparison_dev, diff_method="adjoint")(circuit)
        actual_circuit = qp.qnode(dev, diff_method="adjoint", device_vjp=True)(circuit)

        expected = qp.grad(make_loss_fxn(expected_circuit))(params)
        actual = qp.grad(make_loss_fxn(actual_circuit))(params)
        assert np.allclose(actual, expected, atol=tol, rtol=0)


class TestBatchVectorJacobianProduct:
    """Tests for the batch_vjp function"""

    fixture_params = itertools.product([np.complex64, np.complex128], [None, 2])

    @pytest.fixture(params=fixture_params)
    def dev(self, request):
        return qp.device(device_name, wires=request.param[1], c_dtype=request.param[0])

    def test_one_tape_no_trainable_parameters_1(self, dev):
        """A tape with no trainable parameters will simply return None"""
        with qp.tape.QuantumTape() as tape1:
            qp.RX(0.4, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))

        with qp.tape.QuantumTape() as tape2:
            qp.RX(0.4, wires=0)
            qp.RX(0.6, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))

        tape1.trainable_params = {}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        dys = [np.array([1.0]), np.array([1.0])]

        vjps = dev.compute_vjp(tapes, dys)

        assert len(vjps[0]) == 0
        assert vjps[1] is not None

    def test_all_tapes_no_trainable_parameters_2(self, dev):
        """If all tapes have no trainable parameters all outputs will be None"""
        with qp.tape.QuantumTape() as tape1:
            qp.RX(0.4, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))

        with qp.tape.QuantumTape() as tape2:
            qp.RX(0.4, wires=0)
            qp.RX(0.6, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))

        tape1.trainable_params = set()
        tape2.trainable_params = set()

        tapes = [tape1, tape2]
        dys = [np.array([1.0]), np.array([1.0])]

        vjps = dev.compute_vjp(tapes, dys)

        assert len(vjps[0]) == 0
        assert len(vjps[1]) == 0

    def test_zero_dy(self, dev):
        """A zero dy vector will return no tapes and a zero matrix"""
        with qp.tape.QuantumTape() as tape1:
            qp.RX(0.4, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))

        with qp.tape.QuantumTape() as tape2:
            qp.RX(0.4, wires=0)
            qp.RX(0.6, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))

        tape1.trainable_params = {0}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        dys = [np.array([0.0]), np.array([1.0])]

        vjps = dev.compute_vjp(tapes, dys)

        assert np.allclose(vjps[0], 0)
