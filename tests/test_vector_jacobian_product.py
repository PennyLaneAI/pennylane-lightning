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
Tests for the ``vector_jacobian_product`` method of LightningQubit.
"""
import pytest

import pennylane as qml
from pennylane import numpy as np


class TestComputeVJPTensordot:
    """Tests for the numeric computation of VJPs' Tensordots"""

    @pytest.fixture
    def dev(self):
        return qml.device("lightning.qubit", wires=2)

    def test_computation(self, dev):
        """Test that the correct VJP is returned"""
        dy = np.array([[1.0, 2.0], [3.0, 4.0]])
        jac = np.array([[[1.0, 0.1, 0.2], [0.2, 0.6, 0.1]], [[0.4, -0.7, 1.2], [-0.5, -0.6, 0.7]]])

        vjp = dev._compute_vjp_tensordot(dy, jac)

        assert vjp.shape == (3,)
        assert np.all(vjp == np.tensordot(dy, jac, axes=[[0, 1], [0, 1]]))

    def test_jacobian_is_none(self, dev):
        """A None Jacobian returns a None VJP"""

        dy = np.array([[1.0, 2.0], [3.0, 4.0]])
        jac = None

        vjp = dev._compute_vjp_tensordot(dy, jac)
        assert vjp is None

    def test_zero_dy(self, dev):
        """A zero dy vector will return a zero matrix"""
        dy = np.zeros([2, 2])
        jac = np.array([[[1.0, 0.1, 0.2], [0.2, 0.6, 0.1]], [[0.4, -0.7, 1.2], [-0.5, -0.6, 0.7]]])

        vjp = dev._compute_vjp_tensordot(dy, jac)
        assert np.all(vjp == np.zeros([3]))


class TestVectorJacobianProduct:
    """Tests for the vector_jacobian_product function"""

    @pytest.fixture
    def dev(self):
        return qml.device("lightning.qubit", wires=2)

    def test_no_trainable_parameters(self, dev):
        """A tape with no trainable parameters will simply return None"""
        x = 0.4

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {}
        dy = np.array([1.0])
        vjp = dev.vector_jacobian_product(tape, dy)

        assert vjp is None

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
        vjp = dev.vector_jacobian_product(tape, dy)

        assert np.all(vjp == np.zeros([len(tape.trainable_params)]))

    def test_single_expectation_value(self, tol, dev):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        x = 0.543
        y = -0.654

        with qml.tape.JacobianTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape.trainable_params = {0, 1}
        dy = np.array([1.0])

        vjp = dev.vector_jacobian_product(tape, dy)

        expected = np.array([-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)])
        assert np.allclose(vjp, expected, atol=tol, rtol=0)

    def test_multiple_expectation_values(self, tol, dev):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        x = 0.543
        y = -0.654

        with qml.tape.JacobianTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        tape.trainable_params = {0, 1}
        dy = np.array([1.0, 2.0])

        vjp = dev.vector_jacobian_product(tape, dy)

        expected = np.array([-np.sin(x), 2 * np.cos(y)])
        assert np.allclose(vjp, expected, atol=tol, rtol=0)

    def test_prob_expectation_values(self, tol, dev):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        x = 0.543
        y = -0.654

        with qml.tape.JacobianTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0, 1])

        tape.trainable_params = {0, 1}
        dy = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(qml.QuantumFunctionError, match="Adjoint differentiation method does"):
            dev.vector_jacobian_product(tape, dy)


class TestBatchVectorJacobianProduct:
    """Tests for the batch_vector_jacobian_product function"""

    @pytest.fixture
    def dev(self):
        return qml.device("lightning.qubit", wires=2)

    def test_one_tape_no_trainable_parameters(self, dev):
        """A tape with no trainable parameters will simply return None"""

        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.JacobianTape() as tape2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape1.trainable_params = {}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        dys = [np.array([1.0]), np.array([1.0])]

        vjps = dev.batch_vector_jacobian_product(tapes, dys)

        assert vjps[0] is None
        assert vjps[1] is not None

    def test_all_tapes_no_trainable_parameters(self, dev):
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

        vjps = dev.batch_vector_jacobian_product(tapes, dys)

        assert vjps[0] is None
        assert vjps[1] is None

    def test_zero_dy(self, dev):
        """A zero dy vector will return no tapes and a zero matrix"""

        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.JacobianTape() as tape2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape1.trainable_params = {0}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        dys = [np.array([0.0]), np.array([1.0])]

        vjps = dev.batch_vector_jacobian_product(tapes, dys)

        assert np.allclose(vjps[0], 0)

    def test_reduction_append(self, dev):
        """Test the 'append' reduction strategy"""

        with qml.tape.JacobianTape() as tape1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.JacobianTape() as tape2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape1.trainable_params = {0}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        dys = [np.array([1.0]), np.array([1.0])]

        vjps = dev.batch_vector_jacobian_product(tapes, dys)

        assert len(vjps) == 2
        assert all(isinstance(v, np.ndarray) for v in vjps)
        assert all(len(v) == len(t.trainable_params) for t, v in zip(tapes, vjps))

    def test_reduction_extend(self, dev):
        """Test the 'extend' reduction strategy"""

        with qml.tape.JacobianTape() as tape1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.JacobianTape() as tape2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape1.trainable_params = {0}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        dys = [np.array([1.0]), np.array([1.0])]

        vjps = dev.batch_vector_jacobian_product(tapes, dys)

        assert sum(len(t) for t in vjps) == sum(len(t.trainable_params) for t in tapes)
