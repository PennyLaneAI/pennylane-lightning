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
Tests for the ``vjp`` method of LightningQubit.
"""
from cmath import exp
import pytest

import pennylane as qml
from pennylane import numpy as np

try:
    from pennylane_lightning.lightning_qubit_ops import (
        VectorJacobianProductC64,
        VectorJacobianProductC128,
    )
except (ImportError, ModuleNotFoundError):
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


class TestComputeVJP:
    """Tests for the numeric computation of VJPs"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return qml.device("lightning.qubit", wires=2, c_dtype=request.param)

    def test_computation(self, tol, dev):
        """Test that the correct VJP is returned"""
        dy = np.array([[1.0, 2.0], [3.0, 4.0]])
        jac = np.array([[[1.0, 0.1, 0.2], [0.2, 0.6, 0.1]], [[0.4, -0.7, 1.2], [-0.5, -0.6, 0.7]]])

        vjp = dev.compute_vjp(dy, jac)
        expected = np.tensordot(dy, jac, axes=[[0, 1], [0, 1]])

        assert vjp.shape == (3,)
        assert vjp.dtype == dev.R_DTYPE
        assert np.allclose(vjp, expected, atol=tol, rtol=0)

    def test_computation_num(self, tol, dev):
        """Test that the correct VJP is returned"""
        dy = np.array([[1.0, 2.0], [3.0, 4.0]])
        jac = np.array([[[1.0, 0.1, 0.2], [0.2, 0.6, 0.1]], [[0.4, -0.7, 1.2], [-0.5, -0.6, 0.7]]])

        vjp = dev.compute_vjp(dy, jac, num=4)
        expected = np.tensordot(dy, jac, axes=[[0, 1], [0, 1]])

        assert vjp.shape == (3,)
        assert vjp.dtype == dev.R_DTYPE
        assert np.allclose(vjp, expected, atol=tol, rtol=0)

    def test_computation_num_error(self, dev):
        """Test that the correct VJP is returned"""
        dev._state = dev._asarray(dev._state)

        dy = np.array([[1.0, 2.0], [3.0, 4.0]])
        jac = np.array([[[1.0, 0.1, 0.2], [0.2, 0.6, 0.1]], [[0.4, -0.7, 1.2], [-0.5, -0.6, 0.7]]])

        with pytest.raises(ValueError, match="Invalid size for the gradient-output vector"):
            dev.compute_vjp(dy, jac, num=3)

    def test_jacobian_is_none(self, dev):
        """A None Jacobian returns a None VJP"""
        dev._state = dev._asarray(dev._state)

        dy = np.array([[1.0, 2.0], [3.0, 4.0]])
        jac = None

        vjp = dev.compute_vjp(dy, jac)
        assert vjp is None

    def test_zero_dy(self, dev):
        """A zero dy vector will return a zero matrix"""
        dev._state = dev._asarray(dev._state)

        dy = np.zeros([2, 2])
        jac = np.array([[[1.0, 0.1, 0.2], [0.2, 0.6, 0.1]], [[0.4, -0.7, 1.2], [-0.5, -0.6, 0.7]]])

        vjp = dev.compute_vjp(dy, jac)
        assert np.all(vjp == np.zeros([3]))

    def test_array_dy(self, dev):
        """Test vjp_compute using Python array"""

        dy = [1.0, 1.0, 1.0, 1.0]
        jac = [dy, dy, dy, dy]

        expected = [4.0, 4.0, 4.0, 4.0]
        vjp = dev.compute_vjp(dy, jac)

        assert np.all(vjp == expected)

    def test_torch_tensor_dy(self, dev):
        """Test vjp_compute using the Torch interface"""
        torch = pytest.importorskip("torch")

        if dev.R_DTYPE == np.float32:
            torch_r_dtype = torch.float32
        else:
            torch_r_dtype = torch.float64

        dy = torch.ones(4, dtype=torch_r_dtype)
        jac = torch.ones((4, 4), dtype=torch_r_dtype)

        expected = torch.tensor([4.0, 4.0, 4.0, 4.0], dtype=torch_r_dtype)
        vjp = dev.compute_vjp(dy, jac)

        assert vjp.dtype == torch_r_dtype
        assert torch.all(vjp == expected)

    def test_tf_tensor_dy(self, dev):
        """Test vjp_compute using the Tensorflow interface"""
        tf = pytest.importorskip("tensorflow")

        if dev.R_DTYPE == np.float32:
            tf_r_dtype = tf.float32
        else:
            tf_r_dtype = tf.float64

        dy = tf.ones(4, dtype=tf_r_dtype)
        jac = tf.ones((4, 4), dtype=tf_r_dtype)

        expected = tf.constant([4.0, 4.0, 4.0, 4.0], dtype=tf_r_dtype)
        vjp = dev.compute_vjp(dy, jac)

        assert vjp.dtype == dev.R_DTYPE  # ?
        assert tf.reduce_all(vjp == expected)


class TestVectorJacobianProduct:
    """Tests for the `vjp` function"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return qml.device("lightning.qubit", wires=2, c_dtype=request.param)

    def test_use_device_state(self, tol, dev):
        """Tests that when using the device state, the correct answer is still returned."""
        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        dy = np.array([1.0])

        fn1 = dev.vjp(tape, dy)
        vjp1 = fn1(tape)

        qml.execute([tape], dev, None)
        fn2 = dev.vjp(tape, dy, use_device_state=True)
        vjp2 = fn2(tape)

        assert np.allclose(vjp1, vjp2, atol=tol, rtol=0)

    def test_provide_starting_state(self, tol, dev):
        """Tests provides correct answer when provided starting state."""
        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        dy = np.array([1.0])

        fn1 = dev.vjp(tape, dy)
        vjp1 = fn1(tape)

        qml.execute([tape], dev, None)
        fn2 = dev.vjp(tape, dy, starting_state=dev._pre_rotated_state)
        vjp2 = fn2(tape)

        assert np.allclose(vjp1, vjp2, atol=tol, rtol=0)

    def test_not_expval(self, dev):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.1, wires=0)
            qml.var(qml.PauliZ(0))

        dy = np.array([1.0])

        with pytest.raises(qml.QuantumFunctionError, match="Adjoint differentiation method does"):
            dev.vjp(tape, dy)(tape)

    def test_finite_shots_warns(self):
        """Tests warning raised when finite shots specified"""

        dev = qml.device("lightning.qubit", wires=1, shots=1)

        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        dy = np.array([1.0])

        with pytest.warns(
            UserWarning, match="Requested adjoint differentiation to be computed with finite shots."
        ):
            dev.vjp(tape, dy)(tape)

    from pennylane_lightning import LightningQubit as lq

    @pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_unsupported_op(self, dev):
        """Test if a QuantumFunctionError is raised for an unsupported operation, i.e.,
        multi-parameter operations that are not qml.Rot"""

        with qml.tape.QuantumTape() as tape:
            qml.CRot(0.1, 0.2, 0.3, wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        dy = np.array([1.0])

        with pytest.raises(
            qml.QuantumFunctionError, match="The CRot operation is not supported using the"
        ):
            dev.vjp(tape, dy)(tape)

    @pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_proj_unsupported(self, dev):
        """Test if a QuantumFunctionError is raised for a Projector observable"""

        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0, 1], wires=[0, 1]))

        dy = np.array([1.0])

        with pytest.raises(
            qml.QuantumFunctionError, match="differentiation method does not support the Projector"
        ):
            dev.vjp(tape, dy)(tape)

        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0], wires=[0]) @ qml.PauliZ(0))

        with pytest.raises(
            qml.QuantumFunctionError, match="differentiation method does not support the Projector"
        ):
            dev.vjp(tape, dy)(tape)

    @pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_unsupported_hermitian_expectation(self, dev):
        obs = np.array([[1, 0], [0, -1]], dtype=np.complex128, requires_grad=False)

        with qml.tape.QuantumTape() as tape:
            qml.RY(0.1, wires=(0,))
            qml.expval(qml.Hermitian(obs, wires=(0,)))

        dy = np.array([1.0])

        with pytest.raises(
            qml.QuantumFunctionError, match="Lightning adjoint differentiation method does not"
        ):
            dev.vjp(tape, dy)(tape)

        with qml.tape.QuantumTape() as tape:
            qml.RY(0.1, wires=(0,))
            qml.expval(qml.Hermitian(obs, wires=(0,)) @ qml.PauliZ(wires=1))

        with pytest.raises(
            qml.QuantumFunctionError, match="Lightning adjoint differentiation method does not"
        ):
            dev.vjp(tape, dy)(tape)

    def test_no_trainable_parameters(self, dev):
        """A tape with no trainable parameters will simply return None"""
        x = 0.4

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {}
        dy = np.array([1.0])

        fn = dev.vjp(tape, dy)
        vjp = fn(tape)

        assert vjp is None

    def test_no_trainable_parameters_NEW(self, dev):
        """A tape with no trainable parameters will simply return None"""
        dev._state = dev._asarray(dev._state)

        x = 0.4

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {}
        dy = np.array([1.0])
        fn = dev.vjp(tape, dy)
        vjp = fn(tape)

        assert vjp is None

    def test_no_trainable_parameters_(self, dev):
        """A tape with no trainable parameters will simply return None"""
        x = 0.4

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {}
        dy = np.array([1.0])

        fn = dev.vjp(tape, dy)
        vjp = fn(tape)

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

        fn = dev.vjp(tape, dy)
        vjp = fn(tape)

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

        fn = dev.vjp(tape, dy)
        vjp = fn(tape)

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

        fn = dev.vjp(tape, dy)
        vjp = fn(tape)

        expected = np.array([-np.sin(x), 2 * np.cos(y)])
        assert np.allclose(vjp, expected, atol=tol, rtol=0)

    def test_prob_expectation_values(self, dev):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev._state = dev._asarray(dev._state)

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

        with pytest.raises(qml.QuantumFunctionError, match="Adjoint differentiation method does"):
            dev.vjp(tape, dy)(tape)


class TestBatchVectorJacobianProduct:
    """Tests for the batch_vjp function"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return qml.device("lightning.qubit", wires=2, c_dtype=request.param)

    def test_one_tape_no_trainable_parameters(self, dev):
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

        fn = dev.batch_vjp(tapes, dys)
        vjps = fn(tapes)

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

        fn = dev.batch_vjp(tapes, dys)
        vjps = fn(tapes)

        assert vjps[0] is None
        assert vjps[1] is None

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

        fn = dev.batch_vjp(tapes, dys)
        vjps = fn(tapes)

        assert np.allclose(vjps[0], 0)

    def test_reduction_append(self, dev):
        """Test the 'append' reduction strategy"""
        dev._state = dev._asarray(dev._state)

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
        dys = [np.array([1.0]), np.array([1.0])]

        fn = dev.batch_vjp(tapes, dys, reduction="append")
        vjps = fn(tapes)

        assert len(vjps) == 2
        assert all(isinstance(v, np.ndarray) for v in vjps)
        assert all(len(v) == len(t.trainable_params) for t, v in zip(tapes, vjps))

    def test_reduction_append_callable(self, dev):
        """Test the 'append' reduction strategy"""
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
        dys = [np.array([1.0]), np.array([1.0])]

        fn = dev.batch_vjp(tapes, dys, reduction="append")
        vjps = fn(tapes)

        assert len(vjps) == 2
        assert all(isinstance(v, np.ndarray) for v in vjps)
        assert all(len(v) == len(t.trainable_params) for t, v in zip(tapes, vjps))

    def test_reduction_extend(self, dev):
        """Test the 'extend' reduction strategy"""
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
        dys = [np.array([1.0]), np.array([1.0])]

        fn = dev.batch_vjp(tapes, dys, reduction="extend")
        vjps = fn(tapes)

        assert len(vjps) == sum(len(t.trainable_params) for t in tapes)

    def test_reduction_extend_callable(self, dev):
        """Test the 'extend' reduction strategy"""
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
        dys = [np.array([1.0]), np.array([1.0])]

        fn = dev.batch_vjp(tapes, dys, reduction=list.extend)
        vjps = fn(tapes)

        assert len(vjps) == sum(len(t.trainable_params) for t in tapes)
