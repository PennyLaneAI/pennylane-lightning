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
import math

import pennylane as qml
import pytest
from conftest import LightningDevice as ld
from conftest import device_name
from pennylane import numpy as np

if ld._CPP_BINARY_AVAILABLE:
    if ld._new_API:
        from pennylane_lightning.lightning_qubit_ops import LightningException

else:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


def get_vjp(device, tapes, dy):
    """Helper to get VJP for a tape or batch of tapes"""
    if device._new_API:
        return device.compute_vjp(tapes, dy)

    if isinstance(tapes, qml.tape.QuantumScript):
        return device.vjp(tapes.measurements, dy)(tapes)

    return device.batch_vjp(tapes, dy)(tapes)


def get_jacobian(device, tape):
    """Helper to get Jacobian of a tape"""
    if device._new_API:
        return device.compute_derivatives(tape)
    return device.adjoint_jacobian(tape)


class TestVectorJacobianProduct:
    """Tests for the `vjp` function"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return qml.device(device_name, wires=2, c_dtype=request.param)

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

        dy = np.array([1.0])

        fn1 = dev.vjp(tape.measurements, dy)
        vjp1 = fn1(tape)

        qml.execute([tape], dev, None)
        fn2 = dev.vjp(tape.measurements, dy, use_device_state=True)
        vjp2 = fn2(tape)

        assert np.allclose(vjp1, vjp2, atol=tol, rtol=0)

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

        dy = np.array([1.0])

        fn1 = dev.vjp(tape.measurements, dy)
        vjp1 = fn1(tape)

        qml.execute([tape], dev, None)

        fn2 = dev.vjp(tape.measurements, dy, starting_state=dev.state)
        vjp2 = fn2(tape)

        assert np.allclose(vjp1, vjp2, atol=tol, rtol=0)

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

        vjp = get_vjp(dev, tape1, dy)
        jac = get_jacobian(dev, tape2)

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
            get_vjp(dev, tape1, dy1)

        with pytest.raises(
            ValueError, match="The vjp method only works with a real-valued grad_vec"
        ):
            get_vjp(dev, tape1, dy2)

    def test_not_expval(self, dev):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.1, wires=0)
            qml.var(qml.PauliZ(0))

        dy = np.array([1.0])

        with pytest.raises(
            qml.QuantumFunctionError, match="Adjoint differentiation method does not"
        ):
            get_vjp(dev, tape, dy)

    @pytest.mark.skipif(ld._new_API, reason="Old API required")
    def test_finite_shots_warns(self):
        """Tests warning raised when finite shots specified"""

        dev = qml.device(device_name, wires=1, shots=1)

        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        dy = np.array([1.0])

        with pytest.warns(
            UserWarning,
            match="Requested adjoint differentiation to be computed with finite shots.",
        ):
            dev.vjp(tape.measurements, dy)(tape)

    @pytest.mark.skipif(not ld._new_API, reason="New API required")
    def test_finite_shots_error(self):
        """Test that an error is raised when finite shots specified"""
        dev = qml.device(device_name, wires=1)

        tape = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))], shots=1)
        dy = np.array([1.0])

        with pytest.raises(
            qml.QuantumFunctionError,
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

        if dev._new_API:
            with pytest.raises(
                LightningException,
                match="The operation is not supported using the",
            ):
                dev.compute_vjp(tape, dy)

        else:
            with pytest.raises(
                qml.QuantumFunctionError,
                match="The CRot operation is not supported using the",
            ):
                dev.vjp(tape.measurements, dy)(tape)

    @pytest.mark.skipif(ld._new_API, reason="Old API required")
    @pytest.mark.usefixtures("use_legacy_and_new_opmath")
    def test_proj_unsupported(self, dev):
        """Test if a QuantumFunctionError is raised for a Projector observable"""

        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0, 1], wires=[0, 1]))

        dy = np.array([1.0])

        with pytest.raises(
            qml.QuantumFunctionError,
            match="differentiation method does not support the Projector",
        ):
            dev.vjp(tape.measurements, dy)(tape)

        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0], wires=[0]) @ qml.PauliZ(1))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="differentiation method does not support the Projector",
        ):
            dev.vjp(tape.measurements, dy)(tape)

    def test_hermitian_expectation(self, dev, tol):
        obs = np.array([[1, 0], [0, -1]], dtype=dev.dtype, requires_grad=False)
        dy = np.array([0.8])

        for x in np.linspace(-2 * math.pi, 2 * math.pi, 7):
            with qml.tape.QuantumTape() as tape:
                qml.RY(x, wires=(0,))
                qml.expval(qml.Hermitian(obs, wires=(0,)))

            tape.trainable_params = {0}
            vjp = get_vjp(dev, tape, dy)
            assert np.allclose(vjp, -0.8 * np.sin(x), atol=tol)

    def test_hermitian_tensor_expectation(self, dev, tol):
        obs = np.array([[1, 0], [0, -1]], dtype=dev.dtype, requires_grad=False)
        dy = np.array([0.8])

        for x in np.linspace(-2 * math.pi, 2 * math.pi, 7):
            with qml.tape.QuantumTape() as tape:
                qml.RY(x, wires=(0,))
                qml.expval(qml.Hermitian(obs, wires=(0,)) @ qml.PauliZ(wires=1))

            tape.trainable_params = {0}
            vjp = get_vjp(dev, tape, dy)
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

        vjp = get_vjp(dev, tape, dy)

        assert len(vjp) == 0

    @pytest.mark.skipif(ld._new_API, reason="Old API required")
    def test_no_trainable_parameters_NEW(self, dev):
        """A tape with no trainable parameters will simply return None"""
        _state = dev._asarray(dev.state)
        dev._apply_state_vector(_state, dev.wires)

        x = 0.4

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {}
        dy = np.array([1.0])
        vjp = get_vjp(dev, tape, dy)

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

        vjp = get_vjp(dev, tape, dy)

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

        vjp = get_vjp(dev, tape, dy)

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

        vjp = get_vjp(dev, tape, dy)

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
            qml.QuantumFunctionError,
            match="Adjoint differentiation method does not support",
        ):
            get_vjp(dev, tape, dy)

    def test_device_vjp_qnode_autograd(self, dev, tol):
        """Test that requesting device_vjp=True with lightning device qnodes works as expected"""

        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        dev_ref = qml.device("default.qubit")
        x = qml.numpy.array([0.543, -0.654], requires_grad=True)

        qnode = qml.QNode(circuit, dev, device_vjp=True)
        qnode_ref = qml.QNode(circuit, dev_ref, device_vjp=True)

        tol = 1e-5 if dev.dtype == np.complex64 else 1e-7
        assert np.allclose(qnode(x), qnode_ref(x), atol=tol, rtol=0)

        grad = qml.grad(qnode)(x)
        grad_ref = qml.grad(qnode_ref)(x)
        assert np.allclose(grad, grad_ref, atol=tol, rtol=0)

        jac = qml.jacobian(qnode)(x)
        jac_ref = qml.jacobian(qnode_ref)(x)
        assert np.allclose(jac, jac_ref, atol=tol, rtol=0)

    def test_device_vjp_qnode_torch(self, dev, tol):
        """Test that requesting device_vjp=True with lightning device qnodes works as expected"""
        torch = pytest.importorskip("torch")

        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        dev_ref = qml.device("default.qubit")
        x = torch.tensor([0.543, -0.654], requires_grad=True)
        x_ref = torch.tensor([0.543, -0.654], requires_grad=True)

        qnode = qml.QNode(circuit, dev, device_vjp=True)
        qnode_ref = qml.QNode(circuit, dev_ref, device_vjp=True)

        tol = 1e-5 if dev.dtype == np.complex64 else 1e-7
        res = qnode(x)
        expected = qnode_ref(x_ref)
        assert qml.math.allclose(res, expected, atol=tol, rtol=0)

        res.backward()
        expected.backward()
        assert qml.math.allclose(x.grad, x_ref.grad, atol=tol, rtol=0)

        jac = torch.autograd.functional.jacobian(qnode, (x,))
        jac_ref = torch.autograd.functional.jacobian(qnode_ref, (x_ref,))
        assert qml.math.allclose(jac, jac_ref, atol=tol, rtol=0)

    # TODO: Update the following test after TensorFlow dtype issues are resolved.

    @pytest.mark.skipif(ld._new_API, reason="TensorFlow dtype issues with new API")
    def test_device_vjp_qnode_tf(self, dev, tol):
        """Test that requesting device_vjp=True with lightning device qnodes works as expected"""
        tf = pytest.importorskip("tensorflow")
        dtype = tf.float32 if dev.dtype == np.complex64 else tf.float64

        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        dev_ref = qml.device("default.qubit")
        x = tf.Variable([0.543, -0.654], dtype=dtype)

        qnode = qml.QNode(circuit, dev, device_vjp=True)
        qnode_ref = qml.QNode(circuit, dev_ref, device_vjp=True)

        tol = 1e-5 if dev.dtype == np.complex64 else 1e-7

        with tf.GradientTape(persistent=True) as tape:
            res = qnode(x)
        with tf.GradientTape(persistent=True) as tape_ref:
            expected = qnode_ref(x)

        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad = tape.gradient(res, [x])
        grad_ref = tape_ref.gradient(expected, [x])
        assert np.allclose(grad, grad_ref, atol=tol, rtol=0)

        jac = tape.jacobian(res, [x], experimental_use_pfor=False)
        jac_ref = tape_ref.jacobian(expected, [x], experimental_use_pfor=False)
        assert np.allclose(jac, jac_ref, atol=tol, rtol=0)

    @pytest.mark.parametrize("use_jit", [True, False])
    def test_device_vjp_qnode_jax(self, use_jit, dev, tol):
        """Test that requesting device_vjp=True with lightning device qnodes works as expected"""
        jax = pytest.importorskip("jax")

        if dev.dtype == np.complex128:
            jax.config.update("jax_enable_x64", True)

        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        dev_ref = qml.device("default.qubit")
        x = jax.numpy.array([0.543, -0.654])

        qnode = qml.QNode(circuit, dev, device_vjp=True)
        qnode_ref = qml.QNode(circuit, dev_ref, device_vjp=True)

        tol = 1e-5 if dev.dtype == np.complex64 else 1e-7
        assert np.allclose(qnode(x), qnode_ref(x), atol=tol, rtol=0)

        # We don't test for jacobian with jax due to broadcasted dimensions to vjp dys
        grad = jax.jit(jax.grad(qnode))(x) if use_jit else jax.grad(qnode)(x)
        grad_ref = jax.jit(jax.grad(qnode_ref))(x) if use_jit else jax.grad(qnode_ref)(x)
        assert np.allclose(grad, grad_ref, atol=tol, rtol=0)


class TestBatchVectorJacobianProduct:
    """Tests for the batch_vjp function"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return qml.device(device_name, wires=2, c_dtype=request.param)

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

        vjps = get_vjp(dev, tapes, dys)

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

        vjps = get_vjp(dev, tapes, dys)

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

        vjps = get_vjp(dev, tapes, dys)

        assert np.allclose(vjps[0], 0)

    @pytest.mark.skipif(ld._new_API, reason="Old API required")
    def test_reduction_append(self, dev):
        """Test the 'append' reduction strategy"""
        _state = dev._asarray(dev.state)
        dev._apply_state_vector(_state, dev.wires)

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
        assert len(vjps[1]) == 2
        assert isinstance(vjps[0], np.ndarray)
        assert isinstance(vjps[1][0], np.ndarray)
        assert isinstance(vjps[1][1], np.ndarray)

    @pytest.mark.skipif(ld._new_API, reason="Old API required")
    @pytest.mark.parametrize("reduction_keyword", ("extend", list.extend))
    def test_reduction_extend(self, dev, reduction_keyword):
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

        fn = dev.batch_vjp(tapes, dys, reduction=reduction_keyword)
        vjps = fn(tapes)

        assert len(vjps) == sum(len(t.trainable_params) for t in tapes)
