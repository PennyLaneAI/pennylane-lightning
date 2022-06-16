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
import math

import pennylane as qml
from pennylane import numpy as np

from pennylane_lightning.lightning_qubit import CPP_BINARY_AVAILABLE

if not CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


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

        fn1 = dev.vjp(tape.measurements, dy)
        vjp1 = fn1(tape)

        qml.execute([tape], dev, None)
        fn2 = dev.vjp(tape.measurements, dy, use_device_state=True)
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

        fn1 = dev.vjp(tape.measurements, dy)
        vjp1 = fn1(tape)

        qml.execute([tape], dev, None)
        fn2 = dev.vjp(tape.measurements, dy, starting_state=dev._pre_rotated_state)
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

        fn1 = dev.vjp(tape1.measurements, dy)
        vjp1 = fn1(tape1)

        vjp2 = dev.adjoint_jacobian(tape2)

        assert np.allclose(vjp1, vjp2, atol=tol, rtol=0)

    def test_wrong_dy_expval(self, tol, dev):
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
        tape1.trainable_params = {1, 2, 3}

        with pytest.raises(
            ValueError, match="Number of observables in the tape must be the same as"
        ):
            dev.vjp(tape1.measurements, dy1)

        dy2 = np.array([1.0 + 3.0j, 0.3 + 2.0j, 0.5 + 0.1j])
        with pytest.raises(ValueError, match="The vjp method only works with a real-valued dy"):
            dev.vjp(tape1.measurements, dy2)

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
            dev.vjp(tape.measurements, dy)(tape)

    def test_finite_shots_warns(self):
        """Tests warning raised when finite shots specified"""

        dev = qml.device("lightning.qubit", wires=1, shots=1)

        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        dy = np.array([1.0])

        with pytest.warns(
            UserWarning, match="Requested adjoint differentiation to be computed with finite shots."
        ):
            dev.vjp(tape.measurements, dy)(tape)

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
            dev.vjp(tape.measurements, dy)(tape)

    def test_proj_unsupported(self, dev):
        """Test if a QuantumFunctionError is raised for a Projector observable"""

        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0, 1], wires=[0, 1]))

        dy = np.array([1.0])

        with pytest.raises(
            qml.QuantumFunctionError, match="differentiation method does not support the Projector"
        ):
            dev.vjp(tape.measurements, dy)(tape)

        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0], wires=[0]) @ qml.PauliZ(0))

        with pytest.raises(
            qml.QuantumFunctionError, match="differentiation method does not support the Projector"
        ):
            dev.vjp(tape.measurements, dy)(tape)

    def test_hermitian_expectation(self, dev, tol):
        obs = np.array([[1, 0], [0, -1]], dtype=dev.C_DTYPE, requires_grad=False)
        dy = np.array([0.8])

        fn = dev.vjp([qml.expval(qml.Hermitian(obs, wires=(0,)))], dy)

        for x in np.linspace(-2 * math.pi, 2 * math.pi, 7):
            with qml.tape.QuantumTape() as tape:
                qml.RY(x, wires=(0,))
            vjp = fn(tape)
            assert np.allclose(vjp[0], -0.8 * np.sin(x), atol=tol)

    def test_hermitian_tensor_expectation(self, dev, tol):
        obs = np.array([[1, 0], [0, -1]], dtype=dev.C_DTYPE, requires_grad=False)
        dy = np.array([0.8])

        fn = dev.vjp([qml.expval(qml.Hermitian(obs, wires=(0,)) @ qml.PauliZ(wires=1))], dy)

        for x in np.linspace(-2 * math.pi, 2 * math.pi, 7):
            with qml.tape.QuantumTape() as tape:
                qml.RY(x, wires=(0,))
            assert np.allclose(fn(tape), -0.8 * np.sin(x), atol=tol)

    def test_statevector_ry(self, dev, tol):
        dy = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        fn0 = dev.vjp([qml.state()], dy[0, :])
        fn1 = dev.vjp([qml.state()], dy[1, :])
        fn2 = dev.vjp([qml.state()], dy[2, :])
        fn3 = dev.vjp([qml.state()], dy[3, :])

        for x in np.linspace(-2 * math.pi, 2 * math.pi, 7):
            with qml.tape.QuantumTape() as tape:
                qml.RY(x, wires=(0,))
            assert np.allclose(fn0(tape), -np.sin(x / 2) / 2, atol=tol)
            assert np.allclose(fn1(tape), np.cos(x / 2) / 2, atol=tol)
            assert np.allclose(fn2(tape), 0.0, atol=tol)
            assert np.allclose(fn3(tape), 0.0, atol=tol)

    def test_wrong_dy_statevector(self, tol, dev):
        """Tests raise an exception when dy is incorrect"""
        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.state()

        tape.trainable_params = {1, 2, 3}

        dy1 = np.ones(3, dtype=dev.C_DTYPE)

        with pytest.raises(
            ValueError, match="Size of the provided vector dy must be the same as the size of"
        ):
            dev.vjp(tape.measurements, dy1)

        dy2 = np.ones(4, dtype=dev.R_DTYPE)

        with pytest.warns(UserWarning, match="The vjp method only works with complex-valued dy"):
            dev.vjp(tape.measurements, dy2)

    def test_statevector_complex_circuit(self, dev, tol):
        dy = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        fn0 = dev.vjp([qml.state()], dy[0, :])
        fn1 = dev.vjp([qml.state()], dy[1, :])
        fn2 = dev.vjp([qml.state()], dy[2, :])
        fn3 = dev.vjp([qml.state()], dy[3, :])

        params = [math.pi / 7, 6 * math.pi / 7]

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(np.array([1.0] * 4) / 2, wires=range(2))
            qml.RY(params[0], wires=0)
            qml.RZ(params[1], wires=1)
            qml.CZ(wires=[0, 1])

        tape.trainable_params = {2}  # RZ

        psi_00_diff = (
            (math.cos(params[0] / 2) - math.sin(params[0] / 2))
            * (-math.sin(params[1] / 2) - 1j * math.cos(params[1] / 2))
            / 4
        )
        psi_01_diff = (
            (math.cos(params[0] / 2) + math.sin(params[0] / 2))
            * (-math.sin(params[1] / 2) - 1j * math.cos(params[1] / 2))
            / 4
        )
        psi_10_diff = (
            (math.cos(params[0] / 2) - math.sin(params[0] / 2))
            * (-math.sin(params[1] / 2) + 1j * math.cos(params[1] / 2))
            / 4
        )
        psi_11_diff = (
            -(math.cos(params[0] / 2) + math.sin(params[0] / 2))
            * (-math.sin(params[1] / 2) + 1j * math.cos(params[1] / 2))
            / 4
        )

        assert np.allclose(fn0(tape), psi_00_diff, atol=tol)
        assert np.allclose(fn1(tape), psi_01_diff, atol=tol)
        assert np.allclose(fn2(tape), psi_10_diff, atol=tol)
        assert np.allclose(fn3(tape), psi_11_diff, atol=tol)

    def test_no_trainable_parameters(self, dev):
        """A tape with no trainable parameters will simply return None"""
        x = 0.4

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {}
        dy = np.array([1.0])

        fn = dev.vjp(tape.measurements, dy)
        vjp = fn(tape)

        assert len(vjp) == 0

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
        fn = dev.vjp(tape.measurements, dy)
        vjp = fn(tape)

        assert len(vjp) == 0

    def test_no_trainable_parameters(self, dev):
        """A tape with no trainable parameters will simply return None"""
        x = 0.4

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {}
        dy = np.array([1.0])

        fn = dev.vjp(tape.measurements, dy)
        vjp = fn(tape)

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

        fn = dev.vjp(tape.measurements, dy)
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

        fn = dev.vjp(tape.measurements, dy)
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

        fn = dev.vjp(tape.measurements, dy)
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

        with pytest.raises(
            qml.QuantumFunctionError, match="Adjoint differentiation method does not support"
        ):
            dev.vjp(tape.measurements, dy)(tape)


class TestBatchVectorJacobianProduct:
    """Tests for the batch_vjp function"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return qml.device("lightning.qubit", wires=2, c_dtype=request.param)

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

        fn = dev.batch_vjp(tapes, dys)
        vjps = fn(tapes)

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

        fn = dev.batch_vjp(tapes, dys)
        vjps = fn(tapes)

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
