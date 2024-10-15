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
Unit tests for Measurements in Lightning devices.
"""
import math
from typing import Sequence

import numpy as np
import pennylane as qml
import pytest
from conftest import LightningDevice as ld
from conftest import device_name, lightning_ops, validate_measurements
from flaky import flaky
from pennylane.measurements import Expectation, Shots, Variance

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.mark.skipif(ld._new_API, reason="Old API required")
def test_measurements():
    dev = qml.device(device_name, wires=2)
    m = dev.measurements
    assert isinstance(m, (lightning_ops.MeasurementsC64, lightning_ops.MeasurementsC128))


def test_no_measure():
    """Test that failing to specify a measurement
    raises an exception"""
    dev = qml.device(device_name, wires=2)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.PauliY(0)

    with pytest.raises(qml.QuantumFunctionError, match="must return either a single measurement"):
        circuit(0.65)


class TestProbs:
    """Test Probs in Lightning devices"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return qml.device(device_name, wires=2, c_dtype=request.param)

    @pytest.mark.skipif(ld._new_API, reason="Old API required")
    def test_probs_dtype64(self, dev):
        """Test if probs changes the state dtype"""
        _state = dev._asarray(
            np.array([1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0]).astype(dev.C_DTYPE)
        )
        dev._apply_state_vector(_state, dev.wires)
        p = dev.probability(wires=[0, 1])

        assert dev.state.dtype == dev.C_DTYPE
        assert np.allclose(p, [0.5, 0.5, 0, 0])

    def test_probs_H(self, tol, dev):
        """Test probs with Hadamard"""

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=1)
            return qml.probs(wires=[0, 1])

        assert np.allclose(circuit(), [0.5, 0.5, 0.0, 0.0], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [None, [0.9165164490394898, 0.0, 0.08348355096051052, 0.0]],
            [[], [0.9165164490394898, 0.0, 0.08348355096051052, 0.0]],
        ],
    )
    @pytest.mark.xfail
    def test_probs_tape_nowires(self, cases, tol, dev):
        """Test probs with a circuit on wires=[0]"""

        x, y, z = [0.5, 0.3, -0.7]

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.probs(wires=cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [[0, 1], [0.9165164490394898, 0.0, 0.08348355096051052, 0.0]],
            [0, [0.9165164490394898, 0.08348355096051052]],
            [[0], [0.9165164490394898, 0.08348355096051052]],
        ],
    )
    def test_probs_tape_wire0(self, cases, tol, dev):
        """Test probs with a circuit on wires=[0]"""

        x, y, z = [0.5, 0.3, -0.7]

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.probs(wires=cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.skipif(ld._new_API, reason="Old API required")
    @pytest.mark.parametrize(
        "cases",
        [
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]],
        ],
    )
    def test_fail_probs_tape_unordered_wires(self, cases):
        """Test probs with a circuit on wires=[0] fails for out-of-order wires passed to probs."""

        x, y, z = [0.5, 0.3, -0.7]
        dev = qml.device(device_name, wires=cases[1])

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.probs(wires=cases[0])

        with pytest.raises(
            RuntimeError,
            match="Lightning does not currently support out-of-order indices for probabilities",
        ):
            _ = circuit()

    @pytest.mark.skipif(
        device_name in ("lightning.tensor"),
        reason="lightning.tensor does not support out of order prob.",
    )
    @pytest.mark.parametrize(
        "cases",
        [
            [[1, 0], [1, 0], [0.9165164490394898, 0.08348355096051052, 0.0, 0.0]],
            [[2, 0], [2, 0, 1], [0.9165164490394898, 0.08348355096051052, 0.0, 0.0]],
        ],
    )
    def test_probs_matching_device_wire_order(self, cases, tol):
        """Test probs with a circuit on wires=[0] passes if wires are sorted wrt device wires."""

        x, y, z = [0.5, 0.3, -0.7]
        dev = qml.device(device_name, wires=cases[1])

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.probs(wires=cases[0])

        assert np.allclose(circuit(), cases[2], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [
                [0, 1],
                [
                    0.9178264236525453,
                    0.02096485729264079,
                    0.059841820910257436,
                    0.0013668981445561978,
                ],
            ],
            [0, [0.938791280945186, 0.061208719054813635]],
            [[0], [0.938791280945186, 0.061208719054813635]],
        ],
    )
    def test_probs_tape_wire01(self, cases, tol, dev):
        """Test probs with a circuit on wires=[0,1]"""

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.5, wires=[0])
            qml.RY(0.3, wires=[1])
            return qml.probs(wires=cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.skipif(ld._new_API, reason="Old API required")
    @pytest.mark.parametrize(
        "cases",
        [
            [
                [1, 0],
                [
                    0.9178264236525453,
                    0.059841820910257436,
                    0.02096485729264079,
                    0.0013668981445561978,
                ],
            ],
        ],
    )
    def test_fail_probs_tape_wire01(self, cases, tol, dev):
        """Test probs with a circuit on wires=[0,1]"""

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.5, wires=[0])
            qml.RY(0.3, wires=[1])
            return qml.probs(wires=cases[0])

        with pytest.raises(
            RuntimeError,
            match="Lightning does not currently support out-of-order indices for probabilities",
        ):
            assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.skipif(ld._new_API, reason="Old API required")
    @pytest.mark.parametrize("n_qubits", range(4, 25, 4))
    @pytest.mark.parametrize("n_targets", list(range(1, 9)) + list(range(9, 25, 4)))
    def test_probs_many_wires(self, n_qubits, n_targets, tol):
        """Test probs measuring many wires of a random quantum state."""
        if n_targets >= n_qubits:
            pytest.skip("Number of targets cannot exceed the number of wires.")

        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit", wires=n_qubits)

        init_state = np.random.rand(2**n_qubits) + 1.0j * np.random.rand(2**n_qubits)
        init_state /= np.linalg.norm(init_state)

        def circuit():
            qml.StatePrep(init_state, wires=range(n_qubits))
            return qml.probs(wires=range(0, n_targets))

        res = qml.QNode(circuit, dev)()
        ref = qml.QNode(circuit, dq)()

        assert np.allclose(res, ref, atol=tol, rtol=0)


class TestExpval:
    """Tests for the expval function"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return qml.device(device_name, wires=2, c_dtype=request.param)

    @pytest.mark.skipif(ld._new_API, reason="Old API required")
    def test_expval_dtype64(self, dev):
        """Test if expval changes the state dtype"""
        _state = np.array([1, 0, 0, 0]).astype(dev.C_DTYPE)
        dev._apply_state_vector(_state, dev.wires)
        e = dev.expval(qml.PauliX(0))

        assert dev.state.dtype == dev.C_DTYPE
        assert np.allclose(e, 0.0)

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0), -0.041892271271228736],
            [qml.PauliX(1), 0.0],
            [qml.PauliY(0), -0.5516350865364075],
            [qml.PauliY(1), 0.0],
            [qml.PauliZ(0), 0.8330328980789793],
            [qml.PauliZ(1), 1.0],
        ],
    )
    def test_expval_qml_tape_wire0(self, cases, tol, dev):
        """Test expval with a circuit on wires=[0]"""

        x, y, z = [0.5, 0.3, -0.7]

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.expval(cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0), 0.0],
            [qml.PauliX(1), -0.19866933079506122],
            [qml.PauliY(0), -0.3894183423086505],
            [qml.PauliY(1), 0.0],
            [qml.PauliZ(0), 0.9210609940028852],
            [qml.PauliZ(1), 0.9800665778412417],
        ],
    )
    def test_expval_wire01(self, cases, tol, dev):
        """Test expval with a circuit on wires=[0,1]"""

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.expval(cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.usefixtures("use_legacy_and_new_opmath")
    @pytest.mark.parametrize(
        "obs, coeffs, res",
        [
            ([qml.PauliX(0) @ qml.PauliZ(1)], [1.0], 0.0),
            ([qml.PauliZ(0) @ qml.PauliZ(1)], [1.0], math.cos(0.4) * math.cos(-0.2)),
            (
                [
                    qml.PauliX(0) @ qml.PauliZ(1),
                    (
                        qml.Hermitian(
                            [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 3.0, 0.0, 0.0],
                                [0.0, 0.0, -1.0, 1.0],
                                [0.0, 0.0, 1.0, -2.0],
                            ],
                            wires=[0, 1],
                        )
                        if device_name != "lightning.tensor"
                        else qml.Hermitian([[1.0, 0.0], [0.0, 1.0]], wires=[0])
                    ),
                ],
                [0.3, 1.0],
                0.9319728930156066 if device_name != "lightning.tensor" else 1.0,
            ),
        ],
    )
    def test_expval_hamiltonian(self, obs, coeffs, res, tol, dev):
        """Test expval with Hamiltonian"""
        if not qml.operation.active_new_opmath():
            obs = [
                qml.operation.convert_to_legacy_H(o).ops[0] if isinstance(o, qml.ops.Prod) else o
                for o in obs
            ]
        ham = qml.Hamiltonian(coeffs, obs)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.expval(ham)

        assert np.allclose(circuit(), res, atol=tol, rtol=0)

    def test_value(self, dev, tol):
        """Test that the expval interface works"""

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        x = 0.54
        res = circuit(x)
        expected = -np.sin(x)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_not_an_observable(self, dev):
        """Test that a qml.QuantumFunctionError is raised if the provided
        argument is not an observable"""

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.expval(qml.RX(0.742, wires=[0]))

        with pytest.raises(qml.DeviceError, match="Observable RX.*not supported"):
            circuit()

    def test_observable_return_type_is_expectation(self, dev):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Expectation`"""

        @qml.qnode(dev)
        def circuit():
            res = qml.expval(qml.PauliZ(0))
            assert res.return_type is Expectation
            return res

        circuit()


class TestVar:
    """Tests for the var function"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return qml.device(device_name, wires=2, c_dtype=request.param)

    @pytest.mark.skipif(ld._new_API, reason="Old API required")
    def test_var_dtype64(self, dev):
        """Test if var changes the state dtype"""
        _state = np.array([1, 0, 0, 0]).astype(np.complex64)
        dev._apply_state_vector(_state, dev.wires)
        v = dev.var(qml.PauliX(0))

        assert np.allclose(v, 1.0)

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0), 0.9982450376077382],
            [qml.PauliX(1), 1.0],
            [qml.PauliY(0), 0.6956987716741251],
            [qml.PauliY(1), 1.0],
            [qml.PauliZ(0), 0.3060561907181374],
            [qml.PauliZ(1), -4.440892098500626e-16],
        ],
    )
    def test_var_qml_tape_wire0(self, cases, tol, dev):
        """Test var with a circuit on wires=[0]"""

        x, y, z = [0.5, 0.3, -0.7]

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.var(cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0), 1.0],
            [qml.PauliX(1), 0.9605304970014426],
            [qml.PauliY(0), 0.8483533546735826],
            [qml.PauliY(1), 1.0],
            [qml.PauliZ(0), 0.15164664532641725],
            [qml.PauliZ(1), 0.03946950299855745],
        ],
    )
    def test_var_qml_tape_wire01(self, cases, tol, dev):
        """Test var with a circuit on wires=[0,1]"""

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.var(cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    def test_value(self, dev, tol):
        """Test that the var function works"""

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.var(qml.PauliZ(0))

        x = 0.54
        res = circuit(x)
        expected = np.sin(x) ** 2

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_not_an_observable(self, dev):
        """Test that a qml.QuantumFunctionError is raised if the provided
        argument is not an observable"""

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.var(qml.RX(0.742, wires=[0]))

        with pytest.raises(qml.DeviceError, match="Observable RX.*not supported"):
            circuit()

    def test_observable_return_type_is_variance(self, dev):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Variance`"""

        @qml.qnode(dev)
        def circuit():
            res = qml.var(qml.PauliZ(0))
            assert res.return_type is Variance
            return res

        circuit()


@pytest.mark.parametrize("stat_func", [qml.expval, qml.var])
class TestBetaStatisticsError:
    """Tests for errors arising for the beta statistics functions"""

    def test_not_an_observable(self, stat_func):
        """Test that a qml.QuantumFunctionError is raised if the provided
        argument is not an observable"""
        dev = qml.device(device_name, wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.var(qml.RX(0.742, wires=[0]))

        with pytest.raises(qml.DeviceError, match="Observable RX.*not supported"):
            circuit()


class TestWiresInExpval:
    """Test different Wires settings in Lightning's expval."""

    @pytest.mark.skipif(ld._new_API, reason="Old API required")
    @pytest.mark.parametrize(
        "wires1, wires2",
        [
            ([2, 3, 0], [2, 3, 0]),
            ([0, 1], [0, 1]),
            ([0, 2, 3], [2, 0, 3]),
            (["a", "c", "d"], [2, 3, 0]),
            ([-1, -2, -3], ["q1", "ancilla", 2]),
            (["a", "c"], [3, 0]),
            ([-1, -2], ["ancilla", 2]),
            (["a"], ["nothing"]),
        ],
    )
    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_wires_expval(self, wires1, wires2, C, tol):
        """Test that the expectation of a circuit is independent from the wire labels used."""
        dev1 = qml.device(device_name, wires=wires1, c_dtype=C)
        _state = dev1._asarray(dev1.state, C)
        dev1._apply_state_vector(_state, dev1.wires)

        dev2 = qml.device(device_name, wires=wires2)
        _state = dev2._asarray(dev2.state, C)
        dev2._apply_state_vector(_state, dev2.wires)

        n_wires = len(wires1)

        @qml.qnode(dev1)
        def circuit1():
            qml.RX(0.5, wires=wires1[0 % n_wires])
            qml.RY(2.0, wires=wires1[1 % n_wires])
            if n_wires > 1:
                qml.CNOT(wires=[wires1[0], wires1[1]])
            return [qml.expval(qml.PauliZ(wires=w)) for w in wires1]

        @qml.qnode(dev2)
        def circuit2():
            qml.RX(0.5, wires=wires2[0 % n_wires])
            qml.RY(2.0, wires=wires2[1 % n_wires])
            if n_wires > 1:
                qml.CNOT(wires=[wires2[0], wires2[1]])
            return [qml.expval(qml.PauliZ(wires=w)) for w in wires2]

        assert np.allclose(circuit1(), circuit2(), atol=tol)

    @pytest.mark.skipif(ld._new_API, reason="Old API required")
    @pytest.mark.parametrize(
        "wires1, wires2",
        [
            ([2, 3, 0], [2, 3, 0]),
            ([0, 1], [0, 1]),
            ([0, 2, 3], [2, 0, 3]),
            (["a", "c", "d"], [2, 3, 0]),
            ([-1, -2, -3], ["q1", "ancilla", 2]),
            (["a", "c"], [3, 0]),
            ([-1, -2], ["ancilla", 2]),
        ],
    )
    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_wires_expval_hermitian(self, wires1, wires2, C, tol):
        """Test that the expectation of a circuit is independent from the wire labels used."""
        dev1 = qml.device(device_name, wires=wires1, c_dtype=C)
        _state = dev1._asarray(dev1.state, C)
        dev1._apply_state_vector(_state, dev1.wires)

        dev2 = qml.device(device_name, wires=wires2)
        _state = dev2._asarray(dev2.state, C)
        dev2._apply_state_vector(_state, dev2.wires)

        ob_mat = [
            [1.0, 2.0, 0.0, 1.0],
            [2.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [1.0, 0.0, 0.0, -1.0],
        ]

        n_wires = len(wires1)
        ob1 = qml.Hermitian(ob_mat, wires=[wires1[0 % n_wires], wires1[1 % n_wires]])
        ob2 = qml.Hermitian(ob_mat, wires=[wires2[0 % n_wires], wires2[1 % n_wires]])

        @qml.qnode(dev1)
        def circuit1():
            qml.RX(0.5, wires=wires1[0 % n_wires])
            qml.RY(2.0, wires=wires1[1 % n_wires])
            if n_wires > 1:
                qml.CNOT(wires=[wires1[0], wires1[1]])
            return [qml.expval(ob1)]

        @qml.qnode(dev2)
        def circuit2():
            qml.RX(0.5, wires=wires2[0 % n_wires])
            qml.RY(2.0, wires=wires2[1 % n_wires])
            if n_wires > 1:
                qml.CNOT(wires=[wires2[0], wires2[1]])
            return [qml.expval(ob2)]

        assert np.allclose(circuit1(), circuit2(), atol=tol)


class TestSample:
    """Tests that samples are properly calculated."""

    @pytest.mark.parametrize(
        "shots, wires",
        [
            [10, [0]],
            [12, [1]],
            [17, [0, 1]],
        ],
    )
    def test_sample_dimensions(self, qubit_device, shots, wires):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """
        dev = qubit_device(wires=2, shots=shots)
        ops = [qml.RX(1.5708, wires=[0]), qml.RX(1.5708, wires=[1])]
        obs = qml.PauliZ(wires=[0])
        tape = qml.tape.QuantumScript(ops, [qml.sample(op=obs)], shots=shots)
        s1 = dev.execute(tape)
        assert np.array_equal(s1.shape, (shots,))

    def test_sample_values(self, qubit_device, tol):
        """Tests if the samples returned by sample have
        the correct values
        """
        shots = 1000
        dev = qubit_device(wires=2, shots=shots)
        ops = [qml.RX(1.5708, wires=[0])]
        obs = qml.PauliZ(0)
        tape = qml.tape.QuantumScript(ops, [qml.sample(op=obs)], shots=shots)
        s1 = dev.execute(tape)

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1**2, 1, atol=tol, rtol=0)

    @pytest.mark.parametrize("seed", range(0, 10))
    @pytest.mark.parametrize("nwires", range(1, 11))
    def test_sample_variations(self, qubit_device, nwires, seed):
        """Tests if `sample(wires)` returns correct statistics."""
        shots = 20000
        n_qubits = max(5, nwires + 1)
        np.random.seed(seed)
        wires = qml.wires.Wires(np.random.permutation(nwires))
        state = np.random.rand(2**n_qubits) + 1j * np.random.rand(2**n_qubits)
        state[np.random.randint(0, 2**n_qubits, 1)] += state.size / 10
        state /= np.linalg.norm(state)
        ops = [qml.StatePrep(state, wires=range(n_qubits))]
        tape = qml.tape.QuantumScript(ops, [qml.sample(wires=wires)], shots=shots)

        def reshape_samples(samples):
            return np.atleast_3d(samples) if len(wires) == 1 else np.atleast_2d(samples)

        dev = qubit_device(wires=n_qubits, shots=shots)
        samples = dev.execute(tape)
        probs = qml.measurements.ProbabilityMP(wires=wires).process_samples(
            reshape_samples(samples), wire_order=wires
        )

        dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
        samples = dev.execute(tape)
        ref = qml.measurements.ProbabilityMP(wires=wires).process_samples(
            reshape_samples(samples), wire_order=wires
        )

        assert np.allclose(probs, ref, atol=2.0e-2, rtol=1.0e-4)


class TestWiresInVar:
    """Test different Wires settings in Lightning's var."""

    @pytest.mark.skipif(ld._new_API, reason="Old API required")
    @pytest.mark.parametrize(
        "wires1, wires2",
        [
            (["a", "c", "d"], [2, 3, 0]),
            ([-1, -2, -3], ["q1", "ancilla", 2]),
            (["a", "c"], [3, 0]),
            ([-1, -2], ["ancilla", 2]),
            (["a"], ["nothing"]),
        ],
    )
    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_wires_var(self, wires1, wires2, C, tol):
        """Test that the expectation of a circuit is independent from the wire labels used."""
        dev1 = qml.device(device_name, wires=wires1)
        _state = dev1._asarray(dev1.state, C)
        dev1._apply_state_vector(_state, dev1.wires)

        dev2 = qml.device(device_name, wires=wires2)
        _state = dev2._asarray(dev2.state, C)
        dev2._apply_state_vector(_state, dev2.wires)

        n_wires = len(wires1)

        @qml.qnode(dev1)
        def circuit1():
            qml.RX(0.5, wires=wires1[0 % n_wires])
            qml.RY(2.0, wires=wires1[1 % n_wires])
            if n_wires > 1:
                qml.CNOT(wires=[wires1[0], wires1[1]])
            return [qml.var(qml.PauliZ(wires=w)) for w in wires1]

        @qml.qnode(dev2)
        def circuit2():
            qml.RX(0.5, wires=wires2[0 % n_wires])
            qml.RY(2.0, wires=wires2[1 % n_wires])
            if n_wires > 1:
                qml.CNOT(wires=[wires2[0], wires2[1]])
            return [qml.var(qml.PauliZ(wires=w)) for w in wires2]

        assert np.allclose(circuit1(), circuit2(), atol=tol)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [None, 10000, [10000, 11111]])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
@pytest.mark.parametrize(
    "obs",
    [
        [0],
        [0, 1],
        qml.PauliZ(0),
        qml.PauliY(1),
        qml.PauliZ(0) @ qml.PauliY(1),
        qml.PauliZ(1) @ qml.PauliY(2),
    ],
)
@pytest.mark.parametrize("mcmc", [False, True])
@pytest.mark.parametrize("kernel_name", ["Local", "NonZeroRandom"])
def test_shots_single_measure_obs(shots, measure_f, obs, mcmc, kernel_name):
    """Tests that Lightning handles shots in a circuit where a single measurement of a common observable is performed at the end."""
    n_qubits = 3

    if (
        shots is None or device_name in ("lightning.gpu", "lightning.kokkos", "lightning.tensor")
    ) and (mcmc or kernel_name != "Local"):
        pytest.skip(f"Device {device_name} does not have an mcmc option.")

    if measure_f in (qml.expval, qml.var) and isinstance(obs, Sequence):
        pytest.skip("qml.expval, qml.var do not take wire arguments.")

    if measure_f in (qml.counts, qml.sample) and shots is None:
        pytest.skip("qml.counts, qml.sample do not work with shots = None.")

    if device_name in ("lightning.gpu", "lightning.kokkos", "lightning.tensor"):
        dev = qml.device(device_name, wires=n_qubits, shots=shots)
    else:
        dev = qml.device(
            device_name, wires=n_qubits, shots=shots, mcmc=mcmc, kernel_name=kernel_name
        )
    dq = qml.device("default.qubit", wires=n_qubits, shots=shots)
    params = [np.pi / 4, -np.pi / 4]

    def func(x, y):
        qml.RX(x, 0)
        qml.RX(y, 0)
        qml.RX(y, 1)
        qml.RX(x, 2)
        return measure_f(wires=obs) if isinstance(obs, Sequence) else measure_f(op=obs)

    func1 = qml.QNode(func, dev)
    results1 = func1(*params)

    func2 = qml.QNode(func, dq)
    results2 = func2(*params)

    validate_measurements(measure_f, shots, results1, results2)


# TODO: Add LT after extending the support for shots_vector
@pytest.mark.skipif(
    device_name == "lightning.tensor",
    reason="lightning.tensor does not support single-wire devices.",
)
@pytest.mark.parametrize("shots", ((1, 10), (1, 10, 100), (1, 10, 10, 100, 100, 100)))
def test_shots_bins(shots, qubit_device):
    """Tests that Lightning handles multiple shots."""

    dev = qubit_device(wires=1, shots=shots)

    @qml.qnode(dev)
    def circuit():
        return qml.expval(qml.PauliZ(wires=0))

    if dev.name == "lightning.qubit":
        assert np.sum(shots) == circuit.device.shots.total_shots

    assert np.allclose(circuit(), 1.0)
