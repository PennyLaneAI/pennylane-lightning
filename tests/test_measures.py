# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Unit tests for Measures in lightning.qubit.
"""
import numpy as np
import pennylane as qml
from pennylane.queuing import AnnotatedQueue
from pennylane.measure import (
    Probability,
    Variance,
    Expectation,
    MeasurementProcess,
)

import pytest


def test_no_measure(tol):
    """Test that failing to specify a measurement
    raises an exception"""
    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.PauliY(0)

    with pytest.raises(qml.QuantumFunctionError, match="must return either a single measurement"):
        circuit(0.65)


class TestExpval:
    """Tests for the expval function"""

    def test_value(self, tol):
        """Test that the expval interface works"""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        x = 0.54
        res = circuit(x)
        expected = -np.sin(x)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_not_an_observable(self):
        """Test that a qml.QuantumFunctionError is raised if the provided
        argument is not an observable"""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.expval(qml.CNOT(wires=[0, 1]))

        with pytest.raises(qml.QuantumFunctionError, match="CNOT is not an observable"):
            res = circuit()

    def test_observable_return_type_is_expectation(self):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Expectation`"""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            res = qml.expval(qml.PauliZ(0))
            assert res.return_type is Expectation
            return res

        circuit()


class TestVar:
    """Tests for the var function"""

    def test_value(self, tol):
        """Test that the var function works"""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.var(qml.PauliZ(0))

        x = 0.54
        res = circuit(x)
        expected = np.sin(x) ** 2

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_not_an_observable(self):
        """Test that a qml.QuantumFunctionError is raised if the provided
        argument is not an observable"""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.var(qml.CNOT(wires=[0, 1]))

        with pytest.raises(qml.QuantumFunctionError, match="CNOT is not an observable"):
            res = circuit()

    def test_observable_return_type_is_variance(self):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Variance`"""
        dev = qml.device("lightning.qubit", wires=2)

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
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return stat_func(qml.CNOT(wires=[0, 1]))

        with pytest.raises(qml.QuantumFunctionError, match="CNOT is not an observable"):
            circuit()


class TestWiresInExpval:
    """Test that the device integrates with PennyLane's wire management."""

    @pytest.mark.parametrize(
        "wires1, wires2",
        [
            ([2, 3, 0], [2, 3, 0]),
            ([0, 1], [0, 1]),
            # ([0, 2], [2, 0]),
            # (["a", "c", "d"], [2, 3, 0]),
            ([-1, -2, -3], ["q1", "ancilla", 2]),
            # (["a", "c"], [3, 0]),
            ([-1, -2], ["ancilla", 2]),
            (["a"], ["nothing"]),
        ],
    )
    def test_wires_expval(self, wires1, wires2, tol):
        """Test that the expectation of a circuit is independent from the wire labels used."""
        dev1 = qml.device("lightning.qubit", wires=wires1)
        dev2 = qml.device("lightning.qubit", wires=wires2)

        # circuit1 = circuit_factory(device=dev1, wires=wires1)
        n_wires = len(wires1)

        @qml.qnode(dev1)
        def circuit1():
            qml.RX(0.5, wires=wires1[0 % n_wires])
            qml.RY(2.0, wires=wires1[1 % n_wires])
            if n_wires > 1:
                qml.CNOT(wires=[wires1[0], wires1[1]])
            return [qml.expval(qml.PauliZ(wires=w)) for w in wires1]

        # circuit2 = circuit_factory(device=dev2, wires=wires2)
        @qml.qnode(dev2)
        def circuit2():
            qml.RX(0.5, wires=wires2[0 % n_wires])
            qml.RY(2.0, wires=wires2[1 % n_wires])
            if n_wires > 1:
                qml.CNOT(wires=[wires2[0], wires2[1]])
            return [qml.expval(qml.PauliZ(wires=w)) for w in wires2]

        print(circuit1())
        print(circuit2())
        assert np.allclose(circuit1(), circuit2(), atol=tol)


# class TestProbs:
#     """Test Probs"""

#     @pytest.fixture
#     def dev(self):
#         return qml.device("lightning.qubit", wires=2)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             [None, [0.9165164490394898, 0.0, 0.08348355096051052, 0.0]],
#             [[], [0.9165164490394898, 0.0, 0.08348355096051052, 0.0]],
#             [[0, 1], [0.9165164490394898, 0.0, 0.08348355096051052, 0.0]],
#             [[1, 0], [0.9165164490394898, 0.08348355096051052, 0.0, 0.0]],
#             [0, [0.9165164490394898, 0.08348355096051052]],
#             [[0], [0.9165164490394898, 0.08348355096051052]],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_probs_tape_wire0(self, cases, tol, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         x, y, z = [0.5, 0.3, -0.7]

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.Rot(x, y, z, wires=[0])
#             qml.RY(-0.2, wires=[0])
#             p = qml.probs(cases[0])

#         assert np.allclose(cases[1], p, atol=tol, rtol=0)

# @pytest.mark.parametrize(
#     "cases",
#     [
#         [
#             None,
#             [
#                 0.9178264236525453,
#                 0.02096485729264079,
#                 0.059841820910257436,
#                 0.0013668981445561978,
#             ],
#         ],
#         [
#             [],
#             [
#                 0.9178264236525453,
#                 0.02096485729264079,
#                 0.059841820910257436,
#                 0.0013668981445561978,
#             ],
#         ],
#         [
#             [0, 1],
#             [
#                 0.9178264236525453,
#                 0.02096485729264079,
#                 0.059841820910257436,
#                 0.0013668981445561978,
#             ],
#         ],
#         [
#             [1, 0],
#             [
#                 0.9178264236525453,
#                 0.059841820910257436,
#                 0.02096485729264079,
#                 0.0013668981445561978,
#             ],
#         ],
#         [0, [0.938791280945186, 0.061208719054813635]],
#         [[0], [0.938791280945186, 0.061208719054813635]],
#     ],
# )
# @pytest.mark.parametrize("C", [np.complex64, np.complex128])
# def test_probs_tape_wire01(self, cases, tol, dev, C):
#     dev._state = dev._asarray(dev._state, C)

#     with qml.tape.JacobianTape() as tape:
#         qml.RX(0.5, wires=[0])
#         qml.RY(0.3, wires=[1])

#     p = dev.probs(cases[0])

#     assert np.allclose(cases[1], p, atol=tol, rtol=0)


# class TestExpval:
#     """Test Expval"""

#     @pytest.fixture
#     def dev(self):
#         return qml.device("lightning.qubit", wires=2)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             ["PauliX", [0, 1]],
#             ["PauliY", [0, 1]],
#             ["PauliZ", [0, 1]],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_expval_gate_error(self, cases, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.RY(-0.2, wires=[1])

#         with pytest.raises(
#             ValueError, match="The supplied gate requires 1 wires, but 2 were supplied."
#         ):
#             dev.expval(cases[0], cases[1])

#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_expval_wires_error(self, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.RY(-0.2, wires=[1])

#         with pytest.raises(AttributeError, match="'str' object has no attribute 'name'"):
#             dev.expval("PauliX", wires=None)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             ["PauliX", [0], -0.041892271271228736],
#             ["PauliX", [1], 0.0],
#             ["PauliY", [0], -0.5516350865364075],
#             ["PauliY", [1], 0.0],
#             ["PauliZ", [0], 0.8330328980789793],
#             ["PauliZ", [1], 1.0],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_expval_tape_wire0(self, cases, tol, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         x, y, z = [0.5, 0.3, -0.7]

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.Rot(x, y, z, wires=[0])
#             qml.RY(-0.2, wires=[0])

#         e = dev.expval(cases[0], cases[1])
#         assert np.allclose(e, cases[2], atol=tol, rtol=0)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             ["PauliX", [0], 0.0],
#             ["PauliX", [1], -0.19866933079506122],
#             ["PauliY", [0], -0.3894183423086505],
#             ["PauliY", [1], 0.0],
#             ["PauliZ", [0], 0.9210609940028852],
#             ["PauliZ", [1], 0.9800665778412417],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_expval_tape_wire01(self, cases, tol, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.RY(-0.2, wires=[1])

#         e = dev.expval(cases[0], cases[1])
#         assert np.allclose(e, cases[2], atol=tol, rtol=0)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             [qml.PauliX(0), -0.041892271271228736],
#             [qml.PauliX(1), 0.0],
#             [qml.PauliY(0), -0.5516350865364075],
#             [qml.PauliY(1), 0.0],
#             [qml.PauliZ(0), 0.8330328980789793],
#             [qml.PauliZ(1), 1.0],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_expval_qml_tape_wire0(self, cases, tol, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         x, y, z = [0.5, 0.3, -0.7]

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.Rot(x, y, z, wires=[0])
#             qml.RY(-0.2, wires=[0])

#         e = dev.expval(cases[0])
#         assert np.allclose(e, cases[1], atol=tol, rtol=0)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             [qml.PauliX(0), 0.0],
#             [qml.PauliX(1), -0.19866933079506122],
#             [qml.PauliY(0), -0.3894183423086505],
#             [qml.PauliY(1), 0.0],
#             [qml.PauliZ(0), 0.9210609940028852],
#             [qml.PauliZ(1), 0.9800665778412417],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_expval_qml_tape_wire01(self, cases, tol, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.RY(-0.2, wires=[1])

#         e = dev.expval(cases[0])
#         assert np.allclose(e, cases[1], atol=tol, rtol=0)


# class TestVar:
#     """Test Var"""

#     @pytest.fixture
#     def dev(self):
#         return qml.device("lightning.qubit", wires=2)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             ["PauliX", [0, 1]],
#             ["PauliY", [0, 1]],
#             ["PauliZ", [0, 1]],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_var_gate_error(self, cases, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.RY(-0.2, wires=[1])

#         with pytest.raises(
#             ValueError, match="The supplied gate requires 1 wires, but 2 were supplied."
#         ):
#             dev.var(cases[0], cases[1])

#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_var_wires_error(self, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.RY(-0.2, wires=[1])

#         with pytest.raises(AttributeError, match="'str' object has no attribute 'name'"):
#             dev.var("PauliX", wires=None)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             ["PauliX", [0], 0.9982450376077382],
#             ["PauliX", [1], 1.0],
#             ["PauliY", [0], 0.6956987716741251],
#             ["PauliY", [1], 1.0],
#             ["PauliZ", [0], 0.3060561907181374],
#             ["PauliZ", [1], -4.440892098500626e-16],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_var_tape_wire0(self, cases, tol, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         x, y, z = [0.5, 0.3, -0.7]

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.Rot(x, y, z, wires=[0])
#             qml.RY(-0.2, wires=[0])

#         e = dev.var(cases[0], cases[1])
#         assert np.allclose(e, cases[2], atol=tol, rtol=0)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             ["PauliX", [0], 1.0],
#             ["PauliX", [1], 0.9605304970014426],
#             ["PauliY", [0], 0.8483533546735826],
#             ["PauliY", [1], 1.0],
#             ["PauliZ", [0], 0.15164664532641725],
#             ["PauliZ", [1], 0.03946950299855745],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_var_tape_wire01(self, cases, tol, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.RY(-0.2, wires=[1])

#         e = dev.var(cases[0], cases[1])
#         assert np.allclose(e, cases[2], atol=tol, rtol=0)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             [qml.PauliX(0), 0.9982450376077382],
#             [qml.PauliX(1), 1.0],
#             [qml.PauliY(0), 0.6956987716741251],
#             [qml.PauliY(1), 1.0],
#             [qml.PauliZ(0), 0.3060561907181374],
#             [qml.PauliZ(1), -4.440892098500626e-16],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_var_qml_tape_wire0(self, cases, tol, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         x, y, z = [0.5, 0.3, -0.7]

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.Rot(x, y, z, wires=[0])
#             qml.RY(-0.2, wires=[0])

#         e = dev.var(cases[0])
#         assert np.allclose(e, cases[1], atol=tol, rtol=0)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             [qml.PauliX(0), 1.0],
#             [qml.PauliX(1), 0.9605304970014426],
#             [qml.PauliY(0), 0.8483533546735826],
#             [qml.PauliY(1), 1.0],
#             [qml.PauliZ(0), 0.15164664532641725],
#             [qml.PauliZ(1), 0.03946950299855745],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_var_qml_tape_wire01(self, cases, tol, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.RY(-0.2, wires=[1])

#         e = dev.var(cases[0])
#         assert np.allclose(e, cases[1], atol=tol, rtol=0)
