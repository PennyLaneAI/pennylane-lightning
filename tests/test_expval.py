# Copyright 2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the expval method of the :mod:`pennylane_lightning.LightningQubit` device.
"""
import pytest

import numpy as np
import pennylane as qml

from conftest import U, U2, A


np.random.seed(42)

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
class TestExpval:
    """Test expectation values"""

    def test_identity_expectation(self, theta, phi, qubit_device_3_wires, tol):
        """Test that identity expectation value (i.e. the trace) is 1"""
        dev = qubit_device_3_wires

        O1 = qml.Identity(wires=[0])
        O2 = qml.Identity(wires=[1])

        dev.apply(
            [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            rotations=[*O1.diagonalizing_gates(), *O2.diagonalizing_gates()],
        )

        res = np.array([dev.expval(O1), dev.expval(O2)])
        assert np.allclose(res, np.array([1, 1]), tol)

    def test_pauliz_expectation(self, theta, phi, qubit_device_3_wires, tol):
        """Test that PauliZ expectation value is correct"""
        dev = qubit_device_3_wires
        O1 = qml.PauliZ(wires=[0])
        O2 = qml.PauliZ(wires=[1])

        dev.apply(
            [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            rotations=[*O1.diagonalizing_gates(), *O2.diagonalizing_gates()],
        )

        res = np.array([dev.expval(O1), dev.expval(O2)])
        assert np.allclose(res, np.array([np.cos(theta), np.cos(theta) * np.cos(phi)]), tol)

    def test_paulix_expectation(self, theta, phi, qubit_device_3_wires, tol):
        """Test that PauliX expectation value is correct"""
        dev = qubit_device_3_wires
        O1 = qml.PauliX(wires=[0])
        O2 = qml.PauliX(wires=[1])

        dev.apply(
            [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            rotations=[*O1.diagonalizing_gates(), *O2.diagonalizing_gates()],
        )

        res = np.array([dev.expval(O1), dev.expval(O2)], dtype=dev.C_DTYPE)
        assert np.allclose(
            res, np.array([np.sin(theta) * np.sin(phi), np.sin(phi)], dtype=dev.C_DTYPE)
        )

    def test_pauliy_expectation(self, theta, phi, qubit_device_3_wires, tol):
        """Test that PauliY expectation value is correct"""
        dev = qubit_device_3_wires
        O1 = qml.PauliY(wires=[0])
        O2 = qml.PauliY(wires=[1])

        dev.apply(
            [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            rotations=[*O1.diagonalizing_gates(), *O2.diagonalizing_gates()],
        )

        res = np.array([dev.expval(O1), dev.expval(O2)])
        assert np.allclose(res, np.array([0, -np.cos(theta) * np.sin(phi)]), tol)

    def test_hadamard_expectation(self, theta, phi, qubit_device_3_wires, tol):
        """Test that Hadamard expectation value is correct"""
        dev = qubit_device_3_wires
        O1 = qml.Hadamard(wires=[0])
        O2 = qml.Hadamard(wires=[1])

        dev.apply(
            [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            rotations=[*O1.diagonalizing_gates(), *O2.diagonalizing_gates()],
        )

        res = np.array([dev.expval(O1), dev.expval(O2)])
        expected = np.array(
            [np.sin(theta) * np.sin(phi) + np.cos(theta), np.cos(theta) * np.cos(phi) + np.sin(phi)]
        ) / np.sqrt(2)
        assert np.allclose(res, expected, tol)


@pytest.mark.parametrize("diff_method", ("parameter-shift", "adjoint"))
class TestExpOperatorArithmetic:
    """Test integration of lightning with SProd, Prod, and Sum."""

    dev = qml.device("lightning.qubit", wires=2)

    def test_sprod(self, diff_method):
        """Test the `SProd` class with lightning qubit."""

        @qml.qnode(self.dev, diff_method=diff_method)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.s_prod(0.5, qml.PauliZ(0)))

        x = qml.numpy.array(0.123, requires_grad=True)
        res = circuit(x)
        assert qml.math.allclose(res, 0.5 * np.cos(x))

        g = qml.grad(circuit)(x)
        expected_grad = -0.5 * np.sin(x)
        assert qml.math.allclose(g, expected_grad)

    def test_prod(self, diff_method):
        """Test the `Prod` class with lightning qubit."""

        @qml.qnode(self.dev, diff_method=diff_method)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(1)
            qml.PauliZ(1)
            return qml.expval(qml.prod(qml.PauliZ(0), qml.PauliX(1)))

        x = qml.numpy.array(0.123, requires_grad=True)
        res = circuit(x)
        assert qml.math.allclose(res, -np.cos(x))

        g = qml.grad(circuit)(x)
        expected_grad = np.sin(x)
        assert qml.math.allclose(g, expected_grad)

    def test_sum(self, diff_method):
        """Test the `Sum` class with lightning qubit."""

        @qml.qnode(self.dev, diff_method=diff_method)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return qml.expval(qml.op_sum(qml.PauliZ(0), qml.PauliX(1)))

        x = qml.numpy.array(-3.21, requires_grad=True)
        y = qml.numpy.array(2.34, requires_grad=True)
        res = circuit(x, y)
        assert qml.math.allclose(res, np.cos(x) + np.sin(y))

        g = qml.grad(circuit)(x, y)
        expected = (-np.sin(x), np.cos(y))
        assert qml.math.allclose(g, expected)

    def test_integration(self, diff_method):
        """Test a Combination of `Sum`, `SProd`, and `Prod`."""

        obs = qml.op_sum(
            qml.s_prod(2.3, qml.PauliZ(0)), -0.5 * qml.prod(qml.PauliY(0), qml.PauliZ(1))
        )

        @qml.qnode(self.dev, diff_method=diff_method)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return qml.expval(obs)

        x = qml.numpy.array(0.654, requires_grad=True)
        y = qml.numpy.array(-0.634, requires_grad=True)

        res = circuit(x, y)
        expected = 2.3 * np.cos(x) + 0.5 * np.sin(x) * np.cos(y)
        assert qml.math.allclose(res, expected)

        g = qml.grad(circuit)(x, y)
        expected = (-2.3 * np.sin(x) + 0.5 * np.cos(y) * np.cos(x), -0.5 * np.sin(x) * np.sin(y))
        assert qml.math.allclose(g, expected)


@pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorExpval:
    """Test tensor expectation values"""

    def test_paulix_pauliy(self, theta, phi, varphi, qubit_device_3_wires, tol):
        """Test that a tensor product involving PauliX and PauliY works
        correctly"""
        dev = qubit_device_3_wires
        obs = qml.PauliX(0) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            rotations=obs.diagonalizing_gates(),
        )
        res = dev.expval(obs)

        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)

        assert np.allclose(res, expected, atol=tol)

    def test_pauliz_identity(self, theta, phi, varphi, qubit_device_3_wires, tol):
        """Test that a tensor product involving PauliZ and Identity works
        correctly"""
        dev = qubit_device_3_wires
        obs = qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            rotations=obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = np.cos(varphi) * np.cos(phi)

        assert np.allclose(res, expected, tol)

    def test_pauliz_hadamard_pauliy(self, theta, phi, varphi, qubit_device_3_wires, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard
        works correctly"""
        dev = qubit_device_3_wires
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            rotations=obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)
        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)

        assert np.allclose(res, expected, tol)
