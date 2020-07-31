import pytest

import numpy as np
import pennylane as qml

from conftest import U, U2, A


np.random.seed(42)

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
@pytest.mark.parametrize("shots", [8192])
class TestVar:
    """Tests for the variance"""

    def test_var(self, theta, phi, shots, tol):
        """Tests for variance calculation"""
        dev = qml.device("lightning.qubit", wires=3)

        # test correct variance for <Z> of a rotated state
        observable = qml.PauliZ(wires=[0])

        dev.apply(
            [
                qml.RX(phi, wires=[0]),
                qml.RY(theta, wires=[0]),
            ],
            rotations=[*observable.diagonalizing_gates()]
        )

        dev._samples = dev.generate_samples()

        var = dev.var(observable)
        expected = 0.25 * (3 - np.cos(2 * theta) - 2 * np.cos(theta) ** 2 * np.cos(2 * phi))

        assert np.allclose(var, expected, tol)


@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
@pytest.mark.parametrize("shots", [8192])
class TestTensorVar:
    """Tests for variance of tensor observables"""

    def test_paulix_pauliy(self, theta, phi, varphi, shots, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("lightning.qubit", wires=3)
        obs = qml.PauliX(0) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            rotations=obs.diagonalizing_gates()
        )

        dev._samples = dev.generate_samples()
        res = dev.var(obs)

        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16

        assert np.allclose(res, expected, tol)

    def test_pauliz_hadamard_pauliy(self, theta, phi, varphi, shots, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qml.device("lightning.qubit", wires=3)
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            rotations=obs.diagonalizing_gates()
        )

        dev._samples = dev.generate_samples()
        res = dev.var(obs)

        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4

        assert np.allclose(res, expected, tol)
