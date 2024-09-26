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
Unit tests for the var method of the :mod:`pennylane_lightning.LightningQubit` device.
"""
import numpy as np
import pennylane as qml
import pytest
from conftest import PHI, THETA, VARPHI
from conftest import LightningDevice as ld
from conftest import device_name

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

np.random.seed(42)


@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
class TestVar:
    """Tests for the variance"""

    def test_var(self, theta, phi, qubit_device, tol):
        """Tests for variance calculation"""
        dev = qubit_device(wires=3)

        # test correct variance for <Z> of a rotated state
        obs = qml.PauliZ(wires=[0])
        ops = [
            qml.RX(phi, wires=[0]),
            qml.RY(theta, wires=[0]),
        ]

        tape = qml.tape.QuantumScript(ops, [qml.var(op=obs)])
        var = dev.execute(tape)

        expected = 0.25 * (3 - np.cos(2 * theta) - 2 * np.cos(theta) ** 2 * np.cos(2 * phi))

        assert np.allclose(var, expected, tol)

    @pytest.mark.skipif(
        device_name == "lightning.tensor", reason="lightning.tensor doesn't support projector."
    )
    def test_projector_var(self, theta, phi, qubit_device, tol):
        """Test that Projector variance value is correct"""
        n_qubits = 2
        dev_def = qml.device("default.qubit", wires=n_qubits)
        dev = qubit_device(wires=n_qubits)

        init_state = np.random.rand(2**n_qubits) + 1j * np.random.rand(2**n_qubits)
        init_state /= np.linalg.norm(init_state)
        obs = qml.Projector(np.array([0, 1, 0, 0]) / np.sqrt(2), wires=[0, 1])

        def circuit():
            qml.StatePrep(init_state, wires=range(n_qubits))
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(obs)

        circ = qml.QNode(circuit, dev)
        circ_def = qml.QNode(circuit, dev_def)
        assert np.allclose(circ(), circ_def(), tol)


@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorVar:
    """Tests for variance of tensor observables"""

    def test_paulix_pauliy(self, theta, phi, varphi, qubit_device, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qubit_device(wires=3)
        obs = qml.PauliX(0) @ qml.PauliY(2)
        ops = [
            qml.RX(theta, wires=[0]),
            qml.RX(phi, wires=[1]),
            qml.RX(varphi, wires=[2]),
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=[1, 2]),
        ]
        tape = qml.tape.QuantumScript(ops, [qml.var(op=obs)])
        res = dev.execute(tape)

        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16

        assert np.allclose(res, expected, tol)

    def test_pauliz_hadamard_pauliy(self, theta, phi, varphi, qubit_device, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qubit_device(wires=3)
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)
        ops = [
            qml.RX(theta, wires=[0]),
            qml.RX(phi, wires=[1]),
            qml.RX(varphi, wires=[2]),
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=[1, 2]),
        ]
        tape = qml.tape.QuantumScript(ops, [qml.var(op=obs)])
        res = dev.execute(tape)

        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4

        assert np.allclose(res, expected, tol)
