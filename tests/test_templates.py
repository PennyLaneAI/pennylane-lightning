# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Integration tests for the ``execute`` method of Lightning devices.
"""
import functools

import pennylane as qml
import pytest
from conftest import LightningDevice, device_name
from pennylane import numpy as np

if LightningDevice._new_API and not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


class TestGrover:
    """Test Grover's algorithm (multi-controlled gates, decomposition, etc.)"""

    @pytest.mark.parametrize("num_qubits", range(4, 8))
    def test_grover(self, num_qubits):
        np.random.seed(42)
        omega = np.random.rand(num_qubits) > 0.5
        dev = qml.device(device_name, wires=num_qubits)
        wires = list(range(num_qubits))

        @qml.qnode(dev, diff_method=None)
        def circuit(omega):
            iterations = int(np.round(np.sqrt(2**num_qubits) * np.pi / 4))

            # Initial state preparation
            for wire in wires:
                qml.Hadamard(wires=wire)

            # Grover's iterator
            for _ in range(iterations):
                qml.FlipSign(omega, wires=wires)
                qml.templates.GroverOperator(wires)

            return qml.probs(wires=wires)

        prob = circuit(omega)
        index = omega.astype(int)
        index = functools.reduce(
            lambda sum, x: sum + (1 << x[0]) * x[1],
            zip([i for i in range(len(index) - 1, -1, -1)], index),
            0,
        )
        assert np.allclose(np.sum(prob), 1.0)
        assert prob[index] > 0.95
        assert np.sum(prob) - prob[index] < 0.05


class TestQSVT:
    """Test the QSVT algorithm."""

    def test_qsvt(self):
        dev = qml.device(device_name, wires=2)
        dq = qml.device("default.qubit")
        A = np.array([[0.1]])
        block_encode = qml.BlockEncode(A, wires=[0, 1])
        shifts = [qml.PCPhase(i + 0.1, dim=1, wires=[0, 1]) for i in range(3)]

        def circuit():
            qml.QSVT(block_encode, shifts)
            return qml.expval(qml.Z(0))

        res = qml.QNode(circuit, dev, diff_method=None)()
        ref = qml.QNode(circuit, dq, diff_method=None)()

        assert np.allclose(res, ref)
