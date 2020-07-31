# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Integration tests that
"""
import numpy as np
import pennylane as qml
import pytest


@pytest.fixture
def lightning_qubit_dev(wires):
    """Loads ``lightning.qubit``"""
    return qml.device("lightning.qubit", wires=wires)


@pytest.fixture
def default_qubit_dev(wires):
    """Loads ``default.qubit``"""
    return qml.device("default.qubit", wires=wires)


def one_qubit_block(wires=None):
    """A block containing all of the supported gates in ``lightning.qubit``"""
    qml.PauliX(wires=wires)
    qml.PauliY(wires=wires)
    qml.S(wires=wires)
    qml.Hadamard(wires=wires)
    qml.PauliX(wires=wires)
    qml.T(wires=wires)
    qml.PhaseShift(-1, wires=wires)
    qml.Rot(0.1, 0.2, 0.3, wires=wires)
    qml.RZ(0.11, wires=wires)
    qml.RY(0.22, wires=wires)
    qml.RX(0.33, wires=wires)
    qml.PauliX(wires=wires)


@pytest.mark.usefixtures("lightning_qubit_dev", "default_qubit_dev")
class TestComparison:
    """A test that compares the output states of ``lightning.qubit`` and ``default.qubit`` for a
    variety of different circuits. This uses ``default.qubit`` as a gold standard to compare
    against."""

    @pytest.mark.parametrize("wires", [1])
    def test_one_qubit_circuit(self, wires, lightning_qubit_dev, default_qubit_dev):
        """Test a single qubit circuit"""

        def circuit():
            """A combination of the one_qubit_block and a simple PauliZ measurement"""
            one_qubit_block(wires=0)
            return qml.expval(qml.PauliZ(0))

        lightning = qml.QNode(circuit, lightning_qubit_dev)
        default = qml.QNode(circuit, default_qubit_dev)

        lightning()
        lightning_state = lightning_qubit_dev.state

        default()
        default_state = default_qubit_dev.state

        assert np.allclose(lightning_state, default_state)
