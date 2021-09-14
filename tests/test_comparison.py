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
Integration tests that compare the output states of ``lightning.qubit`` with ``default.qubit``.
"""
import itertools

import numpy as np
import pytest

import pennylane as qml


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

    @pytest.mark.parametrize("basis_state", itertools.product(*[(0, 1)] * 1))
    @pytest.mark.parametrize("wires", [1])
    def test_one_qubit_circuit(self, wires, lightning_qubit_dev, default_qubit_dev, basis_state):
        """Test a single-qubit circuit"""

        def circuit():
            """A combination of the one_qubit_block and a simple PauliZ measurement applied to a
            basis state"""
            qml.BasisState(np.array(basis_state), wires=0)
            one_qubit_block(wires=0)
            return qml.expval(qml.PauliZ(0))

        lightning = qml.QNode(circuit, lightning_qubit_dev)
        default = qml.QNode(circuit, default_qubit_dev)

        lightning()
        lightning_state = lightning_qubit_dev.state

        default()
        default_state = default_qubit_dev.state

        assert np.allclose(lightning_state, default_state)

    @pytest.mark.parametrize("basis_state", itertools.product(*[(0, 1)] * 2))
    @pytest.mark.parametrize("wires", [2])
    def test_two_qubit_circuit(self, wires, lightning_qubit_dev, default_qubit_dev, basis_state):
        """Test a two-qubit circuit"""

        def circuit():
            """A combination of two qubit gates with the one_qubit_block and a simple PauliZ
            measurement applied to an input basis state"""
            qml.BasisState(np.array(basis_state), wires=[0, 1])
            qml.RX(0.5, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            qml.CZ(wires=[1, 0])
            one_qubit_block(wires=0)
            qml.SWAP(wires=[0, 1])
            qml.IsingXX(0.44, wires=[1,0])
            qml.CRX(0.5, wires=[1, 0])
            qml.CRY(0.9, wires=[0, 1])
            one_qubit_block(wires=1)
            qml.IsingZZ(0.66, wires=[0,1])
            qml.CRZ(0.02, wires=[0, 1])
            qml.IsingYY(0.55, wires=[1,0])
            qml.CRot(0.2, 0.3, 0.7, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        lightning = qml.QNode(circuit, lightning_qubit_dev)
        default = qml.QNode(circuit, default_qubit_dev)

        lightning()
        lightning_state = lightning_qubit_dev.state

        default()
        default_state = default_qubit_dev.state
        assert np.allclose(lightning_state, default_state)

    @pytest.mark.parametrize("basis_state", itertools.product(*[(0, 1)] * 3))
    @pytest.mark.parametrize("wires", [3])
    def test_three_qubit_circuit(self, wires, lightning_qubit_dev, default_qubit_dev, basis_state):
        """Test a three-qubit circuit"""

        def circuit():
            """A combination of two and three qubit gates with the one_qubit_block and a simple
            PauliZ measurement applied to an input basis state"""
            qml.BasisState(np.array(basis_state), wires=[0, 1, 2])
            qml.RX(0.5, wires=0)
            qml.Hadamard(wires=1)
            qml.RY(0.9, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[2, 0])
            qml.CZ(wires=[1, 0])
            one_qubit_block(wires=0)
            qml.Toffoli(wires=[1, 0, 2])
            qml.IsingXX(0.398, wires=[2,0])
            one_qubit_block(wires=2)
            qml.SWAP(wires=[0, 1])
            qml.SWAP(wires=[0, 2])
            qml.CRX(0.5, wires=[1, 0])
            qml.IsingYY(0.152, wires=[1,0])
            qml.CSWAP(wires=[2, 1, 0])
            qml.CRY(0.9, wires=[2, 1])
            one_qubit_block(wires=1)
            qml.CRZ(0.02, wires=[0, 1])
            qml.CRot(0.2, 0.3, 0.7, wires=[2, 1])
            qml.IsingZZ(0.452, wires=[0,1])
            qml.RZ(0.4, wires=0)
            qml.Toffoli(wires=[2, 1, 0])
            return qml.expval(qml.PauliZ(0))

        lightning = qml.QNode(circuit, lightning_qubit_dev)
        default = qml.QNode(circuit, default_qubit_dev)

        lightning()
        lightning_state = lightning_qubit_dev.state

        default()
        default_state = default_qubit_dev.state
        assert np.allclose(lightning_state, default_state)

    @pytest.mark.parametrize("basis_state", itertools.product(*[(0, 1)] * 4))
    @pytest.mark.parametrize("wires", [4])
    def test_four_qubit_circuit(self, wires, lightning_qubit_dev, default_qubit_dev, basis_state):
        """Test a four-qubit circuit"""

        def circuit():
            """A combination of two and three qubit gates with the one_qubit_block and a simple
            PauliZ measurement, all acting on a four qubit input basis state"""
            qml.BasisState(np.array(basis_state), wires=[0, 1, 2, 3])
            qml.RX(0.5, wires=0)
            qml.Hadamard(wires=1)
            qml.RY(0.9, wires=2)
            qml.Rot(0.1, -0.2, -0.3, wires=3)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[3, 1])
            one_qubit_block(wires=3)
            qml.CNOT(wires=[2, 0])
            qml.CZ(wires=[1, 0])
            one_qubit_block(wires=0)
            qml.Toffoli(wires=[1, 0, 2])
            one_qubit_block(wires=2)
            qml.SWAP(wires=[0, 1])
            qml.SWAP(wires=[0, 2])
            qml.Toffoli(wires=[1, 3, 2])
            qml.CRX(0.5, wires=[1, 0])
            qml.CSWAP(wires=[2, 1, 0])
            qml.CRY(0.9, wires=[2, 1])
            one_qubit_block(wires=1)
            qml.CRZ(0.02, wires=[0, 1])
            qml.CRY(0.9, wires=[2, 3])
            qml.CRot(0.2, 0.3, 0.7, wires=[2, 1])
            qml.RZ(0.4, wires=0)
            qml.Toffoli(wires=[2, 1, 0])
            return qml.expval(qml.PauliZ(0))

        lightning = qml.QNode(circuit, lightning_qubit_dev)
        default = qml.QNode(circuit, default_qubit_dev)

        lightning()
        lightning_state = lightning_qubit_dev.state

        default()
        default_state = default_qubit_dev.state
        assert np.allclose(lightning_state, default_state)

    @pytest.mark.parametrize("wires", range(1, 17))
    def test_n_qubit_circuit(self, wires, lightning_qubit_dev, default_qubit_dev):
        """Test an n-qubit circuit"""

        vec = np.array([1] * (2 ** wires)) / np.sqrt(2 ** wires)
        w = qml.init.strong_ent_layers_uniform(2, wires)

        def circuit():
            """Prepares the equal superposition state and then applies StronglyEntanglingLayers
            and concludes with a simple PauliZ measurement"""
            qml.QubitStateVector(vec, wires=range(wires))
            qml.templates.StronglyEntanglingLayers(w, wires=range(wires))
            return qml.expval(qml.PauliZ(0))

        lightning = qml.QNode(circuit, lightning_qubit_dev)
        default = qml.QNode(circuit, default_qubit_dev)

        lightning()
        lightning_state = lightning_qubit_dev.state

        default()
        default_state = default_qubit_dev.state
        assert np.allclose(lightning_state, default_state)
