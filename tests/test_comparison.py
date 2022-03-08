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
import os

import pennylane as qml
from pennylane_lightning.lightning_qubit import CPP_BINARY_AVAILABLE


def lightning_qubit_dev(wires):
    """Loads ``lightning.qubit``"""
    return qml.device("lightning.qubit", wires=wires)


def lightning_qubit_batch_obs_dev(wires):
    """Loads ``lightning.qubit``"""
    return qml.device("lightning.qubit", wires=wires, batch_obs=True)


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


class TestComparison:
    """A test that compares the output states of ``lightning.qubit`` and ``default.qubit`` for a
    variety of different circuits. This uses ``default.qubit`` as a gold standard to compare
    against."""

    @pytest.mark.parametrize("basis_state", itertools.product(*[(0, 1)] * 1))
    @pytest.mark.parametrize("wires", [1])
    @pytest.mark.parametrize(
        "lightning_dev_version", [lightning_qubit_dev, lightning_qubit_batch_obs_dev]
    )
    @pytest.mark.skipif(not CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    @pytest.mark.parametrize("num_threads", [1, 2])
    def test_one_qubit_circuit(
        self, monkeypatch, wires, lightning_dev_version, basis_state, num_threads
    ):
        """Test a single-qubit circuit"""

        monkeypatch.setenv("OMP_NUM_THREADS", str(num_threads))

        def circuit():
            """A combination of the one_qubit_block and a simple PauliZ measurement applied to a
            basis state"""
            qml.BasisState(np.array(basis_state), wires=0)
            one_qubit_block(wires=0)
            return qml.expval(qml.PauliZ(0))

        dev_l = lightning_dev_version(wires)
        dev_d = default_qubit_dev(wires)

        lightning = qml.QNode(circuit, dev_l)
        default = qml.QNode(circuit, dev_d)

        lightning()
        lightning_state = dev_l.state

        default()
        default_state = dev_d.state

        assert np.allclose(lightning_state, default_state)
        assert os.getenv("OMP_NUM_THREADS") == str(num_threads)

    @pytest.mark.parametrize("basis_state", itertools.product(*[(0, 1)] * 2))
    @pytest.mark.parametrize("wires", [2])
    @pytest.mark.parametrize(
        "lightning_dev_version", [lightning_qubit_dev, lightning_qubit_batch_obs_dev]
    )
    @pytest.mark.parametrize("num_threads", [1, 2])
    @pytest.mark.skipif(not CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_two_qubit_circuit(
        self, monkeypatch, wires, lightning_dev_version, basis_state, num_threads
    ):
        """Test a two-qubit circuit"""
        monkeypatch.setenv("OMP_NUM_THREADS", str(num_threads))

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
            qml.CRX(0.5, wires=[1, 0])
            qml.CRY(0.9, wires=[0, 1])
            one_qubit_block(wires=1)
            qml.CRZ(0.02, wires=[0, 1])
            qml.CRot(0.2, 0.3, 0.7, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        dev_l = lightning_dev_version(wires)
        dev_d = default_qubit_dev(wires)

        lightning = qml.QNode(circuit, dev_l)
        default = qml.QNode(circuit, dev_d)

        lightning()
        lightning_state = dev_l.state

        default()
        default_state = dev_d.state

        assert np.allclose(lightning_state, default_state)

    @pytest.mark.parametrize("basis_state", itertools.product(*[(0, 1)] * 3))
    @pytest.mark.parametrize("wires", [3])
    @pytest.mark.parametrize(
        "lightning_dev_version", [lightning_qubit_dev, lightning_qubit_batch_obs_dev]
    )
    @pytest.mark.parametrize("num_threads", [1, 2])
    @pytest.mark.skipif(not CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_three_qubit_circuit(
        self, monkeypatch, wires, lightning_dev_version, basis_state, num_threads
    ):
        """Test a three-qubit circuit"""
        monkeypatch.setenv("OMP_NUM_THREADS", str(num_threads))

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
            one_qubit_block(wires=2)
            qml.SWAP(wires=[0, 1])
            qml.SWAP(wires=[0, 2])
            qml.CRX(0.5, wires=[1, 0])
            qml.CSWAP(wires=[2, 1, 0])
            qml.CRY(0.9, wires=[2, 1])
            one_qubit_block(wires=1)
            qml.CRZ(0.02, wires=[0, 1])
            qml.CRot(0.2, 0.3, 0.7, wires=[2, 1])
            qml.RZ(0.4, wires=0)
            qml.Toffoli(wires=[2, 1, 0])
            return qml.expval(qml.PauliZ(0))

        dev_l = lightning_dev_version(wires)
        dev_d = default_qubit_dev(wires)

        lightning = qml.QNode(circuit, dev_l)
        default = qml.QNode(circuit, dev_d)

        lightning()
        lightning_state = dev_l.state

        default()
        default_state = dev_d.state

        assert np.allclose(lightning_state, default_state)

    @pytest.mark.parametrize("basis_state", itertools.product(*[(0, 1)] * 4))
    @pytest.mark.parametrize("wires", [4])
    @pytest.mark.parametrize(
        "lightning_dev_version", [lightning_qubit_dev, lightning_qubit_batch_obs_dev]
    )
    @pytest.mark.parametrize("num_threads", [1, 2])
    @pytest.mark.skipif(not CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_four_qubit_circuit(
        self, monkeypatch, wires, lightning_dev_version, basis_state, num_threads
    ):
        """Test a four-qubit circuit"""
        monkeypatch.setenv("OMP_NUM_THREADS", str(num_threads))

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

        dev_l = lightning_dev_version(wires)
        dev_d = default_qubit_dev(wires)

        lightning = qml.QNode(circuit, dev_l)
        default = qml.QNode(circuit, dev_d)

        lightning()
        lightning_state = dev_l.state

        default()
        default_state = dev_d.state

        assert np.allclose(lightning_state, default_state)

    @pytest.mark.parametrize(
        "lightning_dev_version", [lightning_qubit_dev, lightning_qubit_batch_obs_dev]
    )
    @pytest.mark.parametrize("wires", range(1, 17))
    @pytest.mark.parametrize("num_threads", [1, 2])
    @pytest.mark.skipif(not CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_n_qubit_circuit(self, monkeypatch, wires, lightning_dev_version, num_threads):
        """Test an n-qubit circuit"""
        monkeypatch.setenv("OMP_NUM_THREADS", str(num_threads))

        vec = np.array([1] * (2**wires)) / np.sqrt(2**wires)
        shape = qml.StronglyEntanglingLayers.shape(2, wires)
        w = np.random.uniform(high=2 * np.pi, size=shape)

        def circuit():
            """Prepares the equal superposition state and then applies StronglyEntanglingLayers
            and concludes with a simple PauliZ measurement"""
            qml.QubitStateVector(vec, wires=range(wires))
            qml.StronglyEntanglingLayers(w, wires=range(wires))
            return qml.expval(qml.PauliZ(0))

        dev_l = lightning_dev_version(wires)
        dev_d = default_qubit_dev(wires)

        lightning = qml.QNode(circuit, dev_l)
        default = qml.QNode(circuit, dev_d)

        lightning()
        lightning_state = dev_l.state

        default()
        default_state = dev_d.state

        assert np.allclose(lightning_state, default_state)
