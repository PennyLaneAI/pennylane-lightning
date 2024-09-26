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
Integration tests that compare the output states of the
compiled Lightning device with the ``default.qubit``.
"""
import itertools
import os

import numpy as np
import pennylane as qml
import pytest
from conftest import LightningDevice as ld
from conftest import device_name

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


def lightning_backend_dev(wires):
    """Loads the lightning backend"""
    return qml.device(device_name, wires=wires)


def lightning_backend_batch_obs_dev(wires):
    """Loads the lightning backend"""
    return qml.device(device_name, wires=wires, batch_obs=True)


def default_qubit_dev(wires):
    """Loads ``default.qubit``"""
    return qml.device("default.qubit", wires=wires)


def one_qubit_block(wires=None):
    """A block containing all supported gates"""
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
    """A test that compares the output states of the lightning device and ``default.qubit`` for a
    variety of different circuits. This uses ``default.qubit`` as a reference."""

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device does not support one-qubit circuits",
    )
    @pytest.mark.parametrize("basis_state", itertools.product(*[(0, 1)] * 1))
    @pytest.mark.parametrize("wires", [1])
    @pytest.mark.parametrize(
        "lightning_dev_version", [lightning_backend_dev, lightning_backend_batch_obs_dev]
    )
    @pytest.mark.parametrize("num_threads", [1, 2])
    def test_one_qubit_circuit(
        self, monkeypatch, wires, lightning_dev_version, basis_state, num_threads
    ):
        """Test a single-qubit circuit"""

        monkeypatch.setenv("OMP_NUM_THREADS", str(num_threads))

        def circuit(measurement):
            """A combination of the one_qubit_block and a simple PauliZ measurement applied to a
            basis state"""
            qml.BasisState(np.array(basis_state), wires=0)
            one_qubit_block(wires=0)
            return measurement() if callable(measurement) else measurement

        dev_l = lightning_dev_version(wires)
        dev_d = default_qubit_dev(wires)

        lightning = qml.QNode(circuit, dev_l)
        default = qml.QNode(circuit, dev_d)

        lightning(qml.expval(qml.PauliZ(0)))
        # pylint: disable=protected-access
        lightning_state = dev_l._statevector.state if dev_l._new_API else dev_l.state

        default_state = default(qml.state)

        assert np.allclose(lightning_state, default_state)
        assert os.getenv("OMP_NUM_THREADS") == str(num_threads)

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device does not support direct access to the state",
    )
    @pytest.mark.parametrize("basis_state", itertools.product(*[(0, 1)] * 2))
    @pytest.mark.parametrize("wires", [2])
    @pytest.mark.parametrize(
        "lightning_dev_version",
        ([lightning_backend_dev, lightning_backend_batch_obs_dev]),
    )
    @pytest.mark.parametrize("num_threads", [1, 2])
    def test_two_qubit_circuit(
        self, monkeypatch, wires, lightning_dev_version, basis_state, num_threads
    ):
        """Test a two-qubit circuit"""

        monkeypatch.setenv("OMP_NUM_THREADS", str(num_threads))

        def circuit(measurement):
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
            return measurement() if callable(measurement) else measurement

        dev_l = lightning_dev_version(wires)
        dev_d = default_qubit_dev(wires)

        lightning = qml.QNode(circuit, dev_l)
        default = qml.QNode(circuit, dev_d)

        lightning(qml.expval(qml.PauliZ(0)))

        default_state = default(qml.state)

        # pylint: disable=protected-access
        lightning_state = dev_l._statevector.state if dev_l._new_API else dev_l.state
        assert np.allclose(lightning_state, default_state)

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device does not support the direct access to state",
    )
    @pytest.mark.parametrize("basis_state", itertools.product(*[(0, 1)] * 3))
    @pytest.mark.parametrize("wires", [3])
    @pytest.mark.parametrize(
        "lightning_dev_version",
        ([lightning_backend_dev, lightning_backend_batch_obs_dev]),
    )
    @pytest.mark.parametrize("num_threads", [1, 2])
    def test_three_qubit_circuit(
        self, monkeypatch, wires, lightning_dev_version, basis_state, num_threads
    ):
        """Test a three-qubit circuit"""

        monkeypatch.setenv("OMP_NUM_THREADS", str(num_threads))

        def circuit(measurement):
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
            return measurement() if callable(measurement) else measurement

        dev_l = lightning_dev_version(wires)
        dev_d = default_qubit_dev(wires)

        lightning = qml.QNode(circuit, dev_l)
        default = qml.QNode(circuit, dev_d)

        lightning(qml.expval(qml.PauliZ(0)))

        default_state = default(qml.state)

        # pylint: disable=protected-access
        lightning_state = dev_l._statevector.state if dev_l._new_API else dev_l.state
        assert np.allclose(lightning_state, default_state)

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device does not support the direct access to state",
    )
    @pytest.mark.parametrize("basis_state", itertools.product(*[(0, 1)] * 4))
    @pytest.mark.parametrize("wires", [4])
    @pytest.mark.parametrize(
        "lightning_dev_version",
        ([lightning_backend_dev, lightning_backend_batch_obs_dev]),
    )
    @pytest.mark.parametrize("num_threads", [1, 2])
    def test_four_qubit_circuit(
        self, monkeypatch, wires, lightning_dev_version, basis_state, num_threads
    ):
        """Test a four-qubit circuit"""

        monkeypatch.setenv("OMP_NUM_THREADS", str(num_threads))

        def circuit(measurement):
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
            return measurement() if callable(measurement) else measurement

        dev_l = lightning_dev_version(wires)
        dev_d = default_qubit_dev(wires)

        lightning = qml.QNode(circuit, dev_l)
        default = qml.QNode(circuit, dev_d)

        lightning(qml.expval(qml.PauliZ(0)))

        default_state = default(qml.state)

        # pylint: disable=protected-access
        lightning_state = dev_l._statevector.state if dev_l._new_API else dev_l.state
        assert np.allclose(lightning_state, default_state)

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device does not support does not support the direct access to state",
    )
    @pytest.mark.parametrize(
        "lightning_dev_version", [lightning_backend_dev, lightning_backend_batch_obs_dev]
    )
    @pytest.mark.parametrize("wires", range(1, 17))
    @pytest.mark.parametrize("num_threads", [1, 2])
    @pytest.mark.parametrize("stateprep", [qml.QubitStateVector, qml.StatePrep])
    def test_n_qubit_circuit(
        self, monkeypatch, stateprep, wires, lightning_dev_version, num_threads
    ):
        """Test an n-qubit circuit"""

        monkeypatch.setenv("OMP_NUM_THREADS", str(num_threads))

        vec = np.array([1] * (2**wires)) / np.sqrt(2**wires)
        shape = qml.StronglyEntanglingLayers.shape(2, wires)
        w = np.random.uniform(high=2 * np.pi, size=shape)

        def circuit(measurement):
            """Prepares the equal superposition state and then applies StronglyEntanglingLayers
            and concludes with a simple PauliZ measurement"""
            stateprep(vec, wires=range(wires))
            qml.StronglyEntanglingLayers(w, wires=range(wires))
            return measurement() if callable(measurement) else measurement

        dev_l = lightning_dev_version(wires)
        dev_d = default_qubit_dev(wires)

        lightning = qml.QNode(circuit, dev_l)
        default = qml.QNode(circuit, dev_d)

        lightning(qml.expval(qml.PauliZ(0)))
        # pylint: disable=protected-access
        lightning_state = dev_l._statevector.state if dev_l._new_API else dev_l.state

        default_state = default(qml.state)

        assert np.allclose(lightning_state, default_state)
