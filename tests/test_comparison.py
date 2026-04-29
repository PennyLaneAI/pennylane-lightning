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
import pennylane as qp
import pytest
from conftest import LightningDevice as ld
from conftest import device_name

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


def lightning_backend_dev(wires):
    """Loads the lightning backend"""
    return qp.device(device_name, wires=wires)


def lightning_backend_batch_obs_dev(wires):
    """Loads the lightning backend"""
    return qp.device(device_name, wires=wires, batch_obs=True)


def default_qubit_dev(wires):
    """Loads ``default.qubit``"""
    return qp.device("default.qubit", wires=wires)


def one_qubit_block(wires=None):
    """A block containing all supported gates"""
    qp.PauliX(wires=wires)
    qp.PauliY(wires=wires)
    qp.S(wires=wires)
    qp.Hadamard(wires=wires)
    qp.PauliX(wires=wires)
    qp.T(wires=wires)
    qp.PhaseShift(-1, wires=wires)
    qp.Rot(0.1, 0.2, 0.3, wires=wires)
    qp.RZ(0.11, wires=wires)
    qp.RY(0.22, wires=wires)
    qp.RX(0.33, wires=wires)
    qp.PauliX(wires=wires)


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
            qp.BasisState(np.array(basis_state), wires=0)
            one_qubit_block(wires=0)
            return measurement() if callable(measurement) else qp.apply(measurement)

        dev_l = lightning_dev_version(wires)
        dev_d = default_qubit_dev(wires)

        lightning = qp.QNode(circuit, dev_l)
        default = qp.QNode(circuit, dev_d)

        lightning(qp.expval(qp.PauliZ(0)))
        # pylint: disable=protected-access

        default_state = default(qp.state)

        assert np.allclose(dev_l._statevector.state, default_state)
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
            qp.BasisState(np.array(basis_state), wires=[0, 1])
            qp.RX(0.5, wires=0)
            qp.Hadamard(wires=1)
            qp.CNOT(wires=[0, 1])
            qp.CZ(wires=[1, 0])
            one_qubit_block(wires=0)
            qp.SWAP(wires=[0, 1])
            qp.CRX(0.5, wires=[1, 0])
            qp.CRY(0.9, wires=[0, 1])
            one_qubit_block(wires=1)
            qp.CRZ(0.02, wires=[0, 1])
            qp.CRot(0.2, 0.3, 0.7, wires=[0, 1])
            return measurement() if callable(measurement) else qp.apply(measurement)

        dev_l = lightning_dev_version(wires)
        dev_d = default_qubit_dev(wires)

        lightning = qp.QNode(circuit, dev_l)
        default = qp.QNode(circuit, dev_d)

        lightning(qp.expval(qp.PauliZ(0)))

        default_state = default(qp.state)

        # pylint: disable=protected-access
        assert np.allclose(dev_l._statevector.state, default_state)

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
            qp.BasisState(np.array(basis_state), wires=[0, 1, 2])
            qp.RX(0.5, wires=0)
            qp.Hadamard(wires=1)
            qp.RY(0.9, wires=2)
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[2, 0])
            qp.CZ(wires=[1, 0])
            one_qubit_block(wires=0)
            qp.Toffoli(wires=[1, 0, 2])
            one_qubit_block(wires=2)
            qp.SWAP(wires=[0, 1])
            qp.SWAP(wires=[0, 2])
            qp.CRX(0.5, wires=[1, 0])
            qp.CSWAP(wires=[2, 1, 0])
            qp.CRY(0.9, wires=[2, 1])
            one_qubit_block(wires=1)
            qp.CRZ(0.02, wires=[0, 1])
            qp.CRot(0.2, 0.3, 0.7, wires=[2, 1])
            qp.RZ(0.4, wires=0)
            qp.Toffoli(wires=[2, 1, 0])
            return measurement() if callable(measurement) else qp.apply(measurement)

        dev_l = lightning_dev_version(wires)
        dev_d = default_qubit_dev(wires)

        lightning = qp.QNode(circuit, dev_l)
        default = qp.QNode(circuit, dev_d)

        lightning(qp.expval(qp.PauliZ(0)))

        default_state = default(qp.state)

        # pylint: disable=protected-access
        assert np.allclose(dev_l._statevector.state, default_state)

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
            qp.BasisState(np.array(basis_state), wires=[0, 1, 2, 3])
            qp.RX(0.5, wires=0)
            qp.Hadamard(wires=1)
            qp.RY(0.9, wires=2)
            qp.Rot(0.1, -0.2, -0.3, wires=3)
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[3, 1])
            one_qubit_block(wires=3)
            qp.CNOT(wires=[2, 0])
            qp.CZ(wires=[1, 0])
            one_qubit_block(wires=0)
            qp.Toffoli(wires=[1, 0, 2])
            one_qubit_block(wires=2)
            qp.SWAP(wires=[0, 1])
            qp.SWAP(wires=[0, 2])
            qp.Toffoli(wires=[1, 3, 2])
            qp.CRX(0.5, wires=[1, 0])
            qp.CSWAP(wires=[2, 1, 0])
            qp.CRY(0.9, wires=[2, 1])
            one_qubit_block(wires=1)
            qp.CRZ(0.02, wires=[0, 1])
            qp.CRY(0.9, wires=[2, 3])
            qp.CRot(0.2, 0.3, 0.7, wires=[2, 1])
            qp.RZ(0.4, wires=0)
            qp.Toffoli(wires=[2, 1, 0])
            return measurement() if callable(measurement) else qp.apply(measurement)

        dev_l = lightning_dev_version(wires)
        dev_d = default_qubit_dev(wires)

        lightning = qp.QNode(circuit, dev_l)
        default = qp.QNode(circuit, dev_d)

        lightning(qp.expval(qp.PauliZ(0)))

        default_state = default(qp.state)

        # pylint: disable=protected-access
        assert np.allclose(dev_l._statevector.state, default_state)

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device does not support does not support the direct access to state",
    )
    @pytest.mark.parametrize(
        "lightning_dev_version", [lightning_backend_dev, lightning_backend_batch_obs_dev]
    )
    @pytest.mark.parametrize("wires", range(1, 17))
    @pytest.mark.parametrize("num_threads", [1, 2])
    def test_n_qubit_circuit(self, monkeypatch, wires, lightning_dev_version, num_threads, seed):
        """Test an n-qubit circuit"""

        monkeypatch.setenv("OMP_NUM_THREADS", str(num_threads))

        vec = np.array([1] * (2**wires)) / np.sqrt(2**wires)
        shape = qp.StronglyEntanglingLayers.shape(2, wires)

        rng = np.random.default_rng(seed)
        w = rng.uniform(high=2 * np.pi, size=shape)

        def circuit(measurement):
            """Prepares the equal superposition state and then applies StronglyEntanglingLayers
            and concludes with a simple PauliZ measurement"""
            qp.StatePrep(vec, wires=range(wires))
            qp.StronglyEntanglingLayers(w, wires=range(wires))
            return measurement() if callable(measurement) else qp.apply(measurement)

        dev_l = lightning_dev_version(wires)
        dev_d = default_qubit_dev(wires)

        lightning = qp.QNode(circuit, dev_l)
        default = qp.QNode(circuit, dev_d)

        lightning(qp.expval(qp.PauliZ(0)))
        # pylint: disable=protected-access

        default_state = default(qp.state)

        assert np.allclose(dev_l._statevector.state, default_state)
