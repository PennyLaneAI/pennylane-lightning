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

    @pytest.mark.parametrize("n_qubits", range(4, 8))
    def test_grover(self, n_qubits):
        np.random.seed(42)
        omega = np.random.rand(n_qubits) > 0.5
        dev = qml.device(device_name, wires=n_qubits)
        wires = list(range(n_qubits))

        @qml.qnode(dev, diff_method=None)
        def circuit(omega):
            iterations = int(np.round(np.sqrt(2**n_qubits) * np.pi / 4))

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

    @pytest.mark.parametrize("n_qubits", range(2, 12, 2))
    def test_qsvt(self, n_qubits):
        dev = qml.device(device_name, wires=n_qubits)
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


class TestAngleEmbedding:
    """Test the AngleEmbedding algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 12, 2))
    def test_angleembedding(self, n_qubits):
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(feature_vector):
            qml.AngleEmbedding(features=feature_vector, wires=range(n_qubits), rotation="Z")
            qml.Hadamard(0)
            return qml.probs(wires=range(n_qubits))

        X = np.random.rand(n_qubits)

        res = qml.QNode(circuit, dev, diff_method=None)(X)
        ref = qml.QNode(circuit, dq, diff_method=None)(X)

        assert np.allclose(res, ref)


class TestAmplitudeEmbedding:
    """Test the AmplitudeEmbedding algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 12, 2))
    def test_amplitudeembedding(self, n_qubits):
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(f=None):
            qml.AmplitudeEmbedding(features=f, wires=range(n_qubits))
            if LightningDevice._new_API:
                return qml.expval(qml.Z(0)), qml.state()
            else:
                return qml.state()

        X = np.random.rand(2**n_qubits)
        X /= np.linalg.norm(X)
        if LightningDevice._new_API:
            res, res_state = qml.QNode(circuit, dev, diff_method=None)(f=X)
            ref, ref_state = qml.QNode(circuit, dq, diff_method=None)(f=X)
        else:
            res, res_state = 1.0, qml.QNode(circuit, dev, diff_method=None)(f=X)
            ref, ref_state = 1.0, qml.QNode(circuit, dq, diff_method=None)(f=X)

        assert np.allclose(res, ref)
        assert np.allclose(res_state, ref_state)


class TestBasisEmbedding:
    """Test the BasisEmbedding algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 12, 2))
    def test_basisembedding(self, n_qubits):
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(feature_vector):
            qml.BasisEmbedding(features=feature_vector, wires=range(n_qubits))
            return qml.state()

        X = np.ones(n_qubits)

        res = qml.QNode(circuit, dev, diff_method=None)(X)
        ref = qml.QNode(circuit, dq, diff_method=None)(X)

        assert np.allclose(res, ref)


class TestDisplacementEmbedding:
    """Test the DisplacementEmbedding algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 12, 2))
    @pytest.mark.parametrize("template", [qml.DisplacementEmbedding, qml.SqueezingEmbedding])
    def test_displacementembedding(self, n_qubits, template):
        dev = qml.device(device_name, wires=n_qubits)

        def circuit(feature_vector):
            template(features=feature_vector, wires=range(n_qubits))
            qml.QuadraticPhase(0.1, wires=1)
            return qml.expval(qml.NumberOperator(wires=1))

        X = np.arange(1, n_qubits + 1)

        with pytest.raises(qml._device.DeviceError, match="not supported"):
            _ = qml.QNode(circuit, dev, diff_method=None)(X)


class TestIQPEmbedding:
    """Test the IQPEmbedding algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 12, 2))
    def test_iqpembedding(self, n_qubits):
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(feature_vector):
            qml.IQPEmbedding(feature_vector, wires=range(n_qubits))
            return [qml.expval(qml.Z(w)) for w in range(n_qubits)]

        X = np.arange(1, n_qubits + 1)

        res = qml.QNode(circuit, dev, diff_method=None)(X)
        ref = qml.QNode(circuit, dq, diff_method=None)(X)

        assert np.allclose(res, ref)


class TestQAOAEmbedding:
    """Test the QAOAEmbedding algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 12, 2))
    def test_qaoaembedding(self, n_qubits):
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(feature_vector, weights):
            qml.QAOAEmbedding(features=feature_vector, weights=weights, wires=range(n_qubits))
            return qml.expval(qml.Z(0))

        X = np.random.rand(n_qubits)
        # [1.0, 2.0]
        # layer1 = [0.1, -0.3, 1.5]
        # layer2 = [3.1, 0.2, -2.8]
        weights = np.random.rand(2, 3 if n_qubits == 2 else 2 * n_qubits)

        res = qml.QNode(circuit, dev, diff_method=None)(X, weights)
        ref = qml.QNode(circuit, dq, diff_method=None)(X, weights)

        assert np.allclose(res, ref)


class TestCVNeuralNetLayers:
    """Test the CVNeuralNetLayers algorithm."""

    def test_cvneuralnetlayers(self):
        n_qubits = 2
        dev = qml.device(device_name, wires=n_qubits)

        def circuit(weights):
            qml.CVNeuralNetLayers(*weights, wires=[0, 1])
            return qml.expval(qml.QuadX(0))

        shapes = qml.CVNeuralNetLayers.shape(n_layers=2, n_wires=n_qubits)
        weights = [np.random.random(shape) for shape in shapes]

        with pytest.raises(qml._device.DeviceError, match="not supported"):
            _ = qml.QNode(circuit, dev, diff_method=None)(weights)


class TestRandomLayers:
    """Test the RandomLayers algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 12, 2))
    def test_randomlayers(self, n_qubits):
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(weights):
            qml.RandomLayers(weights=weights, wires=range(n_qubits))
            return qml.expval(qml.Z(0))

        weights = np.array([[0.1, -2.1, 1.4]])

        res = qml.QNode(circuit, dev, diff_method=None)(weights)
        ref = qml.QNode(circuit, dq, diff_method=None)(weights)

        assert np.allclose(res, ref)


class TestStronglyEntanglingLayers:
    """Test the StronglyEntanglingLayers algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 12, 2))
    def test_stronglyentanglinglayers(self, n_qubits):
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(weights):
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
            return qml.expval(qml.Z(0))

        shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n_qubits)
        weights = np.random.random(size=shape)

        res = qml.QNode(circuit, dev, diff_method=None)(weights)
        ref = qml.QNode(circuit, dq, diff_method=None)(weights)

        assert np.allclose(res, ref)


class TestSimplifiedTwoDesign:
    """Test the SimplifiedTwoDesign algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 12, 2))
    def test_simplifiedtwodesign(self, n_qubits):
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(init_weights, weights):
            qml.SimplifiedTwoDesign(
                initial_layer_weights=init_weights, weights=weights, wires=range(n_qubits)
            )
            return [qml.expval(qml.Z(i)) for i in range(n_qubits)]

        init_weights = np.random.rand(n_qubits)
        weights = np.random.rand(2, n_qubits - 1, 2)

        res = qml.QNode(circuit, dev, diff_method=None)(init_weights, weights)
        ref = qml.QNode(circuit, dq, diff_method=None)(init_weights, weights)

        assert np.allclose(res, ref)


class TestBasicEntanglerLayers:
    """Test the BasicEntanglerLayers algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 12, 2))
    def test_basicentanglerlayers(self, n_qubits):
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(weights):
            qml.BasicEntanglerLayers(weights=weights, wires=range(n_qubits))
            return [qml.expval(qml.Z(i)) for i in range(n_qubits)]

        weights = np.random.rand(1, n_qubits)

        res = qml.QNode(circuit, dev, diff_method=None)(weights)
        ref = qml.QNode(circuit, dq, diff_method=None)(weights)

        assert np.allclose(res, ref)


class TestMottonenStatePreparation:
    """Test the MottonenStatePreparation algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 6, 2))
    def test_mottonenstatepreparation(self, n_qubits):
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(state):
            qml.MottonenStatePreparation(state_vector=state, wires=range(n_qubits))
            return qml.state()

        state = np.random.rand(2**n_qubits) + 1j * np.random.rand(2**n_qubits)
        state = state / np.linalg.norm(state)

        res = qml.QNode(circuit, dev, diff_method=None)(state)
        ref = qml.QNode(circuit, dq, diff_method=None)(state)

        assert np.allclose(res, ref)


class TestArbitraryStatePreparation:
    """Test the ArbitraryStatePreparation algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 6, 2))
    def test_arbitrarystatepreparation(self, n_qubits):
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(weights):
            qml.ArbitraryStatePreparation(weights, wires=range(n_qubits))
            return qml.state()

        weights = np.random.rand(2 ** (n_qubits + 1) - 2)

        res = qml.QNode(circuit, dev, diff_method=None)(weights)
        ref = qml.QNode(circuit, dq, diff_method=None)(weights)

        assert np.allclose(res, ref)


class TestCosineWindow:
    """Test the CosineWindow algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 12, 2))
    def test_cosinewindow(self, n_qubits):
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit():
            qml.CosineWindow(wires=range(n_qubits))
            return qml.probs()

        res = qml.QNode(circuit, dev, diff_method=None)()
        ref = qml.QNode(circuit, dq, diff_method=None)()

        assert np.allclose(res, ref)
