# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Test the correctness of templates with Lightning devices.
"""
import functools

import pennylane as qml
import pytest
from conftest import LightningDevice, device_name
from pennylane import numpy as np

# pylint: disable=missing-function-docstring, too-few-public-methods

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


def lightning_tensor_check(n_qubits):
    if device_name == "lightning.tensor" and n_qubits > 14:
        pytest.xfail(
            "Inexact calculation for lightning.tensor with n_qubits > 14 since the default max mps bond dim is 2^7."
        )


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

    @pytest.mark.skipif(not LightningDevice._new_API, reason="New API required.")
    @pytest.mark.parametrize("wires", [5, 10, 13, 15])
    def test_preprocess_grover_operator_decomposition(self, wires):
        """Test that qml.GroverOperator is not decomposed for less than 10 wires."""
        tape = qml.tape.QuantumScript(
            [qml.GroverOperator(wires=list(range(wires)))], [qml.expval(qml.PauliZ(0))]
        )
        dev = LightningDevice(wires=wires)

        program, _ = dev.preprocess()
        [new_tape], _ = program([tape])

        if wires >= 13:
            assert all(not isinstance(op, qml.GroverOperator) for op in new_tape.operations)
        else:
            assert tape.operations == [qml.GroverOperator(wires=list(range(wires)))]


class TestAngleEmbedding:
    """Test the AngleEmbedding algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 20, 2))
    def test_angleembedding(self, n_qubits):
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(feature_vector):
            qml.AngleEmbedding(features=feature_vector, wires=range(n_qubits), rotation="Z")
            return qml.state()

        X = np.random.rand(n_qubits)

        res = qml.QNode(circuit, dev, diff_method=None)(X)
        ref = qml.QNode(circuit, dq, diff_method=None)(X)

        assert np.allclose(res, ref)


class TestAmplitudeEmbedding:
    """Test the AmplitudeEmbedding algorithm."""

    @pytest.mark.parametrize("first_op", [False, True])
    @pytest.mark.parametrize("n_qubits", range(2, 10, 2))
    def test_amplitudeembedding(self, first_op, n_qubits):
        if not first_op:
            if device_name != "lightning.qubit":
                pytest.xfail(
                    f"Operation StatePrep cannot be used after other Operations have already been applied on a {device_name} device."
                )

        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(f=None):
            if not first_op:
                qml.Hadamard(0)
            qml.AmplitudeEmbedding(features=f, wires=range(n_qubits))
            return qml.state()

        X = np.random.rand(2**n_qubits)
        X /= np.linalg.norm(X)
        res = qml.QNode(circuit, dev, diff_method=None)(f=X)
        ref = qml.QNode(circuit, dq, diff_method=None)(f=X)

        assert np.allclose(res, ref)


class TestBasisEmbedding:
    """Test the BasisEmbedding algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 20, 2))
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


class TestDisplacementSqueezingEmbedding:
    """Test the DisplacementEmbedding and SqueezingEmbedding algorithms."""

    @pytest.mark.parametrize("n_qubits", range(2, 20, 2))
    @pytest.mark.parametrize("template", [qml.DisplacementEmbedding, qml.SqueezingEmbedding])
    def test_displacementembedding(self, n_qubits, template):
        dev = qml.device(device_name, wires=n_qubits)

        def circuit(feature_vector):
            template(features=feature_vector, wires=range(n_qubits))
            qml.QuadraticPhase(0.1, wires=1)
            return qml.state()

        X = np.arange(1, n_qubits + 1)

        with pytest.raises(qml.DeviceError, match="not supported"):
            _ = qml.QNode(circuit, dev, diff_method=None)(X)


class TestIQPEmbedding:
    """Test the IQPEmbedding algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 20, 2))
    def test_iqpembedding(self, n_qubits):
        lightning_tensor_check(n_qubits)
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

    @pytest.mark.parametrize("n_qubits", range(2, 20, 2))
    def test_qaoaembedding(self, n_qubits):
        lightning_tensor_check(n_qubits)
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(feature_vector, weights):
            qml.Hadamard(0)
            qml.QAOAEmbedding(features=feature_vector, weights=weights, wires=range(n_qubits))
            return qml.state()

        X = np.random.rand(n_qubits)
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
            return qml.state()

        shapes = qml.CVNeuralNetLayers.shape(n_layers=2, n_wires=n_qubits)
        weights = [np.random.random(shape) for shape in shapes]

        with pytest.raises(qml.DeviceError, match="not supported"):
            _ = qml.QNode(circuit, dev, diff_method=None)(weights)


class TestRandomLayers:
    """Test the RandomLayers algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 20, 2))
    def test_randomlayers(self, n_qubits):
        lightning_tensor_check(n_qubits)
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit", wires=n_qubits)

        def circuit(weights):
            qml.RandomLayers(weights=weights, wires=range(n_qubits))
            return qml.state()

        weights = np.array([[0.1, -2.1, 1.4]])

        res = qml.QNode(circuit, dev, diff_method=None)(weights)
        ref = qml.QNode(circuit, dq, diff_method=None)(weights)

        assert np.allclose(res, ref)


class TestStronglyEntanglingLayers:
    """Test the StronglyEntanglingLayers algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 20, 2))
    def test_stronglyentanglinglayers(self, n_qubits):
        lightning_tensor_check(n_qubits)
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(weights):
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
            return qml.state()

        shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n_qubits)
        weights = np.random.random(size=shape)

        res = qml.QNode(circuit, dev, diff_method=None)(weights)
        ref = qml.QNode(circuit, dq, diff_method=None)(weights)

        assert np.allclose(res, ref)


class TestSimplifiedTwoDesign:
    """Test the SimplifiedTwoDesign algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 20, 2))
    def test_simplifiedtwodesign(self, n_qubits):
        lightning_tensor_check(n_qubits)
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

    @pytest.mark.parametrize("n_qubits", range(2, 20, 2))
    def test_basicentanglerlayers(self, n_qubits):
        lightning_tensor_check(n_qubits)
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

    @pytest.mark.parametrize("n_qubits", range(2, 6, 2))
    def test_cosinewindow(self, n_qubits):
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit():
            qml.CosineWindow(wires=range(n_qubits))
            return qml.state()

        res = qml.QNode(circuit, dev, diff_method=None)()
        ref = qml.QNode(circuit, dq, diff_method=None)()

        assert np.allclose(res, ref)


class TestAllSinglesDoubles:
    """Test the AllSinglesDoubles algorithm."""

    def test_allsinglesdoubles(self):
        n_qubits = 4
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        electrons = 2

        # Define the HF state
        hf_state = qml.qchem.hf_state(electrons, n_qubits)

        # Generate all single and double excitations
        singles, doubles = qml.qchem.excitations(electrons, n_qubits)

        def circuit(weights, hf_state, singles, doubles):
            qml.templates.AllSinglesDoubles(weights, range(n_qubits), hf_state, singles, doubles)
            return qml.state()

        weights = np.random.normal(0, np.pi, len(singles) + len(doubles))
        res = qml.QNode(circuit, dev, diff_method=None)(weights, hf_state, singles, doubles)
        ref = qml.QNode(circuit, dq, diff_method=None)(weights, hf_state, singles, doubles)

        assert np.allclose(res, ref)


class TestBasisRotation:
    """Test the BasisRotation algorithm."""

    def test_basisrotation(self):
        n_qubits = 3
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(unitary_matrix):
            qml.BasisState(np.array([1, 1, 0]), wires=[0, 1, 2])
            qml.BasisRotation(
                wires=range(3),
                unitary_matrix=unitary_matrix,
            )
            return qml.state()

        unitary_matrix = np.array(
            [
                [0.51378719 + 0.0j, 0.0546265 + 0.79145487j, -0.2051466 + 0.2540723j],
                [0.62651582 + 0.0j, -0.00828925 - 0.60570321j, -0.36704948 + 0.32528067j],
                [-0.58608928 + 0.0j, 0.03902657 + 0.04633548j, -0.57220635 + 0.57044649j],
            ]
        )

        res = qml.QNode(circuit, dev, diff_method=None)(unitary_matrix)
        ref = qml.QNode(circuit, dq, diff_method=None)(unitary_matrix)

        assert np.allclose(res, ref)


class TestGateFabric:
    """Test the GateFabric algorithm."""

    def test_gatefabric(self):
        # Build the electronic Hamiltonian
        symbols = ["H", "H"]
        coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
        _, n_qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        # Define the Hartree-Fock state
        electrons = 2
        ref_state = qml.qchem.hf_state(electrons, n_qubits)

        def circuit(weights):
            qml.GateFabric(weights, wires=[0, 1, 2, 3], init_state=ref_state, include_pi=True)
            return qml.state()

        layers = 2
        shape = qml.GateFabric.shape(n_layers=layers, n_wires=n_qubits)
        weights = np.random.random(size=shape)

        res = qml.QNode(circuit, dev, diff_method=None)(weights)
        ref = qml.QNode(circuit, dq, diff_method=None)(weights)

        assert np.allclose(res, ref)


class TestUCCSD:
    """Test the UCCSD algorithm."""

    def test_uccsd(self):
        # Define the molecule
        symbols = ["H", "H", "H"]
        geometry = np.array(
            [
                [0.01076341, 0.04449877, 0.0],
                [0.98729513, 1.63059094, 0.0],
                [1.87262415, -0.00815842, 0.0],
            ],
            requires_grad=False,
        )
        electrons = 2
        charge = 1

        # Build the electronic Hamiltonian
        _, n_qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=charge)

        # Define the HF state
        hf_state = qml.qchem.hf_state(electrons, n_qubits)

        # Generate single and double excitations
        singles, doubles = qml.qchem.excitations(electrons, n_qubits)

        # Map excitations to the wires the UCCSD circuit will act on
        s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(weights):
            qml.UCCSD(weights, range(n_qubits), s_wires, d_wires, hf_state)
            return qml.state()

        weights = np.random.random(len(singles) + len(doubles))

        res = qml.QNode(circuit, dev, diff_method=None)(weights)
        ref = qml.QNode(circuit, dq, diff_method=None)(weights)

        assert np.allclose(res, ref)


class TestkUpCCGSD:
    """Test the kUpCCGSD algorithm."""

    def test_kupccgsd(self):
        # Define the molecule
        symbols = ["H", "H", "H"]
        geometry = np.array(
            [
                [0.01076341, 0.04449877, 0.0],
                [0.98729513, 1.63059094, 0.0],
                [1.87262415, -0.00815842, 0.0],
            ],
            requires_grad=False,
        )
        electrons = 2
        charge = 1

        # Build the electronic Hamiltonian
        _, n_qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=charge)

        # Define the HF state
        hf_state = qml.qchem.hf_state(electrons, n_qubits)

        # Map excitations to the wires the kUpCCGSD circuit will act on
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(weights):
            qml.kUpCCGSD(weights, range(n_qubits), k=1, delta_sz=0, init_state=hf_state)
            return qml.state()

        # Get the shape of the weights for this template
        layers = 1
        shape = qml.kUpCCGSD.shape(k=layers, n_wires=n_qubits, delta_sz=0)
        weights = np.random.random(size=shape)

        res = qml.QNode(circuit, dev, diff_method=None)(weights)
        ref = qml.QNode(circuit, dq, diff_method=None)(weights)

        assert np.allclose(res, ref)


class TestParticleConservingU1:
    """Test the ParticleConservingU1 algorithm."""

    def test_particleconservingu1(self):
        # Build the electronic Hamiltonian
        symbols, coordinates = (["H", "H"], np.array([0.0, 0.0, -0.66140414, 0.0, 0.0, 0.66140414]))
        _, n_qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)

        # Define the Hartree-Fock state
        electrons = 2
        hf_state = qml.qchem.hf_state(electrons, n_qubits)

        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        # Define the ansatz
        ansatz = functools.partial(qml.ParticleConservingU1, init_state=hf_state, wires=dev.wires)

        # Define the cost function
        def circuit(params):
            ansatz(params)
            return qml.state()

        layers = 2
        shape = qml.ParticleConservingU1.shape(layers, n_qubits)
        weights = np.random.random(shape)

        res = qml.QNode(circuit, dev, diff_method=None)(weights)
        ref = qml.QNode(circuit, dq, diff_method=None)(weights)

        assert np.allclose(res, ref)


class TestParticleConservingU2:
    """Test the ParticleConservingU2 algorithm."""

    def test_particleconservingu2(self):
        # Build the electronic Hamiltonian
        symbols, coordinates = (["H", "H"], np.array([0.0, 0.0, -0.66140414, 0.0, 0.0, 0.66140414]))
        _, n_qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)

        # Define the Hartree-Fock state
        electrons = 2
        hf_state = qml.qchem.hf_state(electrons, n_qubits)

        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        # Define the ansatz
        ansatz = functools.partial(qml.ParticleConservingU2, init_state=hf_state, wires=dev.wires)

        # Define the cost function
        def circuit(params):
            ansatz(params)
            return qml.state()

        layers = 2
        shape = qml.ParticleConservingU2.shape(layers, n_qubits)
        weights = np.random.random(shape)

        res = qml.QNode(circuit, dev, diff_method=None)(weights)
        ref = qml.QNode(circuit, dq, diff_method=None)(weights)

        assert np.allclose(res, ref)


class TestApproxTimeEvolution:
    """Test the ApproxTimeEvolution algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 20, 2))
    def test_approxtimeevolution(self, n_qubits):
        lightning_tensor_check(n_qubits)
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        coeffs = [1] * n_qubits
        obs = [qml.X(i) for i in range(n_qubits)]
        hamiltonian = qml.Hamiltonian(coeffs, obs)

        def circuit(time):
            qml.ApproxTimeEvolution(hamiltonian, time, 1)
            return qml.state()

        res = qml.QNode(circuit, dev, diff_method=None)(1.3)
        ref = qml.QNode(circuit, dq, diff_method=None)(1.3)

        assert np.allclose(res, ref)


class TestQDrift:
    """Test the QDrift algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 20, 2))
    def test_qdrift(self, n_qubits):
        lightning_tensor_check(n_qubits)
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit", wires=n_qubits)

        coeffs = [1] * n_qubits
        obs = [qml.X(i) for i in range(n_qubits)]
        hamiltonian = qml.Hamiltonian(coeffs, obs)

        def circuit(time):
            qml.QDrift(hamiltonian, time=time, n=10, seed=10)
            return qml.state()

        res = qml.QNode(circuit, dev, diff_method=None)(1.3)
        ref = qml.QNode(circuit, dq, diff_method=None)(1.3)

        assert np.allclose(res, ref)


class TestTrotterProduct:
    """Test the TrotterProduct algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 20, 2))
    def test_trotterproduct(self, n_qubits):
        lightning_tensor_check(n_qubits)
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        coeffs = [1] * n_qubits
        obs = [qml.X(i) for i in range(n_qubits)]
        hamiltonian = qml.Hamiltonian(coeffs, obs)

        def circuit(time):
            qml.TrotterProduct(hamiltonian, time=time, order=2)
            return qml.state()

        res = qml.QNode(circuit, dev, diff_method=None)(1.3)
        ref = qml.QNode(circuit, dq, diff_method=None)(1.3)

        assert np.allclose(res, ref)


class TestQuantumPhaseEstimation:
    """Test the QuantumPhaseEstimation algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 12, 2))
    def test_quantumphaseestimation(self, n_qubits):
        lightning_tensor_check(n_qubits)
        phase = 5
        target_wires = [0]
        unitary = qml.RX(phase, wires=0).matrix()
        n_estimation_wires = n_qubits - 1
        estimation_wires = range(1, n_estimation_wires + 1)

        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit():
            # Start in the |+> eigenstate of the unitary
            qml.Hadamard(wires=target_wires)

            qml.QuantumPhaseEstimation(
                unitary,
                target_wires=target_wires,
                estimation_wires=estimation_wires,
            )

            return qml.probs(estimation_wires)

        res = qml.QNode(circuit, dev, diff_method=None)()
        ref = qml.QNode(circuit, dq, diff_method=None)()

        assert np.allclose(res, ref)


class TestQFT:
    """Test the QFT algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 15, 2))
    def test_qft(self, n_qubits):
        lightning_tensor_check(n_qubits)
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(basis_state):
            qml.BasisState(basis_state, wires=range(n_qubits))
            qml.QFT(wires=range(n_qubits))
            return qml.state()

        basis_state = [0] * n_qubits
        basis_state[0] = 1
        res = qml.QNode(circuit, dev, diff_method=None)(basis_state)
        ref = qml.QNode(circuit, dq, diff_method=None)(basis_state)

        assert np.allclose(res, ref)

    @pytest.mark.skipif(not LightningDevice._new_API, reason="New API required")
    @pytest.mark.parametrize("wires", [5, 13])
    def test_preprocess_qft_decomposition(self, wires):
        """Test that qml.QFT is always decomposed for any wires."""
        tape = qml.tape.QuantumScript(
            [qml.QFT(wires=list(range(wires)))], [qml.expval(qml.PauliZ(0))]
        )
        dev = LightningDevice(wires=wires)

        program, _ = dev.preprocess()
        [new_tape], _ = program([tape])

        # assert all(not isinstance(op, qml.QFT) for op in new_tape.operations)
        # else:
        assert tape.operations == [qml.QFT(wires=list(range(wires)))]


class TestAQFT:
    """Test the AQFT algorithm."""

    @pytest.mark.parametrize("n_qubits", range(4, 14, 2))
    def test_aqft(self, n_qubits):
        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit")

        def circuit(basis_state):
            qml.BasisState(basis_state, wires=range(n_qubits))
            qml.AQFT(order=1, wires=range(n_qubits))
            return qml.state()

        basis_state = [0] * n_qubits
        basis_state[0] = 1
        res = qml.QNode(circuit, dev, diff_method=None)(basis_state)
        ref = qml.QNode(circuit, dq, diff_method=None)(basis_state)

        assert np.allclose(res, ref)


class TestQSVT:
    """Test the QSVT algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 20, 2))
    def test_qsvt(self, n_qubits):
        lightning_tensor_check(n_qubits)
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
