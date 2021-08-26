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
Unit tests for the serialization helper functions
"""
import pennylane as qml
from pennylane import numpy as np

from pennylane_lightning._serialize import (_serialize_obs, _serialize_ops,
                                            _obs_has_kernel)
import pytest

try:
    from pennylane_lightning.lightning_qubit_ops import ObsStructC128, OpsStructC128
except ImportError:
    pytestmark = pytest.mark.skip


class TestOpsHasKernel:
    """Tests for the _obs_has_kernel function"""

    def test_pauli_z(self):
        """Tests if return is true for a PauliZ observable"""
        o = qml.PauliZ(0)
        assert _obs_has_kernel(o)

    def test_tensor_pauli(self):
        """Tests if return is true for a tensor product of Pauli terms"""
        o = qml.PauliZ(0) @ qml.PauliZ(1)
        assert _obs_has_kernel(o)

    def test_hadamard(self):
        """Tests if return is true for a Hadamard observable"""
        o = qml.Hadamard(0)
        assert _obs_has_kernel(o)

    def test_projector(self):
        """Tests if return is true for a Projector observable"""
        o = qml.Projector([0], wires=0)
        assert _obs_has_kernel(o)

    def test_hermitian(self):
        """Tests if return is false for a Hermitian observable"""
        o = qml.Hermitian(np.eye(2), wires=0)
        assert not _obs_has_kernel(o)

    def test_tensor_product_of_valid_terms(self):
        """Tests if return is true for a tensor product of Pauli, Hadamard, and Projector terms"""
        o = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.Projector([0], wires=2)
        assert _obs_has_kernel(o)

    def test_tensor_product_of_invalid_terms(self):
        """Tests if return is false for a tensor product of Hermitian terms"""
        o = qml.Hermitian(np.eye(2), wires=0) @ qml.Hermitian(np.eye(2), wires=1)
        assert not _obs_has_kernel(o)

    def test_tensor_product_of_mixed_terms(self):
        """Tests if return is false for a tensor product of valid and invalid terms"""
        o = qml.PauliZ(0) @ qml.Hermitian(np.eye(2), wires=1)
        assert not _obs_has_kernel(o)


class TestSerializeObs:
    """Tests for the _serialize_obs function"""

    wires_dict = {i: i for i in range(10)}

    def test_basic_return(self):
        """Test expected serialization for a simple return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        s = _serialize_obs(tape, self.wires_dict)
        s_expected = ([["PauliZ"]], [], [[0]])
        assert s == s_expected

    def test_tensor_return(self):
        """Test expected serialization for a tensor product return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        s = _serialize_obs(tape, self.wires_dict)
        s_expected = ([["PauliZ", "PauliZ"]], [], [[0, 1]])
        assert s == s_expected

    def test_tensor_non_tensor_return(self):
        """Test expected serialization for a mixture of tensor product and non-tensor product
        return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
            qml.expval(qml.Hadamard(1))

        s = _serialize_obs(tape, self.wires_dict)
        s_expected = ([["PauliZ", "PauliX"], ["Hadamard"]], [], [[0, 1], [1]])
        assert s == s_expected

    def test_hermitian_return(self):
        """Test expected serialization for a Hermitian return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.Hermitian(np.eye(4), wires=[0, 1]))

        s = _serialize_obs(tape, self.wires_dict)
        s_expected = ([["Hermitian"]], [np.eye(4)], [[0, 1]])

        assert s[0] == s_expected[0]
        assert np.allclose(s[1], s_expected[1])
        assert s[2] == s_expected[2]

    def test_hermitian_tensor_return(self):
        """Test expected serialization for a Hermitian return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.Hermitian(np.eye(4), wires=[0, 1]) @ qml.Hermitian(np.eye(2), wires=[2]))

        s = _serialize_obs(tape, self.wires_dict)
        s_expected = ([["Hermitian", "Hermitian"]], [np.eye(4), np.eye(2)], [[0, 1, 2]])

        assert s[0] == s_expected[0]
        assert np.allclose(s[1][0], s_expected[1][0])
        assert np.allclose(s[1][1], s_expected[1][1])
        assert s[2] == s_expected[2]

    def test_mixed_tensor_return(self):
        """Test expected serialization for a mixture of Hermitian and Pauli return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.Hermitian(np.eye(4), wires=[0, 1]) @ qml.PauliY(2))

        s = _serialize_obs(tape, self.wires_dict)
        s_expected = ([["Hermitian", "PauliY"]], [np.eye(4)], [[0, 1, 2]])

        assert s[0] == s_expected[0]
        assert np.allclose(s[1][0], s_expected[1][0])
        assert s[2] == s_expected[2]

    def test_integration(self):
        """Test for a comprehensive range of returns"""
        wires_dict = {"a": 0, 1: 1, "b": 2, -1: 3, 3.141: 4, "five": 5, 6: 6, 77: 7, 9: 8}
        I = np.eye(2)
        X = qml.PauliX.matrix
        Y = qml.PauliY.matrix
        Z = qml.PauliZ.matrix

        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ("a") @ qml.PauliX("b"))
            qml.expval(qml.Hermitian(I, wires=1))
            qml.expval(qml.PauliZ(-1) @ qml.Hermitian(X, wires=3.141) @ qml.Hadamard("five"))
            qml.expval(qml.Projector([1, 1], wires=[6, 77]) @ qml.Hermitian(Y, wires=9))
            qml.expval(qml.Hermitian(Z, wires="a") @ qml.Identity(1))

        s = _serialize_obs(tape, wires_dict)
        s_expected = (
            [
                ["PauliZ", "PauliX"],
                ["Hermitian"],
                ["PauliZ", "Hermitian", "Hadamard"],
                ["Projector", "Hermitian"],
                ["Hermitian", "Identity"],
            ],
            [I, X, Y, Z],
            [[0, 2], [1], [3, 4, 5], [6, 7, 8], [0, 1]],
        )

        assert s[0] == s_expected[0]
        assert all(np.allclose(s1, s2) for s1, s2 in zip(s[1], s_expected[1]))
        assert s[2] == s_expected[2]


class TestSerializeOps:
    """Tests for the _serialize_ops function"""

    wires_dict = {i: i for i in range(10)}

    def test_basic_circuit(self):
        """Test expected serialization for a simple circuit"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.RY(0.6, wires=1)
            qml.CNOT(wires=[0, 1])

        s = _serialize_ops(tape, self.wires_dict)
        s_expected = (
            ["RX", "RY", "CNOT"],
            [[0.4], [0.6], []],
            [[0], [1], [0, 1]],
            [False, False, False],
            [[], [], []],
        )
        assert s == s_expected

    def test_skips_prep_circuit(self):
        """Test expected serialization for a simple circuit with state preparation, such that
        the state preparation is skipped"""
        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector([1, 0], wires=0)
            qml.BasisState([1], wires=1)
            qml.RX(0.4, wires=0)
            qml.RY(0.6, wires=1)
            qml.CNOT(wires=[0, 1])

        s = _serialize_ops(tape, self.wires_dict)
        s_expected = (
            ["RX", "RY", "CNOT"],
            [[0.4], [0.6], []],
            [[0], [1], [0, 1]],
            [False, False, False],
            [[], [], []],
        )
        assert s == s_expected

    def test_inverse_circuit(self):
        """Test expected serialization for a simple circuit that includes an inverse gate"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.RY(0.6, wires=1).inv()
            qml.CNOT(wires=[0, 1])

        s = _serialize_ops(tape, self.wires_dict)
        s_expected = (
            ["RX", "RY", "CNOT"],
            [[0.4], [0.6], []],
            [[0], [1], [0, 1]],
            [False, True, False],
            [[], [], []],
        )
        assert s == s_expected

    def test_unsupported_kernel_circuit(self):
        """Test expected serialization for a circuit including gates that do not have a dedicated
        kernel"""
        with qml.tape.QuantumTape() as tape:
            qml.SingleExcitationPlus(0.4, wires=[0, 1])
            qml.SingleExcitationMinus(0.5, wires=[1, 2]).inv()
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.2, wires=2)

        s = _serialize_ops(tape, self.wires_dict)
        s_expected = (
            ["SingleExcitationPlus", "SingleExcitationMinus", "CNOT", "RZ"],
            [[], [], [], [0.2]],
            [[0, 1], [1, 2], [0, 1], [2]],
            [False, False, False, False],
            [
                qml.SingleExcitationPlus(0.4, wires=[0, 1]).matrix,
                qml.SingleExcitationMinus(0.5, wires=[1, 2]).inv().matrix,
                [],
                [],
            ],
        )
        assert s[0] == s_expected[0]
        assert s[1] == s_expected[1]
        assert s[2] == s_expected[2]
        assert s[3] == s_expected[3]

        assert all(np.allclose(s1, s2) for s1, s2 in zip(s[4], s_expected[4]))

    def test_custom_wires_circuit(self):
        """Test expected serialization for a simple circuit with custom wire labels"""
        wires_dict = {"a": 0, 3.2: 1}
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires="a")
            qml.RY(0.6, wires=3.2)
            qml.CNOT(wires=["a", 3.2])

        s = _serialize_ops(tape, wires_dict)
        s_expected = (
            ["RX", "RY", "CNOT"],
            [[0.4], [0.6], []],
            [[0], [1], [0, 1]],
            [False, False, False],
            [[], [], []],
        )
        assert s == s_expected

    def test_integration(self):
        """Test expected serialization for a random circuit"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.RY(0.6, wires=1).inv().inv()
            qml.CNOT(wires=[0, 1])
            qml.QubitUnitary(np.eye(4), wires=[0, 1])
            qml.QFT(wires=[0, 1, 2]).inv()
            qml.DoubleExcitation(0.555, wires=[3, 2, 1, 0])

        s = _serialize_ops(tape, self.wires_dict)
        s_expected = (
            ["RX", "RY", "CNOT", "QubitUnitary", "QFT", "DoubleExcitation"],
            [[0.4], [0.6], [], [], [], []],
            [[0], [1], [0, 1], [0, 1], [0, 1, 2], [3, 2, 1, 0]],
            [False, False, False, False, False, False],
            [
                [],
                [],
                [],
                qml.QubitUnitary(np.eye(4), wires=[0, 1]).matrix,
                qml.QFT(wires=[0, 1, 2]).inv().matrix,
                qml.DoubleExcitation(0.555, wires=[3, 2, 1, 0]).matrix,
            ],
        )
        assert s[0] == s_expected[0]
        assert s[1] == s_expected[1]
        assert s[2] == s_expected[2]
        assert s[3] == s_expected[3]

        assert all(np.allclose(s1, s2) for s1, s2 in zip(s[4], s_expected[4]))
