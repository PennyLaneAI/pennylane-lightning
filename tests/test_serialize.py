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
from pennylane import numpy as np
import pennylane as qml

from pennylane_lightning._serialize import obs_has_kernel, _serialize_obs


class TestOpsHasKernel:
    """Tests for the obs_has_kernel function"""

    def test_pauli_z(self):
        """Tests if return is true for a PauliZ observable"""
        o = qml.PauliZ(0)
        assert obs_has_kernel(o)

    def test_tensor_pauli(self):
        """Tests if return is true for a tensor product of Pauli terms"""
        o = qml.PauliZ(0) @ qml.PauliZ(1)
        assert obs_has_kernel(o)

    def test_hadamard(self):
        """Tests if return is true for a Hadamard observable"""
        o = qml.Hadamard(0)
        assert obs_has_kernel(o)

    def test_projector(self):
        """Tests if return is true for a Projector observable"""
        o = qml.Projector([0], wires=0)
        assert obs_has_kernel(o)

    def test_hermitian(self):
        """Tests if return is false for a Hermitian observable"""
        o = qml.Hermitian(np.eye(2), wires=0)
        assert not obs_has_kernel(o)

    def test_tensor_product_of_valid_terms(self):
        """Tests if return is true for a tensor product of Pauli, Hadamard, and Projector terms"""
        o = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.Projector([0], wires=2)
        assert obs_has_kernel(o)

    def test_tensor_product_of_invalid_terms(self):
        """Tests if return is false for a tensor product of Hermitian terms"""
        o = qml.Hermitian(np.eye(2), wires=0) @ qml.Hermitian(np.eye(2), wires=1)
        assert not obs_has_kernel(o)

    def test_tensor_product_of_mixed_terms(self):
        """Tests if return is false for a tensor product of valid and invalid terms"""
        o = qml.PauliZ(0) @ qml.Hermitian(np.eye(2), wires=1)
        assert not obs_has_kernel(o)


class TestSerializeObs:
    """Tests for the _serialize_obs function"""

    wires_dict = {i: i for i in range(10)}

    def test_basic_return(self):
        """Test expected serialization for a simple return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        s = _serialize_obs(tape, self.wires_dict)
        s_expected = (
            [["PauliZ"]],
            [],
            [[0]]
        )
        assert s == s_expected

    def test_tensor_return(self):
        """Test expected serialization for a tensor product return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        s = _serialize_obs(tape, self.wires_dict)
        s_expected = (
            [["PauliZ", "PauliZ"]],
            [],
            [[0, 1]]
        )
        assert s == s_expected

    def test_tensor_non_tensor_return(self):
        """Test expected serialization for a mixture of tensor product and non-tensor product
        return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
            qml.expval(qml.Hadamard(1))

        s = _serialize_obs(tape, self.wires_dict)
        s_expected = (
            [["PauliZ", "PauliX"], ["Hadamard"]],
            [],
            [[0, 1], [1]]
        )
        assert s == s_expected

    def test_hermitian_return(self):
        """Test expected serialization for a Hermitian return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.Hermitian(np.eye(4), wires=[0, 1]))

        s = _serialize_obs(tape, self.wires_dict)
        s_expected = (
            [["Hermitian"]],
            [np.eye(4)],
            [[0, 1]]
        )

        assert s[0] == s_expected[0]
        assert np.allclose(s[1], s_expected[1])
        assert s[2] == s_expected[2]

    def test_hermitian_tensor_return(self):
        """Test expected serialization for a Hermitian return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.Hermitian(np.eye(4), wires=[0, 1]) @ qml.Hermitian(np.eye(2), wires=[2]))

        s = _serialize_obs(tape, self.wires_dict)
        s_expected = (
            [["Hermitian", "Hermitian"]],
            [np.eye(4), np.eye(2)],
            [[0, 1, 2]]
        )

        assert s[0] == s_expected[0]
        assert np.allclose(s[1][0], s_expected[1][0])
        assert np.allclose(s[1][1], s_expected[1][1])
        assert s[2] == s_expected[2]

    def test_mixed_tensor_return(self):
        """Test expected serialization for a mixture of Hermitian and Pauli return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.Hermitian(np.eye(4), wires=[0, 1]) @ qml.PauliY(2))

        s = _serialize_obs(tape, self.wires_dict)
        s_expected = (
            [["Hermitian", "PauliY"]],
            [np.eye(4)],
            [[0, 1, 2]]
        )

        assert s[0] == s_expected[0]
        assert np.allclose(s[1][0], s_expected[1][0])
        assert s[2] == s_expected[2]

    def test_integration(self):
        """Test for a comprehensive range of returns"""
        I = np.eye(2)
        X = qml.PauliX.matrix
        Y = qml.PauliY.matrix
        Z = qml.PauliZ.matrix

        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0) @ qml.PauliX(2))
            qml.expval(qml.Hermitian(I, wires=1))
            qml.expval(qml.PauliZ(3) @ qml.Hermitian(X, wires=4) @ qml.Hadamard(5))
            qml.expval(qml.Projector([1, 1], wires=[6, 7]) @ qml.Hermitian(Y, wires=8))
            qml.expval(qml.Hermitian(Z, wires=0) @ qml.Identity(1))

        s = _serialize_obs(tape, self.wires_dict)
        s_expected = (
            [["PauliZ", "PauliX"], ["Hermitian"], ["PauliZ", "Hermitian", "Hadamard"], ["Projector", "Hermitian"], ["Hermitian", "Identity"]],
            [I, X, Y, Z],
            [[0, 2], [1], [3, 4, 5], [6, 7, 8], [0, 1]]
        )

        assert s[0] == s_expected[0]
        assert all(np.allclose(s1, s2) for s1, s2 in zip(s[1], s_expected[1]))
        assert s[2] == s_expected[2]




