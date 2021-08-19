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

from pennylane_lightning._serialize import obs_has_kernel


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
