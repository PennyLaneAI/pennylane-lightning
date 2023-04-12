# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
import numpy as np

import pennylane_lightning.experimental as pennylane_lightning
from pennylane_lightning.experimental._serialize import (
    _serialize_ob,
)
import pytest
from unittest import mock

from pennylane_lightning.experimental.lightning_qubit_2 import CPP_BINARY_AVAILABLE

if not CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

from pennylane_lightning.lightning_qubit_ops.adjoint_diff import (
    NamedObsC64,
    NamedObsC128,
    HermitianObsC64,
    HermitianObsC128,
    TensorProdObsC64,
    TensorProdObsC128,
    HamiltonianC64,
    HamiltonianC128,
)


@pytest.mark.parametrize(
    "obs,obs_type",
    [
        (qml.PauliZ(0), NamedObsC64),
        (qml.PauliZ(0) @ qml.PauliZ(1), TensorProdObsC64),
        (qml.Hadamard(0), NamedObsC64),
        (qml.Hermitian(np.eye(2), wires=0), HermitianObsC64),
        (
            qml.PauliZ(0) @ qml.Hadamard(1) @ (0.1 * (qml.PauliZ(2) + qml.PauliX(3))),
            TensorProdObsC64,
        ),
        (
            (
                qml.Hermitian(np.eye(2), wires=0)
                @ qml.Hermitian(np.eye(2), wires=1)
                @ qml.Projector([0], wires=2)
            ),
            TensorProdObsC64,
        ),
        (
            qml.PauliZ(0) @ qml.Hermitian(np.eye(2), wires=1) @ qml.Projector([0], wires=2),
            TensorProdObsC64,
        ),
        (qml.Projector([0], wires=0), HermitianObsC64),
        (qml.Hamiltonian([1], [qml.PauliZ(0)]), HamiltonianC64),
    ],
)
def test_obs_returns_expected_type(obs, obs_type):
    """Tests that observables get serialized to the expected type."""
    assert isinstance(_serialize_ob(obs, True), obs_type)