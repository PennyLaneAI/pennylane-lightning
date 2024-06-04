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
Unit tests for the serialization helper functions.
"""
import numpy as np
import pennylane as qml
import pytest
from conftest import LightningDevice, device_name

from pennylane_lightning.core._serialize import QuantumScriptSerializer, global_phase_diagonal

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

if device_name == "lightning.tensor":
    from pennylane_lightning.lightning_tensor_ops.observables import (
        HamiltonianC64,
        HamiltonianC128,
        HermitianObsC64,
        HermitianObsC128,
        NamedObsC64,
        NamedObsC128,
        TensorProdObsC64,
        TensorProdObsC128,
    )


def test_wrong_device_name():
    """Test the device name is not a valid option"""

    with pytest.raises(qml.DeviceError, match="The device name"):
        QuantumScriptSerializer("thunder.qubit")


@pytest.mark.usefixtures("use_legacy_and_new_opmath")
@pytest.mark.parametrize(
    "obs,obs_type",
    [
        (qml.PauliZ(0), NamedObsC128),
        (qml.PauliZ(0) @ qml.PauliZ(1), TensorProdObsC128),
        (qml.Hadamard(0), NamedObsC128),
        (qml.Hermitian(np.eye(2), wires=0), HermitianObsC128),
        # (
        #    qml.PauliZ(0) @ qml.Hadamard(1) @ (0.1 * (qml.PauliZ(2) + qml.PauliX(3))),
        #    TensorProdObsC128,
        # ),
        (
            qml.PauliZ(0) @ qml.PauliY(1) @ qml.PauliX(2),
            TensorProdObsC128,
        ),
        # (
        #    qml.PauliZ(0) @ qml.PauliY(1) @ (0.1 * (qml.PauliZ(2) + qml.PauliX(3))),
        #    HamiltonianC128,
        # ),
        (
            (
                qml.Hermitian(np.eye(2), wires=0)
                @ qml.Hermitian(np.eye(2), wires=1)
                @ qml.Projector([0], wires=2)
            ),
            TensorProdObsC128,
        ),
        (
            (
                qml.Hermitian(np.eye(2), wires=0)
                @ qml.Hermitian(np.eye(2), wires=1)
                @ qml.Projector([0], wires=1)
            ),
            HermitianObsC128,
        ),
        (
            qml.PauliZ(0) @ qml.Hermitian(np.eye(2), wires=1) @ qml.Projector([0], wires=2),
            TensorProdObsC128,
        ),
        (qml.Projector([0], wires=0), HermitianObsC128),
        (qml.Hamiltonian([1], [qml.PauliZ(0)]), NamedObsC128),
        (qml.sum(qml.Hadamard(0), qml.PauliX(1)), HamiltonianC128),
        (2.5 * qml.PauliZ(0), HamiltonianC128),
    ],
)
def test_obs_returns_expected_type(obs, obs_type):
    """Tests that observables get serialized to the expected type, with and without wires map"""
    serializer = QuantumScriptSerializer(device_name)
    assert isinstance(serializer._ob(obs, dict(enumerate(obs.wires))), obs_type)
    assert isinstance(serializer._ob(obs), obs_type)


@pytest.mark.parametrize(
    "obs",
    [qml.SparseHamiltonian(qml.Hamiltonian([1], [qml.PauliZ(0)]).sparse_matrix(), wires=[0])],
)
def test_unsupported_obs_returns_expected_type(obs):
    """Tests that observables get serialized to the expected type, with and without wires map"""
    serializer = QuantumScriptSerializer(device_name)
    with pytest.raises(
        NotImplementedError,
        match="SparseHamiltonian is not supported on the lightning.tensor device.",
    ):
        serializer._ob(obs, dict(enumerate(obs.wires)))
