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
Unit tests for the serialization helper functions.
"""
import pytest
from conftest import device_name, LightningDevice as ld

import pennylane as qml
import numpy as np
import pennylane_lightning

from pennylane_lightning.core._serialize import QuantumScriptSerializer

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


class TestSerializeObs:
    """Tests for the _serialize_observables function"""

    wires_dict = {i: i for i in range(10)}

    @pytest.mark.parametrize("use_csingle", [True, False])
    @pytest.mark.parametrize("obs_chunk", list(range(1, 5)))
    def test_chunk_obs(self, use_csingle, obs_chunk):
        """Test chunking of observable array"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(
                0.5 * qml.PauliX(0) @ qml.PauliZ(1)
                + 0.7 * qml.PauliZ(0) @ qml.PauliX(1)
                + 1.2 * qml.PauliY(0) @ qml.PauliY(1)
            )
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
            qml.expval(qml.PauliY(wires=1))
            qml.expval(qml.PauliX(0) @ qml.Hermitian([[0, 1], [1, 0]], wires=3) @ qml.Hadamard(2))
            qml.expval(qml.Hermitian(qml.PauliZ.compute_matrix(), wires=0) @ qml.Identity(1))
        s, offsets = QuantumScriptSerializer(
            device_name, use_csingle, split_obs=True
        ).serialize_observables(tape, self.wires_dict)
        obtained_chunks = pennylane_lightning.core.lightning_base._chunk_iterable(s, obs_chunk)
        assert len(list(obtained_chunks)) == int(np.ceil(len(s) / obs_chunk))
        assert [0, 3, 4, 5, 6, 7] == offsets
