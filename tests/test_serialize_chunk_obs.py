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

import pennylane as qp
import pytest
from conftest import LightningDevice as ld
from conftest import device_name

from pennylane_lightning.lightning_base._serialize import QuantumScriptSerializer

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

if device_name == "lightning.tensor":
    pytest.skip(
        "Lightning Tensor serialization is tested separately in tests/lightning_tensor/test_serialize_chunk_obs_tensor.py",
        allow_module_level=True,
    )


class TestSerializeObs:
    """Tests for the _serialize_observables function"""

    wires_dict = {i: i for i in range(10)}

    @pytest.mark.parametrize("use_csingle", [True, False])
    @pytest.mark.parametrize("obs_chunk, expected", [(1, 5), (2, 6), (3, 7), (7, 7)])
    def test_chunk_obs(self, use_csingle, obs_chunk, expected):
        """Test chunking of observable array"""
        with qp.tape.QuantumTape() as tape:
            qp.expval(
                0.5 * qp.PauliX(0) @ qp.PauliZ(1)
                + 0.7 * qp.PauliZ(0) @ qp.PauliX(1)
                + 1.2 * qp.PauliY(0) @ qp.PauliY(1)
            )
            qp.expval(qp.PauliZ(0) @ qp.PauliX(1))
            qp.expval(qp.PauliY(wires=1))
            qp.expval(qp.PauliX(0) @ qp.Hermitian([[0, 1], [1, 0]], wires=3) @ qp.Hadamard(2))
            qp.expval(qp.Hermitian(qp.PauliZ.compute_matrix(), wires=0) @ qp.Identity(1))
        s, obs_idx = QuantumScriptSerializer(
            device_name, use_csingle, split_obs=obs_chunk
        ).serialize_observables(tape, self.wires_dict)
        assert expected == len(s)
        assert [0] * (expected - 4) + [1, 2, 3, 4] == obs_idx
