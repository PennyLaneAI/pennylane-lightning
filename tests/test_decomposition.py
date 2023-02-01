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
Unit tests for operation decomposition.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane_lightning import LightningQubit

from pennylane_lightning.lightning_qubit import CPP_BINARY_AVAILABLE

if not CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


class TestDenseMatrixDecompositionThreshold:
    """Tests, for QFT and Grover operators, the automatic transition from full matrix to decomposition
    on calculations."""

    input = [
        (qml.QFT, 8, True),
        (qml.QFT, 10, False),
        (qml.GroverOperator, 8, True),
        (qml.GroverOperator, 13, False),
    ]

    @pytest.mark.parametrize("op, n_wires, condition", input)
    def test_threshold(self, op, n_wires, condition):
        wires = np.linspace(0, n_wires - 1, n_wires, dtype=int)
        op = op(wires=wires)
        assert LightningQubit.stopping_condition.__get__(op)(op) == condition
