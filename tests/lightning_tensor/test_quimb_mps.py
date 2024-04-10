# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Unit tests for the ``quimb`` interface.
"""


import numpy as np
import pytest
from conftest import LightningDevice  # tested device
import quimb.tensor as qtn
from pennylane.wires import Wires

from pennylane_lightning.lightning_tensor import LightningTensor

if not LightningDevice._new_API:
    pytest.skip("Exclusive tests for new API. Skipping.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.mark.parametrize("num_wires", [None, 4])
@pytest.mark.parametrize("c_dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("backend", ["quimb"])
@pytest.mark.parametrize("method", ["mps"])
def test_device_init(num_wires, c_dtype, backend, method):
    """Test the class initialization and returned properties."""
    wires = Wires(range(num_wires)) if num_wires else None
    dev = LightningTensor(wires=wires, backend=backend, method=method, c_dtype=c_dtype)
    assert isinstance(dev.state, qtn.MatrixProductState)
    assert isinstance(dev.state_to_array(), np.ndarray)
