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
Unit tests for the generic lightning tensor class.
"""


import numpy as np
import pytest
from conftest import LightningDevice  # tested device
from pennylane.wires import Wires

from pennylane_lightning.lightning_tensor import LightningTensor

if not LightningDevice._new_API:
    pytest.skip("Exclusive tests for new API. Skipping.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.mark.parametrize("num_wires", [None, 4])
@pytest.mark.parametrize("c_dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("device_name", ["lightning.tensor"])
def test_device_name_and_init(num_wires, c_dtype, device_name):
    """Test the class initialization and returned properties."""
    wires = Wires(range(num_wires)) if num_wires else None
    dev = LightningTensor(wires=wires, c_dtype=c_dtype)
    assert dev.name == device_name
    assert dev.c_dtype == c_dtype
    assert dev.wires == wires


# def test_wrong_device_name():
#    """Test an invalid device name"""
#    with pytest.raises(qml.DeviceError, match="The device name"):
#       LightningTensor(3, device_name="thunder.tensor")


# @pytest.mark.parametrize("dtype", [np.double])
# def test_wrong_dtype(dtype):
#    """Test if the class returns a TypeError for a wrong dtype"""
#    with pytest.raises(TypeError, match="Unsupported complex type:"):
#        assert LightningTensor(wires=3, dtype=dtype)
