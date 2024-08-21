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
Unit tests for the tensornet functions.
"""

import math

import numpy as np
import pennylane as qml
import pytest
from conftest import LightningDevice, device_name  # tested device
from pennylane.wires import Wires

if device_name != "lightning.tensor":
    pytest.skip("Skipping tests for the tensornet class.", allow_module_level=True)
else:
    from pennylane_lightning.lightning_tensor._tensornet import LightningTensorNet

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.mark.parametrize("num_wires", range(1, 4))
@pytest.mark.parametrize("bondDims", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("device_name", ["lightning.tensor"])
def test_device_name_and_init(num_wires, bondDims, dtype, device_name):
    """Test the class initialization and returned properties."""
    if num_wires < 2:
        with pytest.raises(ValueError, match="Number of wires must be greater than 1."):
            LightningTensorNet(num_wires, bondDims, c_dtype=dtype, device_name=device_name)
        return
    else:
        tensornet = LightningTensorNet(num_wires, bondDims, c_dtype=dtype, device_name=device_name)
        assert tensornet.dtype == dtype
        assert tensornet.device_name == device_name
        assert tensornet.num_wires == num_wires


def test_wrong_device_name():
    """Test an invalid device name"""
    with pytest.raises(qml.DeviceError, match="The device name"):
        LightningTensorNet(3, 5, device_name="thunder.tensor")


def test_errors_basis_state():
    """Test that errors are raised when applying a BasisState operation."""
    with pytest.raises(ValueError, match="BasisState parameter must consist of 0 or 1 integers."):
        tensornet = LightningTensorNet(3, 5)
        tensornet.apply_operations([qml.BasisState(np.array([-0.2, 4.2]), wires=[0, 1])])
    with pytest.raises(ValueError, match="BasisState parameter and wires must be of equal length."):
        tensornet = LightningTensorNet(3, 5)
        tensornet.apply_operations([qml.BasisState(np.array([0, 1]), wires=[0])])
