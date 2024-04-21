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
import pennylane as qml
import pytest
from conftest import LightningDevice  # tested device
from pennylane.wires import Wires

from pennylane_lightning.lightning_tensor import LightningTensor

# if LightningDevice._CPP_BINARY_AVAILABLE:
#    pytest.skip("Device doesn't have C++ support yet.", allow_module_level=True)


@pytest.mark.parametrize("num_wires", [None, 4])
@pytest.mark.parametrize("c_dtype", [np.complex64, np.complex128])
def test_device_name_and_init(num_wires, c_dtype):
    """Test the class initialization and returned properties."""
    wires = Wires(range(num_wires)) if num_wires else None
    dev = LightningTensor(wires=wires, c_dtype=c_dtype)
    assert dev.name == "lightning.tensor"
    assert dev.c_dtype == c_dtype
    assert dev.wires == wires
    if num_wires is None:
        assert dev.num_wires == 0
    else:
        assert dev.num_wires == num_wires


@pytest.mark.parametrize("backend", ["fake_backend"])
def test_invalid_backend(backend):
    """Test an invalid backend."""
    with pytest.raises(ValueError, match=f"Unsupported backend: {backend}"):
        LightningTensor(backend=backend)


@pytest.mark.parametrize("method", ["fake_method"])
def test_invalid_method(method):
    """Test an invalid method."""
    with pytest.raises(ValueError, match=f"Unsupported method: {method}"):
        LightningTensor(method=method)


def test_invalid_keyword_arg():
    """Test an invalid keyword argument."""
    with pytest.raises(
        TypeError,
        match=f"Unexpected argument: fake_arg during initialization of the LightningTensor device.",
    ):
        LightningTensor(fake_arg=None)


def test_invalid_shots():
    """Test that an error is raised if finite number of shots are requestd."""
    with pytest.raises(ValueError, match="LightningTensor does not support the `shots` parameter."):
        LightningTensor(shots=5)
