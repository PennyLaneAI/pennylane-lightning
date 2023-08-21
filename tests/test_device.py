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
Unit tests for Lightning devices creation.
"""
import pytest
from conftest import device_name, LightningDevice as ld

import numpy as np
import pennylane as qml

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


def test_create_device():
    dev = qml.device(device_name, wires=1)


@pytest.mark.parametrize("C", [np.complex64, np.complex128])
def test_create_device_with_dtype(C):
    dev = qml.device(device_name, wires=1, c_dtype=C)


@pytest.mark.skipif(
    not hasattr(np, "complex256"), reason="Numpy only defines complex256 in Linux-like system"
)
def test_create_device_with_unsupported_dtype():
    with pytest.raises(TypeError, match="Unsupported complex Type:"):
        dev = qml.device(device_name, wires=1, c_dtype=np.complex256)
