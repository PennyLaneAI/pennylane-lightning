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
Array tests for Lightning devices.
"""
import numpy as np
import pytest
from conftest import LightningDevice as ld
from conftest import device_name, lightning_ops

if device_name in ("lightning_gpu", "lightning.tensor") or not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.mark.parametrize("dt", [np.dtype(np.complex64), np.dtype(np.complex128)])
def test_allocate_aligned_array_unset(dt):
    arr = lightning_ops.allocate_aligned_array(1024, dt, False)
    assert arr.dtype == dt


@pytest.mark.parametrize("dt", [np.dtype(np.complex64), np.dtype(np.complex128)])
def test_allocate_aligned_array_set(dt):
    arr = lightning_ops.allocate_aligned_array(1024, dt, True)
    assert arr.dtype == dt
    assert np.all(arr == 0)
