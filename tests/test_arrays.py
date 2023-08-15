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
import pytest
from conftest import LightningDevice as ld

import numpy as np

try:
    from pennylane_lightning.lightning_qubit_ops import allocate_aligned_array
except (ImportError, ModuleNotFoundError):
    try:
        from pennylane_lightning.lightning_kokkos_ops import allocate_aligned_array
    except (ImportError, ModuleNotFoundError):
        pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.mark.skipif(not ld._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
@pytest.mark.parametrize("dt", [np.dtype(np.complex64), np.dtype(np.complex128)])
def test_allocate_aligned_array(dt):
    arr = allocate_aligned_array(1024, dt)
    assert arr.dtype == dt
