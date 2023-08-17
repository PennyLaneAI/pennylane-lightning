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
from conftest import device_name, LightningDevice

from pennylane_lightning.core._serialize import QuantumScriptSerializer

if LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("Binary module found. Skipping.", allow_module_level=True)


@pytest.mark.skipif(LightningDevice._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
def test_no_binaries():
    """Test no binaries were found for the device"""

    with pytest.raises(ImportError, match="Pre-compiled binaries for "):
        QuantumScriptSerializer(device_name)
