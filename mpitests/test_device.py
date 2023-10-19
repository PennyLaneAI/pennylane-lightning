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
from mpi4py import MPI
    
if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

if device_name != "lightning.gpu":
    pytest.skip("Only lightning.gpu supports MPI. Skipping.", allow_module_level=True)

def test_create_device():
    dev = qml.device(device_name, mpi=True, wires=4)


@pytest.mark.skipif(
    device_name != "lightning.gpu" or not ld._CPP_BINARY_AVAILABLE,
    reason="Only lightning.gpu has a kwarg mpi_buf_size.",
)
def test_create_device_with_unsupported_mpi_buf_size():
    with pytest.raises(TypeError, match="Unsupported mpi_buf_size value"):
        dev = qml.device(device_name, mpi=True, wires=4, mpi_buf_size=-1)
    with pytest.raises(TypeError, match="Unsupported mpi_buf_size value"):
        dev = qml.device(device_name, mpi=True, wires=4, mpi_buf_size=3)
    with pytest.warns(
        RuntimeWarning, match="The MPI buffer size is larger than the local state vector size"
    ):
        dev = qml.device(device_name, mpi=True, wires=4, mpi_buf_size=2**4)
    with pytest.raises(
        ValueError,
        match="Number of processes should be smaller than the number of statevector elements",
    ):
        dev = qml.device(device_name, mpi=True, wires=1)
