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
# pylint: disable=protected-access,unused-variable,missing-function-docstring,c-extension-no-member

import pennylane as qml
import pytest
from conftest import LightningDevice as ld
from conftest import device_name
from mpi4py import MPI

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


def test_create_device():
    if MPI.COMM_WORLD.Get_size() > 2:
        with pytest.raises(
            ValueError,
            match="Number of devices should be larger than or equal to the number of processes on each node.",
        ):
            dev = qml.device(device_name, mpi=True, wires=4)
    else:
        dev = qml.device(device_name, mpi=True, wires=4)


def test_unsupported_mpi_buf_size():
    with pytest.raises(ValueError, match="Unsupported mpi_buf_size value"):
        dev = qml.device(device_name, mpi=True, wires=4, mpi_buf_size=-1)
    with pytest.raises(ValueError, match="Unsupported mpi_buf_size value"):
        dev = qml.device(device_name, mpi=True, wires=4, mpi_buf_size=3)
    with pytest.raises(
        RuntimeError,
        match="The MPI buffer size is larger than the local state vector size.",
    ):
        dev = qml.device(device_name, mpi=True, wires=4, mpi_buf_size=2**4)
    with pytest.raises(
        ValueError,
        match="Number of processes should be smaller than the number of statevector elements",
    ):
        dev = qml.device(device_name, mpi=True, wires=1)
