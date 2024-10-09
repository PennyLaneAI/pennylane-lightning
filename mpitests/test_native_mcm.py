# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for default qubit preprocessing."""
from mpi4py import MPI
import numpy as np
import pennylane as qml
import pytest
from conftest import LightningDevice, device_name

if device_name not in ("lightning.qubit", "lightning.kokkos", "lightning.gpu"):
    pytest.skip("Native MCM not supported. Skipping.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


def test_unspported_mid_measurement():
    """Test unsupported mid_measurement for lightning.gpu-mpi."""
    comm = MPI.COMM_WORLD
    dev = qml.device(device_name, wires=2, mpi=True, shots=1000)
    params = np.pi / 4 * np.ones(2)

    @qml.qnode(dev)
    def func(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0)
        qml.cond(m0, qml.RY)(y, wires=1)
        return qml.probs(wires=0)

    comm.Barrier()

    with pytest.raises(
        qml.DeviceError,
        match=f"LightningGPU-MPI does not support Mid-circuit measurements.",
    ):
        func(*params)

