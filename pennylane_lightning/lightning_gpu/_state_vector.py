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
Class implementation for state-vector manipulation.
"""

import numpy as np
import pennylane as qml
from pennylane import DeviceError
from pennylane.wires import Wires

# from ._measurements import LightningMeasurements


class LightningStateVector:
    """Lightning state-vector class.

    Interfaces with C++ python binding methods for state-vector manipulation.

    Args:
        num_wires(int): the number of wires to initialize the device with
        dtype: Datatypes for state-vector representation. Must be one of
            ``np.complex64`` or ``np.complex128``. Default is ``np.complex128``
        device_name(string): state vector device name. Options: ["lightning.qubit"]
    """

    def __init__(self, num_wires, dtype=np.complex128, device_name="lightning.qubit"):
        self._num_wires = num_wires
        self._wires = Wires(range(num_wires))
        self._dtype = dtype

        if dtype not in [np.complex64, np.complex128]:
            raise TypeError(f"Unsupported complex type: {dtype}")

        if device_name != "lightning.qubit":
            raise DeviceError(f'The device name "{device_name}" is not a valid option.')

        self._device_name = device_name

    @property
    def dtype(self):
        """Returns the state vector data type."""
        return self._dtype

    @property
    def device_name(self):
        """Returns the state vector device name."""
        return self._device_name

    @property
    def wires(self):
        """All wires that can be addressed on this device"""
        return self._wires

    @property
    def num_wires(self):
        """Number of wires addressed on this device"""
        return self._num_wires