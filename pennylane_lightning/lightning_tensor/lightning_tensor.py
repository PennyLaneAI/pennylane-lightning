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
This module contains the LightningTensor class that inherits from the new device interface.
"""

import numpy as np
import pennylane as qml
from pennylane.devices import Device

from pennylane.wires import Wires


from ._state_tensor import LightningStateTensor


class LightningTensor(Device):
    """PennyLane Lightning Tensor device.

    A device that interfaces with C++ to perform fast linear algebra calculations.
    """

    _new_API = True

    # TODO: add `max_bond_dim` parameter
    def __init__(
        self,
        *,
        wires=None,
        backend="quimb",
        method="mps",
        c_dtype=np.complex128,
        shots=None,
    ):

        if backend not in ["quimb", "cutensornet"]:
            raise TypeError(f"Unsupported backend: {backend}")

        if backend == "cutensornet":
            raise NotImplementedError(
                f"The cutensornet backend has not yet been implemented."
            )

        if method not in ["mps", "tn"]:
            raise TypeError(f"Unsupported method: {method}")

        if method == "tn":
            raise NotImplementedError(
                f"The tensor network method has not yet been implemented."
            )

        # Should we accept cases in which shots=0, shots=1?
        if shots is not None:
            raise ValueError("LightningTensor does not support the `shots` parameter.")

        super().__init__(wires=wires, shots=shots)

        self._num_wires = len(self.wires) if self.wires else 0

        self._c_dtype = c_dtype

        self._statetensor = LightningStateTensor(
            num_wires=self.num_wires, dtype=self._c_dtype
        )

    @property
    def name(self):
        """The name of the device."""
        return "lightning.tensor"

    @property
    def num_wires(self):
        """Number of wires addressed on this device"""
        return self._num_wires

    @property
    def c_dtype(self):
        """State vector complex data type."""
        return self._c_dtype

    dtype = c_dtype

    def execute():
        pass
