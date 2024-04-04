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
Class implementation for state-tensor manipulation.
"""

from typing import Iterable, Union
import quimb.tensor as qtn

import numpy as np
import pennylane as qml
from pennylane import DeviceError

from pennylane.wires import Wires


class LightningStateTensor:
    """Lightning state-tensor class.

    Interfaces with C++ python binding methods for state-tensor manipulation.
    """

    def __init__(self, num_wires, dtype=np.complex128, device_name="lightning.tensor"):
        self._num_wires = num_wires
        self._wires = Wires(range(num_wires))
        self._dtype = dtype

        if dtype not in [np.complex64, np.complex128]:  # pragma: no cover
            raise TypeError(f"Unsupported complex type: {dtype}")

        if device_name != "lightning.tensor":
            raise DeviceError(f'The device name "{device_name}" is not a valid option.')

        self._device_name = device_name
        # TODO: add binding to Lightning Managed state tensor C++ class.
        # self._tensor_state = self._state_dtype()(self._num_wires)

        # TODO: change name
        self._quimb_state = self._create_initial_state(self._wires)

    @property
    def dtype(self):
        """Returns the state tensor data type."""
        return self._dtype

    @property
    def device_name(self):
        """Returns the state tensor device name."""
        return self._device_name

    @property
    def wires(self):
        """All wires that can be addressed on this device"""
        return self._wires

    @property
    def num_wires(self):
        """Number of wires addressed on this device"""
        return self._num_wires

    @property
    def state_tensor(self):
        """Returns a handle to the state tensor."""
        return self._tensor_state

    # TODO implement
    @property
    def state(self):
        """Copy the state tensor data to a numpy array."""
        pass

    # TODO implement
    def _state_dtype(self):
        """Binding to Lightning Managed state tensor C++ class.

        Returns: the state tensor class
        """
        pass

    def _create_initial_state(self, wires: Union[qml.wires.Wires, Iterable]):
        r"""
        Returns an initial state to :math:`\ket{0}`.

        Args:
            wires (Union[Wires, Iterable]): The wires to be present in the initial state.

        Returns:
            array: The initial state of a circuit.
        """

        return qtn.MPS_computational_state(
            "0" * max(1, len(wires)),
            dtype=self._dtype.__name__,
            tags=[str(l) for l in wires.labels],
        )
