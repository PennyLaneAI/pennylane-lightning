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

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from pennylane import BasisState, StatePrep
from pennylane.measurements import MidMeasureMP
from pennylane.tape import QuantumScript
from pennylane.wires import Wires


class LightningBaseStateVector(ABC):
    """Lightning [Device] state-vector class.

    A class that serves as a base class for Lightning state-vector simulators.
    Interfaces with C++ python binding methods for state-vector manipulation.

    Args:
        num_wires(int): the number of wires to initialize the device with
        dtype: Datatypes for state-vector representation. Must be one of
            ``np.complex64`` or ``np.complex128``. Default is ``np.complex128``
    """

    def __init__(self, num_wires: int, dtype: Union[np.complex128, np.complex64]):
        if dtype not in [np.complex64, np.complex128]:
            raise TypeError(f"Unsupported complex type: {dtype}")

        self._num_wires = num_wires
        self._wires = Wires(range(num_wires))
        self._dtype = dtype

        # Dummy for the device name
        self._device_name = None
        # Dummy for the C++ bindings
        self._qubit_state = None

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

    @property
    def state_vector(self):
        """Returns a handle to the state vector."""
        return self._qubit_state

    @property
    @abstractmethod
    def state(self):
        """Copy the state vector data to a numpy array.

        **Example**

        >>> dev = qml.device('lightning.[Device]', wires=1)
        >>> dev.apply([qml.PauliX(wires=[0])])
        >>> print(dev.state)
        [0.+0.j 1.+0.j]
        """

    @abstractmethod
    def _state_dtype(self):
        """Binding to Lightning Managed state vector C++ class.

        Returns: the state vector class
        """

    def reset_state(self):
        """Reset the device's state"""
        # init the state vector to |00..0>
        self._qubit_state.resetStateVector()

    @abstractmethod
    def _apply_state_vector(self, state, device_wires: Wires, sync: Optional[bool] = None):
        """Initialize the internal state vector in a specified state.
        Args:
            state (array[complex]): normalized input state of length ``2**len(wires)``
                or broadcasted state of shape ``(batch_size, 2**len(wires))``
            device_wires (Wires): wires that get initialized in the state
        """

    def _apply_basis_state(self, state, wires):
        """Initialize the state vector in a specified computational basis state.

        Args:
            state (array[int]): computational basis state of shape ``(wires,)``
                consisting of 0s and 1s.
            wires (Wires): wires that the provided computational state should be
                initialized on
            use_async(Optional[bool]): immediately sync with host-sv after applying operation.

        Note: This function does not support broadcasted inputs yet.
        """
        if not set(state.tolist()).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

        if len(state) != len(wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        # Return a computational basis state over all wires.
        self._qubit_state.setBasisState(list(state), list(wires))

    @abstractmethod
    def _apply_lightning_controlled(self, operation):
        """Apply an arbitrary controlled operation to the state tensor.

        Args:
            operation (~pennylane.operation.Operation): controlled operation to apply

        Returns:
            None
        """

    @abstractmethod
    def _apply_lightning_midmeasure(
        self, operation: MidMeasureMP, mid_measurements: dict, postselect_mode: str
    ):
        """Execute a MidMeasureMP operation and return the sample in mid_measurements.

        Args:
            operation (~pennylane.operation.Operation): mid-circuit measurement
            mid_measurements (None, dict): Dictionary of mid-circuit measurements
            postselect_mode (str): Configuration for handling shots with mid-circuit measurement
                postselection. Use ``"hw-like"`` to discard invalid shots and ``"fill-shots"`` to
                keep the same number of shots.

        Returns:
            None
        """

    @abstractmethod
    def _apply_lightning(
        self, operations, mid_measurements: dict = None, postselect_mode: str = None
    ):
        """Apply a list of operations to the state tensor.

        Args:
            operations (list[~pennylane.operation.Operation]): operations to apply
            mid_measurements (None, dict): Dictionary of mid-circuit measurements
            postselect_mode (str): Configuration for handling shots with mid-circuit measurement
                postselection. Use ``"hw-like"`` to discard invalid shots and ``"fill-shots"`` to
                keep the same number of shots. Default is ``None``.

        Returns:
            None
        """

    def apply_operations(
        self, operations, mid_measurements: dict = None, postselect_mode: str = None
    ):
        """Applies operations to the state vector."""
        # State preparation is currently done in Python
        if operations:  # make sure operations[0] exists
            if isinstance(operations[0], StatePrep):
                self._apply_state_vector(operations[0].parameters[0].copy(), operations[0].wires)
                operations = operations[1:]
            elif isinstance(operations[0], BasisState):
                self._apply_basis_state(operations[0].parameters[0], operations[0].wires)
                operations = operations[1:]
        self._apply_lightning(
            operations, mid_measurements=mid_measurements, postselect_mode=postselect_mode
        )

    def get_final_state(
        self,
        circuit: QuantumScript,
        mid_measurements: dict = None,
        postselect_mode: str = None,
    ):
        """
        Get the final state that results from executing the given quantum script.

        This is an internal function that will be called by the successor to ``lightning.[Device]``.

        Args:
            circuit (QuantumScript): The single circuit to simulate
            mid_measurements (None, dict): Dictionary of mid-circuit measurements
            postselect_mode (str): Configuration for handling shots with mid-circuit measurement
                postselection. Use ``"hw-like"`` to discard invalid shots and ``"fill-shots"`` to
                keep the same number of shots. Default is ``None``.

        Returns:
            Lightning [Device] StateVector: Lightning final state class.

        """
        self.apply_operations(
            circuit.operations, mid_measurements=mid_measurements, postselect_mode=postselect_mode
        )

        return self
