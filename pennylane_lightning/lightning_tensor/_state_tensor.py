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
"""
Class implementation for state-tensor manipulation.
"""

# pylint: disable=import-error, no-name-in-module, ungrouped-imports
try:
    from pennylane_lightning.lightning_tensor_ops import StateTensorC64, StateTensorC128
except ImportError:
    pass


import numpy as np
import pennylane as qml
from pennylane import BasisState, DeviceError, StatePrep
from pennylane.ops.op_math import Adjoint
from pennylane.tape import QuantumScript
from pennylane.wires import Wires


class LightningStateTensor:
    """Lightning state-tensor class.

    Interfaces with C++ python binding methods for state-tensor manipulation.

    Args:
        num_wires(int): the number of wires to initialize the device with
        maxBondDim(int): maximum bond dimension for the state tensor
        dtype: Datatypes for state-tensor representation. Must be one of
            ``np.complex64`` or ``np.complex128``. Default is ``np.complex128``
        device_name(string): state tensor device name. Options: ["lightning.tensor"]
    """

    def __init__(self, num_wires, maxBondDim, dtype=np.complex128, device_name="lightning.tensor"):
        self._num_wires = num_wires
        self._wires = Wires(range(num_wires))
        self._maxBondDim = maxBondDim
        self._dtype = dtype

        if dtype not in [np.complex64, np.complex128]:  # pragma: no cover
            raise TypeError(f"Unsupported complex type: {dtype}")

        if device_name != "lightning.tensor":
            raise DeviceError(f'The device name "{device_name}" is not a valid option.')

        self._device_name = device_name
        self._tensor_state = self._state_dtype()(self._num_wires, self._maxBondDim)

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

    def _state_dtype(self):
        """Binding to Lightning Managed state tensor C++ class.

        Returns: the state tensor class
        """
        return StateTensorC128 if self.dtype == np.complex128 else StateTensorC64

    def reset_state(self):
        """Reset the device's state tensor"""
        # init the state tensor to |00..0>
        self._tensor_state.reset()

    def _apply_basis_state(self, state, wires):
        """Initialize the state tensor in a specified computational basis state.

        Args:
            state (array[int]): computational basis state of shape ``(wires,)``
                consisting of 0s and 1s.
            wires (Wires): wires that the provided computational state should be
                initialized on

        Note: This function does not support broadcasted inputs yet.
        """
        # length of basis state parameter
        n_basis_state = len(state)

        if not set(state.tolist()).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

        if n_basis_state != len(wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        self._tensor_state.setBasisState(state)

    def _apply_lightning(self, operations):
        """Apply a list of operations to the state tensor.

        Args:
            operations (list[~pennylane.operation.Operation]): operations to apply

        Returns:
            None
        """
        state = self.state_tensor

        # Skip over identity operations instead of performing
        # matrix multiplication with it.
        for operation in operations:
            if isinstance(operation, qml.Identity):
                continue
            if isinstance(operation, Adjoint):
                raise DeviceError("Adjoint operations are not supported.")
            name = operation.name
            method = getattr(state, name, None)
            wires = list(operation.wires)

            if method is not None:  # apply specialized gate
                param = operation.parameters
                method(wires, False, param)
            else:  # apply gate as a matrix
                # Inverse can be set to False since qml.matrix(operation) is already in
                # inverted form
                method = getattr(state, "applyMatrix")
                try:
                    method(qml.matrix(operation), wires, False)
                except AttributeError:  # pragma: no cover
                    # To support older versions of PL
                    method(operation.matrix, wires, False)

    def apply_operations(self, operations):
        """Append operations to underly graph."""
        # State preparation is currently done in Python
        if operations:  # make sure operations[0] exists
            if isinstance(operations[0], StatePrep):
                raise DeviceError(
                    "lightning.tensor does not support initialization with a state vector."
                )
            if isinstance(operations[0], BasisState):
                self._apply_basis_state(operations[0].parameters[0], operations[0].wires)
                operations = operations[1:]

        self._apply_lightning(operations)

    def get_final_state(self, circuit: QuantumScript):
        """
        Get the final state that results from executing the given quantum script.

        This is an internal function that will be called by the successor to ``lightning.tensor``.

        Args:
            circuit (QuantumScript): The single circuit to simulate

        Returns:
            LightningStateTensor: Lightning final state class.

        """
        self.apply_operations(circuit.operations)
        self.state_tensor.getFinalState()

        return self
