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

try:
    from pennylane_lightning.lightning_qubit_ops import (
        StateVectorC64,
        StateVectorC128,
        allocate_aligned_array,
    )
except ImportError:
    pass

from itertools import product

import numpy as np
import pennylane as qml
from pennylane import BasisState, DeviceError, StatePrep
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional
from pennylane.ops.op_math import Adjoint
from pennylane.tape import QuantumScript
from pennylane.wires import Wires

from ._measurements import LightningMeasurements


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

        if dtype not in [np.complex64, np.complex128]:  # pragma: no cover
            raise TypeError(f"Unsupported complex type: {dtype}")

        if device_name != "lightning.qubit":
            raise DeviceError(f'The device name "{device_name}" is not a valid option.')

        self._device_name = device_name
        self._qubit_state = self._state_dtype()(self._num_wires)

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
    def state(self):
        """Copy the state vector data to a numpy array.

        **Example**

        >>> dev = qml.device('lightning.qubit', wires=1)
        >>> dev.apply([qml.PauliX(wires=[0])])
        >>> print(dev.state)
        [0.+0.j 1.+0.j]
        """
        state = np.zeros(2**self._num_wires, dtype=self.dtype)
        self._qubit_state.getState(state)
        return state

    def _state_dtype(self):
        """Binding to Lightning Managed state vector C++ class.

        Returns: the state vector class
        """
        return StateVectorC128 if self.dtype == np.complex128 else StateVectorC64

    def _create_basis_state(self, index):
        """Return a computational basis state over all wires.

        Args:
            index (int): integer representing the computational basis state.
        """
        self._qubit_state.setBasisState(index)

    def reset_state(self):
        """Reset the device's state"""
        # init the state vector to |00..0>
        self._qubit_state.resetStateVector()

    def _preprocess_state_vector(self, state, device_wires):
        """Initialize the internal state vector in a specified state.

        Args:
            state (array[complex]): normalized input state of length ``2**len(wires)``
                or broadcasted state of shape ``(batch_size, 2**len(wires))``
            device_wires (Wires): wires that get initialized in the state

        Returns:
            array[int]: indices for which the state is changed to input state vector elements
            array[complex]: normalized input state of length ``2**len(wires)``
                or broadcasted state of shape ``(batch_size, 2**len(wires))``
        """
        # special case for integral types
        if state.dtype.kind == "i":
            state = np.array(state, dtype=self.dtype)

        if len(device_wires) == self._num_wires and Wires(sorted(device_wires)) == device_wires:
            return None, state

        # generate basis states on subset of qubits via the cartesian product
        basis_states = np.array(list(product([0, 1], repeat=len(device_wires))))

        # get basis states to alter on full set of qubits
        unravelled_indices = np.zeros((2 ** len(device_wires), self._num_wires), dtype=int)
        unravelled_indices[:, device_wires] = basis_states

        # get indices for which the state is changed to input state vector elements
        ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self._num_wires)
        return ravelled_indices, state

    def _get_basis_state_index(self, state, wires):
        """Returns the basis state index of a specified computational basis state.

        Args:
            state (array[int]): computational basis state of shape ``(wires,)``
                consisting of 0s and 1s
            wires (Wires): wires that the provided computational state should be initialized on

        Returns:
            int: basis state index
        """
        # length of basis state parameter
        n_basis_state = len(state)

        if not set(state.tolist()).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

        if n_basis_state != len(wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        # get computational basis state number
        basis_states = 2 ** (self._num_wires - 1 - np.array(wires))
        basis_states = qml.math.convert_like(basis_states, state)
        return int(qml.math.dot(state, basis_states))

    def _apply_state_vector(self, state, device_wires: Wires):
        """Initialize the internal state vector in a specified state.
        Args:
            state (array[complex]): normalized input state of length ``2**len(wires)``
                or broadcasted state of shape ``(batch_size, 2**len(wires))``
            device_wires (Wires): wires that get initialized in the state
        """

        if isinstance(state, self._qubit_state.__class__):
            state_data = allocate_aligned_array(state.size, np.dtype(self.dtype), True)
            state.getState(state_data)
            state = state_data

        ravelled_indices, state = self._preprocess_state_vector(state, device_wires)

        # translate to wire labels used by device
        output_shape = [2] * self._num_wires

        if len(device_wires) == self._num_wires and Wires(sorted(device_wires)) == device_wires:
            # Initialize the entire device state with the input state
            state = np.reshape(state, output_shape).ravel(order="C")
            self._qubit_state.UpdateData(state)
            return

        self._qubit_state.setStateVector(ravelled_indices, state)

    def _apply_basis_state(self, state, wires):
        """Initialize the state vector in a specified computational basis state.

        Args:
            state (array[int]): computational basis state of shape ``(wires,)``
                consisting of 0s and 1s.
            wires (Wires): wires that the provided computational state should be
                initialized on

        Note: This function does not support broadcasted inputs yet.
        """
        num = self._get_basis_state_index(state, wires)
        self._create_basis_state(num)

    def _apply_lightning_controlled(self, operation):
        """Apply an arbitrary controlled operation to the state tensor.

        Args:
            operation (~pennylane.operation.Operation): controlled operation to apply

        Returns:
            None
        """
        state = self.state_vector

        basename = operation.base.name
        method = getattr(state, f"{basename}", None)
        control_wires = list(operation.control_wires)
        control_values = operation.control_values
        target_wires = list(operation.target_wires)
        if method is not None:  # apply n-controlled specialized gate
            inv = False
            param = operation.parameters
            method(control_wires, control_values, target_wires, inv, param)
        else:  # apply gate as an n-controlled matrix
            method = getattr(state, "applyControlledMatrix")
            method(
                qml.matrix(operation.base),
                control_wires,
                control_values,
                target_wires,
                False,
            )

    def _apply_lightning_midmeasure(self, operation: MidMeasureMP, mid_measurements: dict):
        """Execute a MidMeasureMP operation and return the sample in mid_measurements.
        Args:
            operation (~pennylane.operation.Operation): mid-circuit measurement
            mid_measurements (None, dict): Dictionary of mid-circuit measurements
        Returns:
            None
        """
        wires = self.wires.indices(operation.wires)
        wire = list(wires)[0]
        circuit = QuantumScript([], [qml.sample(wires=operation.wires)], shots=1)
        sample = LightningMeasurements(self).measure_final_state(circuit)
        sample = np.squeeze(sample)
        if operation.postselect is not None and sample != operation.postselect:
            mid_measurements[operation] = -1
            return
        mid_measurements[operation] = sample
        getattr(self.state_vector, "collapse")(wire, bool(sample))
        if operation.reset and bool(sample):
            self.apply_operations([qml.PauliX(operation.wires)], mid_measurements=mid_measurements)

    def _apply_lightning(self, operations, mid_measurements: dict = None):
        """Apply a list of operations to the state tensor.

        Args:
            operations (list[~pennylane.operation.Operation]): operations to apply
            mid_measurements (None, dict): Dictionary of mid-circuit measurements

        Returns:
            None
        """
        state = self.state_vector

        # Skip over identity operations instead of performing
        # matrix multiplication with it.
        for operation in operations:
            if isinstance(operation, qml.Identity):
                continue
            if isinstance(operation, Adjoint):
                name = operation.base.name
                invert_param = True
            else:
                name = operation.name
                invert_param = False
            method = getattr(state, name, None)
            wires = list(operation.wires)

            if isinstance(operation, Conditional):
                if operation.meas_val.concretize(mid_measurements):
                    self._apply_lightning([operation.then_op])
            elif isinstance(operation, MidMeasureMP):
                self._apply_lightning_midmeasure(operation, mid_measurements)
            elif method is not None:  # apply specialized gate
                param = operation.parameters
                method(wires, invert_param, param)
            elif isinstance(operation, qml.ops.Controlled):  # apply n-controlled gate
                self._apply_lightning_controlled(operation)
            else:  # apply gate as a matrix
                # Inverse can be set to False since qml.matrix(operation) is already in
                # inverted form
                method = getattr(state, "applyMatrix")
                try:
                    method(qml.matrix(operation), wires, False)
                except AttributeError:  # pragma: no cover
                    # To support older versions of PL
                    method(operation.matrix, wires, False)

    def apply_operations(self, operations, mid_measurements: dict = None):
        """Applies operations to the state vector."""
        # State preparation is currently done in Python
        if operations:  # make sure operations[0] exists
            if isinstance(operations[0], StatePrep):
                self._apply_state_vector(operations[0].parameters[0].copy(), operations[0].wires)
                operations = operations[1:]
            elif isinstance(operations[0], BasisState):
                self._apply_basis_state(operations[0].parameters[0], operations[0].wires)
                operations = operations[1:]

        self._apply_lightning(operations, mid_measurements=mid_measurements)

    def get_final_state(self, circuit: QuantumScript, mid_measurements: dict = None):
        """
        Get the final state that results from executing the given quantum script.

        This is an internal function that will be called by the successor to ``lightning.qubit``.

        Args:
            circuit (QuantumScript): The single circuit to simulate
            mid_measurements (None, dict): Dictionary of mid-circuit measurements

        Returns:
            LightningStateVector: Lightning final state class.

        """
        self.apply_operations(circuit.operations, mid_measurements=mid_measurements)

        return self
