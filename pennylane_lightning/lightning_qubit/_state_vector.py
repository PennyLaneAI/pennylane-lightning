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
Class implementation for lightning_qubit state-vector manipulation.
"""
from warnings import warn

try:
    from pennylane_lightning.lightning_qubit_ops import (
        StateVectorC64,
        StateVectorC128,
        allocate_aligned_array,
    )
except ImportError as ex:
    warn(str(ex), UserWarning)

from typing import Union

import numpy as np
import pennylane as qml
import scipy as sp
from numpy.random import Generator
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional
from pennylane.ops.op_math import Adjoint
from pennylane.wires import Wires

# pylint: disable=ungrouped-imports
from pennylane_lightning.lightning_base._state_vector import LightningBaseStateVector

from ._measurements import LightningMeasurements


class LightningStateVector(LightningBaseStateVector):  # pylint: disable=too-few-public-methods
    """Lightning Qubit state-vector class.

    Interfaces with C++ python binding methods for state-vector manipulation.

    Args:
        num_wires(int): the number of wires to initialize the device with
        dtype: Datatypes for state-vector representation. Must be one of
            ``np.complex64`` or ``np.complex128``. Default is ``np.complex128``
        rng (Generator): random number generator to use for seeding sampling measurement.
    """

    def __init__(
        self,
        num_wires: int,
        dtype: Union[np.complex128, np.complex64] = np.complex128,
        rng: Generator = None,
    ):

        super().__init__(num_wires, dtype, rng)

        self._device_name = "lightning.qubit"

        # Initialize the state vector
        self._qubit_state = self._state_dtype()(num_wires)

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

    @staticmethod
    def _operation_is_sparse(operation):
        """Check if the operation is a sparse matrix operation.

        Args:
            operation (Operation): operation to check

        Returns:
            bool: True if the operation is a sparse matrix operation, False otherwise
        """
        return operation.has_sparse_matrix and not operation.has_matrix

    def _apply_state_vector(self, state, device_wires: Wires):
        """Initialize the internal state vector in a specified state.
        Args:
            state (Union[array[complex], scipy.SparseABC]): normalized input state of length ``2**len(wires)`` as a dense array or Scipy sparse array.
            device_wires (Wires): wires that get initialized in the state.
        """

        if sp.sparse.issparse(state):
            state = state.toarray().flatten()
        elif isinstance(state, self._qubit_state.__class__):
            state_data = allocate_aligned_array(state.size(), np.dtype(self.dtype), True)
            state.getState(state_data)
            state = state_data

        # Convert PennyLane tensor to NumPy array if needed
        if hasattr(state, "numpy"):
            state = state.numpy()

        if len(device_wires) == self._num_wires and Wires(sorted(device_wires)) == device_wires:
            # Initialize the entire device state with the input state
            output_shape = (2,) * self._num_wires
            state = np.reshape(state, output_shape).ravel(order="C")
            self._qubit_state.updateData(state)
            return

        self._qubit_state.setStateVector(state, list(device_wires))

    def _apply_lightning_controlled(self, operation, adjoint):
        """Apply an arbitrary controlled operation to the state tensor.

        Args:
            operation (~pennylane.operation.Operation): controlled operation to apply
            adjoint (bool): Apply the adjoint of the operation if True

        Returns:
            None
        """
        state = self.state_vector

        if isinstance(operation.base, Adjoint):
            base_operation = operation.base.base
            adjoint = not adjoint
        else:
            base_operation = operation.base

        method = getattr(state, f"{base_operation.name}", None)

        control_wires = list(operation.control_wires)
        control_values = operation.control_values
        target_wires = list(operation.target_wires)

        if method is not None:  # apply n-controlled specialized gate
            param = base_operation.parameters

            if isinstance(base_operation, qml.PCPhase):
                hyper = float(base_operation.hyperparameters["dimension"][0])
                param = np.array([base_operation.parameters[0], hyper])

            method(control_wires, control_values, target_wires, adjoint, param)
        else:  # apply gate as an n-controlled matrix
            method = getattr(state, "applyControlledMatrix")
            method(
                qml.matrix(base_operation),
                control_wires,
                control_values,
                target_wires,
                adjoint,
            )

    def _apply_lightning_controlled_sparse(self, operation):
        """Apply an arbitrary controlled operation to the state tensor.

        Args:
            operation (~pennylane.operation.Operation): controlled operation to apply
        Returns:
            None
        """
        state = self.state_vector

        if isinstance(operation.base, Adjoint):
            base_operation = operation.base.base
        else:
            base_operation = operation.base

        CSR_SparseHamiltonian = base_operation.sparse_matrix()

        control_wires = list(operation.control_wires)
        control_values = operation.control_values
        target_wires = list(operation.target_wires)

        method = getattr(state, "applyControlledSparseMatrix")
        method(
            CSR_SparseHamiltonian.indptr,
            CSR_SparseHamiltonian.indices,
            CSR_SparseHamiltonian.data,
            control_wires,
            control_values,
            target_wires,
            False,
        )

    # pylint: disable=too-many-branches
    def _apply_lightning(
        self, operations, mid_measurements: dict = None, postselect_mode: str = None
    ):  # pylint: disable=protected-access
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
        state = self.state_vector
        # Skip over identity operations instead of performing
        # matrix multiplication with it.
        for operation in operations:
            if isinstance(operation, qml.Identity):
                continue
            if isinstance(operation, Adjoint):
                op_adjoint_base = operation.base
                invert_param = True
            else:
                op_adjoint_base = operation
                invert_param = False

            name = op_adjoint_base.name
            method = getattr(state, name, None)
            wires = list(operation.wires)

            if isinstance(operation, Conditional):
                if operation.meas_val.concretize(mid_measurements):
                    self._apply_lightning([operation.base])
            elif isinstance(operation, MidMeasureMP):
                self._apply_lightning_midmeasure(
                    LightningMeasurements(self).measure_final_state,
                    operation,
                    mid_measurements,
                    postselect_mode=postselect_mode,
                )
            elif isinstance(operation, qml.PauliRot):
                method = getattr(state, "applyPauliRot")
                paulis = operation._hyperparameters[  # pylint: disable=protected-access
                    "pauli_word"
                ]
                wires = [i for i, w in zip(wires, paulis) if w != "I"]
                word = "".join(p for p in paulis if p != "I")
                method(wires, invert_param, operation.parameters, word)
            elif self._operation_is_sparse(operation):
                # Inverse can be set to False since operation.sparse_matrix() is already in inverted form
                if isinstance(op_adjoint_base, qml.ops.Controlled):
                    self._apply_lightning_controlled_sparse(op_adjoint_base)
                # If the operation is not controlled, apply it as a sparse matrix.
                else:
                    CSR_SparseHamiltonian = operation.sparse_matrix()
                    method = getattr(state, "applySparseMatrix")
                    method(
                        CSR_SparseHamiltonian.indptr,
                        CSR_SparseHamiltonian.indices,
                        CSR_SparseHamiltonian.data,
                        wires,
                        False,
                    )
            elif method is not None:  # apply specialized gate
                param = op_adjoint_base.parameters

                if isinstance(op_adjoint_base, qml.PCPhase):
                    hyper = float(op_adjoint_base.hyperparameters["dimension"][0])
                    param = np.array([op_adjoint_base.parameters[0], hyper])

                method(wires, invert_param, param)
            elif isinstance(op_adjoint_base, qml.ops.Controlled):  # apply n-controlled gate
                self._apply_lightning_controlled(op_adjoint_base, invert_param)
            else:
                # apply gate as a matrix
                # Inverse can be set to False since qml.matrix(operation) is already in inverted form
                method = getattr(state, "applyMatrix")
                method(qml.matrix(operation), wires, False)
