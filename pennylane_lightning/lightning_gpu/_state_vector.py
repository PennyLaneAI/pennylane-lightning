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
Class implementation for lightning_gpu state-vector manipulation.
"""
try:
    from pennylane_lightning.lightning_gpu_ops import (
        StateVectorC64,
        StateVectorC128,
    )
    
    try: # Try to import the MPI modules
        # pylint: disable=no-name-in-module
        from pennylane_lightning.lightning_gpu_ops import (
            StateVectorMPIC64,
            StateVectorMPIC128,
        )

        MPI_SUPPORT = True
    except ImportError:
        MPI_SUPPORT = False
        
except ImportError:
    pass

from itertools import product

import numpy as np
import pennylane as qml
from pennylane import DeviceError
from pennylane.ops.op_math import Adjoint
from pennylane.wires import Wires
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional


from pennylane_lightning.core._state_vector_base import LightningBaseStateVector

from ._measurements import LightningGPUMeasurements


class LightningGPUStateVector(LightningBaseStateVector):
    """Lightning GPU state-vector class.

    Interfaces with C++ python binding methods for state-vector manipulation.

    Args:
        num_wires(int): the number of wires to initialize the device with
        dtype: Datatypes for state-vector representation. Must be one of
            ``np.complex64`` or ``np.complex128``. Default is ``np.complex128``
        device_name(string): state vector device name. Options: ["lightning.gpu"]
    """

    def __init__(self, num_wires, dtype=np.complex128, device_name="lightning.gpu", 
                 mpi_handler = None, 
                 sync=True,
                 ):

        if device_name != "lightning.gpu":
            raise DeviceError(f'The device name "{device_name}" is not a valid option.')

        super().__init__(num_wires, dtype)

        self._device_name = device_name

        self._num_global_wires = self._mpi_handler.num_global_wires
        self._num_local_wires = self._mpi_handler.num_local_wires

        self._dtype = dtype
        self._mpi_handler = mpi_handler
        self._sync = sync

        self._wires = Wires(range(num_wires))
        
        if self._mpi_handler.use_mpi:
            self._lgpu_state = self._state_dtype()(
                self._mpi_handler.mpi_manager,
                self._mpi_handler.devtag,
                self._mpi_handler.mpi_buf_size,
                self._mpi_handler.num_global_wires,
                self._mpi_handler.num_local_wires,
            )

        if not self._mpi_handler.use_mpi:
            self._lgpu_state = self._state_dtype()(self.num_wires)
            
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
        return self._lgpu_state

    @property
    def state(self):
        """Copy the state vector data from the device to the host.

        A state vector Numpy array is explicitly allocated on the host to store and return the data.

        **Example**

        >>> dev = qml.device('lightning.gpu', wires=1)
        >>> dev.apply([qml.PauliX(wires=[0])])
        >>> print(dev.state)
        [0.+0.j 1.+0.j]
        """
        state = np.zeros(1 << self._num_local_wires, dtype=self.dtype)
        self.syncD2H(state)
        return state

    def _state_dtype(self):
        """Binding to Lightning Managed state vector C++ class.

        Returns: the state vector class
        """
        if self._mpi_handler.use_mpi:
            return StateVectorMPIC128 if self.dtype == np.complex128 else StateVectorMPIC64
        else:
            return StateVectorC128 if self.dtype == np.complex128 else StateVectorC64

    def reset_state(self):
        """Reset the device's state"""
        # init the state vector to |00..0>
        self._gpu_state.resetGPU(False)  # Sync reset

        self._lgpu_state.resetStateVector()

    def syncD2H(self, state_vector, use_async=False):
        """Copy the state vector data on device to a state vector on the host provided by the user
        Args:
            state_vector(array[complex]): the state vector array on host
            use_async(bool): indicates whether to use asynchronous memory copy from host to device or not.
            Note: This function only supports synchronized memory copy.

        **Example**
        >>> dev = qml.device('lightning.gpu', wires=1)
        >>> dev.apply([qml.PauliX(wires=[0])])
        >>> state_vector = np.zeros(2**dev.num_wires).astype(dev.C_DTYPE)
        >>> dev.syncD2H(state_vector)
        >>> print(state_vector)
        [0.+0.j 1.+0.j]
        """
        self._lgpu_state.DeviceToHost(state_vector.ravel(order="C"), use_async)

    def syncH2D(self, state_vector, use_async=False):
        """Copy the state vector data on host provided by the user to the state vector on the device
        Args:
            state_vector(array[complex]): the state vector array on host.
            use_async(bool): indicates whether to use asynchronous memory copy from host to device or not.
            Note: This function only supports synchronized memory copy.

        **Example**
        >>> dev = qml.device('lightning.gpu', wires=3)
        >>> obs = qml.Identity(0) @ qml.PauliX(1) @ qml.PauliY(2)
        >>> obs1 = qml.Identity(1)
        >>> H = qml.Hamiltonian([1.0, 1.0], [obs1, obs])
        >>> state_vector = np.array([0.0 + 0.0j, 0.0 + 0.1j, 0.1 + 0.1j, 0.1 + 0.2j,
            0.2 + 0.2j, 0.3 + 0.3j, 0.3 + 0.4j, 0.4 + 0.5j,], dtype=np.complex64,)
        >>> dev.syncH2D(state_vector)
        >>> res = dev.expval(H)
        >>> print(res)
        1.0
        """
        self._lgpu_state.HostToDevice(state_vector.ravel(order="C"), use_async)

    def _apply_state_vector(self, state, device_wires, use_async=False):
        """Initialize the state vector on GPU with a specified state on host.
        Note that any use of this method will introduce host-overheads.
        Args:
        state (array[complex]): normalized input state (on host) of length ``2**len(wires)``
                or broadcasted state of shape ``(batch_size, 2**len(wires))``
        device_wires (Wires): wires that get initialized in the state
        use_async(bool): indicates whether to use asynchronous memory copy from host to device or not.
        Note: This function only supports synchronized memory copy from host to device.
        """
        # translate to wire labels used by device
        # device_wires = self.map_wires(device_wires)

        # state = self._asarray(state, dtype=self.C_DTYPE)  # this operation on host
        output_shape = [2] * self._num_local_wires

        if len(device_wires) == self.num_wires and Wires(sorted(device_wires)) == device_wires:
            # Initialize the entire device state with the input state
            if self.num_wires == self._num_local_wires:
                self.syncH2D(self._reshape(state, output_shape))
                return
            local_state = np.zeros(1 << self._num_local_wires, dtype=self.C_DTYPE)
            self._mpi_handler.mpi_manager.Scatter(state, local_state, 0)
            # Initialize the entire device state with the input state
            # self.syncH2D(self._reshape(local_state, output_shape))
            self.syncH2D(np.reshape(local_state, output_shape))
            return

        # generate basis states on subset of qubits via the cartesian product
        basis_states = np.array(list(product([0, 1], repeat=len(device_wires))))

        # get basis states to alter on full set of qubits
        unravelled_indices = np.zeros((2 ** len(device_wires), self.num_wires), dtype=int)
        unravelled_indices[:, device_wires] = basis_states

        # get indices for which the state is changed to input state vector elements
        ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self.num_wires)

        # set the state vector on GPU with the unravelled_indices and their corresponding values
        self._lgpu_state.setStateVector(
            ravelled_indices, state, use_async
        )  # this operation on device

    def _apply_basis_state(self, state, wires):
        """Initialize the state vector in a specified computational basis state on GPU directly.
            Args:
            state (array[int]): computational basis state (on host) of shape ``(wires,)``
                consisting of 0s and 1s.
            wires (Wires): wires that the provided computational state should be initialized on
        Note: This function does not support broadcasted inputs yet.
        """
        # # translate to wire labels used by device
        # device_wires = self.map_wires(wires)

        # length of basis state parameter
        if not set(state.tolist()).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

        if len(state) != len(wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        self._lgpu_state.setStateVector(list(state), list(wires))
        # # get computational basis state number
        # basis_states = 2 ** (self.num_wires - 1 - np.array(list(wires)))
        # basis_states = qml.math.convert_like(basis_states, state)
        # num = int(qml.math.dot(state, basis_states))

        # self._create_basis_state(num)

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
                    self._apply_lightning([operation.base])
            elif isinstance(operation, MidMeasureMP):
                self._apply_lightning_midmeasure(
                    operation, mid_measurements, postselect_mode=postselect_mode
                )
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
