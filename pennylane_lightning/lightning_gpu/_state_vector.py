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
from warnings import warn

try:
    from pennylane_lightning.lightning_gpu_ops import StateVectorC64, StateVectorC128

    try:  # Try to import the MPI modules
        from pennylane_lightning.lightning_gpu_ops import StateVectorMPIC64, StateVectorMPIC128

        mpi_error = None
        MPI_SUPPORT = True
    except ImportError as ex_mpi:
        mpi_error = ex_mpi
        MPI_SUPPORT = False

except ImportError as ex:
    warn(str(ex), UserWarning)

from typing import Union

import numpy as np
import pennylane as qml
import scipy as sp
from numpy.random import Generator
from pennylane.exceptions import DeviceError
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional
from pennylane.ops.op_math import Adjoint
from pennylane.wires import Wires

# pylint: disable=ungrouped-imports
from pennylane_lightning.lightning_base._state_vector import LightningBaseStateVector

from ._measurements import LightningGPUMeasurements
from ._mpi_handler import MPIHandler

gate_cache_needs_hash = (
    qml.BlockEncode,
    qml.ControlledQubitUnitary,
    qml.DiagonalQubitUnitary,
    qml.MultiControlledX,
    qml.OrbitalRotation,
    qml.PSWAP,
    qml.QubitUnitary,
)


class LightningGPUStateVector(LightningBaseStateVector):
    """Lightning GPU state-vector class.

    Interfaces with C++ python binding methods for state-vector manipulation.

    Args:
        num_wires(int): the number of wires to initialize the device with
        dtype: Datatypes for state-vector representation. Must be one of
            ``np.complex64`` or ``np.complex128``. Default is ``np.complex128``
        rng (Generator): random number generator to use for seeding sampling measurement.
        mpi_handler(MPIHandler): MPI handler for PennyLane Lightning GPU device.
            Provides functionality to distribute the state-vector to multiple devices.
        use_async (bool): is host-device data copy asynchronized or not.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        num_wires: int,
        dtype: Union[np.complex128, np.complex64] = np.complex128,
        rng: Generator = None,
        mpi_handler: MPIHandler = None,
        use_async: bool = False,
    ):

        super().__init__(num_wires, dtype, rng)

        self._device_name = "lightning.gpu"

        # Initialize GPU and MPI variables
        if mpi_handler is None:
            mpi_handler = MPIHandler(False, 0, num_wires, dtype)

        self._num_global_wires = mpi_handler.num_global_wires
        self._num_local_wires = mpi_handler.num_local_wires

        self._mpi_handler = mpi_handler
        self._use_async = use_async

        # Initialize the state vector
        if self._mpi_handler.use_mpi:  # using MPI
            self._qubit_state = self._state_dtype()(
                self._mpi_handler.mpi_manager,
                self._mpi_handler.devtag,
                self._mpi_handler.mpi_buf_size,
                self._mpi_handler.num_global_wires,
                self._mpi_handler.num_local_wires,
            )
        else:  # without MPI
            self._qubit_state = self._state_dtype()(num_wires)

    def _state_dtype(self):
        """Binding to Lightning Managed state vector C++ class.

        Returns: the state vector class
        """
        if self._mpi_handler.use_mpi:
            if not MPI_SUPPORT:
                warn(str(mpi_error), UserWarning)

            return StateVectorMPIC128 if self.dtype == np.complex128 else StateVectorMPIC64

        # without MPI
        return StateVectorC128 if self.dtype == np.complex128 else StateVectorC64

    def syncD2H(self, state_vector, use_async: bool = False):
        """Copy the state vector data on device to a state vector on the host provided by the user.
        Args:
            state_vector(array[complex]): the state vector array on host.
            use_async(bool): indicates whether to use asynchronous memory copy from host to device or not.
            Note: This function only supports synchronized memory copy.

        **Example**

        >>> dev = qml.device('lightning.gpu', wires=1)
        >>> dev.apply([qml.PauliX(wires=[0])])
        >>> state_vector = np.zeros(2**dev.num_wires).astype(dev.c_type)
        >>> dev.syncD2H(state_vector)
        >>> print(state_vector)
        [0.+0.j 1.+0.j]
        """
        self._qubit_state.DeviceToHost(state_vector.ravel(order="C"), use_async)

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
        state = np.zeros(2**self._num_local_wires, dtype=self.dtype)
        self.syncD2H(state)
        return state

    def syncH2D(self, state_vector, use_async: bool = False):
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
        self._qubit_state.HostToDevice(state_vector.ravel(order="C"), use_async)

    @staticmethod
    def _asarray(arr, dtype=None):
        arr = np.asarray(arr)  # arr is not copied

        if arr.dtype.kind not in ["f", "c"]:
            return arr

        if not dtype:
            dtype = arr.dtype

        return arr

    # pylint: disable=unused-argument
    @staticmethod
    def _operation_is_sparse(operation):
        """Check if the operation is a sparse matrix operation.

        Args:
            operation (_): operation to check

        Returns:
            bool: True if the operation is a sparse matrix operation, False otherwise
        """
        # Currently there is not support for sparse matrices in the LightningGPU device.
        return False

    def _apply_state_vector(self, state, device_wires: Wires, **kwargs):
        """Initialize the internal state vector in a specified state.
        Args:
            state (Union[array[complex], scipy.SparseABC]): normalized input state of length ``2**len(wires)`` as a dense array or Scipy sparse array.
            device_wires (Wires): wires that get initialized in the state
            use_async(bool): indicates whether to use asynchronous memory copy from host to device or not.
        Note: This function only supports synchronized memory copy from host to device.
        """
        use_async = kwargs.get("use_async", False)

        if isinstance(state, self._qubit_state.__class__):
            raise DeviceError("LightningGPU does not support allocate external state_vector.")

            # TODO
            # Create an implementation in the C++ backend and binding to be able
            # to allocate memory for a new statevector and copy the data
            # from an external state vector.
            # state_data = allocate_aligned_array(state.size, np.dtype(self.dtype), True)
            # state.getState(state_data)
            # state = state_data

        if sp.sparse.issparse(state):
            state = state.toarray().flatten()  # this operation is on host
        else:
            state = self._asarray(state, dtype=self.dtype)  # this operation is on host

        output_shape = [2] * self._num_local_wires

        if len(device_wires) == self.num_wires and Wires(sorted(device_wires)) == device_wires:
            # Initialize the entire device state with the input state
            output_shape = [2] * self._num_local_wires
            if self.num_wires == self._num_local_wires:
                self.syncH2D(np.reshape(state, output_shape))
                return
            local_state = np.zeros(2**self._num_local_wires, dtype=self._dtype)
            self._mpi_handler.mpi_manager.Scatter(state, local_state, 0)
            self.syncH2D(np.reshape(local_state, output_shape))
            return

        # set the state vector on GPU with provided state and their corresponding wires
        self._qubit_state.setStateVector(state, list(device_wires), use_async)

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

        if method:  # apply n-controlled specialized gate
            param = operation.parameters
            if isinstance(base_operation, qml.PCPhase):
                # PCPhase has hyperparameters for dimension
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

    # pylint: disable=unused-argument, too-many-branches
    def _apply_lightning(
        self, operations, mid_measurements: dict = None, postselect_mode: str = None
    ):
        """Apply a list of operations to the state vector.

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
                    LightningGPUMeasurements(self).measure_final_state,
                    operation,
                    mid_measurements,
                    postselect_mode=postselect_mode,
                )
            elif method is not None:  # apply specialized gate
                param = operation.parameters
                if isinstance(op_adjoint_base, qml.PCPhase):
                    # PCPhase has hyperparameters for dimension
                    hyper = float(op_adjoint_base.hyperparameters["dimension"][0])
                    param = np.array([op_adjoint_base.parameters[0], hyper])

                method(wires, invert_param, param)
            elif (
                isinstance(op_adjoint_base, qml.ops.Controlled) and not self._mpi_handler.use_mpi
            ):  # MPI backend does not have native controlled gates support
                self._apply_lightning_controlled(op_adjoint_base, invert_param)
            elif (
                self._mpi_handler.use_mpi
                and isinstance(op_adjoint_base, qml.ops.Controlled)
                and isinstance(op_adjoint_base.base, qml.GlobalPhase)
            ):
                # TODO: To move this line to the _apply_lightning_controlled method once the MPI backend supports controlled gates natively
                raise DeviceError(
                    "Lightning-GPU-MPI does not support Controlled GlobalPhase gates."
                )
            else:  # apply gate as a matrix
                try:
                    mat = qml.matrix(operation)
                except AttributeError:  # pragma: no cover
                    # To support older versions of PL
                    mat = operation.matrix
                r_dtype = np.float32 if self.dtype == np.complex64 else np.float64
                param = (
                    [[r_dtype(operation.hash)]]
                    if isinstance(operation, gate_cache_needs_hash)
                    else []
                )
                if len(mat) == 0:
                    raise ValueError("Unsupported operation")

                self._qubit_state.apply(
                    name,
                    wires,
                    False,
                    param,
                    mat.ravel(order="C"),  # inv = False: Matrix already in correct form;
                )  # Parameters can be ignored for explicit matrices; F-order for cuQuantum
