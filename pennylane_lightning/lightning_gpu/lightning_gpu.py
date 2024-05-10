# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
This module contains the :class:`~.LightningGPU` class, a PennyLane simulator device that
interfaces with the NVIDIA cuQuantum cuStateVec simulator library for GPU-enabled calculations.
"""

from ctypes.util import find_library
from importlib import util as imp_util
from itertools import product
from pathlib import Path
from typing import List, Union
from warnings import warn

import numpy as np
import pennylane as qml
from pennylane import BasisState, DeviceError, QuantumFunctionError, Rot, StatePrep, math
from pennylane.measurements import Expectation, State
from pennylane.ops.op_math import Adjoint
from pennylane.wires import Wires

from pennylane_lightning.core._serialize import QuantumScriptSerializer, global_phase_diagonal
from pennylane_lightning.core._version import __version__

# pylint: disable=import-error, no-name-in-module, ungrouped-imports
from pennylane_lightning.core.lightning_base import LightningBase

try:

    from pennylane_lightning.lightning_gpu_ops import (
        DevPool,
        MeasurementsC64,
        MeasurementsC128,
        StateVectorC64,
        StateVectorC128,
        backend_info,
        get_gpu_arch,
        is_gpu_supported,
    )
    from pennylane_lightning.lightning_gpu_ops.algorithms import (
        AdjointJacobianC64,
        AdjointJacobianC128,
        create_ops_listC64,
        create_ops_listC128,
    )

    try:
        # pylint: disable=no-name-in-module
        from pennylane_lightning.lightning_gpu_ops import (
            DevTag,
            MeasurementsMPIC64,
            MeasurementsMPIC128,
            MPIManager,
            StateVectorMPIC64,
            StateVectorMPIC128,
        )
        from pennylane_lightning.lightning_gpu_ops.algorithmsMPI import (
            AdjointJacobianMPIC64,
            AdjointJacobianMPIC128,
            create_ops_listMPIC64,
            create_ops_listMPIC128,
        )

        MPI_SUPPORT = True
    except ImportError:
        MPI_SUPPORT = False

    if find_library("custatevec") is None and not imp_util.find_spec(
        "cuquantum"
    ):  # pragma: no cover
        raise ImportError(
            "custatevec libraries not found. Please pip install the appropriate custatevec library in a virtual environment."
        )
    if not DevPool.getTotalDevices():  # pragma: no cover
        raise ValueError("No supported CUDA-capable device found")

    if not is_gpu_supported():  # pragma: no cover
        raise ValueError(f"CUDA device is an unsupported version: {get_gpu_arch()}")

    LGPU_CPP_BINARY_AVAILABLE = True
except (ImportError, ValueError) as e:
    warn(str(e), UserWarning)
    backend_info = None
    LGPU_CPP_BINARY_AVAILABLE = False


def _gpu_dtype(dtype, mpi=False):
    if dtype not in [np.complex128, np.complex64]:  # pragma: no cover
        raise ValueError(f"Data type is not supported for state-vector computation: {dtype}")
    if mpi:
        return StateVectorMPIC128 if dtype == np.complex128 else StateVectorMPIC64
    return StateVectorC128 if dtype == np.complex128 else StateVectorC64


def _adj_dtype(use_csingle, mpi=False):
    if mpi:
        return AdjointJacobianMPIC64 if use_csingle else AdjointJacobianMPIC128
    return AdjointJacobianC64 if use_csingle else AdjointJacobianC128


def _mebibytesToBytes(mebibytes):
    return mebibytes * 1024 * 1024


allowed_operations = {
    "Identity",
    "BasisState",
    "QubitStateVector",
    "StatePrep",
    "QubitUnitary",
    "ControlledQubitUnitary",
    "MultiControlledX",
    "DiagonalQubitUnitary",
    "PauliX",
    "PauliY",
    "PauliZ",
    "MultiRZ",
    "GlobalPhase",
    "C(GlobalPhase)",
    "Hadamard",
    "S",
    "Adjoint(S)",
    "T",
    "Adjoint(T)",
    "SX",
    "Adjoint(SX)",
    "CNOT",
    "SWAP",
    "ISWAP",
    "PSWAP",
    "Adjoint(ISWAP)",
    "SISWAP",
    "Adjoint(SISWAP)",
    "SQISW",
    "CSWAP",
    "Toffoli",
    "CY",
    "CZ",
    "PhaseShift",
    "ControlledPhaseShift",
    "RX",
    "RY",
    "RZ",
    "Rot",
    "CRX",
    "CRY",
    "CRZ",
    "CRot",
    "IsingXX",
    "IsingYY",
    "IsingZZ",
    "IsingXY",
    "SingleExcitation",
    "SingleExcitationPlus",
    "SingleExcitationMinus",
    "DoubleExcitation",
    "DoubleExcitationPlus",
    "DoubleExcitationMinus",
    "QubitCarry",
    "QubitSum",
    "OrbitalRotation",
    "QFT",
    "ECR",
    "BlockEncode",
}

allowed_observables = {
    "PauliX",
    "PauliY",
    "PauliZ",
    "Hadamard",
    "SparseHamiltonian",
    "Hamiltonian",
    "LinearCombination",
    "Hermitian",
    "Identity",
    "Sum",
    "Prod",
    "SProd",
}

gate_cache_needs_hash = (
    qml.BlockEncode,
    qml.ControlledQubitUnitary,
    qml.DiagonalQubitUnitary,
    qml.MultiControlledX,
    qml.OrbitalRotation,
    qml.PSWAP,
    qml.QubitUnitary,
)


class LightningGPU(LightningBase):  # pylint: disable=too-many-instance-attributes
    """PennyLane Lightning GPU device.

    A GPU-backed Lightning device using NVIDIA cuQuantum SDK.

    Use of this device requires pre-built binaries or compilation from source. Check out the
    :doc:`/lightning_gpu/installation` guide for more details.

    Args:
        wires (int): the number of wires to initialize the device with
        mpi (bool): enable MPI support. MPI support will be enabled if ``mpi`` is set as``True``.
        mpi_buf_size (int): size of GPU memory (in MiB) set for MPI operation and its default value is 64 MiB.
        sync (bool): immediately sync with host-sv after applying operations
        c_dtype: Datatypes for statevector representation. Must be one of ``np.complex64`` or ``np.complex128``.
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically.
        batch_obs (Union[bool, int]): determine whether to use multiple GPUs within the same node or not
    """

    name = "Lightning GPU PennyLane plugin"
    short_name = "lightning.gpu"

    operations = allowed_operations
    observables = allowed_observables
    _backend_info = backend_info
    config = Path(__file__).parent / "lightning_gpu.toml"
    _CPP_BINARY_AVAILABLE = LGPU_CPP_BINARY_AVAILABLE

    def __init__(
        self,
        wires,
        *,
        mpi: bool = False,
        mpi_buf_size: int = 0,
        sync=False,
        c_dtype=np.complex128,
        shots=None,
        batch_obs: Union[bool, int] = False,
    ):  # pylint: disable=too-many-arguments
        if c_dtype is np.complex64:
            self.use_csingle = True
        elif c_dtype is np.complex128:
            self.use_csingle = False
        else:
            raise TypeError(f"Unsupported complex type: {c_dtype}")

        super().__init__(wires, shots=shots, c_dtype=c_dtype)

        self._dp = DevPool()

        if not mpi:
            self._mpi = False
            self._num_local_wires = self.num_wires
            self._gpu_state = _gpu_dtype(c_dtype)(self._num_local_wires)
        else:
            self._mpi = True
            self._mpi_init_helper(self.num_wires)

            if mpi_buf_size < 0:
                raise TypeError(f"Unsupported mpi_buf_size value: {mpi_buf_size}")

            if mpi_buf_size:
                if mpi_buf_size & (mpi_buf_size - 1):
                    raise TypeError(
                        f"Unsupported mpi_buf_size value: {mpi_buf_size}. mpi_buf_size should be power of 2."
                    )
                # Memory size in bytes
                sv_memsize = np.dtype(c_dtype).itemsize * (1 << self._num_local_wires)
                if _mebibytesToBytes(mpi_buf_size) > sv_memsize:
                    w_msg = "The MPI buffer size is larger than the local state vector size."
                    warn(
                        w_msg,
                        RuntimeWarning,
                    )

            self._gpu_state = _gpu_dtype(c_dtype, mpi)(
                self._mpi_manager,
                self._devtag,
                mpi_buf_size,
                self._num_global_wires,
                self._num_local_wires,
            )

        self._sync = sync
        self._batch_obs = batch_obs
        self._create_basis_state(0)

    def _mpi_init_helper(self, num_wires):
        """Set up MPI checks."""
        if not MPI_SUPPORT:
            raise ImportError("MPI related APIs are not found.")
        # initialize MPIManager and config check in the MPIManager ctor
        self._mpi_manager = MPIManager()
        # check if number of GPUs per node is larger than
        # number of processes per node
        numDevices = self._dp.getTotalDevices()
        numProcsNode = self._mpi_manager.getSizeNode()
        if numDevices < numProcsNode:
            raise ValueError(
                "Number of devices should be larger than or equal to the number of processes on each node."
            )
        # check if the process number is larger than number of statevector elements
        if self._mpi_manager.getSize() > (1 << (num_wires - 1)):
            raise ValueError(
                "Number of processes should be smaller than the number of statevector elements."
            )
        # set the number of global and local wires
        commSize = self._mpi_manager.getSize()
        self._num_global_wires = commSize.bit_length() - 1
        self._num_local_wires = num_wires - self._num_global_wires
        # set GPU device
        rank = self._mpi_manager.getRank()
        deviceid = rank % numProcsNode
        self._dp.setDeviceID(deviceid)
        self._devtag = DevTag(deviceid)

    @staticmethod
    def _asarray(arr, dtype=None):
        arr = np.asarray(arr)  # arr is not copied

        if arr.dtype.kind not in ["f", "c"]:
            return arr

        if not dtype:
            dtype = arr.dtype

        return arr

    # pylint disable=missing-function-docstring
    def reset(self):
        """Reset the device"""
        super().reset()
        # init the state vector to |00..0>
        self._gpu_state.resetGPU(False)  # Sync reset

    @property
    def state(self):
        # pylint disable=missing-function-docstring
        """Copy the state vector data from the device to the host.

        A state vector Numpy array is explicitly allocated on the host to store and return the data.

        **Example**

        >>> dev = qml.device('lightning.gpu', wires=1)
        >>> dev.apply([qml.PauliX(wires=[0])])
        >>> print(dev.state)
        [0.+0.j 1.+0.j]
        """
        state = np.zeros(1 << self._num_local_wires, dtype=self.C_DTYPE)
        state = self._asarray(state, dtype=self.C_DTYPE)
        self.syncD2H(state)
        return state

    @property
    def create_ops_list(self):
        """Returns create_ops_list function of the matching precision."""
        if self._mpi:
            return create_ops_listMPIC64 if self.use_csingle else create_ops_listMPIC128
        return create_ops_listC64 if self.use_csingle else create_ops_listC128

    @property
    def measurements(self):
        """Returns Measurements constructor of the matching precision."""
        if self._mpi:
            return (
                MeasurementsMPIC64(self._gpu_state)
                if self.use_csingle
                else MeasurementsMPIC128(self._gpu_state)
            )
        return (
            MeasurementsC64(self._gpu_state)
            if self.use_csingle
            else MeasurementsC128(self._gpu_state)
        )

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
        self._gpu_state.DeviceToHost(state_vector.ravel(order="C"), use_async)

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
        self._gpu_state.HostToDevice(state_vector.ravel(order="C"), use_async)

    def _create_basis_state(self, index, use_async=False):
        """Return a computational basis state over all wires.
        Args:
            index (int): integer representing the computational basis state.
            use_async(bool): indicates whether to use asynchronous memory copy from host to device or not.
            Note: This function only supports synchronized memory copy.
        """
        self._gpu_state.setBasisState(index, use_async)

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
        device_wires = self.map_wires(device_wires)

        state = self._asarray(state, dtype=self.C_DTYPE)  # this operation on host
        output_shape = [2] * self._num_local_wires

        if len(device_wires) == self.num_wires and Wires(sorted(device_wires)) == device_wires:
            # Initialize the entire device state with the input state
            if self.num_wires == self._num_local_wires:
                self.syncH2D(self._reshape(state, output_shape))
                return
            local_state = np.zeros(1 << self._num_local_wires, dtype=self.C_DTYPE)
            self._mpi_manager.Scatter(state, local_state, 0)
            # Initialize the entire device state with the input state
            self.syncH2D(self._reshape(local_state, output_shape))
            return

        # generate basis states on subset of qubits via the cartesian product
        basis_states = np.array(list(product([0, 1], repeat=len(device_wires))))

        # get basis states to alter on full set of qubits
        unravelled_indices = np.zeros((2 ** len(device_wires), self.num_wires), dtype=int)
        unravelled_indices[:, device_wires] = basis_states

        # get indices for which the state is changed to input state vector elements
        ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self.num_wires)

        # set the state vector on GPU with the unravelled_indices and their corresponding values
        self._gpu_state.setStateVector(
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
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        # length of basis state parameter
        n_basis_state = len(state)
        state = state.tolist() if hasattr(state, "tolist") else state
        if not set(state).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

        if n_basis_state != len(device_wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        # get computational basis state number
        basis_states = 2 ** (self.num_wires - 1 - np.array(device_wires))
        basis_states = qml.math.convert_like(basis_states, state)
        num = int(qml.math.dot(state, basis_states))

        self._create_basis_state(num)

    def apply_lightning(self, operations):
        """Apply a list of operations to the state tensor.

        Args:
            operations (list[~pennylane.operation.Operation]): operations to apply
            dtype (type): Type of numpy ``complex`` to be used. Can be important
            to specify for large systems for memory allocation purposes.

        Returns:
            array[complex]: the output state tensor
        """
        # Skip over identity operations instead of performing
        # matrix multiplication with the identity.
        for ops in operations:
            if isinstance(ops, qml.Identity):
                continue
            if isinstance(ops, Adjoint):
                name = ops.base.name
                invert_param = True
            else:
                name = ops.name
                invert_param = False
            method = getattr(self._gpu_state, name, None)
            wires = self.wires.indices(ops.wires)

            if isinstance(ops, qml.ops.op_math.Controlled) and isinstance(
                ops.base, qml.GlobalPhase
            ):
                controls = ops.control_wires
                control_values = ops.control_values
                param = ops.base.parameters[0]
                matrix = global_phase_diagonal(param, self.wires, controls, control_values)
                self._gpu_state.apply(name, wires, False, [], matrix)
            elif method is None:
                # Inverse can be set to False since qml.matrix(ops) is already in inverted form
                try:
                    mat = qml.matrix(ops)
                except AttributeError:  # pragma: no cover
                    # To support older versions of PL
                    mat = ops.matrix
                r_dtype = np.float32 if self.use_csingle else np.float64
                param = [[r_dtype(ops.hash)]] if isinstance(ops, gate_cache_needs_hash) else []
                if len(mat) == 0:
                    raise ValueError("Unsupported operation")
                self._gpu_state.apply(
                    name,
                    wires,
                    False,
                    param,
                    mat.ravel(order="C"),  # inv = False: Matrix already in correct form;
                )  # Parameters can be ignored for explicit matrices; F-order for cuQuantum

            else:
                param = ops.parameters
                method(wires, invert_param, param)

    # pylint: disable=unused-argument
    def apply(self, operations, rotations=None, **kwargs):
        """Applies a list of operations to the state tensor."""
        # State preparation is currently done in Python
        if operations:  # make sure operations[0] exists
            if isinstance(operations[0], StatePrep):
                self._apply_state_vector(operations[0].parameters[0].copy(), operations[0].wires)
                operations = operations[1:]
            elif isinstance(operations[0], BasisState):
                self._apply_basis_state(operations[0].parameters[0], operations[0].wires)
                operations = operations[1:]

        for operation in operations:
            if isinstance(operation, (StatePrep, BasisState)):
                raise DeviceError(
                    f"Operation {operation.name} cannot be used after other "
                    + f"Operations have already been applied on a {self.short_name} device."
                )

        self.apply_lightning(operations)

    @staticmethod
    def _check_adjdiff_supported_operations(operations):
        """Check Lightning adjoint differentiation method support for a tape.

        Raise ``QuantumFunctionError`` if ``tape`` contains not supported measurements,
        observables, or operations by the Lightning adjoint differentiation method.

        Args:
            tape (.QuantumTape): quantum tape to differentiate.
        """
        for op in operations:
            if op.num_params > 1 and not isinstance(op, Rot):
                raise QuantumFunctionError(
                    f"The {op.name} operation is not supported using "
                    'the "adjoint" differentiation method'
                )

    def _init_process_jacobian_tape(self, tape, starting_state, use_device_state):
        """Generate an initial state vector for ``_process_jacobian_tape``."""
        if starting_state is not None:
            if starting_state.size != 2 ** len(self.wires):
                raise QuantumFunctionError(
                    "The number of qubits of starting_state must be the same as "
                    "that of the device."
                )
            self._apply_state_vector(starting_state, self.wires)
        elif not use_device_state:
            self.reset()
            self.apply(tape.operations)
        return self._gpu_state

    # pylint: disable=too-many-branches
    def adjoint_jacobian(self, tape, starting_state=None, use_device_state=False):
        """Implements the adjoint method outlined in
        `Jones and Gacon <https://arxiv.org/abs/2009.02823>`__ to differentiate an input tape.

        After a forward pass, the circuit is reversed by iteratively applying adjoint
        gates to scan backwards through the circuit.
        """
        if self.shots is not None:
            warn(
                "Requested adjoint differentiation to be computed with finite shots."
                " The derivative is always exact when using the adjoint differentiation method.",
                UserWarning,
            )

        tape_return_type = self._check_adjdiff_supported_measurements(tape.measurements)

        if not tape_return_type:  # the tape does not have measurements
            return np.array([], dtype=self.state.dtype)

        if tape_return_type is State:  # pragma: no cover
            raise QuantumFunctionError(
                "Adjoint differentiation method does not support measurement StateMP."
                "Use vjp method instead for this purpose."
            )

        # Check adjoint diff support
        self._check_adjdiff_supported_operations(tape.operations)

        processed_data = self._process_jacobian_tape(
            tape, starting_state, use_device_state, self._mpi, self._batch_obs
        )

        if not processed_data:  # training_params is empty
            return np.array([], dtype=self.state.dtype)

        trainable_params = processed_data["tp_shift"]
        # pylint: disable=pointless-string-statement
        """
        This path enables controlled batching over the requested observables, be they explicit, or part of a Hamiltonian.
        The traditional path will assume there exists enough free memory to preallocate all arrays and run through each observable iteratively.
        However, for larger system, this becomes impossible, and we hit memory issues very quickly. the batching support here enables several functionalities:
        - Pre-allocate memory for all observables on the primary GPU (`batch_obs=False`, default behaviour): This is the simplest path, and works best for few observables, and moderate qubit sizes. All memory is preallocated for each observable, and run through iteratively on a single GPU.
        - Evenly distribute the observables over all available GPUs (`batch_obs=True`): This will evenly split the data into ceil(num_obs/num_gpus) chunks, and allocate enough space on each GPU up-front before running through them concurrently. This relies on C++ threads to handle the orchestration.
        - Allocate at most `n` observables per GPU (`batch_obs=n`): Providing an integer value restricts each available GPU to at most `n` copies of the statevector, and hence `n` given observables for a given batch. This will iterate over the data in chnuks of size `n*num_gpus`.
        """
        adjoint_jacobian = _adj_dtype(self.use_csingle, self._mpi)()

        if self._batch_obs:  # Batching of Measurements
            if not self._mpi:  # Single-node path, controlled batching over available GPUs
                num_obs = len(processed_data["obs_serialized"])
                batch_size = (
                    num_obs
                    if isinstance(self._batch_obs, bool)
                    else self._batch_obs * self._dp.getTotalDevices()
                )
                jac = []
                for chunk in range(0, num_obs, batch_size):
                    obs_chunk = processed_data["obs_serialized"][chunk : chunk + batch_size]
                    jac_chunk = adjoint_jacobian.batched(
                        self._gpu_state,
                        obs_chunk,
                        processed_data["ops_serialized"],
                        trainable_params,
                    )
                    jac.extend(jac_chunk)
            else:  # MPI path, restrict memory per known GPUs
                jac = adjoint_jacobian.batched(
                    self._gpu_state,
                    processed_data["obs_serialized"],
                    processed_data["ops_serialized"],
                    trainable_params,
                )

        else:
            jac = adjoint_jacobian(
                self._gpu_state,
                processed_data["obs_serialized"],
                processed_data["ops_serialized"],
                trainable_params,
            )

        jac = np.array(jac)  # only for parameters differentiable with the adjoint method
        jac = jac.reshape(-1, len(trainable_params))
        jac_r = np.zeros((len(tape.observables), processed_data["all_params"]))
        if not self._batch_obs:
            jac_r[:, processed_data["record_tp_rows"]] = jac
        else:
            # Reduce over decomposed expval(H), if required.
            for idx in range(len(processed_data["obs_idx_offsets"][0:-1])):
                if (
                    processed_data["obs_idx_offsets"][idx + 1]
                    - processed_data["obs_idx_offsets"][idx]
                ) > 1:
                    jac_r[idx, :] = np.sum(
                        jac[
                            processed_data["obs_idx_offsets"][idx] : processed_data[
                                "obs_idx_offsets"
                            ][idx + 1],
                            :,
                        ],
                        axis=0,
                    )
                else:
                    jac_r[idx, :] = jac[
                        processed_data["obs_idx_offsets"][idx] : processed_data["obs_idx_offsets"][
                            idx + 1
                        ],
                        :,
                    ]

        return self._adjoint_jacobian_processing(jac_r)

    # pylint: disable=inconsistent-return-statements, line-too-long, missing-function-docstring
    def vjp(self, measurements, grad_vec, starting_state=None, use_device_state=False):
        """Generate the processing function required to compute the vector-Jacobian products
        of a tape.

        This function can be used with multiple expectation values or a quantum state.
        When a quantum state is given,

        .. code-block:: python

            vjp_f = dev.vjp([qml.state()], grad_vec)
            vjp = vjp_f(tape)

        computes :math:`w = (w_1,\\cdots,w_m)` where

        .. math::

            w_k = \\langle v| \\frac{\\partial}{\\partial \\theta_k} | \\psi_{\\pmb{\\theta}} \\rangle.

        Here, :math:`m` is the total number of trainable parameters,
        :math:`\\pmb{\\theta}` is the vector of trainable parameters and
        :math:`\\psi_{\\pmb{\\theta}}` is the output quantum state.

        Args:
            measurements (list): List of measurement processes for vector-Jacobian product.
                Now it must be expectation values or a quantum state.
            grad_vec (tensor_like): Gradient-output vector. Must have shape matching the output
                shape of the corresponding tape, i.e. number of measurements if the return
                type is expectation or :math:`2^N` if the return type is statevector
            starting_state (tensor_like): post-forward pass state to start execution with.
                It should be complex-valued. Takes precedence over ``use_device_state``.
            use_device_state (bool): use current device state to initialize.
                A forward pass of the same circuit should be the last thing the device
                has executed. If a ``starting_state`` is provided, that takes precedence.

        Returns:
            The processing function required to compute the vector-Jacobian products of a tape.
        """
        if self.shots is not None:
            warn(
                "Requested adjoint differentiation to be computed with finite shots."
                " The derivative is always exact when using the adjoint differentiation method.",
                UserWarning,
            )

        tape_return_type = self._check_adjdiff_supported_measurements(measurements)

        if math.allclose(grad_vec, 0) or tape_return_type is None:
            return lambda tape: math.convert_like(np.zeros(len(tape.trainable_params)), grad_vec)

        if tape_return_type is Expectation:
            if len(grad_vec) != len(measurements):
                raise ValueError(
                    "Number of observables in the tape must be the same as the length of grad_vec in the vjp method"
                )

            if np.iscomplexobj(grad_vec):
                raise ValueError(
                    "The vjp method only works with a real-valued grad_vec when the tape is returning an expectation value"
                )

            ham = qml.Hamiltonian(grad_vec, [m.obs for m in measurements])

            # pylint: disable=protected-access
            def processing_fn(tape):
                nonlocal ham
                num_params = len(tape.trainable_params)

                if num_params == 0:
                    return np.array([], dtype=self.state.dtype)

                new_tape = tape.copy()
                new_tape._measurements = [qml.expval(ham)]

                return self.adjoint_jacobian(new_tape, starting_state, use_device_state)

            return processing_fn

    # pylint: disable=attribute-defined-outside-init
    def sample(self, observable, shot_range=None, bin_size=None, counts=False):
        """Return samples of an observable."""
        diagonalizing_gates = observable.diagonalizing_gates()
        if diagonalizing_gates:
            self.apply(diagonalizing_gates)
        if not isinstance(observable, qml.PauliZ):
            self._samples = self.generate_samples()
        results = super().sample(
            observable, shot_range=shot_range, bin_size=bin_size, counts=counts
        )
        if diagonalizing_gates:
            self.apply([qml.adjoint(g, lazy=False) for g in reversed(diagonalizing_gates)])
        return results

    def generate_samples(self):
        """Generate samples

        Returns:
            array[int]: array of samples in binary representation with shape
            ``(dev.shots, dev.num_wires)``
        """
        return self.measurements.generate_samples(len(self.wires), self.shots).astype(
            int, copy=False
        )

    # pylint: disable=protected-access
    def expval(self, observable, shot_range=None, bin_size=None):
        """Expectation value of the supplied observable.

        Args:
            observable: A PennyLane observable.
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.

        Returns:
            Expectation value of the observable
        """
        if self.shots is not None:
            # estimate the expectation value
            samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
            return np.squeeze(np.mean(samples, axis=0))

        if isinstance(observable, qml.SparseHamiltonian):
            if self._mpi:
                # Identity for CSR_SparseHamiltonian to pass to processes with rank != 0 to reduce
                # host(cpu) memory requirements
                obs = qml.Identity(0)
                Hmat = qml.Hamiltonian([1.0], [obs]).sparse_matrix()
                H_sparse = qml.SparseHamiltonian(Hmat, wires=range(1))
                CSR_SparseHamiltonian = H_sparse.sparse_matrix().tocsr()
                # CSR_SparseHamiltonian for rank == 0
                if self._mpi_manager.getRank() == 0:
                    CSR_SparseHamiltonian = observable.sparse_matrix().tocsr()
            else:
                CSR_SparseHamiltonian = observable.sparse_matrix().tocsr()

            return self.measurements.expval(
                CSR_SparseHamiltonian.indptr,
                CSR_SparseHamiltonian.indices,
                CSR_SparseHamiltonian.data,
            )

        # use specialized functors to compute expval(Hermitian)
        if isinstance(observable, qml.Hermitian):
            observable_wires = self.map_wires(observable.wires)
            if self._mpi and len(observable_wires) > self._num_local_wires:
                raise RuntimeError(
                    "MPI backend does not support Hermitian with number of target wires larger than local wire number."
                )
            matrix = observable.matrix()
            return self.measurements.expval(matrix, observable_wires)

        if (
            isinstance(observable, qml.ops.Hamiltonian)
            or (observable.arithmetic_depth > 0)
            or isinstance(observable.name, List)
        ):
            ob_serialized = QuantumScriptSerializer(
                self.short_name, self.use_csingle, self._mpi
            )._ob(observable, self.wire_map)
            return self.measurements.expval(ob_serialized)

        # translate to wire labels used by device
        observable_wires = self.map_wires(observable.wires)

        return self.measurements.expval(observable.name, observable_wires)

    def probability_lightning(self, wires=None):
        """Return the probability of each computational basis state.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided are traced out of the system.

        Returns:
            array[float]: list of the probabilities
        """
        # translate to wire labels used by device
        observable_wires = self.map_wires(wires)
        # Device returns as col-major orderings, so perform transpose on data for bit-index shuffle for now.
        local_prob = self.measurements.probs(observable_wires)
        if len(local_prob) > 0:
            num_local_wires = len(local_prob).bit_length() - 1 if len(local_prob) > 0 else 0
            return local_prob.reshape([2] * num_local_wires).transpose().reshape(-1)
        return local_prob

    def var(self, observable, shot_range=None, bin_size=None):
        """Variance of the supplied observable.

        Args:
            observable: A PennyLane observable.
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.

        Returns:
            Variance of the observable
        """
        if self.shots is not None:
            # estimate the var
            # Lightning doesn't support sampling yet
            samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
            return np.squeeze(np.var(samples, axis=0))

        if isinstance(observable, qml.SparseHamiltonian):
            csr_hamiltonian = observable.sparse_matrix(wire_order=self.wires).tocsr(copy=False)
            return self.measurements.var(
                csr_hamiltonian.indptr,
                csr_hamiltonian.indices,
                csr_hamiltonian.data,
            )

        if (
            isinstance(observable, (qml.Hermitian, qml.ops.Hamiltonian))
            or (observable.arithmetic_depth > 0)
            or isinstance(observable.name, List)
        ):
            ob_serialized = QuantumScriptSerializer(
                self.short_name, self.use_csingle, self._mpi
            )._ob(observable, self.wire_map)
            return self.measurements.var(ob_serialized)

        # translate to wire labels used by device
        observable_wires = self.map_wires(observable.wires)

        return self.measurements.var(observable.name, observable_wires)
