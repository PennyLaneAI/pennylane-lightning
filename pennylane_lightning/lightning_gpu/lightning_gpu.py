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
from warnings import warn
import numpy as np

from pennylane_lightning.core.lightning_base import (
    LightningBase,
    LightningBaseFallBack,
)

try:
    from pennylane_lightning.lightning_gpu_ops import (
        allocate_aligned_array,
        backend_info,
        best_alignment,
        get_alignment,
        StateVectorC128,
        StateVectorC64,
        MeasurementsC128,
        MeasurementsC64,
        is_gpu_supported,
        get_gpu_arch,
        DevPool,
    )

    from ctypes.util import find_library
    from importlib import util as imp_util

    if find_library("custatevec") == None and not imp_util.find_spec("cuquantum"):
        raise ImportError(
            'cuQuantum libraries not found. Please check your "LD_LIBRARY_PATH" environment variable,'
            'or ensure you have installed the appropriate distributable "cuQuantum" package.'
        )
    if not DevPool.getTotalDevices():
        raise ValueError(f"No supported CUDA-capable device found")
    if not is_gpu_supported():
        raise ValueError(f"CUDA device is an unsupported version: {get_gpu_arch()}")

    LGPU_CPP_BINARY_AVAILABLE = True
# except (ModuleNotFoundError, ImportError, ValueError) as e:
except (ImportError, ValueError) as e:
    warn(str(e), UserWarning)
    LGPU_CPP_BINARY_AVAILABLE = False

if LGPU_CPP_BINARY_AVAILABLE:
    from typing import List, Union
    from itertools import product

    from pennylane import (
        math,
        BasisState,
        StatePrep,
        DeviceError,
        Projector,
        Rot,
        QuantumFunctionError,
    )
    from pennylane.operation import Tensor
    from pennylane.ops.op_math import Adjoint
    from pennylane.measurements import Expectation, MeasurementProcess, State
    from pennylane.wires import Wires

    import pennylane as qml

    from pennylane_lightning.core._serialize import QuantumScriptSerializer
    from pennylane_lightning.core._version import __version__
    from pennylane_lightning.lightning_gpu_ops.algorithms import (
        AdjointJacobianC64,
        create_ops_listC64,
        AdjointJacobianC128,
        create_ops_listC128,
    )

    def _gpu_dtype(dtype):
        if dtype not in [np.complex128, np.complex64]:
            raise ValueError(f"Data type is not supported for state-vector computation: {dtype}")
        return StateVectorC128 if dtype == np.complex128 else StateVectorC64

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
        "CPhase",
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
    }

    allowed_observables = {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "SparseHamiltonian",
        "Hamiltonian",
        "Hermitian",
        "Identity",
        "Sum",
        "Prod",
        "SProd",
    }

    class LightningGPU(LightningBase):
        """PennyLane-Lightning-GPU device.
        Args:
            wires (int): the number of wires to initialize the device with
            sync (bool): immediately sync with host-sv after applying operations
            c_dtype: Datatypes for statevector representation. Must be one of ``np.complex64`` or ``np.complex128``.
            shots (int): How many times the circuit should be evaluated (or sampled) to estimate
                the expectation values. Defaults to ``None`` if not specified. Setting
                to ``None`` results in computing statistics like expectation values and
                variances analytically.
            batch_obs (Union[bool, int]): determine whether to use multiple GPUs within the same node or not
        """

        name = "PennyLane plugin for GPU-backed Lightning device using NVIDIA cuQuantum SDK"
        short_name = "lightning.gpu"

        operations = allowed_operations
        observables = allowed_observables
        _backend_info = backend_info

        def __init__(
            self,
            wires,
            *,
            sync=False,
            c_dtype=np.complex128,
            shots=None,
            batch_obs: Union[bool, int] = False,
        ):
            if c_dtype is np.complex64:
                r_dtype = np.float32
                self.use_csingle = True
            elif c_dtype is np.complex128:
                r_dtype = np.float64
                self.use_csingle = False
            else:
                raise TypeError(f"Unsupported complex Type: {c_dtype}")

            super().__init__(wires, shots=shots, c_dtype=c_dtype)

            self._dp = DevPool()
            self._sync = sync
            self._batch_obs = batch_obs

            self._num_local_wires = self.num_wires
            self._gpu_state = _gpu_dtype(c_dtype)(self._num_local_wires)

            self._create_basis_state(0)

        @staticmethod
        def _asarray(arr, dtype=None):
            arr = np.asarray(arr)  # arr is not copied

            if arr.dtype.kind not in ["f", "c"]:
                return arr

            if not dtype:
                dtype = arr.dtype

            # We allocate a new aligned memory and copy data to there if alignment
            # or dtype mismatches
            # Note that get_alignment does not necessarily return CPUMemoryModel(Unaligned) even for
            # numpy allocated memory as the memory location happens to be aligned.
            if int(get_alignment(arr)) < int(best_alignment()) or arr.dtype != dtype:
                new_arr = allocate_aligned_array(arr.size, np.dtype(dtype)).reshape(arr.shape)
                np.copyto(new_arr, arr)
                arr = new_arr
            return arr

        def reset(self):
            super().reset()
            # init the state vector to |00..0>
            self._gpu_state.resetGPU(False)  # Sync reset

        @property
        def state(self):
            """Copy the state vector data from the device to the host. A state vector Numpy array is explicitly allocated on the host to store and return the data.
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
            return create_ops_listC64 if self.use_csingle else create_ops_listC128

        @property
        def measurements(self):
            """Returns Measurements constructor of the matching precision."""
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
            dim = 2 ** len(device_wires)

            state = self._asarray(state, dtype=self.C_DTYPE)  # this operation on host
            batch_size = self._get_batch_size(state, (dim,), dim)  # this operation on host
            output_shape = [2] * self._num_local_wires

            if batch_size is not None:
                output_shape.insert(0, batch_size)

            if len(device_wires) == self.num_wires and Wires(sorted(device_wires)) == device_wires:
                # Initialize the entire device state with the input state
                if self.num_wires == self._num_local_wires:
                    self.syncH2D(self._reshape(state, output_shape))
                    return
                local_state = np.zeros(1 << self._num_local_wires, dtype=self.C_DTYPE)
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

        def apply_cq(self, operations):
            # Skip over identity operations instead of performing
            # matrix multiplication with the identity.
            skipped_ops = ["Identity"]
            invert_param = False

            for o in operations:
                if str(o.name) in skipped_ops:
                    continue
                name = o.name
                if isinstance(o, Adjoint):
                    name = o.base.name
                    invert_param = True
                method = getattr(self._gpu_state, name, None)
                wires = self.wires.indices(o.wires)

                if method is None:
                    # Inverse can be set to False since qml.matrix(o) is already in inverted form
                    try:
                        mat = qml.matrix(o)
                    except AttributeError:  # pragma: no cover
                        # To support older versions of PL
                        mat = o.matrix

                    if len(mat) == 0:
                        raise Exception("Unsupported operation")
                    self._gpu_state.apply(
                        name,
                        wires,
                        False,
                        [],
                        mat.ravel(order="C"),  # inv = False: Matrix already in correct form;
                    )  # Parameters can be ignored for explicit matrices; F-order for cuQuantum

                else:
                    param = o.parameters
                    method(wires, invert_param, param)

        def apply(self, operations, rotations=None, **kwargs):
            # State preparation is currently done in Python
            if operations:  # make sure operations[0] exists
                if isinstance(operations[0], StatePrep):
                    self._apply_state_vector(
                        operations[0].parameters[0].copy(), operations[0].wires
                    )
                    operations = operations[1:]
                elif isinstance(operations[0], BasisState):
                    self._apply_basis_state(operations[0].parameters[0], operations[0].wires)
                    operations = operations[1:]

            for operation in operations:
                if isinstance(operation, (StatePrep, BasisState)):
                    raise DeviceError(
                        "Operation {} cannot be used after other Operations have already been "
                        "applied on a {} device.".format(operation.name, self.short_name)
                    )

            self.apply_cq(operations)

        @staticmethod
        def _check_adjdiff_supported_measurements(measurements: List[MeasurementProcess]):
            """Check whether given list of measurement is supported by adjoint_diff.
            Args:
                measurements (List[MeasurementProcess]): a list of measurement processes to check.
            Returns:
                Expectation or State: a common return type of measurements.
            """
            if len(measurements) == 0:
                return None

            if len(measurements) == 1 and measurements[0].return_type is State:
                # return State
                raise QuantumFunctionError(
                    "Adjoint differentiation does not support State measurements."
                )

            # The return_type of measurement processes must be expectation
            if not all([m.return_type is Expectation for m in measurements]):
                raise QuantumFunctionError(
                    "Adjoint differentiation method does not support expectation return type "
                    "mixed with other return types"
                )

            for m in measurements:
                if not isinstance(m.obs, Tensor):
                    if isinstance(m.obs, Projector):
                        raise QuantumFunctionError(
                            "Adjoint differentiation method does not support the Projector observable"
                        )
                else:
                    if any([isinstance(o, Projector) for o in m.obs.non_identity_obs]):
                        raise QuantumFunctionError(
                            "Adjoint differentiation method does not support the Projector observable"
                        )
            return Expectation

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

        def adjoint_jacobian(self, tape, starting_state=None, use_device_state=False, **kwargs):
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
                    "This method does not support statevector return type. "
                    "Use vjp method instead for this purpose."
                )

            # Check adjoint diff support
            self._check_adjdiff_supported_operations(tape.operations)

            processed_data = self._process_jacobian_tape(tape, starting_state, use_device_state)

            if not processed_data:  # training_params is empty
                return np.array([], dtype=self.state.dtype)

            trainable_params = processed_data["tp_shift"]
            """
            This path enables controlled batching over the requested observables, be they explicit, or part of a Hamiltonian.
            The traditional path will assume there exists enough free memory to preallocate all arrays and run through each observable iteratively.
            However, for larger system, this becomes impossible, and we hit memory issues very quickly. the batching support here enables several functionalities:
            - Pre-allocate memory for all observables on the primary GPU (`batch_obs=False`, default behaviour): This is the simplest path, and works best for few observables, and moderate qubit sizes. All memory is preallocated for each observable, and run through iteratively on a single GPU.
            - Evenly distribute the observables over all available GPUs (`batch_obs=True`): This will evenly split the data into ceil(num_obs/num_gpus) chunks, and allocate enough space on each GPU up-front before running through them concurrently. This relies on C++ threads to handle the orchestration.
            - Allocate at most `n` observables per GPU (`batch_obs=n`): Providing an integer value restricts each available GPU to at most `n` copies of the statevector, and hence `n` given observables for a given batch. This will iterate over the data in chnuks of size `n*num_gpus`.
            """
            adjoint_jacobian = AdjointJacobianC64() if self.use_csingle else AdjointJacobianC128()

            if self._batch_obs:
                adjoint_jacobian = adjoint_jacobian.batched

            jac = adjoint_jacobian(
                processed_data["state_vector"],
                processed_data["obs_serialized"],
                processed_data["ops_serialized"],
                trainable_params,
            )

            jac = np.array(jac)  # only for parameters differentiable with the adjoint method
            jac = jac.reshape(-1, len(trainable_params))
            jac_r = np.zeros((jac.shape[0], processed_data["all_params"]))

            jac_r[:, processed_data["record_tp_rows"]] = jac

            if hasattr(qml, "active_return"):
                return self._adjoint_jacobian_processing(jac_r) if qml.active_return() else jac_r
            return self._adjoint_jacobian_processing(jac_r)

        def vjp(self, measurements, dy, starting_state=None, use_device_state=False):
            """Generate the processing function required to compute the vector-Jacobian products of a tape."""
            if self.shots is not None:
                warn(
                    "Requested adjoint differentiation to be computed with finite shots."
                    " The derivative is always exact when using the adjoint differentiation method.",
                    UserWarning,
                )

            tape_return_type = self._check_adjdiff_supported_measurements(measurements)

            if math.allclose(dy, 0) or tape_return_type is None:
                return lambda tape: math.convert_like(np.zeros(len(tape.trainable_params)), dy)

            if tape_return_type is Expectation:
                if len(dy) != len(measurements):
                    raise ValueError(
                        "Number of observables in the tape must be the same as the length of dy in the vjp method"
                    )

                if np.iscomplexobj(dy):
                    raise ValueError(
                        "The vjp method only works with a real-valued grad_vec when the tape is returning an expectation value"
                    )

                ham = qml.Hamiltonian(dy, [m.obs for m in measurements])

                def processing_fn(tape):
                    nonlocal ham
                    num_params = len(tape.trainable_params)

                    if num_params == 0:
                        return np.array([], dtype=self.state.dtype)

                    new_tape = tape.copy()
                    new_tape._measurements = [qml.expval(ham)]

                    return self.adjoint_jacobian(new_tape, starting_state, use_device_state)

                return processing_fn

        def sample(self, observable, shot_range=None, bin_size=None, counts=False):
            if observable.name != "PauliZ":
                self.apply_cq(observable.diagonalizing_gates())
                self._samples = self.generate_samples()
            return super().sample(
                observable, shot_range=shot_range, bin_size=bin_size, counts=counts
            )

        def generate_samples(self):
            """Generate samples

            Returns:
                array[int]: array of samples in binary representation with shape
                ``(dev.shots, dev.num_wires)``
            """
            return self.measurements.generate_samples(len(self.wires), self.shots).astype(
                int, copy=False
            )

        def expval(self, observable, shot_range=None, bin_size=None):
            if observable.name in [
                "Projector",
            ]:
                return super().expval(observable, shot_range=shot_range, bin_size=bin_size)

            if self.shots is not None:
                # estimate the expectation value
                samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
                return np.squeeze(np.mean(samples, axis=0))

            if observable.name in ["SparseHamiltonian"]:
                CSR_SparseHamiltonian = observable.sparse_matrix().tocsr()
                return self.measurements.expval(
                    CSR_SparseHamiltonian.indptr,
                    CSR_SparseHamiltonian.indices,
                    CSR_SparseHamiltonian.data,
                )

            # use specialized functors to compute expval(Hermitian)
            if observable.name == "Hermitian":
                observable_wires = self.map_wires(observable.wires)
                matrix = observable.matrix()
                return self.measurements.expval(matrix, observable_wires)

            if (
                observable.name in ["Hermitian", "Hamiltonian"]
                or (observable.arithmetic_depth > 0)
                or isinstance(observable.name, List)
            ):
                ob_serialized = QuantumScriptSerializer(self.short_name, self.use_csingle)._ob(
                    observable, self.wire_map
                )
                return self.measurements.expval(ob_serialized)

            # translate to wire labels used by device
            observable_wires = self.map_wires(observable.wires)

            return self.measurements.expval(observable.name, observable_wires)

        def probability_lightning(self, wires=None, shot_range=None, bin_size=None):
            # translate to wire labels used by device
            observable_wires = self.map_wires(wires)
            # Device returns as col-major orderings, so perform transpose on data for bit-index shuffle for now.
            local_prob = self.measurements.probs(observable_wires)
            num_local_wires = len(local_prob).bit_length() - 1 if len(local_prob) > 0 else 0
            return local_prob.reshape([2] * num_local_wires).transpose().reshape(-1)

        def generate_samples(self):
            """Generate samples

            Returns:
                array[int]: array of samples in binary representation with shape ``(dev.shots, dev.num_wires)``
            """
            return self.measurements.generate_samples(len(self.wires), self.shots).astype(int)

        def var(self, observable, shot_range=None, bin_size=None):
            if self.shots is not None:
                # estimate the var
                # Lightning doesn't support sampling yet
                samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
                return np.squeeze(np.var(samples, axis=0))

            if observable.name == "SparseHamiltonian":
                csr_hamiltonian = observable.sparse_matrix(wire_order=self.wires).tocsr(copy=False)
                return self.measurements.var(
                    csr_hamiltonian.indptr,
                    csr_hamiltonian.indices,
                    csr_hamiltonian.data,
                )

            if (
                observable.name in ["Hamiltonian", "Hermitian"]
                or (observable.arithmetic_depth > 0)
                or isinstance(observable.name, List)
            ):
                ob_serialized = QuantumScriptSerializer(self.short_name, self.use_csingle)._ob(
                    observable, self.wire_map
                )
                return self.measurements.var(ob_serialized)

            # translate to wire labels used by device
            observable_wires = self.map_wires(observable.wires)

            return self.measurements.var(observable.name, observable_wires)

else:  # LGPU_CPP_BINARY_AVAILABLE:

    class LightningGPU(LightningBaseFallBack):
        name = "PennyLane plugin for GPU-backed Lightning device using NVIDIA cuQuantum SDK: [No binaries found - Fallback: default.qubit]"
        short_name = "lightning.gpu"

        def __init__(self, wires, *, c_dtype=np.complex128, **kwargs):
            w_msg = """
                "Pre-compiled binaries for lightning.gpu are not available. Falling back to "
                "using the Python-based default.qubit implementation. To manually compile from "
                "source, follow the instructions at "
                "https://pennylane-lightning.readthedocs.io/en/latest/installation.html.",
            """
            warn(
                w_msg,
                UserWarning,
            )
            super().__init__(wires, c_dtype=c_dtype, **kwargs)
