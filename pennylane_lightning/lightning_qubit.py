# Copyright 2021 Xanadu Quantum Technologies Inc.

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
This module contains the :class:`~.LightningQubit` class, a PennyLane simulator device that
interfaces with C++ for fast linear algebra calculations.
"""
from typing import List
from warnings import warn
from os import getenv
from itertools import islice, product

import numpy as np
from pennylane import (
    QubitDevice,
    math,
    BasisState,
    QubitStateVector,
    Projector,
    Rot,
    QuantumFunctionError,
    DeviceError,
)
from pennylane.devices import DefaultQubit
from pennylane.operation import Tensor, Operation
from pennylane.measurements import MeasurementProcess, Expectation, State
from pennylane.wires import Wires

# tolerance for numerical errors
tolerance = 1e-10

import pennylane as qml

from ._version import __version__

try:
    from .lightning_qubit_ops import (
        adjoint_diff,
        MeasuresC64,
        StateVectorC64,
        MeasuresC128,
        StateVectorC128,
        Kokkos_info,
        allocate_aligned_array,
        get_alignment,
        best_alignment,
    )

    from ._serialize import _serialize_ob, _serialize_observables, _serialize_ops

    CPP_BINARY_AVAILABLE = True
except ModuleNotFoundError:
    CPP_BINARY_AVAILABLE = False


def _chunk_iterable(it, num_chunks):
    "Lazy-evaluated chunking of given iterable from https://stackoverflow.com/a/22045226"
    it = iter(it)
    return iter(lambda: tuple(islice(it, num_chunks)), ())


allowed_operations = {
    "Identity",
    "BasisState",
    "QubitStateVector",
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
    "Hermitian",
    "Identity",
    "Projector",
    "SparseHamiltonian",
    "Hamiltonian",
    "Sum",
    "SProd",
    "Prod",
    "Exp",
}


class LightningQubit(QubitDevice):
    """PennyLane Lightning device.

    A device that interfaces with C++ to perform fast linear algebra calculations.

    Use of this device requires pre-built binaries or compilation from source. Check out the
    :doc:`/installation` guide for more details.

    Args:
        wires (int): the number of wires to initialize the device with
        c_dtype: Datatypes for statevector representation. Must be one of ``np.complex64`` or ``np.complex128``.
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically.
        batch_obs (bool): Determine whether we process observables parallelly when computing the
            jacobian. This value is only relevant when the lightning qubit is built with OpenMP.
    """

    name = "Lightning Qubit PennyLane plugin"
    short_name = "lightning.qubit"
    pennylane_requires = ">=0.26"
    version = __version__
    author = "Xanadu Inc."
    _CPP_BINARY_AVAILABLE = True
    operations = allowed_operations
    observables = allowed_observables

    def __init__(self, wires, *, c_dtype=np.complex128, shots=None, batch_obs=False, analytic=None):
        if c_dtype is np.complex64:
            r_dtype = np.float32
            self.use_csingle = True
        elif c_dtype is np.complex128:
            r_dtype = np.float64
            self.use_csingle = False
        else:
            raise TypeError(f"Unsupported complex Type: {c_dtype}")
        super().__init__(wires, shots=shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)
        self._batch_obs = batch_obs

        # Create the initial state. Internally, we store the
        # state as an array of dimension [2]*wires.
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

    @property
    def stopping_condition(self):
        """.BooleanFn: Returns the stopping condition for the device. The returned
        function accepts a queuable object (including a PennyLane operation
        and observable) and returns ``True`` if supported by the device."""

        def accepts_obj(obj):
            if obj.name == "QFT" and len(obj.wires) < 10:
                return True
            if obj.name == "GroverOperator" and len(obj.wires) < 13:
                return True
            return (not isinstance(obj, qml.tape.QuantumTape)) and getattr(
                self, "supports_operation", lambda name: False
            )(obj.name)

        return qml.BooleanFn(accepts_obj)

    def _create_basis_state(self, index):
        """Return a computational basis state over all wires.
        Args:
            index (int): integer representing the computational basis state
        Returns:
            array[complex]: complex array of shape ``[2]*self.num_wires``
            representing the statevector of the basis state
        Note: This function does not support broadcasted inputs yet.
        """
        state = np.zeros(2**self.num_wires, dtype=np.complex128)
        state[index] = 1
        state = self._asarray(state, dtype=self.C_DTYPE)
        return self._reshape(state, [2] * self.num_wires)

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qubit",
            supports_inverse_operations=True,
            supports_analytic_computation=True,
            supports_broadcasting=False,
            returns_state=True,
        )
        return capabilities

    @staticmethod
    def _asarray(arr, dtype=None):
        arr = np.asarray(arr)  # arr is not copied

        if arr.dtype.kind not in ["f", "c"]:
            return arr

        if not dtype:
            dtype = arr.dtype

        # We allocate a new aligned memory and copy data to there if alignment or dtype mismatches
        # Note that get_alignment does not necessarily return CPUMemoryModel(Unaligned) even for
        # numpy allocated memory as the memory location happens to be aligned.
        if int(get_alignment(arr)) < int(best_alignment()) or arr.dtype != dtype:
            new_arr = allocate_aligned_array(arr.size, np.dtype(dtype)).reshape(arr.shape)
            np.copyto(new_arr, arr)
            arr = new_arr
        return arr

    def reset(self):
        """Reset the device"""
        super().reset()

        # init the state vector to |00..0>
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

    @property
    def state(self):
        # Flattening the state.
        shape = (1 << self.num_wires,)
        return self._reshape(self._pre_rotated_state, shape)

    def _apply_state_vector(self, state, device_wires):
        """Initialize the internal state vector in a specified state.
        Args:
            state (array[complex]): normalized input state of length ``2**len(wires)``
                or broadcasted state of shape ``(batch_size, 2**len(wires))``
            device_wires (Wires): wires that get initialized in the state
        """

        # translate to wire labels used by device
        device_wires = self.map_wires(device_wires)

        state = self._asarray(state, dtype=self.C_DTYPE)
        output_shape = [2] * self.num_wires

        if not qml.math.is_abstract(state):
            norm = qml.math.linalg.norm(state, axis=-1, ord=2)
            if not qml.math.allclose(norm, 1.0, atol=tolerance):
                raise ValueError("Sum of amplitudes-squared does not equal one.")

        if len(device_wires) == self.num_wires and Wires(sorted(device_wires)) == device_wires:
            # Initialize the entire device state with the input state
            self._state = self._reshape(state, output_shape)
            return

        # generate basis states on subset of qubits via the cartesian product
        basis_states = np.array(list(product([0, 1], repeat=len(device_wires))))

        # get basis states to alter on full set of qubits
        unravelled_indices = np.zeros((2 ** len(device_wires), self.num_wires), dtype=int)
        unravelled_indices[:, device_wires] = basis_states

        # get indices for which the state is changed to input state vector elements
        ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self.num_wires)

        state = self._scatter(ravelled_indices, state, [2**self.num_wires])
        state = self._reshape(state, output_shape)
        self._state = self._asarray(state, dtype=self.C_DTYPE)

    def _apply_basis_state(self, state, wires):
        """Initialize the state vector in a specified computational basis state.

        Args:
            state (array[int]): computational basis state of shape ``(wires,)``
                consisting of 0s and 1s.
            wires (Wires): wires that the provided computational state should be initialized on

        Note: This function does not support broadcasted inputs yet.
        """
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        # length of basis state parameter
        n_basis_state = len(state)

        if not set(state.tolist()).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

        if n_basis_state != len(device_wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        # get computational basis state number
        basis_states = 2 ** (self.num_wires - 1 - np.array(device_wires))
        basis_states = qml.math.convert_like(basis_states, state)
        num = int(qml.math.dot(state, basis_states))

        self._state = self._create_basis_state(num)

    # To be able to validate the adjoint method [_validate_adjoint_method(device)],
    #  the qnode requires the definition of:
    # ["_apply_operation", "_apply_unitary", "adjoint_jacobian"]
    def _apply_operation():
        pass

    def _apply_unitary():
        pass

    def apply_lightning(self, state, operations):
        """Apply a list of operations to the state tensor.

        Args:
            state (array[complex]): the input state tensor
            operations (list[~pennylane.operation.Operation]): operations to apply
            dtype (type): Type of numpy ``complex`` to be used. Can be important
            to specify for large systems for memory allocation purposes.

        Returns:
            array[complex]: the output state tensor
        """
        state_vector = np.ravel(state)

        if self.use_csingle:
            # use_csingle
            sim = StateVectorC64(state_vector)
        else:
            # self.C_DTYPE is np.complex128 by default
            sim = StateVectorC128(state_vector)

        # Skip over identity operations instead of performing
        # matrix multiplication with the identity.
        skipped_ops = ["Identity"]

        for o in operations:
            if o.base_name in skipped_ops:
                continue
            name = o.name.split(".")[0]  # The split is because inverse gates have .inv appended
            method = getattr(sim, name, None)

            wires = self.wires.indices(o.wires)
            if method is None:
                # Inverse can be set to False since qml.matrix(o) is already in inverted form
                method = getattr(sim, "applyMatrix")
                try:
                    method(qml.matrix(o), wires, False)
                except AttributeError:  # pragma: no cover
                    # To support older versions of PL
                    method(o.matrix, wires, False)
            else:
                inv = o.inverse
                param = o.parameters
                method(wires, inv, param)

        return np.reshape(state_vector, state.shape)

    def apply(self, operations, rotations=None, **kwargs):
        # State preparation is currently done in Python
        if operations:  # make sure operations[0] exists
            if isinstance(operations[0], QubitStateVector):
                self._apply_state_vector(operations[0].parameters[0].copy(), operations[0].wires)
                del operations[0]
            elif isinstance(operations[0], BasisState):
                self._apply_basis_state(operations[0].parameters[0], operations[0].wires)
                del operations[0]

        for operation in operations:
            if isinstance(operation, (QubitStateVector, BasisState)):
                raise DeviceError(
                    "Operation {} cannot be used after other Operations have already been "
                    "applied on a {} device.".format(operation.name, self.short_name)
                )

        if operations:
            self._pre_rotated_state = self.apply_lightning(self._state, operations)
        else:
            self._pre_rotated_state = self._state

        if rotations:
            self._state = self.apply_lightning(np.copy(self._pre_rotated_state), rotations)
        else:
            self._state = self._pre_rotated_state

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
            return State

        # Now the return_type of measurement processes must be expectation
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

    def _process_jacobian_tape(self, tape, starting_state, use_device_state):
        # To support np.complex64 based on the type of self._state
        if self.use_csingle:
            create_ops_list = adjoint_diff.create_ops_list_C64
        else:
            create_ops_list = adjoint_diff.create_ops_list_C128

        # Initialization of state
        if starting_state is not None:
            if starting_state.size != 2 ** len(self.wires):
                raise QuantumFunctionError(
                    "The number of qubits of starting_state must be the same as "
                    "that of the device."
                )
            ket = self._asarray(starting_state, dtype=self.C_DTYPE)
        else:
            if not use_device_state:
                self.reset()
                self.apply(tape.operations)
            ket = self._pre_rotated_state

        obs_serialized = _serialize_observables(tape, self.wire_map, use_csingle=self.use_csingle)
        ops_serialized, use_sp = _serialize_ops(tape, self.wire_map)

        ops_serialized = create_ops_list(*ops_serialized)

        # We need to filter out indices in trainable_params which do not
        # correspond to operators.
        trainable_params = sorted(tape.trainable_params)
        if len(trainable_params) == 0:
            return None

        tp_shift = []
        record_tp_rows = []
        all_params = 0

        for op_idx, tp in enumerate(trainable_params):
            op, _ = tape.get_operation(
                op_idx
            )  # get op_idx-th operator among differentiable operators
            if isinstance(op, Operation) and not isinstance(op, (BasisState, QubitStateVector)):
                # We now just ignore non-op or state preps
                tp_shift.append(tp)
                record_tp_rows.append(all_params)
            all_params += 1

        if use_sp:
            # When the first element of the tape is state preparation. Still, I am not sure
            # whether there must be only one state preparation...
            tp_shift = [i - 1 for i in tp_shift]

        ket = ket.reshape(-1)
        state_vector = StateVectorC64(ket) if self.use_csingle else StateVectorC128(ket)
        return {
            "state_vector": state_vector,
            "obs_serialized": obs_serialized,
            "ops_serialized": ops_serialized,
            "tp_shift": tp_shift,
            "record_tp_rows": record_tp_rows,
            "all_params": all_params,
        }

    def adjoint_jacobian(self, tape, starting_state=None, use_device_state=False):
        if self.shots is not None:
            warn(
                "Requested adjoint differentiation to be computed with finite shots."
                " The derivative is always exact when using the adjoint differentiation method.",
                UserWarning,
            )

        tape_return_type = self._check_adjdiff_supported_measurements(tape.measurements)

        if not tape_return_type:  # the tape does not have measurements
            return np.array([], dtype=self._state.dtype)

        if tape_return_type is State:
            raise QuantumFunctionError(
                "This method does not support statevector return type. "
                "Use vjp method instead for this purpose."
            )

        self._check_adjdiff_supported_operations(tape.operations)

        processed_data = self._process_jacobian_tape(tape, starting_state, use_device_state)

        if not processed_data:  # training_params is empty
            return np.array([], dtype=self._state.dtype)

        trainable_params = processed_data["tp_shift"]

        # If requested batching over observables, chunk into OMP_NUM_THREADS sized chunks.
        # This will allow use of Lightning with adjoint for large-qubit numbers AND large
        # numbers of observables, enabling choice between compute time and memory use.
        requested_threads = int(getenv("OMP_NUM_THREADS", "1"))

        if self._batch_obs and requested_threads > 1:
            obs_partitions = _chunk_iterable(processed_data["obs_serialized"], requested_threads)
            jac = []
            for obs_chunk in obs_partitions:
                jac_local = adjoint_diff.adjoint_jacobian(
                    processed_data["state_vector"],
                    obs_chunk,
                    processed_data["ops_serialized"],
                    trainable_params,
                )
                jac.extend(jac_local)
        else:
            jac = adjoint_diff.adjoint_jacobian(
                processed_data["state_vector"],
                processed_data["obs_serialized"],
                processed_data["ops_serialized"],
                trainable_params,
            )
        jac = np.array(jac)
        jac = jac.reshape(-1, len(trainable_params))
        jac_r = np.zeros((jac.shape[0], processed_data["all_params"]))
        jac_r[:, processed_data["record_tp_rows"]] = jac
        return jac_r

    def vjp(self, measurements, dy, starting_state=None, use_device_state=False):
        """Generate the processing function required to compute the vector-Jacobian products of a tape.

        This function can be used with multiple expectation values or a quantum state. When a quantum state
        is given,

        .. code-block:: python

            vjp_f = dev.vjp([qml.state()], dy)
            vjp = vjp_f(tape)

        computes :math:`w = (w_1,\\cdots,w_m)` where

        .. math::

            w_k = \\langle v| \\frac{\\partial}{\\partial \\theta_k} | \\psi_{\\pmb{\\theta}} \\rangle.

        Here, :math:`m` is the total number of trainable parameters, :math:`\\pmb{\\theta}` is the vector of trainable parameters and :math:`\\psi_{\\pmb{\\theta}}`
        is the output quantum state.

        Args:
            measurements (list): List of measurement processes for vector-Jacobian product. Now it must be expectation values or a quantum state.
            dy (tensor_like): Gradient-output vector. Must have shape matching the output shape of the corresponding tape, i.e. number of measrurements if the return type is expectation or :math:`2^N` if the return type is statevector
            starting_state (tensor_like): post-forward pass state to start execution with. It should be
                complex-valued. Takes precedence over ``use_device_state``.
            use_device_state (bool): use current device state to initialize. A forward pass of the same
                circuit should be the last thing the device has executed. If a ``starting_state`` is
                provided, that takes precedence.

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

        if math.allclose(dy, 0) or tape_return_type is None:
            return lambda tape: math.convert_like(np.zeros(len(tape.trainable_params)), dy)

        if tape_return_type is Expectation:
            if len(dy) != len(measurements):
                raise ValueError(
                    "Number of observables in the tape must be the same as the length of dy in the vjp method"
                )

            if np.iscomplexobj(dy):
                raise ValueError(
                    "The vjp method only works with a real-valued dy when the tape is returning an expectation value"
                )

            ham = qml.Hamiltonian(dy, [m.obs for m in measurements])

            def processing_fn(tape):
                nonlocal ham
                num_params = len(tape.trainable_params)

                if num_params == 0:
                    return np.array([], dtype=self._state.dtype)

                new_tape = tape.copy()
                new_tape._measurements = [qml.expval(ham)]

                return self.adjoint_jacobian(new_tape, starting_state, use_device_state).reshape(-1)

            return processing_fn

        if tape_return_type is State:
            if len(dy) != 2 ** len(self.wires):
                raise ValueError(
                    "Size of the provided vector dy must be the same as the size of the statevector"
                )
            if np.isrealobj(dy):
                warn(
                    "The vjp method only works with complex-valued dy when the tape is returning a statevector. Upcasting dy."
                )

            dy = dy.astype(self.C_DTYPE)

            def processing_fn(tape):
                nonlocal dy
                processed_data = self._process_jacobian_tape(tape, starting_state, use_device_state)
                return adjoint_diff.statevector_vjp(
                    processed_data["state_vector"],
                    processed_data["ops_serialized"],
                    dy,
                    processed_data["tp_shift"],
                )

            return processing_fn

    def batch_vjp(
        self, tapes, dys, reduction="append", starting_state=None, use_device_state=False
    ):
        """Generate the processing function required to compute the vector-Jacobian products
        of a batch of tapes.

        Args:
            tapes (Sequence[.QuantumTape]): sequence of quantum tapes to differentiate
            dys (Sequence[tensor_like]): Sequence of gradient-output vectors ``dy``. Must be the
                same length as ``tapes``. Each ``dy`` tensor should have shape
                matching the output shape of the corresponding tape.
            reduction (str): Determines how the vector-Jacobian products are returned.
                If ``append``, then the output of the function will be of the form
                ``List[tensor_like]``, with each element corresponding to the VJP of each
                input tape. If ``extend``, then the output VJPs will be concatenated.
            starting_state (tensor_like): post-forward pass state to start execution with. It should be
                complex-valued. Takes precedence over ``use_device_state``.
            use_device_state (bool): use current device state to initialize. A forward pass of the same
                circuit should be the last thing the device has executed. If a ``starting_state`` is
                provided, that takes precedence.

        Returns:
            The processing function required to compute the vector-Jacobian products of a batch of tapes.
        """
        fns = []

        # Loop through the tapes and dys vector
        for tape, dy in zip(tapes, dys):
            fn = self.vjp(
                tape.measurements,
                dy,
                starting_state=starting_state,
                use_device_state=use_device_state,
            )
            fns.append(fn)

        def processing_fns(tapes):
            vjps = []
            for t, f in zip(tapes, fns):
                vjp = f(t)

                if isinstance(reduction, str):
                    getattr(vjps, reduction)(vjp)
                elif callable(reduction):
                    reduction(vjps, vjp)

            return vjps

        return processing_fns

    def probability(self, wires=None, shot_range=None, bin_size=None):
        """Return the probability of each computational basis state.

        Devices that require a finite number of shots always return the
        estimated probability.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided are traced out of the system.
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.

        Returns:
            array[float]: list of the probabilities
        """
        if self.shots is not None:
            return self.estimate_probability(wires=wires, shot_range=shot_range, bin_size=bin_size)

        wires = wires or self.wires
        wires = Wires(wires)

        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        # To support np.complex64 based on the type of self._state
        dtype = self._state.dtype
        ket = np.ravel(self._state)
        state_vector = StateVectorC64(ket) if self.use_csingle else StateVectorC128(ket)
        M = MeasuresC64(state_vector) if self.use_csingle else MeasuresC128(state_vector)
        return M.probs(device_wires)

    def generate_samples(self):
        """Generate samples

        Returns:
            array[int]: array of samples in binary representation with shape ``(dev.shots, dev.num_wires)``
        """

        # Initialization of state
        ket = np.ravel(self._state)

        state_vector = StateVectorC64(ket) if self.use_csingle else StateVectorC128(ket)
        M = MeasuresC64(state_vector) if self.use_csingle else MeasuresC128(state_vector)

        return M.generate_samples(len(self.wires), self.shots).astype(int, copy=False)

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
        if (
            (observable.arithmetic_depth > 0)
            or isinstance(observable.name, List)
            or observable.name
            in [
                "Identity",
                "Projector",
            ]
        ):
            return super().expval(observable, shot_range=shot_range, bin_size=bin_size)

        if self.shots is not None:
            # estimate the expectation value
            # LightningQubit doesn't support sampling yet
            samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
            return np.squeeze(np.mean(samples, axis=0))

        # Initialization of state
        ket = np.ravel(self._pre_rotated_state)

        state_vector = StateVectorC64(ket) if self.use_csingle else StateVectorC128(ket)
        M = MeasuresC64(state_vector) if self.use_csingle else MeasuresC128(state_vector)
        if observable.name == "SparseHamiltonian":
            if Kokkos_info()["USE_KOKKOS"] == True:
                # converting COO to CSR sparse representation.
                CSR_SparseHamiltonian = observable.data[0].tocsr(copy=False)
                return M.expval(
                    CSR_SparseHamiltonian.indptr,
                    CSR_SparseHamiltonian.indices,
                    CSR_SparseHamiltonian.data,
                )
            raise NotImplementedError(
                "The expval of a SparseHamiltonian requires Kokkos and Kokkos Kernels."
            )

        if observable.name in ["Hamiltonian", "Hermitian"]:
            ob_serialized = _serialize_ob(observable, self.wire_map, use_csingle=self.use_csingle)
            return M.expval(ob_serialized)

        # translate to wire labels used by device
        observable_wires = self.map_wires(observable.wires)

        return M.expval(observable.name, observable_wires)

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
        if isinstance(observable.name, List) or observable.name in [
            "Identity",
            "Projector",
            "Hermitian",
        ]:
            return super().var(observable, shot_range=shot_range, bin_size=bin_size)

        if self.shots is not None:
            # estimate the var
            # LightningQubit doesn't support sampling yet
            samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
            return np.squeeze(np.var(samples, axis=0))

        # Initialization of state
        ket = np.ravel(self._pre_rotated_state)

        state_vector = StateVectorC64(ket) if self.use_csingle else StateVectorC128(ket)
        M = MeasuresC64(state_vector) if self.use_csingle else MeasuresC128(state_vector)

        # translate to wire labels used by device
        observable_wires = self.map_wires(observable.wires)

        return M.var(observable.name, observable_wires)


if not CPP_BINARY_AVAILABLE:

    class LightningQubit(DefaultQubit):  # pragma: no cover
        name = "Lightning Qubit PennyLane plugin"
        short_name = "lightning.qubit"
        pennylane_requires = ">=0.26"
        version = __version__
        author = "Xanadu Inc."
        _CPP_BINARY_AVAILABLE = False

        def __init__(self, wires, *, c_dtype=np.complex128, **kwargs):
            warn(
                "Pre-compiled binaries for lightning.qubit are not available. Falling back to "
                "using the Python-based default.qubit implementation. To manually compile from "
                "source, follow the instructions at "
                "https://pennylane-lightning.readthedocs.io/en/latest/installation.html.",
                UserWarning,
            )

            if c_dtype is np.complex64:
                r_dtype = np.float32
            elif c_dtype is np.complex128:
                r_dtype = np.float64
            else:
                raise TypeError(f"Unsupported complex Type: {c_dtype}")
            super().__init__(wires, r_dtype=r_dtype, c_dtype=c_dtype, **kwargs)
