# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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

import os
import sys
from os import getenv
from pathlib import Path
from typing import List
from warnings import warn

import numpy as np
import pennylane as qml
from pennylane import BasisState, DeviceError, QuantumFunctionError, Rot, StatePrep, math
from pennylane.measurements import Expectation, MidMeasureMP, State
from pennylane.ops import Conditional
from pennylane.ops.op_math import Adjoint
from pennylane.wires import Wires

from pennylane_lightning.core._serialize import QuantumScriptSerializer, global_phase_diagonal
from pennylane_lightning.core._version import __version__
from pennylane_lightning.core.lightning_base import LightningBase, _chunk_iterable

try:
    # pylint: disable=import-error, no-name-in-module
    from pennylane_lightning.lightning_kokkos_ops import (
        InitializationSettings,
        MeasurementsC64,
        MeasurementsC128,
        StateVectorC64,
        StateVectorC128,
        allocate_aligned_array,
        backend_info,
        print_configuration,
    )

    # pylint: disable=import-error, no-name-in-module, ungrouped-imports
    from pennylane_lightning.lightning_kokkos_ops.algorithms import (
        AdjointJacobianC64,
        AdjointJacobianC128,
        create_ops_listC64,
        create_ops_listC128,
    )

    LK_CPP_BINARY_AVAILABLE = True
except ImportError:
    LK_CPP_BINARY_AVAILABLE = False
    backend_info = None


def _kokkos_dtype(dtype):
    if dtype not in [np.complex128, np.complex64]:  # pragma: no cover
        raise ValueError(f"Data type is not supported for state-vector computation: {dtype}")
    return StateVectorC128 if dtype == np.complex128 else StateVectorC64


def _kokkos_configuration():
    return print_configuration()


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
    "C(BlockEncode)",
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
    "LinearCombination",
    "Sum",
    "SProd",
    "Prod",
    "Exp",
}


class LightningKokkos(LightningBase):
    """PennyLane Lightning Kokkos device.

    A device that interfaces with C++ to perform fast linear algebra calculations.

    Use of this device requires pre-built binaries or compilation from source. Check out the
    :doc:`/lightning_kokkos/installation` guide for more details.

    Args:
        wires (int): the number of wires to initialize the device with
        sync (bool): immediately sync with host-sv after applying operations
        c_dtype: Datatypes for statevector representation. Must be one of
            ``np.complex64`` or ``np.complex128``.
        kokkos_args (InitializationSettings): binding for Kokkos::InitializationSettings
            (threading parameters).
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically.
    """

    name = "Lightning Kokkos PennyLane plugin"
    short_name = "lightning.kokkos"
    kokkos_config = {}
    operations = allowed_operations
    observables = allowed_observables
    _backend_info = backend_info
    config = Path(__file__).parent / "lightning_kokkos.toml"
    _CPP_BINARY_AVAILABLE = LK_CPP_BINARY_AVAILABLE

    def __init__(
        self,
        wires,
        *,
        sync=True,
        c_dtype=np.complex128,
        shots=None,
        batch_obs=False,
        kokkos_args=None,
    ):  # pylint: disable=unused-argument, too-many-arguments
        super().__init__(wires, shots=shots, c_dtype=c_dtype)

        if kokkos_args is None:
            self._kokkos_state = _kokkos_dtype(c_dtype)(self.num_wires)
        elif isinstance(kokkos_args, InitializationSettings):
            self._kokkos_state = _kokkos_dtype(c_dtype)(self.num_wires, kokkos_args)
        else:
            type0 = type(InitializationSettings())
            raise TypeError(
                f"Argument kokkos_args must be of type {type0} but it is of {type(kokkos_args)}."
            )
        self._sync = sync

        if not LightningKokkos.kokkos_config:
            LightningKokkos.kokkos_config = _kokkos_configuration()

    @property
    def stopping_condition(self):
        """.BooleanFn: Returns the stopping condition for the device. The returned
        function accepts a queueable object (including a PennyLane operation
        and observable) and returns ``True`` if supported by the device."""
        fun = super().stopping_condition

        def accepts_obj(obj):
            return fun(obj) or isinstance(obj, (qml.measurements.MidMeasureMP, qml.ops.Conditional))

        return qml.BooleanFn(accepts_obj)

    # pylint: disable=missing-function-docstring
    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(supports_mid_measure=True)
        return capabilities

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
        if arr.dtype != dtype:
            new_arr = allocate_aligned_array(arr.size, np.dtype(dtype), False).reshape(arr.shape)
            np.copyto(new_arr, arr)
            arr = new_arr
        return arr

    def _create_basis_state(self, index):
        """Return a computational basis state over all wires.
        Args:
            index (int): integer representing the computational basis state
        Returns:
            array[complex]: complex array of shape ``[2]*self.num_wires``
            representing the statevector of the basis state
        Note: This function does not support broadcasted inputs yet.
        """
        self._kokkos_state.setBasisState(index)

    def reset(self):
        """Reset the device"""
        super().reset()

        # init the state vector to |00..0>
        self._kokkos_state.resetStateVector()  # Sync reset

    def sync_h2d(self, state_vector):
        """Copy the state vector data on host provided by the user to the state
        vector on the device

        Args:
            state_vector(array[complex]): the state vector array on host.


        **Example**

        >>> dev = qml.device('lightning.kokkos', wires=3)
        >>> obs = qml.Identity(0) @ qml.PauliX(1) @ qml.PauliY(2)
        >>> obs1 = qml.Identity(1)
        >>> H = qml.Hamiltonian([1.0, 1.0], [obs1, obs])
        >>> state_vector = np.array([0.0 + 0.0j, 0.0 + 0.1j, 0.1 + 0.1j, 0.1 + 0.2j, 0.2 + 0.2j, 0.3 + 0.3j, 0.3 + 0.4j, 0.4 + 0.5j,], dtype=np.complex64)
        >>> dev.sync_h2d(state_vector)
        >>> res = dev.expval(H)
        >>> print(res)
        1.0
        """
        self._kokkos_state.HostToDevice(state_vector.ravel(order="C"))

    def sync_d2h(self, state_vector):
        """Copy the state vector data on device to a state vector on the host provided
        by the user

        Args:
            state_vector(array[complex]): the state vector array on host


        **Example**

        >>> dev = qml.device('lightning.kokkos', wires=1)
        >>> dev.apply([qml.PauliX(wires=[0])])
        >>> state_vector = np.zeros(2**dev.num_wires).astype(dev.C_DTYPE)
        >>> dev.sync_d2h(state_vector)
        >>> print(state_vector)
        [0.+0.j 1.+0.j]
        """
        self._kokkos_state.DeviceToHost(state_vector.ravel(order="C"))

    @property
    def create_ops_list(self):
        """Returns create_ops_list function of the matching precision."""
        return create_ops_listC64 if self.use_csingle else create_ops_listC128

    @property
    def measurements(self):
        """Returns Measurements constructor of the matching precision."""
        state_vector = self.state_vector
        return MeasurementsC64(state_vector) if self.use_csingle else MeasurementsC128(state_vector)

    @property
    def state(self):
        """Copy the state vector data from the device to the host.

        A state vector Numpy array is explicitly allocated on the host to store and return
        the data.

        **Example**

        >>> dev = qml.device('lightning.kokkos', wires=1)
        >>> dev.apply([qml.PauliX(wires=[0])])
        >>> print(dev.state)
        [0.+0.j 1.+0.j]
        """
        state = np.zeros(2**self.num_wires, dtype=self.C_DTYPE)
        state = self._asarray(state, dtype=self.C_DTYPE)
        self.sync_d2h(state)
        return state

    @property
    def state_vector(self):
        """Returns a handle to the statevector."""
        return self._kokkos_state

    def _apply_state_vector(self, state, device_wires):
        """Initialize the internal state vector in a specified state.

        Args:
            state (array[complex]): normalized input state of length ``2**len(wires)``
                or broadcasted state of shape ``(batch_size, 2**len(wires))``
            device_wires (Wires): wires that get initialized in the state
        """

        if isinstance(state, self._kokkos_state.__class__):
            state_data = allocate_aligned_array(state.size, np.dtype(self.C_DTYPE), True)
            state.DeviceToHost(state_data)
            state = state_data

        ravelled_indices, state = self._preprocess_state_vector(state, device_wires)

        # translate to wire labels used by device
        device_wires = self.map_wires(device_wires)
        output_shape = [2] * self.num_wires

        if len(device_wires) == self.num_wires and Wires(sorted(device_wires)) == device_wires:
            # Initialize the entire device state with the input state
            self.sync_h2d(self._reshape(state, output_shape))
            return

        self._kokkos_state.setStateVector(ravelled_indices, state)  # this operation on device

    def _apply_basis_state(self, state, wires):
        """Initialize the state vector in a specified computational basis state.

        Args:
            state (array[int]): computational basis state of shape ``(wires,)``
                consisting of 0s and 1s.
            wires (Wires): wires that the provided computational state should be initialized on

        Note: This function does not support broadcasted inputs yet.
        """
        num = self._get_basis_state_index(state, wires)
        self._create_basis_state(num)

    def _apply_lightning_midmeasure(
        self, operation: MidMeasureMP, mid_measurements: dict, postselect_mode: str
    ):
        """Execute a MidMeasureMP operation and return the sample in mid_measurements.

        Args:
            operation (~pennylane.operation.Operation): mid-circuit measurement

        Returns:
            None
        """
        wires = self.wires.indices(operation.wires)
        wire = list(wires)[0]
        if postselect_mode == "fill-shots" and operation.postselect is not None:
            sample = operation.postselect
        else:
            sample = qml.math.reshape(self.generate_samples(shots=1), (-1,))[wire]
        mid_measurements[operation] = sample
        getattr(self.state_vector, "collapse")(wire, bool(sample))
        if operation.reset and bool(sample):
            self.apply([qml.PauliX(operation.wires)], mid_measurements=mid_measurements)

    def apply_lightning(self, operations, mid_measurements=None, postselect_mode=None):
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
        state = self.state_vector

        for ops in operations:
            if isinstance(ops, Adjoint):
                name = ops.base.name
                invert_param = True
            else:
                name = ops.name
                invert_param = False
            if isinstance(ops, qml.Identity):
                continue
            method = getattr(state, name, None)
            wires = self.wires.indices(ops.wires)

            if isinstance(ops, Conditional):
                if ops.meas_val.concretize(mid_measurements):
                    self.apply_lightning([ops.base])
            elif isinstance(ops, MidMeasureMP):
                self._apply_lightning_midmeasure(ops, mid_measurements, postselect_mode)
            elif isinstance(ops, qml.ops.op_math.Controlled) and isinstance(
                ops.base, qml.GlobalPhase
            ):
                controls = ops.control_wires
                control_values = ops.control_values
                param = ops.base.parameters[0]
                matrix = global_phase_diagonal(param, self.wires, controls, control_values)
                state.apply(name, wires, False, [[param]], matrix)
            elif method is None:
                # Inverse can be set to False since qml.matrix(ops) is already in inverted form
                try:
                    mat = qml.matrix(ops)
                except AttributeError:  # pragma: no cover
                    # To support older versions of PL
                    mat = ops.matrix

                if len(mat) == 0:
                    raise ValueError("Unsupported operation")
                state.apply(
                    name,
                    wires,
                    False,
                    [],
                    mat.ravel(order="C"),  # inv = False: Matrix already in correct form;
                )  # Parameters can be ignored for explicit matrices; F-order for cuQuantum
            else:
                param = ops.parameters
                method(wires, invert_param, param)

    # pylint: disable=unused-argument
    def apply(self, operations, rotations=None, mid_measurements=None, **kwargs):
        """Applies a list of operations to the state tensor."""
        # State preparation is currently done in Python
        if operations:  # make sure operations[0] exists
            if isinstance(operations[0], StatePrep):
                self._apply_state_vector(operations[0].parameters[0].copy(), operations[0].wires)
                operations = operations[1:]
            elif isinstance(operations[0], BasisState):
                self._apply_basis_state(operations[0].parameters[0], operations[0].wires)
                operations = operations[1:]

        postselect_mode = kwargs.get("postselect_mode", None)

        for operation in operations:
            if isinstance(operation, (StatePrep, BasisState)):
                raise DeviceError(
                    f"Operation {operation.name} cannot be used after other "
                    + f"Operations have already been applied on a {self.short_name} device."
                )

        self.apply_lightning(
            operations, mid_measurements=mid_measurements, postselect_mode=postselect_mode
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
        if isinstance(observable, qml.Projector):
            diagonalizing_gates = observable.diagonalizing_gates()
            if self.shots is None and diagonalizing_gates:
                self.apply(diagonalizing_gates)
            results = super().expval(observable, shot_range=shot_range, bin_size=bin_size)
            if self.shots is None and diagonalizing_gates:
                self.apply([qml.adjoint(g, lazy=False) for g in reversed(diagonalizing_gates)])
            return results

        if self.shots is not None:
            # estimate the expectation value
            # LightningQubit doesn't support sampling yet
            samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
            return np.squeeze(np.mean(samples, axis=0))

        # Initialization of state
        measure = (
            MeasurementsC64(self.state_vector)
            if self.use_csingle
            else MeasurementsC128(self.state_vector)
        )
        if isinstance(observable, qml.SparseHamiltonian):
            csr_hamiltonian = observable.sparse_matrix(wire_order=self.wires).tocsr(copy=False)
            return measure.expval(
                csr_hamiltonian.indptr,
                csr_hamiltonian.indices,
                csr_hamiltonian.data,
            )

        # use specialized functors to compute expval(Hermitian)
        if isinstance(observable, qml.Hermitian):
            observable_wires = self.map_wires(observable.wires)
            matrix = observable.matrix()
            return measure.expval(matrix, observable_wires)

        if (
            isinstance(observable, qml.ops.Hamiltonian)
            or (observable.arithmetic_depth > 0)
            or isinstance(observable.name, List)
        ):
            ob_serialized = QuantumScriptSerializer(self.short_name, self.use_csingle)._ob(
                observable, self.wire_map
            )
            return measure.expval(ob_serialized)

        # translate to wire labels used by device
        observable_wires = self.map_wires(observable.wires)

        return measure.expval(observable.name, observable_wires)

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
        if isinstance(observable, qml.Projector):
            diagonalizing_gates = observable.diagonalizing_gates()
            if self.shots is None and diagonalizing_gates:
                self.apply(diagonalizing_gates)
            results = super().var(observable, shot_range=shot_range, bin_size=bin_size)
            if self.shots is None and diagonalizing_gates:
                self.apply([qml.adjoint(g, lazy=False) for g in reversed(diagonalizing_gates)])
            return results

        if self.shots is not None:
            # estimate the var
            # LightningKokkos doesn't support sampling yet
            samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
            return np.squeeze(np.var(samples, axis=0))

        # Initialization of state
        measure = (
            MeasurementsC64(self.state_vector)
            if self.use_csingle
            else MeasurementsC128(self.state_vector)
        )

        if isinstance(observable, qml.SparseHamiltonian):
            csr_hamiltonian = observable.sparse_matrix(wire_order=self.wires).tocsr(copy=False)
            return measure.var(
                csr_hamiltonian.indptr,
                csr_hamiltonian.indices,
                csr_hamiltonian.data,
            )

        if (
            isinstance(observable, (qml.Hamiltonian, qml.Hermitian))
            or (observable.arithmetic_depth > 0)
            or isinstance(observable.name, List)
        ):
            ob_serialized = QuantumScriptSerializer(self.short_name, self.use_csingle)._ob(
                observable, self.wire_map
            )
            return measure.var(ob_serialized)

        # translate to wire labels used by device
        observable_wires = self.map_wires(observable.wires)

        return measure.var(observable.name, observable_wires)

    def generate_samples(self, shots=None):
        """Generate samples

        Returns:
            array[int]: array of samples in binary representation with shape
            ``(dev.shots, dev.num_wires)``
        """
        shots = self.shots if shots is None else shots

        shots = shots if isinstance(shots, int) else shots.total_shots
        
        measure = (
            MeasurementsC64(self._kokkos_state)
            if self.use_csingle
            else MeasurementsC128(self._kokkos_state)
        )
        return measure.generate_samples(len(self.wires), shots.total_shots).astype(int, copy=False)

    def probability_lightning(self, wires):
        """Return the probability of each computational basis state.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided are traced out of the system.

        Returns:
            array[float]: list of the probabilities
        """
        return self.measurements.probs(wires)

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

    @staticmethod
    def _check_adjdiff_supported_operations(operations):
        """Check Lightning adjoint differentiation method support for a tape.

        Raise ``QuantumFunctionError`` if ``tape`` contains not supported measurements,
        observables, or operations by the Lightning adjoint differentiation method.

        Args:
            tape (.QuantumTape): quantum tape to differentiate.
        """
        for operation in operations:
            if operation.num_params > 1 and not isinstance(operation, Rot):
                raise QuantumFunctionError(
                    f"The {operation.name} operation is not supported using "
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
        return self.state_vector

    def adjoint_jacobian(self, tape, starting_state=None, use_device_state=False):
        """Implements the adjoint method outlined in
        `Jones and Gacon <https://arxiv.org/abs/2009.02823>`__ to differentiate an input tape.

        After a forward pass, the circuit is reversed by iteratively applying adjoint
        gates to scan backwards through the circuit.
        """
        if self.shots is not None:
            warn(
                "Requested adjoint differentiation to be computed with finite shots."
                " The derivative is always exact when using the adjoint "
                "differentiation method.",
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

        self._check_adjdiff_supported_operations(tape.operations)

        processed_data = self._process_jacobian_tape(tape, starting_state, use_device_state)

        if not processed_data:  # training_params is empty
            return np.array([], dtype=self.state.dtype)

        trainable_params = processed_data["tp_shift"]

        # If requested batching over observables, chunk into OMP_NUM_THREADS sized chunks.
        # This will allow use of Lightning with adjoint for large-qubit numbers AND large
        # numbers of observables, enabling choice between compute time and memory use.
        requested_threads = int(getenv("OMP_NUM_THREADS", "1"))

        adjoint_jacobian = AdjointJacobianC64() if self.use_csingle else AdjointJacobianC128()

        if self._batch_obs and requested_threads > 1:  # pragma: no cover
            obs_partitions = _chunk_iterable(processed_data["obs_serialized"], requested_threads)
            jac = []
            for obs_chunk in obs_partitions:
                jac_local = adjoint_jacobian(
                    processed_data["state_vector"],
                    obs_chunk,
                    processed_data["ops_serialized"],
                    trainable_params,
                )
                jac.extend(jac_local)
        else:
            jac = adjoint_jacobian(
                processed_data["state_vector"],
                processed_data["obs_serialized"],
                processed_data["ops_serialized"],
                trainable_params,
            )
        jac = np.array(jac)
        jac = jac.reshape(-1, len(trainable_params))
        jac_r = np.zeros((jac.shape[0], processed_data["all_params"]))
        jac_r[:, processed_data["record_tp_rows"]] = jac
        if hasattr(qml, "active_return"):  # pragma: no cover
            return self._adjoint_jacobian_processing(jac_r) if qml.active_return() else jac_r
        return self._adjoint_jacobian_processing(jac_r)

    # pylint: disable=inconsistent-return-statements, line-too-long
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
                " The derivative is always exact when using the adjoint "
                "differentiation method.",
                UserWarning,
            )

        tape_return_type = self._check_adjdiff_supported_measurements(measurements)

        if math.allclose(grad_vec, 0) or tape_return_type is None:
            return lambda tape: math.convert_like(np.zeros(len(tape.trainable_params)), grad_vec)

        if tape_return_type is Expectation:
            if len(grad_vec) != len(measurements):
                raise ValueError(
                    "Number of observables in the tape must be the same as the "
                    "length of grad_vec in the vjp method"
                )

            if np.iscomplexobj(grad_vec):
                raise ValueError(
                    "The vjp method only works with a real-valued grad_vec when "
                    "the tape is returning an expectation value"
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

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """

        # The shared object file extension varies depending on the underlying operating system
        file_extension = ""
        OS = sys.platform
        if OS == "linux":
            file_extension = ".so"
        elif OS == "darwin":
            file_extension = ".dylib"
        else:
            raise RuntimeError(
                f"'LightningKokkosSimulator' shared library not available for '{OS}' platform"
            )

        lib_name = "liblightning_kokkos_catalyst" + file_extension
        package_root = Path(__file__).parent

        # The absolute path of the plugin shared object varies according to the installation mode.

        # Wheel mode:
        # Fixed location at the root of the project
        wheel_mode_location = package_root.parent / lib_name
        if wheel_mode_location.is_file():
            return "LightningKokkosSimulator", wheel_mode_location.as_posix()

        # Editable mode:
        # The build directory contains a folder which varies according to the platform:
        #   lib.<system>-<architecture>-<python-id>"
        # To avoid mismatching the folder name, we search for the shared object instead.
        # TODO: locate where the naming convention of the folder is decided and replicate it here.
        editable_mode_path = package_root.parent.parent / "build"
        for path, _, files in os.walk(editable_mode_path):
            if lib_name in files:
                lib_location = (Path(path) / lib_name).as_posix()
                return "LightningKokkosSimulator", lib_location

        raise RuntimeError("'LightningKokkosSimulator' shared library not found")
