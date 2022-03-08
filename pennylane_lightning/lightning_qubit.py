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
from itertools import islice

import numpy as np
from pennylane import (
    math,
    gradients,
    BasisState,
    QubitStateVector,
    QubitUnitary,
    Projector,
    Hermitian,
    Rot,
    QuantumFunctionError,
    DeviceError,
)
from pennylane.devices import DefaultQubit
from pennylane.operation import Expectation, Tensor
from pennylane.wires import Wires

# Remove after the next release of PL
# Add from pennylane import matrix
import pennylane as qml

from ._version import __version__

try:
    from .lightning_qubit_ops import (
        MeasuresC64,
        StateVectorC64,
        AdjointJacobianC64,
        VectorJacobianProductC64,
        MeasuresC128,
        StateVectorC128,
        AdjointJacobianC128,
        VectorJacobianProductC128,
        DEFAULT_KERNEL_FOR_OPS,
        EXPORTED_KERNEL_OPS,
    )

    from ._serialize import _serialize_obs, _serialize_ops, _is_lightning_gate

    CPP_BINARY_AVAILABLE = True
except ModuleNotFoundError:
    CPP_BINARY_AVAILABLE = False


UNSUPPORTED_PARAM_GATES_ADJOINT = (
    "SingleExcitation",
    "SingleExcitationPlus",
    "SingleExcitationMinus",
    "DoubleExcitation",
    "DoubleExcitationPlus",
    "DoubleExcitationMinus",
)


def _chunk_iterable(it, num_chunks):
    "Lazy-evaluated chunking of given iterable from https://stackoverflow.com/a/22045226"
    it = iter(it)
    return iter(lambda: tuple(islice(it, num_chunks)), ())


def _remove_snapshot_from_operations(operations):
    operations = operations.copy()
    operations.discard("Snapshot")
    return operations


class LightningQubit(DefaultQubit):
    """PennyLane Lightning device.

    An extension of PennyLane's built-in ``default.qubit`` device that interfaces with C++ to
    perform fast linear algebra calculations.

    Use of this device requires pre-built binaries or compilation from source. Check out the
    :doc:`/installation` guide for more details.

    Args:
        wires (int): the number of wires to initialize the device with
        kernel_for_ops (dict): Optional argument which kernel to run for a gate operation.
            For example, if {'PauliX': 'LM', 'RX': 'PI'} is passed, the less memory (LM) kernel
            is used for PauliX whereas precomputed indices (PI) kernel is used for RX.
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically.
    """

    name = "Lightning Qubit PennyLane plugin"
    short_name = "lightning.qubit"
    pennylane_requires = ">=0.15"
    version = __version__
    author = "Xanadu Inc."
    _CPP_BINARY_AVAILABLE = True
    operations = _remove_snapshot_from_operations(DefaultQubit.operations)

    def __init__(self, wires, *, kernel_for_ops=None, shots=None, batch_obs=False):
        self._kernel_for_ops = DEFAULT_KERNEL_FOR_OPS
        if kernel_for_ops is not None:
            if not isinstance(kernel_for_ops, dict):
                raise ValueError("Argument kernel_for_ops must be a dictionary.")

            for gate_op, kernel in kernel_for_ops.items():
                if (kernel, gate_op) not in EXPORTED_KERNEL_OPS:
                    raise ValueError(
                        f"The given kernel {kernel} does not implement {gate_op} gate."
                    )
                self._kernel_for_ops[gate_op] = kernel

        super().__init__(wires, shots=shots)
        self._batch_obs = batch_obs

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qubit",
            supports_reversible_diff=False,
            supports_inverse_operations=True,
            supports_analytic_computation=True,
            returns_state=True,
        )
        capabilities.pop("passthru_devices", None)
        return capabilities

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

        # Get the Type of self._state
        # as the reference type
        dtype = self._state.dtype

        if operations:
            self._pre_rotated_state = self.apply_lightning(self._state, operations, dtype=dtype)
        else:
            self._pre_rotated_state = self._state

        if rotations:
            if any(isinstance(r, QubitUnitary) for r in rotations):
                super().apply(operations=[], rotations=rotations)
            else:
                self._state = self.apply_lightning(
                    np.copy(self._pre_rotated_state), rotations, dtype=dtype
                )
        else:
            self._state = self._pre_rotated_state

    def apply_lightning(self, state, operations, dtype=np.complex128):
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

        if dtype == np.complex64:
            # use_csingle
            sim = StateVectorC64(state_vector)
        elif dtype == np.complex128:
            # self.C_DTYPE is np.complex128 by default
            sim = StateVectorC128(state_vector)
        else:
            raise TypeError(f"Unsupported complex Type: {dtype}")

        for o in operations:
            name = o.name.split(".")[0]  # The split is because inverse gates have .inv appended
            if _is_lightning_gate(name):
                kernel = self._kernel_for_ops[name]
                method = getattr(sim, f"{name}_{kernel}".format(), None)
            else:
                method = None

            wires = self.wires.indices(o.wires)

            if method is None:
                # Inverse can be set to False since qml.matrix(o) is already in inverted form
                method = getattr(sim, "applyMatrix_{}".format(self._kernel_for_ops["Matrix"]))
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

    def adjoint_diff_support_check(self, tape):
        """Check Lightning adjoint differentiation method support for a tape.

        Raise ``QuantumFunctionError`` if ``tape`` contains not supported measurements,
        observables, or operations by the Lightning adjoint differentiation method.

        Args:
            tape (.QuantumTape): quantum tape to differentiate
        """
        for m in tape.measurements:
            if m.return_type is not Expectation:
                raise QuantumFunctionError(
                    "Adjoint differentiation method does not support"
                    f" measurement {m.return_type.value}"
                )
            if not isinstance(m.obs, Tensor):
                if isinstance(m.obs, Projector):
                    raise QuantumFunctionError(
                        "Adjoint differentiation method does not support the Projector observable"
                    )
                if isinstance(m.obs, Hermitian):
                    raise QuantumFunctionError(
                        "Lightning adjoint differentiation method does not currently support the Hermitian observable"
                    )
            else:
                if any([isinstance(o, Projector) for o in m.obs.non_identity_obs]):
                    raise QuantumFunctionError(
                        "Adjoint differentiation method does not support the Projector observable"
                    )
                if any([isinstance(o, Hermitian) for o in m.obs.non_identity_obs]):
                    raise QuantumFunctionError(
                        "Lightning adjoint differentiation method does not currently support the Hermitian observable"
                    )

        for op in tape.operations:
            if (
                op.num_params > 1 and not isinstance(op, Rot)
            ) or op.name in UNSUPPORTED_PARAM_GATES_ADJOINT:
                raise QuantumFunctionError(
                    f"The {op.name} operation is not supported using "
                    'the "adjoint" differentiation method'
                )

    def adjoint_jacobian(self, tape, starting_state=None, use_device_state=False):
        if self.shots is not None:
            warn(
                "Requested adjoint differentiation to be computed with finite shots."
                " The derivative is always exact when using the adjoint differentiation method.",
                UserWarning,
            )

        # To support np.complex64 based on the type of self._state
        dtype = self._state.dtype
        if dtype == np.complex64:
            use_csingle = True
        elif dtype == np.complex128:
            use_csingle = False
        else:
            raise TypeError(f"Unsupported complex Type: {dtype}")

        if len(tape.trainable_params) == 0:
            return np.array(0)

        # Check adjoint diff support
        self.adjoint_diff_support_check(tape)

        # Initialization of state
        if starting_state is not None:
            ket = np.ravel(starting_state)
        else:
            if not use_device_state:
                self.reset()
                self.execute(tape)
            ket = np.ravel(self._pre_rotated_state)

        if use_csingle:
            adj = AdjointJacobianC64()
            ket = ket.astype(np.complex64)
        else:
            adj = AdjointJacobianC128()

        obs_serialized = _serialize_obs(tape, self.wire_map, use_csingle=use_csingle)
        ops_serialized, use_sp = _serialize_ops(tape, self.wire_map, use_csingle=use_csingle)

        ops_serialized = adj.create_ops_list(*ops_serialized)

        trainable_params = sorted(tape.trainable_params)
        first_elem = 1 if trainable_params[0] == 0 else 0

        tp_shift = (
            trainable_params if not use_sp else [i - 1 for i in trainable_params[first_elem:]]
        )  # exclude first index if explicitly setting sv

        state_vector = StateVectorC64(ket) if use_csingle else StateVectorC128(ket)

        # If requested batching over observables, chunk into OMP_NUM_THREADS sized chunks.
        # This will allow use of Lightning with adjoint for large-qubit numbers AND large
        # numbers of observables, enabling choice between compute time and memory use.
        requested_threads = int(getenv("OMP_NUM_THREADS", "1"))

        if self._batch_obs and requested_threads > 1:
            obs_partitions = _chunk_iterable(obs_serialized, requested_threads)
            jac = []
            for obs_chunk in obs_partitions:
                jac_local = adj.adjoint_jacobian(
                    state_vector,
                    obs_chunk,
                    ops_serialized,
                    tp_shift,
                    tape.num_params,
                )
                jac.extend(jac_local)
            jac = np.array(jac)
        else:
            jac = adj.adjoint_jacobian(
                state_vector,
                obs_serialized,
                ops_serialized,
                tp_shift,
                tape.num_params,
            )
        return jac.reshape(-1, tape.num_params)

    def compute_vjp(self, dy, jac, num=None):
        """Convenience function to compute the vector-Jacobian product for a given
        vector of gradient outputs and a Jacobian.
        Args:
            dy (tensor_like): vector of gradient outputs
            jac (tensor_like): Jacobian matrix. For an n-dimensional ``dy``
                vector, the first n-dimensions of ``jac`` should match
                the shape of ``dy``.
        Keyword Args:
        num (int): The length of the flattened ``dy`` argument. This is an
            optional argument, but can be useful to provide if ``dy`` potentially
            has no shape (for example, due to tracing or just-in-time compilation).
        Returns:
            tensor_like: the vector-Jacobian product
        """
        if jac is None:
            return None

        if not isinstance(dy, np.ndarray) or not isinstance(jac, np.ndarray):
            return gradients.compute_vjp(dy, jac)

        dy_row = math.reshape(dy, [-1])

        if num is None:
            num = math.shape(dy_row)[0]

        jac = math.reshape(jac, [num, -1])
        num_params = jac.shape[1]

        if math.allclose(dy, 0):
            return math.convert_like(np.zeros([num_params]), dy)

        # To support np.complex64 based on the type of self._state
        dtype = self._state.dtype
        if dtype == np.complex64:
            VJP = VectorJacobianProductC64()
        elif dtype == np.complex128:
            VJP = VectorJacobianProductC128()
        else:
            raise TypeError(f"Unsupported complex Type: {dtype}")

        vjp_tensor = VJP.compute_vjp_from_jac(
            math.reshape(jac, [-1]),
            dy_row,
            num,
            num_params,
        )
        return vjp_tensor

    def vjp(self, tape, dy, starting_state=None, use_device_state=False):
        """Generate the processing function required to compute the vector-Jacobian products of a tape.
        Args:
            tape (.QuantumTape): quantum tape to differentiate
            dy (tensor_like): Gradient-output vector. Must have shape
                matching the output shape of the corresponding tape.
        Keyword Args:
            starting_state (tensor_like): post-forward pass state to start execution with. It should be
                complex-valued. Takes precedence over ``use_device_state``.
            use_device_state (bool): use current device state to initialize. A forward pass of the same
                circuit should be the last thing the device has executed. If a ``starting_state`` is
                provided, that takes precedence.
        Returns:
            The processing function required to compute the vector-Jacobian
            products of a tape.
        """
        if self.shots is not None:
            warn(
                "Requested adjoint differentiation to be computed with finite shots."
                " The derivative is always exact when using the adjoint differentiation method.",
                UserWarning,
            )

        num_params = len(tape.trainable_params)

        if num_params == 0:
            return lambda _: None

        if math.allclose(dy, 0):
            return lambda _: math.convert_like(np.zeros([num_params]), dy)

        # To support np.complex64 based on the type of self._state
        dtype = self._state.dtype
        if dtype == np.complex64:
            use_csingle = True
        elif dtype == np.complex128:
            use_csingle = False
        else:
            raise TypeError(f"Unsupported complex Type: {dtype}")

        V = VectorJacobianProductC64() if use_csingle else VectorJacobianProductC128()

        fn = V.vjp_fn(math.reshape(dy, [-1]), tape.num_params)

        def processing_fn(tape):
            # Check adjoint diff support
            self.adjoint_diff_support_check(tape)

            # Initialization of state
            if starting_state is not None:
                ket = np.ravel(starting_state)
            else:
                if not use_device_state:
                    self.reset()
                    self.execute(tape)
                ket = np.ravel(self._pre_rotated_state)

            if use_csingle:
                ket = ket.astype(np.complex64)

            obs_serialized = _serialize_obs(tape, self.wire_map, use_csingle=use_csingle)
            ops_serialized, use_sp = _serialize_ops(tape, self.wire_map, use_csingle=use_csingle)

            ops_serialized = V.create_ops_list(*ops_serialized)

            trainable_params = sorted(tape.trainable_params)
            first_elem = 1 if trainable_params[0] == 0 else 0

            tp_shift = (
                trainable_params if not use_sp else [i - 1 for i in trainable_params[first_elem:]]
            )  # exclude first index if explicitly setting sv

            state_vector = StateVectorC64(ket) if use_csingle else StateVectorC128(ket)

            return fn(state_vector, obs_serialized, ops_serialized, tp_shift)

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
        Keyword Args:
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
                tape, dy, starting_state=starting_state, use_device_state=use_device_state
            )
            fns.append(fn)

        def processing_fns(tapes):
            vjps = []
            for t, f in zip(tapes, fns):
                vjp = f(t)

                if vjp is None:
                    if reduction == "append":
                        vjps.append(None)
                    continue

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
        if dtype == np.complex64:
            use_csingle = True
        elif dtype == np.complex128:
            use_csingle = False
        else:
            raise TypeError(f"Unsupported complex Type: {dtype}")

        # Initialization of state
        ket = np.ravel(self._state)

        if use_csingle:
            ket = ket.astype(np.complex64)

        state_vector = StateVectorC64(ket) if use_csingle else StateVectorC128(ket)
        M = MeasuresC64(state_vector) if use_csingle else MeasuresC128(state_vector)

        return M.probs(device_wires)

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
        if isinstance(observable.name, List) or observable.name in [
            "Identity",
            "Projector",
            "Hermitian",
            "Hamiltonian",
            "SparseHamiltonian",
        ]:
            return super().expval(observable, shot_range=shot_range, bin_size=bin_size)

        if self.shots is not None:
            # estimate the expectation value
            # LightningQubit doesn't support sampling yet
            samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
            return np.squeeze(np.mean(samples, axis=0))

        # To support np.complex64 based on the type of self._state
        dtype = self._state.dtype
        if dtype == np.complex64:
            use_csingle = True
        elif dtype == np.complex128:
            use_csingle = False
        else:
            raise TypeError(f"Unsupported complex Type: {dtype}")

        # Initialization of state
        ket = np.ravel(self._pre_rotated_state)

        if use_csingle:
            ket = ket.astype(np.complex64)

        state_vector = StateVectorC64(ket) if use_csingle else StateVectorC128(ket)
        M = MeasuresC64(state_vector) if use_csingle else MeasuresC128(state_vector)

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

        # To support np.complex64 based on the type of self._state
        dtype = self._state.dtype
        if dtype == np.complex64:
            use_csingle = True
        elif dtype == np.complex128:
            use_csingle = False
        else:
            raise TypeError(f"Unsupported complex Type: {dtype}")

        # Initialization of state
        ket = np.ravel(self._pre_rotated_state)

        if use_csingle:
            ket = ket.astype(np.complex64)

        state_vector = StateVectorC64(ket) if use_csingle else StateVectorC128(ket)
        M = MeasuresC64(state_vector) if use_csingle else MeasuresC128(state_vector)

        # translate to wire labels used by device
        observable_wires = self.map_wires(observable.wires)

        return M.var(observable.name, observable_wires)


if not CPP_BINARY_AVAILABLE:

    class LightningQubit(DefaultQubit):  # pragma: no cover
        name = "Lightning Qubit PennyLane plugin"
        short_name = "lightning.qubit"
        pennylane_requires = ">=0.15"
        version = __version__
        author = "Xanadu Inc."
        _CPP_BINARY_AVAILABLE = False

        def __init__(self, *args, **kwargs):
            warn(
                "Pre-compiled binaries for lightning.qubit are not available. Falling back to "
                "using the Python-based default.qubit implementation. To manually compile from "
                "source, follow the instructions at "
                "https://pennylane-lightning.readthedocs.io/en/latest/installation.html.",
                UserWarning,
            )
            super().__init__(*args, **kwargs)
