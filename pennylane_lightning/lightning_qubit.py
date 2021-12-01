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
from warnings import warn
import platform, os, sys

import numpy as np
from pennylane import (
    math,
    BasisState,
    DeviceError,
    QuantumFunctionError,
    QubitStateVector,
    QubitUnitary,
)
import pennylane as qml
from pennylane.devices import DefaultQubit
from pennylane.operation import Expectation

from ._version import __version__

try:
    if platform.system() == "Windows" and sys.version_info[:2] >= (3, 8):  # pragma: no cover
        # Add the current directory to DLL path.
        # See https://docs.python.org/3/whatsnew/3.8.html#bpo-36085-whatsnew
        os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))
        from lightning_qubit_ops import (
            apply,
            StateVectorC64,
            StateVectorC128,
            AdjointJacobianC128,
            VectorJacobianProductC128,
        )
    else:
        from .lightning_qubit_ops import (
            apply,
            StateVectorC64,
            StateVectorC128,
            AdjointJacobianC128,
            VectorJacobianProductC128,
        )
    from ._serialize import _serialize_obs, _serialize_ops

    CPP_BINARY_AVAILABLE = True
except ModuleNotFoundError:
    CPP_BINARY_AVAILABLE = False

UNSUPPORTED_PARAM_GATES_ADJOINT = (
    "MultiRZ",
    "IsingXX",
    "IsingYY",
    "IsingZZ",
    "SingleExcitation",
    "SingleExcitationPlus",
    "SingleExcitationMinus",
    "DoubleExcitation",
    "DoubleExcitationPlus",
    "DoubleExcitationMinus",
)


class LightningQubit(DefaultQubit):
    """PennyLane Lightning device.

    An extension of PennyLane's built-in ``default.qubit`` device that interfaces with C++ to
    perform fast linear algebra calculations.

    Use of this device requires pre-built binaries or compilation from source. Check out the
    :doc:`/installation` guide for more details.

    Args:
        wires (int): the number of wires to initialize the device with
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

    def __init__(self, wires, *, shots=None):
        super().__init__(wires, shots=shots)

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

        if operations:
            self._pre_rotated_state = self.apply_lightning(self._state, operations)
        else:
            self._pre_rotated_state = self._state

        if rotations:
            if any(isinstance(r, QubitUnitary) for r in rotations):
                super().apply(operations=[], rotations=rotations)
            else:
                self._state = self.apply_lightning(np.copy(self._pre_rotated_state), rotations)
        else:
            self._state = self._pre_rotated_state

    def apply_lightning(self, state, operations):
        """Apply a list of operations to the state tensor.

        Args:
            state (array[complex]): the input state tensor
            operations (list[~pennylane.operation.Operation]): operations to apply

        Returns:
            array[complex]: the output state tensor
        """
        assert state.dtype == np.complex128
        state_vector = np.ravel(state)
        sim = StateVectorC128(state_vector)

        for o in operations:
            name = o.name.split(".")[0]  # The split is because inverse gates have .inv appended
            method = getattr(sim, name, None)

            wires = self.wires.indices(o.wires)

            if method is None:
                # Inverse can be set to False since o.matrix is already in inverted form
                sim.applyMatrix(o.matrix, wires, False)
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
            if not isinstance(m.obs, qml.operation.Tensor):
                if isinstance(m.obs, qml.Projector):
                    raise QuantumFunctionError(
                        "Adjoint differentiation method does not support the Projector observable"
                    )
                if isinstance(m.obs, qml.Hermitian):
                    raise QuantumFunctionError(
                        "Lightning adjoint differentiation method does not currently support the Hermitian observable"
                    )
            else:
                if any([isinstance(o, qml.Projector) for o in m.obs.non_identity_obs]):
                    raise QuantumFunctionError(
                        "Adjoint differentiation method does not support the Projector observable"
                    )
                if any([isinstance(o, qml.Hermitian) for o in m.obs.non_identity_obs]):
                    raise QuantumFunctionError(
                        "Lightning adjoint differentiation method does not currently support the Hermitian observable"
                    )

        for op in tape.operations:
            if (
                op.num_params > 1 and not isinstance(op, qml.Rot)
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

        adj = AdjointJacobianC128()

        obs_serialized = _serialize_obs(tape, self.wire_map)
        ops_serialized, use_sp = _serialize_ops(tape, self.wire_map)

        ops_serialized = adj.create_ops_list(*ops_serialized)

        trainable_params = sorted(tape.trainable_params)
        first_elem = 1 if trainable_params[0] == 0 else 0

        tp_shift = (
            trainable_params if not use_sp else [i - 1 for i in trainable_params[first_elem:]]
        )  # exclude first index if explicitly setting sv

        jac = adj.adjoint_jacobian(
            StateVectorC128(ket),
            obs_serialized,
            ops_serialized,
            tp_shift,
            tape.num_params,
        )
        return jac

    def vector_jacobian_product(self, tape, dy, starting_state=None, use_device_state=False):
        """Generate the the vector-Jacobian products of a tape.

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
            tuple[array or None, tensor_like or None]: A tuple of the adjoint-jacobian and the Vector-Jacobian
            product. Returns ``None`` if the tape has no trainable parameters.
        """
        if self.shots is not None:
            warn(
                "Requested adjoint differentiation to be computed with finite shots."
                " The derivative is always exact when using the adjoint differentiation method.",
                UserWarning,
            )

        num_params = len(tape.trainable_params)

        if num_params == 0:
            return None, None

        if math.allclose(dy, 0):
            return None, math.convert_like(np.zeros([num_params]), dy)

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

        VJP = VectorJacobianProductC128()

        obs_serialized = _serialize_obs(tape, self.wire_map)
        ops_serialized, use_sp = _serialize_ops(tape, self.wire_map)

        ops_serialized = VJP.create_ops_list(*ops_serialized)

        trainable_params = sorted(tape.trainable_params)
        first_elem = 1 if trainable_params[0] == 0 else 0

        tp_shift = (
            trainable_params if not use_sp else [i - 1 for i in trainable_params[first_elem:]]
        )  # exclude first index if explicitly setting sv

        jac, vjp = VJP.vjp(
            math.reshape(dy, [-1]),
            StateVectorC128(ket),
            obs_serialized,
            ops_serialized,
            tp_shift,
            tape.num_params,
        )
        return jac, vjp

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

        dy_row = math.reshape(dy, [-1])

        if num is None:
            num = math.shape(dy_row)[0]

        if not isinstance(dy_row, np.ndarray):
            jac = math.convert_like(jac, dy_row)

        jac = math.reshape(jac, [num, -1])
        num_params = jac.shape[1]

        if math.allclose(dy, 0):
            return math.convert_like(np.zeros([num_params]), dy)

        VJP = VectorJacobianProductC128()

        vjp_tensor = VJP.compute_vjp_from_jac(
            math.reshape(jac, [-1]),
            dy_row,
            num,
            num_params,
        )
        return vjp_tensor

    def batch_vjp(
        self, tapes, dys, reduction="append", starting_state=None, use_device_state=False
    ):
        """Generate the the vector-Jacobian products of a batch of tapes.

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
            tuple[List[array or None], List[tensor_like or None]]: A tuple containing a list
            of adjoint-jacobians and a list of vector-Jacobian products. ``None`` elements corresponds
            to tapes with no trainable parameters.
        """
        vjps = []
        jacs = []

        # Loop through the tapes and dys vector
        for tape, dy in zip(tapes, dys):
            jac, vjp = self.vector_jacobian_product(
                tape,
                dy,
                starting_state=starting_state,
                use_device_state=use_device_state,
            )
            if vjp is None:
                if reduction == "append":
                    vjps.append(None)
                    jacs.append(jac)
                continue
            if isinstance(reduction, str):
                getattr(vjps, reduction)(vjp)
                getattr(jacs, reduction)(jac)
            elif callable(reduction):
                reduction(vjps, vjp)
                reduction(jacs, jac)

        return jacs, vjps


if not CPP_BINARY_AVAILABLE:

    class LightningQubit(DefaultQubit):

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
