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
"""
Internal methods for adjoint Jacobian differentiation method.
"""
from typing import List

import numpy as np
from pennylane import (
    BasisState,
    QubitStateVector,
    Projector,
    Rot,
    QuantumFunctionError,
)
from pennylane.operation import Tensor, Operation
from pennylane.measurements import MeasurementProcess, Expectation, State
from pennylane.tape import QuantumTape
from pennylane.devices.qubit.initialize_state import create_initial_state
from pennylane import active_return

from ._serialize import _serialize_observables, _serialize_ops
from ._simulate import asarray
from ._apply_operations import apply_operations

from ..lightning_qubit_ops import (
    adjoint_diff,
    StateVectorC64,
    StateVectorC128,
)


def _check_supported_measurements(measurements: List[MeasurementProcess]):
    """Check whether given list of measurement is supported by adjoint_diff.

    Args:
        measurements (List[MeasurementProcess]): a list of measurement processes to check.

    Raises:
        QuantumFunctionError: if adjoint method is not supported for any measurement
    """
    if len(measurements) == 1 and measurements[0].return_type is State:
        raise QuantumFunctionError(
            "This method does not support statevector return type. "
            "Use vjp method instead for this purpose."
        )

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


def _check_supported_operations(operations):
    """Check Lightning adjoint differentiation method support for a tape.

    Args:
        operations (List): list with operations

    Raises:
        QuantumFunctionError:  if ``tape`` contains observables or operations,
    not supported by the Lightning adjoint differentiation method.
    """

    for op in operations:
        if op.num_params > 1 and not isinstance(op, Rot):
            raise QuantumFunctionError(
                f"The {op.name} operation is not supported using "
                'the "adjoint" differentiation method'
            )


def _check_adjoint_method_supported(tape: QuantumTape):
    """Check measurement and operation lists for adjoint Jacobian support in Lightning.

    Args:
        tape (QuantumTape): A quantum tape recording a variational quantum program.
    """
    _check_supported_measurements(tape.measurements)
    _check_supported_operations(tape.operations)


def _process_jacobian_tape(tape, state):
    """Process a Jacobian tape before calculation.

    Args:
        tape (QuantumTape): A quantum tape recording a variational quantum program.
        state(np.array): unravelled initial state (1D)

    Returns:
        Set: A set with a tape processed for Lightning.
    """
    use_csingle = True if state.dtype == np.complex64 else False
    # To support np.complex64 based on the type of self._state
    create_ops_list = (
        adjoint_diff.create_ops_list_C64 if use_csingle else adjoint_diff.create_ops_list_C128
    )

    obs_serialized = _serialize_observables(tape, use_csingle=use_csingle)
    ops_serialized, use_sp = _serialize_ops(tape)

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
            op_idx, False
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

    state_vector = StateVectorC64(state) if use_csingle else StateVectorC128(state)
    return {
        "state_vector": state_vector,
        "obs_serialized": obs_serialized,
        "ops_serialized": ops_serialized,
        "tp_shift": tp_shift,
        "record_tp_rows": record_tp_rows,
        "all_params": all_params,
    }


def _adjoint_jacobian_processing(jac):
    """
    Post-process the Jacobian matrix returned by ``adjoint_jacobian`` for
    the new return type system.
    """
    jac = np.squeeze(jac)

    if jac.ndim == 0:
        return np.array(jac)

    if jac.ndim == 1:
        return tuple(np.array(j) for j in jac)

    # must be 2-dimensional
    return tuple(tuple(np.array(j_) for j_ in j) for j in jac)


def adjoint_jacobian(tape, c_dtype=np.complex128):
    """Calculates the Adjoint Jacobian for a given tape.

    Args:
        tape (QuantumTape): A quantum tape recording a variational quantum program.
        c_dtype (Complex data type, Optional): Default to ``np.complex128``.
        starting_state (np.array, Optional): unravelled initial state (1D). Default to None.

    Returns:
        np.array: An array results.
    """
    state = create_initial_state(tape.wires, tape._prep[0] if tape._prep else None)
    state = np.ravel(asarray(state, c_dtype))
    state = apply_operations(tape._ops, state)

    if len(tape.measurements) == 0:  # the tape does not have measurements
        return np.array([], dtype=state.dtype)

    processed_data = _process_jacobian_tape(tape, state)

    if not processed_data:  # training_params is empty
        return np.array([], dtype=state.dtype)

    trainable_params = processed_data["tp_shift"]

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
    return _adjoint_jacobian_processing(jac_r) if active_return() else jac_r
