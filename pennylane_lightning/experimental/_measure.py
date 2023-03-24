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
Internal code relevant for performing measurements on a state.
"""

import numpy as np
from typing import Callable, List

from pennylane.measurements import StateMeasurement, MeasurementProcess, ExpectationMP
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ._apply_operations import apply_operations

from ..lightning_qubit_ops import (
    MeasuresC64,
    StateVectorC64,
    MeasuresC128,
    StateVectorC128,
    Kokkos_info,
)

from ._serialize import _serialize_ob


def state_diagonalizing_gates(measurementprocess: StateMeasurement, state: np.array) -> TensorLike:
    """Apply a measurement to state when the measurement process has an observable with diagonalizing gates.

    Args:
        measurementprocess (StateMeasurement): measurement to apply to the state
        state (np.array): unravelled state (1D) to apply the measurement to

    Returns:
        TensorLike: the result of the measurement
    """
    state = apply_operations(measurementprocess.diagonalizing_gates(), np.copy(state))

    total_wires = int(np.log2(state.size))
    wires = Wires(range(total_wires))
    return measurementprocess.process_state(state, wires)


def expval(measurementprocess: MeasurementProcess, state: np.array):
    """Expectation value of the supplied observable contained in the MeasurementProcess.

    Args:
        measurementprocess (StateMeasurement): measurement to apply to the state
        state (np.array): unravelled state (1D) to apply the measurement to

    Returns:
        Expectation value of the observable
    """
    if state.dtype == np.complex64:
        state_vector = StateVectorC64(state)
        M = MeasuresC64(state_vector)
    else:
        state_vector = StateVectorC128(state)
        M = MeasuresC128(state_vector)

    if measurementprocess.obs.name == "SparseHamiltonian":
        if Kokkos_info()["USE_KOKKOS"] == True:
            # ensuring CSR sparse representation.
            total_wires = int(np.log2(state.size))
            CSR_SparseHamiltonian = measurementprocess.obs.sparse_matrix(
                wire_order=list(range(total_wires))
            ).tocsr(copy=False)
            return M.expval(
                CSR_SparseHamiltonian.indptr,
                CSR_SparseHamiltonian.indices,
                CSR_SparseHamiltonian.data,
            )
        raise NotImplementedError(
            "The expval of a SparseHamiltonian requires Kokkos and Kokkos Kernels."
        )

    if (
        measurementprocess.obs.name in ["Hamiltonian", "Hermitian"]
        or (measurementprocess.obs.arithmetic_depth > 0)
        or isinstance(measurementprocess.obs.name, List)
    ):
        ob_serialized = _serialize_ob(measurementprocess.obs, state.dtype == np.complex64)
        return M.expval(ob_serialized)

    return M.expval(measurementprocess.obs.name, measurementprocess.obs.wires)


def get_measurement_function(
    measurementprocess: MeasurementProcess, state: np.array
) -> Callable[[MeasurementProcess, TensorLike], TensorLike]:
    """Get the appropriate method for performing a measurement.

    Args:
        measurementprocess (MeasurementProcess): measurement process to apply to the state
        state (np.array): unravelled state (1D) to be measured

    Returns:
        Callable: function that returns the measurement result
    """
    if isinstance(measurementprocess, StateMeasurement):
        if isinstance(measurementprocess, ExpectationMP):
            if measurementprocess.obs.name in [
                "Identity",
                "Projector",
            ]:
                return state_diagonalizing_gates
            return expval

        if measurementprocess.obs is None or measurementprocess.obs.has_diagonalizing_gates:
            return state_diagonalizing_gates

    raise NotImplementedError


def measure(measurementprocess: MeasurementProcess, state: np.array) -> TensorLike:
    """Apply a measurement process to a state.

    Args:
        measurementprocess (MeasurementProcess): measurement process to apply to the state
        state (np.array): unravelled state (1D) to be measured

    Returns:
        TensorLike: the result of the measurement
    """
    return get_measurement_function(measurementprocess, state)(measurementprocess, state)
