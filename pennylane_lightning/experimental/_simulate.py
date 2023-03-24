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
Internal methods to simulate a quantum script.
"""

import numpy as np
from typing import Union

import pennylane as qml

from pennylane.tape import QuantumScript
from pennylane.devices.qubit.initialize_state import create_initial_state
from pennylane.typing import TensorLike

from ._measure import measure
from ._apply_operations import apply_operations

from ..lightning_qubit_ops import (
    allocate_aligned_array,
    get_alignment,
    best_alignment,
)


def asarray(arr, dtype=np.complex128):
    """Convert the input to a numpy array.
       Data is guaranteed to be aligned.

    Args:
        arr (array[complex]): input float or complex array
        dtype (numpy data type, optional): array data type. Defaults to ``np.complex128``.

    Returns:
        array[dtype]: an array with data aligned in memory.
    """
    arr = np.asarray(arr)  # arr is not copied

    # We allocate a new aligned memory and copy data to there if alignment or dtype mismatches
    # Note that get_alignment does not necessarily return CPUMemoryModel(Unaligned) even for
    # numpy allocated memory as the memory location happens to be aligned.
    if int(get_alignment(arr)) < int(best_alignment()) or arr.dtype != dtype:
        new_arr = allocate_aligned_array(arr.size, np.dtype(dtype)).reshape(arr.shape)
        np.copyto(new_arr, arr)
        arr = new_arr
    return arr


def _execute_single_script(circuit: QuantumScript, c_dtype=np.complex128) -> Union[tuple, TensorLike]:
    """Execute a single quantum script [Internal Function].

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        c_dtype: Complex data type. Default to ``np.complex128``.

    Returns:
        tuple(TensorLike): The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.

    It does currently not support sampling or observables without diagonalizing gates.

    This function assumes that all operations provide matrices.

    >>> qs = qml.tape.QuantumScript([qml.RX(1.2, wires=0)], [qml.expval(qml.PauliZ(0)), qml.probs(wires=(0,1))])
    >>> _execute_single_script(qs)
    (0.36235775447667357,
    tensor([0.68117888, 0.        , 0.31882112, 0.        ], requires_grad=True))

    """
    if set(circuit.wires) != set(range(circuit.num_wires)):
        wire_map = {w: i for i, w in enumerate(circuit.wires)}
        circuit = qml.map_wires(circuit, wire_map)

    state = create_initial_state(circuit.wires, circuit._prep[0] if circuit._prep else None)
    state = np.ravel(asarray(state, c_dtype))
    state = apply_operations(circuit._ops, state)

    if len(circuit.measurements) == 1:
        return measure(circuit.measurements[0], state)

    measures = []
    for mp in circuit.measurements:
        measure_val = measure(mp, state)
        measures += [
            measure_val,
        ]
    return tuple(measures)
