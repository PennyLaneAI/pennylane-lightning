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

import pennylane as qml

from pennylane.tape import QuantumScript
from pennylane.devices.qubit.initialize_state import create_initial_state
from pennylane.typing import TensorLike

from ._measurements import measurement
from ._apply_operations import apply_operations

try:
    from pennylane_lightning.lightning_qubit_ops import (
        allocate_aligned_array,
        get_alignment,
        best_alignment,
    )
except ImportError:
    pass

from pennylane.typing import Result


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


def get_final_state(circuit: QuantumScript, c_dtype=np.complex128, debugger=None):
    """
    Get the final state that results from executing the given quantum script.

    This is an internal function that will be called by the successor to ``lightning.qubit``.

    Args:
        circuit (QuantumScript): The single circuit to simulate
        debugger (._Debugger): The debugger to use

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """
    circuit = circuit.map_to_standard_wires()

    prep = None
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        prep = circuit[0]

    state = create_initial_state(circuit.wires, prep)
    state = np.ravel(asarray(state, c_dtype))
    state = apply_operations(circuit._ops, state, prep)

    # # initial state is batched only if the state preparation (if it exists) is batched
    # is_state_batched = False
    # if prep and prep.batch_size is not None:
    #     is_state_batched = True

    # for op in circuit.operations[bool(prep) :]:
    #     state = apply_operations(op, state)

    #     # new state is batched if i) the old state is batched, or ii) the new op adds a batch dim
    #     is_state_batched = is_state_batched or op.batch_size is not None

    return state, False  # is_state_batched


def measure_final_state(
    circuit: QuantumScript, state: TensorLike, is_state_batched, c_dtype=np.complex128, rng=None
) -> Result:
    """
    Perform the measurements required by the circuit on the provided state.

    This is an internal function that will be called by the successor to ``lightning.qubit``.

    Args:
        circuit (QuantumScript): The single circuit to simulate
        state (TensorLike): The state to perform measurement on
        is_state_batched (bool): Whether the state has a batch dimension or not.
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.

    Returns:
        Tuple[TensorLike]: The measurement results
    """
    if set(circuit.wires) != set(range(circuit.num_wires)):
        wire_map = {w: i for i, w in enumerate(circuit.wires)}
        circuit = qml.map_wires(circuit, wire_map)

    if not circuit.shots:
        # analytic case
        if len(circuit.measurements) == 1:
            return measurement(circuit.measurements[0], state)

        results = []
        for mp in circuit.measurements:
            measure_val = measurement(mp, state)
            results += [
                measure_val,
            ]
        return tuple(results)

    # # finite-shot case

    # rng = default_rng(rng)
    # results = measure_with_samples(
    #     circuit.measurements, state, shots=circuit.shots, is_state_batched=is_state_batched, rng=rng
    # )

    # if len(circuit.measurements) == 1:
    #     if circuit.shots.has_partitioned_shots:
    #         return tuple(res[0] for res in results)

    #     return results[0]
