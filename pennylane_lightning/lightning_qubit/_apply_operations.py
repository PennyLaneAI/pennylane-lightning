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
from pennylane import matrix

try:
    from pennylane_lightning.lightning_qubit_ops import (
        StateVectorC64,
        StateVectorC128,
    )
except ImportError:
    pass


def apply_operations(operations, state, prep=False):
    """Apply a list of operations to the state tensor.

    Args:
        operations (List[ops] or None): A list containing operations to be applied.
        state (np.array): The starting state (1D).
        prep (bool): if the first operation is a state preparation.

    Returns:
        array[complex]: the output state.
    """
    sim = StateVectorC64(state) if state.dtype == np.complex64 else StateVectorC128(state)

    # Skip over identity operations instead of performing
    # matrix multiplication with the identity.
    skipped_ops = ["Identity"]
    for op in operations[bool(prep) :]:
        if op.name in skipped_ops:
            continue

        method = getattr(sim, op.name, None)

        wires = op.wires
        if method is None:
            # Inverse can be set to False since qml.matrix(o) is already in inverted form
            method = getattr(sim, "applyMatrix")
            method(matrix(op), wires, False)
        else:
            inv = False
            param = op.parameters
            method(wires, inv, param)

    return state
