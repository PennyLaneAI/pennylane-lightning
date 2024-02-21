# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
This module contains the LightningQubit2 class that inherits from the new device interface.

"""
from typing import Union, Sequence, Optional
from dataclasses import replace
import numpy as np

import pennylane as qml


from pennylane.tape import QuantumScript
from pennylane.typing import Result

from ._state_vector import LightningStateVector
from ._measurements import LightningMeasurements


try:
    # pylint: disable=import-error, no-name-in-module
    from pennylane_lightning.lightning_qubit_ops import (
        StateVectorC64,
        StateVectorC128,
    )

    LQ_CPP_BINARY_AVAILABLE = True
except ImportError:
    LQ_CPP_BINARY_AVAILABLE = False


def simulate(circuit: QuantumScript, state: LightningStateVector, dtype=np.complex128) -> Result:
    """Simulate a single quantum script.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        state (LightningStateVector): handle to Lightning state vector
        dtype: Datatypes for state-vector representation. Must be one of
            ``np.complex64`` or ``np.complex128``.

    Returns:
        tuple(TensorLike): The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.

    """
    state.reset_state()
    final_state = state.get_final_state(circuit)
    return LightningMeasurements(final_state).measure_final_state(circuit)


def dummy_jacobian(circuit: QuantumScript):
    return np.array(0.0)


def simulate_and_jacobian(circuit: QuantumScript):
    return np.array(0.0), np.array(0.0)
