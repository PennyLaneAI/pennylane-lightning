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
Class implementation for the Quimb MPS interface for simulating quantum circuits while keeping the state always in MPS form.
"""

import numpy as np
import pennylane as qml
import quimb.tensor as qtn
from pennylane.wires import Wires

_operations = frozenset({})  # pragma: no cover
# The set of supported operations.

_observables = frozenset({})  # pragma: no cover
# The set of supported observables.


def stopping_condition(op: qml.operation.Operator) -> bool:
    """A function that determines if an operation is supported by ``lightning.tensor`` for this interface."""
    return op.name in _operations  # pragma: no cover


def accepted_observables(obs: qml.operation.Operator) -> bool:
    """A function that determines if an observable is supported by ``lightning.tensor`` for this interface."""
    return obs.name in _observables  # pragma: no cover


class QuimbMPS:
    """Quimb MPS class.

    Used internally by the `LightningTensor` device.
    Interfaces with `quimb` for MPS manipulation, and provides methods to execute quantum circuits.

    Args:
        num_wires (int): the number of wires in the circuit.
        interf_opts (dict): dictionary containing the interface options.
        dtype (np.dtype): the complex type used for the MPS.
    """

    def __init__(self, num_wires, interf_opts, dtype=np.complex128):

        if dtype not in [np.complex64, np.complex128]:  # pragma: no cover
            raise TypeError(f"Unsupported complex type: {dtype}")

        self._wires = Wires(range(num_wires))
        self._dtype = dtype

        self._init_state_ops = {
            "binary": "0" * max(1, len(self._wires)),
            "dtype": self._dtype.__name__,
            "tags": [str(l) for l in self._wires.labels],
        }

        self._gate_opts = {
            "contract": "swap+split",
            "parametrize": None,
            "cutoff": interf_opts["cutoff"],
            "max_bond": interf_opts["max_bond_dim"],
        }

        self._expval_opts = {
            "dtype": self._dtype.__name__,
            "simplify_sequence": "ADCRS",
            "simplify_atol": 0.0,
        }

        self._circuitMPS = qtn.CircuitMPS(psi0=self._initial_mps())

    @property
    def state(self):
        """Current MPS handled by the interface."""
        return self._circuitMPS.psi

    def state_to_array(self) -> np.ndarray:
        """Contract the MPS into a dense array."""
        return self._circuitMPS.to_dense()

    def _initial_mps(self) -> qtn.MatrixProductState:
        r"""
        Returns an initial state to :math:`\ket{0}`.

        Internally, it uses `quimb`'s `MPS_computational_state` method.

        Returns:
            MatrixProductState: The initial MPS of a circuit.
        """

        return qtn.MPS_computational_state(**self._init_state_ops)
