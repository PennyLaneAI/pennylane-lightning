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
Class implementation for MPS manipulation based on the `quimb` Python package.
"""

import numpy as np
import quimb.tensor as qtn
from pennylane.wires import Wires

# TODO: understand if supporting all operations and observables is feasible for the first release
# I comment the following lines since otherwise Codecov complaints

# _operations = frozenset({})
# The set of supported operations.

# _observables = frozenset({})
# The set of supported observables.


class QuimbMPS:
    """Quimb MPS class.

    Interfaces with `quimb` for MPS manipulation.
    """

    # TODO: add more details in description during implementation phase (next PR)

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
            "rehearse": interf_opts["rehearse"],
        }

        self._return_tn = interf_opts["return_tn"]

        self._circuitMPS = qtn.CircuitMPS(psi0=self._initial_mps())

    @property
    def state(self):
        """Current MPS handled by the interface."""
        return self._circuitMPS.psi

    def state_to_array(self, digits: int = 5):
        """Contract the MPS into a dense array."""
        return self._circuitMPS.to_dense().round(digits)

    def _initial_mps(self):
        r"""
        Returns an initial state to :math:`\ket{0}`.

        Returns:
            array: The initial state of a circuit.
        """

        return qtn.MPS_computational_state(**self._init_state_ops)
