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
from pennylane import DeviceError
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

    def __init__(
        self,
        num_wires,
        dtype=np.complex128,
        device_name="lightning.tensor",
        **kwargs,
    ):

        if dtype not in [np.complex64, np.complex128]:  # pragma: no cover
            raise TypeError(f"Unsupported complex type: {dtype}")

        if device_name != "lightning.tensor":
            raise DeviceError(f'The device name "{device_name}" is not a valid option.')

        self._device_name = device_name
        self._num_wires = num_wires
        self._wires = Wires(range(num_wires))
        self._dtype = dtype

        # options for MPS
        self._max_bond_dim = kwargs.get("max_bond_dim", None)
        self._cutoff = kwargs.get("cutoff", 1e-16)
        self._measure_algorithm = kwargs.get("measure_algorithm", None)

        # TODO: allows users to specify initial state
        self._circuit = qtn.CircuitMPS(psi0=self._set_initial_mps())

    @property
    def state(self):
        """Current MPS handled by the device."""
        return self._circuit.psi

    def state_to_array(self, digits: int = 5):
        """Contract the MPS into a dense array."""
        return self._circuit.to_dense().round(digits)

    def _set_initial_mps(self):
        r"""
        Returns an initial state to :math:`\ket{0}`.

        Args:
            wires (Union[Wires, Iterable]): The wires to be present in the initial state.

        Returns:
            array: The initial state of a circuit.
        """

        return qtn.MPS_computational_state(
            "0" * max(1, self._num_wires),
            dtype=self._dtype.__name__,
            tags=[str(l) for l in self._wires.labels],
        )
