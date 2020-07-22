# Copyright 2020 Xanadu Quantum Technologies Inc.

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
The default plugin is meant to be used as a template for writing PennyLane device
plugins for new qubit-based backends.

It implements the necessary :class:`~pennylane._device.Device` methods as well as some built-in
:mod:`qubit operations <pennylane.ops.qubit>`, and provides a very simple pure state
simulation of a qubit-based quantum circuit architecture.
"""
from pennylane.plugins import DefaultQubit
from .src.lightning_qubit_ops import apply_2q
import numpy as np


class LightningQubit(DefaultQubit):
    """TODO"""

    def apply(self, operations, rotations=None, **kwargs):
        # super().apply(operations, rotations, **kwargs)
        if rotations:  # We should support this!
            raise NotImplementedError("Rotations not yet supported")

        op_names = [o.name for o in operations]
        op_wires = [o.wires for o in operations]
        op_param = [o.params for o in operations]
        state_vector = np.ravel(self._state, order="F")
        state_vector_updated = apply_2q(state_vector, op_names, op_wires, op_param)
        self._state = np.reshape(state_vector_updated, self._state.shape, order="F")
        self._pre_rotated_state = self._state  # TODO
