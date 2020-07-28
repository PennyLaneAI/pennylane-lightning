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
from .src.lightning_qubit_ops import apply
import numpy as np
from pennylane import QubitStateVector, BasisState, DeviceError


class LightningQubit(DefaultQubit):
    """TODO"""

    _capabilities = {"inverse_operations": False}  # we should look at supporting

    operations = {
        "BasisState",
        "QubitStateVector",
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "S",
        "T",
        "CNOT",
        "SWAP",
        "CSWAP",
        "Toffoli",
        "CZ",
        "PhaseShift",
        "RX",
        "RY",
        "RZ",
        "Rot",
        "CRX",
        "CRY",
        "CRZ",
        "CRot",
    }

    def apply(self, operations, rotations=None, **kwargs):

        for i, operation in enumerate(operations):  # State preparation is currently done in Python
            if isinstance(operation, (QubitStateVector, BasisState)):
                if i == 0:
                    self._apply_operation(operation)
                    del operations[0]
                else:
                    raise DeviceError(
                        "Operation {} cannot be used after other Operations have already been "
                        "applied on a {} device.".format(operation.name, self.short_name)
                    )

        self._pre_rotated_state = self.apply_lightning(self._state, operations)
        if rotations:
            self._state = self.apply_lightning(self._pre_rotated_state, rotations)
        else:
            self._state = self._pre_rotated_state

    def apply_lightning(self, state, operations):
        """TODO"""
        op_names = [o.name for o in operations]
        op_wires = [o.wires for o in operations]
        op_param = [o.parameters for o in operations]
        state_vector = np.ravel(state, order="F")
        state_vector_updated = apply(state_vector, op_names, op_wires, op_param, self.num_wires)
        return np.reshape(state_vector_updated, state.shape, order="F")
