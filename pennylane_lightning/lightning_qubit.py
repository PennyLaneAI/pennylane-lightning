# Copyright 2021 Xanadu Quantum Technologies Inc.

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
This module contains the :class:`~.LightningQubit` class, a PennyLane simulator device that
interfaces with C++ for fast linear algebra calculations.
"""
from pennylane.devices import DefaultQubit
from .lightning_qubit_ops import apply, adjoint_jacobian
import numpy as np
from pennylane import QubitStateVector, BasisState, DeviceError, QubitUnitary, QuantumFunctionError
from pennylane.operation import Expectation

from ._version import __version__


class LightningQubit(DefaultQubit):
    """PennyLane Lightning device.

    An extension of PennyLane's built-in ``default.qubit`` device that interfaces with C++ to
    perform fast linear algebra calculations.

    Use of this device requires pre-built binaries or compilation from source. Check out the
    :doc:`/installation` guide for more details.

    Args:
        wires (int): the number of wires to initialize the device with
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically.
    """

    name = "Lightning Qubit PennyLane plugin"
    short_name = "lightning.qubit"
    pennylane_requires = ">=0.15"
    version = __version__
    author = "Xanadu Inc."

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

    observables = {"PauliX", "PauliY", "PauliZ", "Hadamard", "Hermitian", "Identity"}

    def __init__(self, wires, *, shots=None):
        super().__init__(wires, shots=shots)

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qubit",
            supports_reversible_diff=False,
            supports_inverse_operations=True,
            supports_analytic_computation=True,
            returns_state=True,
        )
        capabilities.pop("passthru_devices", None)
        return capabilities

    def apply(self, operations, rotations=None, **kwargs):

        # State preparation is currently done in Python
        if operations:  # make sure operations[0] exists
            if isinstance(operations[0], QubitStateVector):
                self._apply_state_vector(operations[0].parameters[0].copy(), operations[0].wires)
                del operations[0]
            elif isinstance(operations[0], BasisState):
                self._apply_basis_state(operations[0].parameters[0], operations[0].wires)
                del operations[0]

        for operation in operations:
            if isinstance(operation, (QubitStateVector, BasisState)):
                raise DeviceError(
                    "Operation {} cannot be used after other Operations have already been "
                    "applied on a {} device.".format(operation.name, self.short_name)
                )

        if operations:
            self._pre_rotated_state = self.apply_lightning(self._state, operations)
        else:
            self._pre_rotated_state = self._state

        if rotations:
            if any(isinstance(r, QubitUnitary) for r in rotations):
                super().apply(operations=[], rotations=rotations)
            else:
                self._state = self.apply_lightning(np.copy(self._pre_rotated_state), rotations)
        else:
            self._state = self._pre_rotated_state

    def apply_lightning(self, state, operations):
        """Apply a list of operations to the state tensor.

        Args:
            state (array[complex]): the input state tensor
            operations (list[~pennylane.operation.Operation]): operations to apply

        Returns:
            array[complex]: the output state tensor
        """
        op_names = [self._remove_inverse_string(o.name) for o in operations]
        op_wires = [self.wires.indices(o.wires) for o in operations]
        op_param = [o.parameters for o in operations]
        op_inverse = [o.inverse for o in operations]

        state_vector = np.ravel(state)
        apply(state_vector, op_names, op_wires, op_param, op_inverse, self.num_wires)
        return np.reshape(state_vector, state.shape)

    def adjoint_jacobian(self, tape):
        for m in tape.measurements:
            if m.return_type is not Expectation:
                raise QuantumFunctionError(
                    "Adjoint differentiation method does not support"
                    f" measurement {m.return_type.value}"
                )

            if not hasattr(m.obs, "base_name"):
                m.obs.base_name = None  # This is needed for when the observable is a tensor product

        param_number = len(tape._par_info) - 1

        def _unwrap(param_list):
            for i, l in enumerate(param_list):
                try:
                    param_list[i] = l.unwrap()
                    if isinstance(param_list[i], (int, float)):
                        param_list[i] = [float(param_list[i])]
                    else:
                        param_list[i] = list(param_list[i])
                except AttributeError:
                    continue
            # if len(param_list) == 1:
            #     return param_list[0]
            return param_list

        op_data = [(self._remove_inverse_string(op.name), _unwrap(op.parameters), op.wires.tolist()) for op in tape.operations]
        operations, op_params, op_wires = map(list, zip(*op_data))

        obs_data = [(self._remove_inverse_string(obs.name), _unwrap(obs.parameters), obs.wires.tolist()) for obs in tape.observables]
        observables, obs_params, obs_wires = map(list, zip(*obs_data))

        trainable_params = list(tape.trainable_params)

        # send in flattened array of zeros to be populated by adjoint_jacobian
        jac = np.zeros(len(tape.observables) * len(tape.trainable_params))
        adjoint_jacobian(
            self.state,       # numpy.ndarray[numpy.complex128]
            jac,              # numpy.ndarray[numpy.float64]
            observables,      # List[str]
            obs_params,       # List[List[float]]
            obs_wires,        # List[List[int]]
            operations,       # List[str]
            op_params,        # List[List[float]]
            op_wires,         # List[List[int]]
            trainable_params, # List[int]
            param_number,     # int
        )
        return jac.reshape((len(tape.observables), len(tape.trainable_params)))

    @staticmethod
    def _remove_inverse_string(string):
        """Removes the ``.inv`` appended to the end of inverse gates.

        Args:
            string (str): name of operation

        Returns:
            str: name of operation with ``.inv`` removed (if present)
        """
        return string.replace(".inv", "")
