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
import itertools

import numpy as np

from pennylane import QubitDevice, DeviceError, QubitStateVector, BasisState
# from .lightning_qubit_utils import mvp
from .lightning_qubit_ops import mvp
from . import lightning_ops


# tolerance for numerical errors
tolerance = 1e-10


class LightningQubit(QubitDevice):
    """Default qubit device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to 1000 if not specified.
            If ``analytic == True``, then the number of shots is ignored
            in the calculation of expectation values and variances, and only controls the number
            of samples returned by ``sample``.
        analytic (bool): indicates if the device should calculate expectations
            and variances analytically
    """

    name = "Default qubit PennyLane plugin"
    short_name = "lightning.qubit"
    pennylane_requires = ">=0.9.0"
    version = "0.9.0"
    author = "Xanadu Inc."
    _capabilities = {"inverse_operations": True}

    operations = {
        "BasisState",
        "QubitStateVector",
        "QubitUnitary",
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

    def __init__(self, wires, *, shots=1000, analytic=True):
        self.eng = None
        self.analytic = analytic

        self._state = np.zeros(2 ** wires, dtype=complex)
        self._state[0] = 1
        self._state = np.reshape(self._state, [2] * wires)
        self._pre_rotated_state = self._state


        super().__init__(wires, shots, analytic)

    def apply(self, operations, rotations=None, **kwargs):
        rotations = rotations or []

        # apply the circuit operations
        for i, operation in enumerate(operations):
            # number of wires on device
            wires = operation.wires
            par = operation.params

            if i > 0 and operation.name in ("QubitStateVector", "BasisState"):
                raise DeviceError(
                    "Operation {} cannot be used after other Operations have already been applied "
                    "on a {} device.".format(operation.name, self.short_name)
                )

            if operation.name is "QubitStateVector":
                input_state = np.asarray(par[0], dtype=np.complex128)
                self.apply_state_vector(input_state, wires)

            elif operation.name is "BasisState":
                basis_state = par[0]
                self.apply_basis_state(basis_state, wires)

            else:
                matrix_tensor = self._get_matrix_tensor(operation)
                self._state = mvp(matrix_tensor, self._state, wires)

        # store the pre-rotated state
        self._pre_rotated_state = self._state

        # apply the circuit rotations
        for operation in rotations:
            wires = operation.wires

            matrix_tensor = self._get_matrix_tensor(operation)
            self._state = mvp(matrix_tensor, self._state, wires)

    def _get_matrix_tensor(self, operation):

        if operation.parameters:
            if not operation.inverse:
                op = getattr(lightning_ops, operation.name)
                lightning_op = op(*operation.parameters, wires=operation.wires)
            else:
                op = getattr(lightning_ops, operation.name[:-4])
                lightning_op = op(*operation.parameters, wires=operation.wires).inv()
        else:
            if not operation.inverse:
                op = getattr(lightning_ops, operation.name)
                lightning_op = op(wires=operation.wires)
            else:
                op = getattr(lightning_ops, operation.name[:-4])
                lightning_op = op(wires=operation.wires).inv()

        return lightning_op.matrix_tensor

    @property
    def state(self):
        return self._pre_rotated_state.reshape(2 ** self.num_wires)

    def apply_state_vector(self, input_state, wires):
        """Initialize the internal state vector in a specified state.

        Args:
            input_state (array[complex]): normalized input state of length
                ``2**len(wires)``
            wires (list[int]): list of wires where the provided state should
                be initialized
        """
        if not np.isclose(np.linalg.norm(input_state, 2), 1.0, atol=tolerance):
            raise ValueError("Sum of amplitudes-squared does not equal one.")

        n_state_vector = input_state.shape[0]

        if input_state.ndim == 1 and n_state_vector == 2 ** len(wires):
            # generate basis states on subset of qubits via the cartesian product
            basis_states = np.array(list(itertools.product([0, 1], repeat=len(wires))))

            # get basis states to alter on full set of qubits
            unravelled_indices = np.zeros((2 ** len(wires), self.num_wires), dtype=int)
            unravelled_indices[:, wires] = basis_states

            # get indices for which the state is changed to input state vector elements
            ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self.num_wires)
            s_temp = self._state.reshape(2 ** self.num_wires)
            s_temp = np.zeros_like(s_temp)
            s_temp[ravelled_indices] = input_state
            self._state = s_temp.reshape([2] * self.num_wires)
        else:
            raise ValueError("State vector must be of length 2**wires.")

    def apply_basis_state(self, state, wires):
        """Initialize the state vector in a specified computational basis state.

        Args:
            state (array[int]): computational basis state of shape ``(wires,)``
                consisting of 0s and 1s.
            wires (list[int]): list of wires where the provided computational state should
                be initialized
        """
        # length of basis state parameter
        n_basis_state = len(state)

        if not set(state).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

        if n_basis_state != len(wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        # get computational basis state number
        basis_states = 2 ** (self.num_wires - 1 - np.array(wires))
        num = int(np.dot(state, basis_states))

        s_temp = self._state.reshape(2 ** self.num_wires)
        s_temp = np.zeros_like(s_temp)
        s_temp[num] = 1.0
        self._state = s_temp.reshape([2] * self.num_wires)

    def mat_vec_product(self, mat_t, vec, wires, num_wires):
        r"""Apply multiplication of a matrix to subsystems of the quantum state.

        Args:
            mat_t (array): matrix to multiply in tensor form
            vec (array): state vector to multiply
            wires (Sequence[int]): target subsystems
            num_wires: total number of wires in circuit

        Returns:
            array: output vector after applying ``mat`` to input ``vec`` on specified subsystems
        """
        return mvp(mat_t, vec, wires)

    def reset(self):
        """Reset the device"""
        # init the state vector to |00..0>
        super().reset()

        s_temp = np.zeros(2 ** self.num_wires, dtype=complex)
        s_temp[0] = 1
        self._pre_rotated_state = s_temp.reshape([2] * self.num_wires)
        self._state = s_temp.reshape([2] * self.num_wires)

    def analytic_probability(self, wires=None):
        """Return the (marginal) analytic probability of each computational basis state."""
        if self._state is None:
            return None

        wires = wires or range(self.num_wires)
        s_temp = self._state.reshape(2 ** self.num_wires)
        prob = self.marginal_prob(np.abs(s_temp) ** 2, wires)
        return prob
