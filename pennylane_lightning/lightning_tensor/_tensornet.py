# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Class implementation for tensornet manipulation.
"""

# pylint: disable=import-error, no-name-in-module, ungrouped-imports
try:
    from pennylane_lightning.lightning_tensor_ops import TensorNetC64, TensorNetC128
except ImportError:
    pass

from itertools import product

import numpy as np
import pennylane as qml
from pennylane import BasisState, DeviceError, StatePrep
from pennylane.ops.op_math import Adjoint
from pennylane.tape import QuantumScript
from pennylane.wires import Wires


def get_ranks(lst):
    # Create a sorted list of unique elements
    sorted_unique = sorted(set(lst), reverse=True)

    # Create a rank dictionary
    rank_dict = {value: rank for rank, value in enumerate(sorted_unique)}

    # Map the original list elements to their ranks
    return [rank_dict[value] for value in lst]


def svd_split(M, bond_dim):
    """SVD split a matrix into a matrix product state via numpy linalg."""
    U, S, Vd = np.linalg.svd(M, full_matrices=False)
    U = U @ np.diag(S)  # Append singular values to U
    bonds = len(S)
    Vd = Vd.reshape(bonds, 2, -1)
    U = U.reshape((-1, 2, bonds))

    # keep only chi bonds
    chi = np.min([bonds, bond_dim])
    U, S, Vd = U[:, :, :chi], S[:chi], Vd[:chi]
    return U, S, Vd


def split(M):
    U, S, Vd = np.linalg.svd(M, full_matrices=False)
    bonds = len(S)
    U = U @ np.diag(S)
    Vd = Vd.reshape(bonds, 2, 2, -1)
    U = U.reshape((-1, 2, 2, bonds))

    return U, Vd


def dense_to_mpo(psi, n_wires):
    Ms = []

    psi = np.reshape(psi, (2**2, -1))
    U, Vd = split(psi)  # psi[2, (2x2x..)] = U[4, mu] S[mu] Vd[mu, (2x2x2x..)]

    Ms.append(U)
    bondL = Vd.shape[0]
    psi = Vd

    for _ in range(n_wires - 2):
        psi = np.reshape(psi, (4 * bondL, -1))  # reshape psi[2 * bondL, (2x2x2...)]
        U, Vd = split(psi)  # psi[2, (2x2x..)] = U[2, mu] S[mu] Vd[mu, (2x2x2x..)]
        Ms.append(U)

        psi = Vd
        bondL = Vd.shape[0]

    Ms.append(Vd)

    return Ms


def dense_to_mps(psi, n_wires, bond_dim):
    """Convert a dense state vector to a matrix product state."""
    Ms = []

    psi = np.reshape(psi, (2, -1))  # split psi[2, 2, 2, 2..] = psi[2, (2x2x2...)]
    U, S, Vd = svd_split(psi, bond_dim)  # psi[2, (2x2x..)] = U[2, mu] S[mu] Vd[mu, (2x2x2x..)]

    Ms.append(U)
    bondL = Vd.shape[0]
    psi = Vd

    for _ in range(n_wires - 2):
        psi = np.reshape(psi, (2 * bondL, -1))  # reshape psi[2 * bondL, (2x2x2...)]
        U, S, Vd = svd_split(psi, bond_dim)  # psi[2, (2x2x..)] = U[2, mu] S[mu] Vd[mu, (2x2x2x..)]
        Ms.append(U)

        psi = Vd
        bondL = Vd.shape[0]

    Ms.append(Vd)

    return Ms


# pylint: disable=too-many-instance-attributes
class LightningTensorNet:
    """Lightning tensornet class.

    Interfaces with C++ python binding methods for tensornet manipulation.

    Args:
        num_wires(int): the number of wires to initialize the device with
        c_dtype: Datatypes for tensor network representation. Must be one of
            ``np.complex64`` or ``np.complex128``. Default is ``np.complex128``
        method(string): tensor network method. Options: ["mps"]. Default is "mps".
        max_bond_dim(int): maximum bond dimension for the tensor network
        cutoff(float): threshold for singular value truncation. Default is 0.
        cutoff_mode(string): singular value truncation mode. Options: ["rel", "abs"].
        device_name(string): tensor network device name. Options: ["lightning.tensor"]
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        num_wires,
        method: str = "mps",
        c_dtype=np.complex128,
        max_bond_dim: int = 128,
        cutoff: float = 0,
        cutoff_mode: str = "abs",
        device_name="lightning.tensor",
    ):
        self._num_wires = num_wires
        self._max_bond_dim = max_bond_dim
        self._method = method
        self._cutoff = cutoff
        self._cutoff_mode = cutoff_mode
        self._c_dtype = c_dtype

        if device_name != "lightning.tensor":
            raise DeviceError(f'The device name "{device_name}" is not a valid option.')

        if num_wires < 2:
            raise ValueError("Number of wires must be greater than 1.")

        self._device_name = device_name
        self._tensornet = self._tensornet_dtype()(self._num_wires, self._max_bond_dim)

    @property
    def dtype(self):
        """Returns the tensor network data type."""
        return self._c_dtype

    @property
    def device_name(self):
        """Returns the tensor network device name."""
        return self._device_name

    @property
    def num_wires(self):
        """Number of wires addressed on this device"""
        return self._num_wires

    @property
    def tensornet(self):
        """Returns a handle to the tensor network."""
        return self._tensornet

    @property
    def state(self):
        """Copy the state vector data to a numpy array."""
        state = np.zeros(2**self._num_wires, dtype=self.dtype)
        self._tensornet.getState(state)
        return state

    def _tensornet_dtype(self):
        """Binding to Lightning Managed tensor network C++ class.

        Returns: the tensor network class
        """
        return TensorNetC128 if self.dtype == np.complex128 else TensorNetC64

    def reset_state(self):
        """Reset the device's initial quantum state"""
        # init the quantum state to |00..0>
        self._tensornet.reset()

    def _preprocess_state_vector(self, state, device_wires):
        """Convert a specified state to a full internal state vector.

        Args:
            state (array[complex]): normalized input state of length ``2**len(wires)``
            device_wires (Wires): wires that get initialized in the state

        Returns:
            array[complex]: normalized input state of length ``2**len(wires)``
        """
        output_shape = [2] * self._num_wires
        # special case for integral types
        if state.dtype.kind == "i":
            state = np.array(state, dtype=self.dtype)

        if len(device_wires) == self._num_wires and Wires(sorted(device_wires)) == device_wires:
            return np.reshape(state, output_shape).ravel(order="C")

        # generate basis states on subset of qubits via the cartesian product
        basis_states = np.array(list(product([0, 1], repeat=len(device_wires))))

        # get basis states to alter on full set of qubits
        unravelled_indices = np.zeros((2 ** len(device_wires), self._num_wires), dtype=int)
        unravelled_indices[:, device_wires] = basis_states

        # get indices for which the state is changed to input state vector elements
        ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self._num_wires)

        # get full state vector to be factorized into MPS
        full_state = np.zeros(2**self._num_wires, dtype=self.dtype)
        for i in range(len(state)):
            full_state[ravelled_indices[i]] = state[i]
        return np.reshape(full_state, output_shape).ravel(order="C")

    def _apply_state_vector(self, state, device_wires: Wires):
        """Convert a specified state to MPS sites.
        Args:
            state (array[complex]): normalized input state of length ``2**len(wires)``
                or broadcasted state of shape ``(batch_size, 2**len(wires))``
            device_wires (Wires): wires that get initialized in the state
        """

        state = self._preprocess_state_vector(state, device_wires)

        M = dense_to_mps(state, self._num_wires, self._max_bond_dim)

        for i in range(len(M)):
            self._tensornet.updateIthSite(i, M[i])

    def _apply_basis_state(self, state, wires):
        """Initialize the quantum state in a specified computational basis state.

        Args:
            state (array[int]): computational basis state of shape ``(wires,)``
                consisting of 0s and 1s.
            wires (Wires): wires that the provided computational state should be
                initialized on

        Note: This function does not support broadcasted inputs yet.
        """
        # length of basis state parameter
        n_basis_state = len(state)

        if not set(state.tolist()).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

        if n_basis_state != len(wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        self._tensornet.setBasisState(state)

    def _apply_lightning(self, operations):
        """Apply a list of operations to the quantum state.

        Args:
            operations (list[~pennylane.operation.Operation]): operations to apply

        Returns:
            None
        """
        tensornet = self._tensornet

        # Skip over identity operations instead of performing
        # matrix multiplication with it.
        for operation in operations:
            if isinstance(operation, qml.Identity):
                continue
            if isinstance(operation, Adjoint):
                name = operation.base.name
                invert_param = True
            else:
                name = operation.name
                invert_param = False
            method = getattr(tensornet, name, None)
            wires = list(operation.wires)

            if self._method == "mps":
                if method is not None and len(wires) <= 2:
                    param = operation.parameters
                    method(wires, invert_param, param)
                else:
                    # Inverse can be set to False since qml.matrix(operation) is already in
                    # inverted form
                    method = getattr(tensornet, "applyMPOOperator")
                    try:
                        gate_ops_data = qml.matrix(operation)
                    except AttributeError:
                        gate_ops_data = operation.matrix
                    # Check if gate permutation is required
                    # Decompose the gate into MPOs
                    gate_data = np.transpose(gate_ops_data, axes=(1, 0)).flatten()
                    gate_data_shape = [4] * len(wires)
                    gate_data = np.array(gate_ops_data).reshape(gate_data_shape)

                    ranks = get_ranks(wires)

                    sorted_wires = sorted(wires)

                    gate_data = np.transpose(gate_data, axes=ranks).flatten()

                    MPOs = dense_to_mpo(gate_data, len(wires))

                    for i in range(len(MPOs)):
                        if i == 0:
                            MPOs[i] = np.transpose(MPOs[i], axes=(3, 2, 1, 0)).flatten()
                        elif i < len(MPOs) - 2:
                            MPOs[i] = np.transpose(MPOs[i], axes=(0, 1, 3, 2)).flatten()
                        else:
                            MPOs[i] = MPOs[i].flatten()

                    # Append the MPOs to the tensor network
                    method(MPOs, sorted_wires)
            else:
                if method is not None:
                    param = operation.parameters
                    method(wires, invert_param, param)
                else:
                    # Inverse can be set to False since qml.matrix(operation) is already in
                    # inverted form
                    method = getattr(tensornet, "applyMatrix")
                    try:
                        method(qml.matrix(operation), wires, False)
                    except AttributeError:
                        method(operation.matrix, wires, False)

    def apply_operations(self, operations):
        """Append operations to the tensor network graph."""
        # State preparation is currently done in Python
        if operations:  # make sure operations[0] exists
            if isinstance(operations[0], StatePrep):
                self._apply_state_vector(operations[0].parameters[0].copy(), operations[0].wires)
                operations = operations[1:]
            elif isinstance(operations[0], BasisState):
                self._apply_basis_state(operations[0].parameters[0], operations[0].wires)
                operations = operations[1:]

        self._apply_lightning(operations)

    def set_tensor_network(self, circuit: QuantumScript):
        """
        Set the tensor network that results from executing the given quantum script.

        This is an internal function that will be called by the successor to ``lightning.tensor``.

        Args:
            circuit (QuantumScript): The single circuit to simulate
        """
        self.apply_operations(circuit.operations)
        self.appendMPSFinalState()

    def appendMPSFinalState(self):
        """
        Append the final state to the tensor network for the MPS backend. This is an function to be called
        by once apply_operations is called.
        """
        if self._method == "mps":
            self._tensornet.appendMPSFinalState(self._cutoff, self._cutoff_mode)
