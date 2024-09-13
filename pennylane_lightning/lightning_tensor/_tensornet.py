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

# pylint: disable=ungrouped-imports
from pennylane_lightning.core._serialize import global_phase_diagonal


def svd_split(Mat, site_shape, max_bond_dim):
    """SVD decomposition of a matrix via numpy linalg. Note that this function is to be moved to the C++ layer."""
    U, S, Vd = np.linalg.svd(Mat, full_matrices=False)
    U = U @ np.diag(S)  # Append singular values to U
    bonds = len(S)

    Vd = Vd.reshape(tuple([bonds] + site_shape + [-1]))
    U = U.reshape(tuple([-1] + site_shape + [bonds]))

    # keep only chi bonds
    chi = np.min([bonds, max_bond_dim])
    U, Vd = U[..., :chi], Vd[:chi]
    return U, Vd


def decompose_dense(psi, n_wires, site_shape, max_bond_dim):
    """Decompose a dense state vector/gate matrix into MPS/MPO sites."""
    Ms = [[] for _ in range(n_wires)]
    site_len = np.prod(site_shape)
    psi = np.reshape(psi, (site_len, -1))

    U, Vd = svd_split(psi, site_shape, max_bond_dim)

    Ms[0] = U.reshape(tuple(site_shape + [-1]))
    bondL = Vd.shape[0]
    psi = Vd

    for i in range(1, n_wires - 1):
        psi = np.reshape(psi, (site_len * bondL, -1))
        U, Vd = svd_split(psi, site_shape, max_bond_dim)
        Ms[i] = U

        psi = Vd
        bondL = Vd.shape[0]

    Ms[-1] = Vd.reshape(tuple([-1] + site_shape))

    return Ms


def gate_matrix_decompose(gate_ops_matrix, wires, c_dtype):
    """Permute and decompose a gate matrix into MPO sites."""
    sorted_indexed_wires = sorted(enumerate(wires), key=lambda x: x[1])

    sorted_wires = []
    original_axes = []
    for index, wire in sorted_indexed_wires:
        sorted_wires.append(wire)
        original_axes.append(index)

    tensor_shape = [2] * len(wires) * 2

    matrix = gate_ops_matrix.astype(c_dtype)

    # Convert the gate matrix to the correct shape and complex dtype
    gate_tensor = matrix.reshape(tensor_shape)

    # Create the correct order of indices for the gate tensor to be decomposed
    indices_order = []
    for i in range(len(wires)):
        indices_order.extend([original_axes[i] + len(wires), original_axes[i]])

    # Permutation of the gate tensor
    gate_tensor = np.transpose(gate_tensor, axes=indices_order)

    mpo_site_shape = [2] * 2
    max_mpo_bond_dim = 2 ** len(wires)  # Exact SVD decomposition for MPO
    MPOs = decompose_dense(gate_tensor, len(wires), mpo_site_shape, max_mpo_bond_dim)

    mpos = []
    for i in range(len(MPOs)):
        if i == 0:
            # [bond, bra, ket] -> [ket, bond, bra]
            mpos.append(np.transpose(MPOs[len(MPOs) - 1 - i], axes=(2, 0, 1)))
        elif i == len(MPOs) - 1:
            # [bra, ket, bond] -> [ket, bra, bond]
            mpos.append(np.transpose(MPOs[len(MPOs) - 1 - i], axes=(1, 0, 2)))
        else:
            # sites between MSB and LSB [bondL, bra, ket, bondR] -> [ket, bondL, bra, bondR]
            # To match the order of cutensornet backend
            mpos.append(np.transpose(MPOs[len(MPOs) - 1 - i], axes=(2, 0, 1, 3)))

    return mpos, sorted_wires


def create_swap_queue(wires):
    """Create a swap ops queue non-local target wires gates applied to the MPS tensor network."""
    swap_wire_pairs = []
    wires_size = len(wires)
    if (wires[-1] - wires[0]) == wires_size - 1:
        target_wires = wires
        return target_wires, swap_wire_pairs
    else:
        fixed_pos = wires_size // 2
        fixed_gate_wire_id = wires[fixed_pos]
        op_wires_queue = []
        target_wires = [fixed_gate_wire_id]

        left_pos = fixed_pos - 1
        right_pos = fixed_pos + 1

        while left_pos >= 0 or right_pos < wires_size:
            if left_pos >= 0:
                wire_pair_queue = []
                print()
                for i in range(wires[left_pos], wires[fixed_pos] - (fixed_pos - left_pos)):
                    wire_pair_queue.append([i, i + 1])
                if wire_pair_queue:
                    op_wires_queue.append(wire_pair_queue)
                target_wires = [target_wires[0] - 1] + target_wires
                left_pos -= 1

            if right_pos < wires_size:
                wire_pair_queue = []
                for i in range(wires[right_pos], wires[fixed_pos] + right_pos - fixed_pos, -1):
                    wire_pair_queue.append([i, i - 1])
                if wire_pair_queue:
                    op_wires_queue.append(wire_pair_queue)
                target_wires += [target_wires[-1] + 1]
                right_pos += 1
    return target_wires, op_wires_queue


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

        self._wires = Wires(range(num_wires))

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
    def wires(self):
        """All wires that can be addressed on this device"""
        return self._wires

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
            state (array[complex]): normalized input state of length ``2**len(device_wires)``
            device_wires (Wires): wires that get initialized in the state

        Returns:
            array[complex]: normalized input state of length ``2**len(device_wires)``
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
        for i, value in enumerate(state):
            full_state[ravelled_indices[i]] = value
        return np.reshape(full_state, output_shape).ravel(order="C")

    def _apply_state_vector(self, state, device_wires: Wires):
        """Convert a specified state to MPS sites.
        Args:
            state (array[complex]): normalized input state of length ``2**len(device_wires)``
                or broadcasted state of shape ``(batch_size, 2**len(device_wires))``
            device_wires (Wires): wires that get initialized in the state
        """

        state = self._preprocess_state_vector(state, device_wires)
        mps_site_shape = [2]
        M = decompose_dense(state, self._num_wires, mps_site_shape, self._max_bond_dim)

        self._tensornet.updateMPSSitesData(M)

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

    def _apply_MPO(self, gate_matrix, wires):
        """Apply a matrix product operator to the quantum state.

        Args:
            gate_matrix (array[complex/float]): matrix representation of the MPO
            wires (Wires): wires that the MPO should be applied to
        Returns:
            None
        """
        # Get sorted wires and MPO site tensor
        mpos, sorted_wires = gate_matrix_decompose(gate_matrix, wires, self._c_dtype)

        # Check if SWAP operation should be applied
        local_target_wires, swap_pair_queue = create_swap_queue(sorted_wires)

        # TODO: This following part can be moved to the C++ layer in 2024 Q4
        # Apply SWAP operation to ensure the target wires are local
        for swap_wire_pairs in swap_pair_queue:
            for swap_wires in swap_wire_pairs:
                swap_op = getattr(self._tensornet, "SWAP", None)
                swap_op(swap_wires, False, [])

        max_mpo_bond_dim = 2 ** len(wires)  # Exact SVD decomposition for MPO

        self._tensornet.applyMPOOperator(mpos, local_target_wires, max_mpo_bond_dim)

        # Apply SWAP operation to restore the original wire order
        for swap_wire_pairs in swap_pair_queue[::-1]:
            for swap_wires in swap_wire_pairs[::-1]:
                swap_op = getattr(self._tensornet, "SWAP", None)
                swap_op(swap_wires, False, [])

    # pylint: disable=too-many-branches
    def _apply_lightning_controlled(self, operation):
        """Apply an arbitrary controlled operation to the state tensor. Note that `cutensornet` only supports controlled gates with a single wire target.

        Args:
            operation (~pennylane.operation.Operation): controlled operation to apply

        Returns:
            None
        """
        tensornet = self._tensornet

        basename = operation.base.name
        method = getattr(tensornet, f"{basename}", None)
        control_wires = list(operation.control_wires)
        control_values = operation.control_values
        target_wires = list(operation.target_wires)

        if method is not None and basename not in ("GlobalPhase", "MultiRZ"):
            inv = False
            param = operation.parameters
            method(control_wires, control_values, target_wires, inv, param)
        else:  # apply gate as an n-controlled matrix
            method = getattr(tensornet, "applyControlledMatrix")
            method(qml.matrix(operation.base), control_wires, control_values, target_wires, False)

    # pylint: disable=too-many-statements
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

            if isinstance(operation, qml.ops.Controlled) and len(list(operation.target_wires)) == 1:
                self._apply_lightning_controlled(operation)
            elif isinstance(operation, qml.GlobalPhase):
                matrix = np.eye(2) * operation.matrix().flatten()[0]
                method = getattr(tensornet, "applyMatrix")
                method(
                    matrix, [0], False
                )  # GlobalPhase is always applied to the first wire in the tensor network
            elif len(wires) <= 2 and not isinstance(operation, qml.MultiRZ):
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
                        method(operation.matrix(), wires, False)
            else:
                try:
                    gate_ops_matrix = qml.matrix(operation)
                except AttributeError:
                    gate_ops_matrix = operation.matrix()

                self._apply_MPO(gate_ops_matrix, wires)

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
        return self

    def appendMPSFinalState(self):
        """
        Append the final state to the tensor network for the MPS backend. This is an function to be called
        by once apply_operations is called.
        """
        if self._method == "mps":
            self._tensornet.appendMPSFinalState(self._cutoff, self._cutoff_mode)
