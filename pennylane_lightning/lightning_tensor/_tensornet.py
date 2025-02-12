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
    from pennylane_lightning.lightning_tensor_ops import (
        exactTensorNetC64,
        exactTensorNetC128,
        mpsTensorNetC64,
        mpsTensorNetC128,
    )
except ImportError:
    pass

import numpy as np
import pennylane as qml
from pennylane import BasisState, DeviceError, MPSPrep, StatePrep
from pennylane.ops.op_math import Adjoint
from pennylane.tape import QuantumScript
from pennylane.wires import Wires


def svd_split(Mat, site_shape, max_bond_dim, direction="right"):
    """SVD decomposition of a matrix via numpy linalg. Note that this function is to be moved to the C++ layer."""
    # TODO: Check if cutensornet allows us to remove all zero (or < tol) singular values and the respective rows and columns of U and Vd
    
    U, S, Vd = np.linalg.svd(Mat, full_matrices=False)
    
    bonds = len(S)
    chi = min([bonds, max_bond_dim])

    if direction == "right":
        U = U * S  # Append singular values to U
        
    if direction == "left":
        Vd = S * Vd.T # Append singular values to Vd
        Vd = Vd.T

    Vd = Vd.reshape([bonds] + site_shape + [-1])
    U = U.reshape([-1] + site_shape + [bonds])

    # keep only chi bonds
    U, Vd = U[..., :chi], Vd[:chi]

    return U, Vd


def decompose_dense(psi, n_wires, site_shape, max_bond_dim, direction="right"):
    """Decompose a dense state vector/gate matrix into MPS/MPO sites using Right-canonical matrix product state."""
    Ms = [[] for _ in range(n_wires)]
    site_len = np.prod(site_shape)

    if direction == "right":
        
        psi = np.reshape(psi, (-1,site_len)) 
        U, Vd = svd_split(psi, site_shape, max_bond_dim, direction=direction)  

        Ms[0] = Vd
        bondL = U.shape[-1]
        psi = U

        for i in range(1, n_wires - 1):
            psi = np.reshape(psi, (-1,site_len * bondL)) 
            U, Vd = svd_split(psi, site_shape, max_bond_dim, direction=direction)  
            Ms[i] = Vd

            psi = U
            bondL = U.shape[-1]

        Ms[-1] = U
        
        Ms.reverse()
        
    elif direction == "left":
        
        psi = np.reshape(psi, (site_len,-1))  
        U, Vd = svd_split(psi, site_shape, max_bond_dim, direction=direction)  
        
        Ms[0] = U
        bondL = Vd.shape[0]
        psi = Vd

        for i in range(1, n_wires - 1):
            psi = np.reshape(psi, (site_len * bondL, -1))  
            U, Vd = svd_split(psi, site_shape, max_bond_dim, direction=direction)
            Ms[i] = U
            psi = Vd
            bondL = Vd.shape[0]
            
        Ms[-1] = Vd
    
    first_shape = Ms[0].shape[1:]
    Ms[0] = np.reshape(Ms[0], first_shape)

    last_shape = Ms[-1].shape[:-1]    
    Ms[-1] = np.reshape(Ms[-1], last_shape)
        
    return Ms


def gate_matrix_decompose(gate_ops_matrix, wires, max_mpo_bond_dim, c_dtype):
    """Permute and decompose a gate matrix into MPO sites. This method return the MPO sites in the Fortran order of the ``cutensornet`` backend. Note that MSB in the Pennylane convention is the LSB in the ``cutensornet`` convention."""
    sorted_indexed_wires = sorted(enumerate(wires), key=lambda x: x[1])

    original_axes, sorted_wires = zip(*sorted_indexed_wires)

    tensor_shape = [2] * len(wires) * 2

    matrix = gate_ops_matrix.astype(c_dtype)

    # Convert the gate matrix to the correct shape and complex dtype
    gate_tensor = matrix.reshape(tensor_shape)

    # Create the correct order of indices for the gate tensor to be decomposed
    indices_order = []
    for i in range(len(wires)):
        indices_order.extend([original_axes[i], original_axes[i] + len(wires)])
    # Reverse the indices order to match the target wire order of cutensornet backend
    indices_order.reverse()

    # Permutation of the gate tensor
    gate_tensor = np.transpose(gate_tensor, axes=indices_order)

    mpo_site_shape = [2] * 2

    # The indices order of MPOs: 1. left-most site: [ket, bra, bondR]; 2. right-most sites: [bondL, ket, bra]; 3. sites in-between: [bondL, ket, bra, bondR].
    MPOs = decompose_dense(gate_tensor, len(wires), mpo_site_shape, max_mpo_bond_dim)

    # Convert the MPOs to the correct order for the cutensornet backend
    mpos = []
    for index, MPO in enumerate(MPOs):
        if index == 0:
            # [ket, bra, bond](0, 1, 2) -> [ket, bond, bra](0, 2, 1) -> Fortran order or reverse indices(1, 2, 0) to match the order requirement of cutensornet backend.
            mpos.append(np.transpose(MPO, axes=(1, 2, 0)))
        elif index == len(MPOs) - 1:
            # [bond, ket, bra](0, 1, 2) -> Fortran order or reverse indices(2, 1, 0) to match the order requirement of cutensornet backend.
            mpos.append(np.transpose(MPO, axes=(2, 1, 0)))
        else:
            # [bondL, ket, bra, bondR](0, 1, 2, 3) -> [bondL, ket, bondR, bra](0, 1, 3, 2) -> Fortran order or reverse indices(2, 3, 1, 0) to match the requirement of cutensornet backend.
            mpos.append(np.transpose(MPO, axes=(2, 3, 1, 0)))

    return mpos, sorted_wires

def check_canonical_form(mps, direction="left"):

    canon_values = []
        
    for sites in mps:
        
        C = None
        tensor_product_axes = [[-1,-2],[0,1]]
        
        if direction == "left":
            C = np.tensordot(sites.conj().T, sites, axes=tensor_product_axes)
        else: # direction == "right"
            C = np.tensordot(sites, sites.conj().T, axes=tensor_product_axes)
        
        # Compare C with the identity matrix
        eye = np.eye(C.shape[0], dtype=C.dtype)
        close = np.allclose(C, eye, atol=1e-10)
        
        canon_values.append(close)

    # Return True if all the values of canon_values are True
    return all(canon_values)


def expand_mps_top_onsite(state_MPS, max_bond_dim=128):    
    """Expand the only_state_MPS to match the ctrl_plus_state_MPS."""
    
    expanded_MPS = state_MPS.copy()
    
    n_sites_change = (len(expanded_MPS)+1)//2
            
    for i in range(n_sites_change-1):        
        # Create the next site for expanded_only_state_MPS    
        site_test = state_MPS[i]
        
        target_l,_, target_r = state_MPS[i+1].shape

        if len(state_MPS) % 2 == 1: # odd
            target_r = target_l*2 if target_l*2 < max_bond_dim else max_bond_dim
            
        site_r = site_test.shape[-1]

        # Horizontal padding   
        horizontal_pad = 2**i if 2**i < max_bond_dim else 0    
        site_test = np.pad(site_test, ((0,horizontal_pad),(0, 0),(0, 0)), mode="constant")

        # Vertical padding
        site_test = site_test.reshape(target_l, 2, site_r)
        site_test = np.pad(site_test, ((0,0),(0, 0),(0, target_r-site_r)), mode="constant")

        # Assign the new site        
        expanded_MPS[i] = site_test
    
    # Padding last site
    site_test = state_MPS[n_sites_change-1]
    
    target_l,_, target_r = state_MPS[n_sites_change].shape
    
    if len(state_MPS) % 2 == 1: # odd
        target_l = target_l*2
        target_r = target_r*2
    else: # even
        target_r =target_l
    
    site_r = site_test.shape[-1]

    # Horizontal padding
    horizontal_pad = 2**(n_sites_change-1) if 2**(n_sites_change-1) < max_bond_dim else 0
    site_test = np.pad(site_test, ((0,horizontal_pad),(0, 0),(0, 0)), mode="constant")

    # Vertical padding
    site_test = site_test.reshape(target_l, 2, target_r)
    site_test = np.pad(site_test, ((0,0),(0, 0),(0, target_r-site_r)), mode="constant")

    # Assign the last new site
    expanded_MPS[n_sites_change-1] = site_test

    # Add the initial site
    expanded_MPS.insert(0, np.eye(2, dtype=state_MPS[0].dtype).reshape(1,2,2))
    
    return expanded_MPS

def restore_left_canonical_form(mps,site_shape):
    
    new_mps = []
    Vd = np.eye(1, dtype=mps[0].dtype)
    
    for site in mps:
        site_p = np.tensordot(Vd, site, axes=[[-1], [0]])
        site_p = site_p.reshape(-1, site.shape[-1])

        U, S, Vd = np.linalg.svd(site_p, full_matrices=False)

        bonds = len(S)

        Vd = S * Vd 
        U = U.reshape([-1] + site_shape + [bonds])

        new_mps.append(U)
        
    return new_mps

def restore_right_canonical_form(mps,site_shape):

    mps.reverse()

    new_mps = []
    U = np.eye(1, dtype=mps[0].dtype)
    
    for site in mps:
        site_p = np.tensordot(site, U, axes=[[-1], [0]])
        site_p = site_p.reshape( site.shape[0],-1)

        U, S, Vd = np.linalg.svd(site_p, full_matrices=False)

        bonds = len(S)

        U = U * S 
        Vd = Vd.reshape([bonds] + site_shape + [-1])

        new_mps.append(Vd)
    
    new_mps.reverse()
    mps.reverse()
    
    return new_mps


def print_mps(mps, full=False, name=None):
    """Print the MPOs."""
    print("-"*100)
    if name is not None:
        print("MPS name:", name)
    
    max_bond_dim = 0
    for i, site in enumerate(mps):
        max_bond_dim = max(max_bond_dim, site.shape[-1])
    
    print("Sites:", len(mps), "| Max bond dim:", max_bond_dim)
    for i in range(len(mps)):
        print(f"| {'site '+str(i):^12}",end="")
    
    print()
    for i, site in enumerate(mps):
        print(f"| {str(site.shape):^12}", end="")
    print()
    
    if full:
        print("~"*100)
        for i, site in enumerate(mps):
            # print("Site", i, ":\n", site)
            print("Site", i, " | Shape:", site.shape)            
            print(site)
            print("~"*100)
    print("-"*100)


# pylint: disable=too-many-instance-attributes
class LightningTensorNet:
    """Lightning tensornet class.

    Interfaces with C++ python binding methods for tensornet manipulation.

    Args:
        num_wires(int): the number of wires to initialize the device with
        c_dtype: Datatypes for tensor network representation. Must be one of
            ``np.complex64`` or ``np.complex128``. Default is ``np.complex128``
        method(string): tensor network method. Supported methods are "mps" (Matrix Product State) and
            "tn" (Exact Tensor Network). Options: ["mps", "tn"].
        device_name(string): tensor network device name. Options: ["lightning.tensor"]
    Keyword Args:
        max_bond_dim (int): The maximum bond dimension to be used in the MPS simulation. Default is 128.
            The accuracy of the wavefunction representation comes with a memory tradeoff which can be
            tuned with `max_bond_dim`. The larger the internal bond dimension, the more entanglement can
            be described but the larger the memory requirements. Note that GPUs are ill-suited (i.e. less
            competitive compared with CPUs) for simulating circuits with low bond dimensions and/or circuit
            layers with a single or few gates because the arithmetic intensity is lower.
        cutoff (float): The threshold used to truncate the singular values of the MPS tensors. Default is 0.
        cutoff_mode (str): Singular value truncation mode for MPS tensors can be done either by
            considering the absolute values of the singular values (``"abs"``) or by considering
            the relative values of the singular values (``"rel"``). Default is ``"abs"``.
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        num_wires=None,
        method: str = "mps",
        c_dtype=np.complex128,
        device_name="lightning.tensor",
        **kwargs,
    ):
        if device_name != "lightning.tensor":
            raise DeviceError(f'The device name "{device_name}" is not a valid option.')

        if num_wires < 2:
            raise ValueError("Number of wires must be greater than 1.")

        self._num_wires = num_wires
        self._method = method
        self._c_dtype = c_dtype
        self._device_name = device_name

        self._wires = Wires(range(num_wires))

        if self._method == "mps":
            self._max_bond_dim = kwargs.get("max_bond_dim", 128)
            self._cutoff = kwargs.get("cutoff", 0)
            self._cutoff_mode = kwargs.get("cutoff_mode", "abs")
            self._tensornet = self._tensornet_dtype()(self._num_wires, self._max_bond_dim)
        elif self._method == "tn":
            self._tensornet = self._tensornet_dtype()(self._num_wires)
        elif self._method == "custom_mps":
            self._max_bond_dim = kwargs.get("max_bond_dim", 128)
            self._cutoff = kwargs.get("cutoff", 0)
            self._cutoff_mode = kwargs.get("cutoff_mode", "abs")
            self.bond_dim = kwargs.get("bond_dim", 0)
            self._tensornet = self._tensornet_dtype()(self._num_wires, self._max_bond_dim, self.bond_dim)
            self._method = "mps"

        else:
            raise DeviceError(f"The method {self._method} is not supported.")

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
        """Returns the number of wires addressed on this device"""
        return self._num_wires

    @property
    def method(self):
        """Returns the method (mps or tn) for evaluating the tensor network."""
        return self._method

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
        if self.method == "tn":  # Using "tn" method
            return exactTensorNetC128 if self.dtype == np.complex128 else exactTensorNetC64
        # Using "mps" method
        return mpsTensorNetC128 if self.dtype == np.complex128 else mpsTensorNetC64

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

        local_dev_wires = device_wires.tolist().copy()
        local_dev_wires = local_dev_wires[::-1]

        # generate basis states on subset of qubits via broadcasting as substitute of cartesian product.

        # Allocate a single row as a base to avoid a large array allocation with
        # the cartesian product algorithm.
        # Initialize the base with the pattern [0 1 0 1 ...].
        base = np.tile([0, 1], 2 ** (len(local_dev_wires) - 1)).astype(dtype=np.int64)
        # Allocate the array where it will accumulate the value of the indexes depending on
        # the value of the basis.
        indexes = np.zeros(2 ** (len(local_dev_wires)), dtype=np.int64)

        max_dev_wire = self._num_wires - 1

        # Iterate over all device wires.
        for i, wire in enumerate(local_dev_wires):

            # Accumulate indexes from the basis.
            indexes += base * 2 ** (max_dev_wire - wire)

            if i == len(local_dev_wires) - 1:
                continue

            two_n = 2 ** (i + 1)  # Compute the value of the base.

            # Update the value of the base without reallocating a new array.
            # Reshape the basis to swap the internal columns.
            base = base.reshape(-1, two_n * 2)
            swapper_A = two_n // 2
            swapper_B = swapper_A + two_n

            base[:, swapper_A:swapper_B] = base[:, swapper_A:swapper_B][:, ::-1]
            # Flatten the base array
            base = base.reshape(-1)

        # get full state vector to be factorized into MPS
        full_state = np.zeros(2**self._num_wires, dtype=self.dtype)
        for i, value in enumerate(state):
            full_state[indexes[i]] = value
        return np.reshape(full_state, output_shape).ravel(order="C")

    def _apply_state_vector(self, state, device_wires: Wires):
        """Convert a specified state to MPS sites.
        Args:
            state (array[complex]): normalized input state of length ``2**len(device_wires)``
                or broadcasted state of shape ``(batch_size, 2**len(device_wires))``
            device_wires (Wires): wires that get initialized in the state
        """
        if self.method == "tn":
            raise DeviceError("Exact Tensor Network does not support StatePrep")

        if self.method == "mps":
            state = self._preprocess_state_vector(state, device_wires)
            mps_site_shape = [2]
            M = decompose_dense(state, self._num_wires, mps_site_shape, self._max_bond_dim)
            self._tensornet.updateMPSSitesData(M)
            
    def _apply_mps_state(self, mps: tuple, target_wires: Wires) -> None:

        if len(target_wires) == self._num_wires and Wires(sorted(target_wires)) == target_wires:
            self._tensornet.updateMPSSitesData(mps)
            return

        trgt_wires = target_wires.tolist().copy()
        
        # Sort wires in ascending order
        trgt_wires.sort()
        
        # check if 0 is prensent in trgt_wires
        zero_trgt = 0 in trgt_wires
        single_wire = self._num_wires - len(trgt_wires) > 1
        
        if single_wire and not zero_trgt:
            raise DeviceError("MPSPrep only support to append a single wire at the beginning of the MPS.")
        
        new_mps = list(mps)
        
        if len(mps[0].shape) != 3:
            new_mps[0] = new_mps[0].reshape(1,2,2)
            
        if len(mps[-1].shape) != 3:
            new_mps[-1] = new_mps[-1].reshape(2,2,1)
        
        # Check the canonical form of the MPS
        canon_left = check_canonical_form(new_mps, direction="left")
        canon_right = check_canonical_form(new_mps, direction="right")
        
        # Expand the current MPS to match the size of the target wires
        new_mps = expand_mps_top_onsite(new_mps, self._max_bond_dim)
        
        if canon_left:
            new_mps = restore_left_canonical_form(new_mps, [2])
        elif canon_right:
            new_mps = restore_right_canonical_form(new_mps, [2])
            
        print_mps(new_mps, full=False, name="new_mps")
            
        # Restore dimension of first and last sites
        new_mps[0] = new_mps[0].reshape(2,2)
        new_mps[-1] = new_mps[-1].reshape(2,2)
        
        # Update the MPS sites in the tensornet
        self._tensornet.updateMPSSitesData(new_mps)
        

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
        """Apply a matrix product operator to the quantum state (MPS method only).

        Args:
            gate_matrix (array[complex/float]): matrix representation of the MPO
            wires (Wires): wires that the MPO should be applied to
        Returns:
            None
        """
        # TODO: Discuss if public interface for max_mpo_bond_dim argument
        max_mpo_bond_dim = self._max_bond_dim

        # Get sorted wires and MPO site tensor
        mpos, sorted_wires = gate_matrix_decompose(
            gate_matrix, wires, max_mpo_bond_dim, self._c_dtype
        )

        self._tensornet.applyMPOOperation(mpos, sorted_wires, max_mpo_bond_dim)

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
                # GlobalPhase is always applied to the first wire in the tensor network
                method(matrix, [0], False)
            elif len(wires) <= 2:
                if method is not None:
                    param = operation.parameters
                    method(wires, invert_param, param)
                else:
                    # Inverse can be set to False since qml.matrix(operation) is already in
                    # inverted form
                    method = getattr(tensornet, "applyMatrix")
                    try:
                        method(qml.matrix(operation), wires, False)
                    except AttributeError:  # pragma: no cover
                        # To support older versions of PL
                        method(operation.matrix(), wires, False)
            else:
                try:
                    gate_ops_matrix = qml.matrix(operation)
                except AttributeError:  # pragma: no cover
                    # To support older versions of PL
                    gate_ops_matrix = operation.matrix()

                if self.method == "mps":
                    self._apply_MPO(gate_ops_matrix, wires)
                if self.method == "tn":
                    method = getattr(tensornet, "applyMatrix")
                    method(gate_ops_matrix, wires, False)

    def apply_operations(self, operations):
        """Append operations to the tensor network graph."""
        # State preparation is currently done in Python
        if operations:  # make sure operations[0] exists
            if isinstance(operations[0], StatePrep):
                if self.method == "mps":
                    self._apply_state_vector(
                        operations[0].parameters[0].copy(), operations[0].wires
                    )
                    operations = operations[1:]
                if self.method == "tn":
                    raise DeviceError("Exact Tensor Network does not support StatePrep")
            elif isinstance(operations[0], BasisState):
                self._apply_basis_state(operations[0].parameters[0], operations[0].wires)
                operations = operations[1:]
            elif isinstance(operations[0], MPSPrep):
                if self.method == "mps":
                    self._apply_mps_state(operations[0].mps, operations[0].wires)
                    
                    operations = operations[1:]

                if self.method == "tn":
                    raise DeviceError("Exact Tensor Network does not support MPSPrep")

        self._apply_lightning(operations)

    def set_tensor_network(self, circuit: QuantumScript):
        """
        Set the tensor network that results from executing the given quantum script.

        This is an internal function that will be called by the successor to ``lightning.tensor``.

        Args:
            circuit (QuantumScript): The single circuit to simulate
        """
        self.apply_operations(circuit.operations)
        self.appendFinalState()

        return self

    def appendFinalState(self):
        """
        Append the final state to the tensor network. This function should be called once when apply_operations is called. It only applies to the MPS method and is an empty call for the Exact Tensor Network method.
        """
        if self.method == "mps":
            self._tensornet.appendMPSFinalState(self._cutoff, self._cutoff_mode)
