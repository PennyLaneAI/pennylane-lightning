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
from ._tensornet_MPS import LightningTensorNetMPS
from ._tensornet_ExactTN import LightningTensorNetExactTN

import numpy as np
import pennylane as qml
from pennylane import BasisState, DeviceError, StatePrep
from pennylane.ops.op_math import Adjoint
from pennylane.tape import QuantumScript
from pennylane.wires import Wires


def svd_split(Mat, site_shape, max_bond_dim):
    """SVD decomposition of a matrix via numpy linalg. Note that this function is to be moved to the C++ layer."""
    # TODO: Check if cutensornet allows us to remove all zero (or < tol) singular values and the respective rows and columns of U and Vd
    U, S, Vd = np.linalg.svd(Mat, full_matrices=False)
    U = U * S  # Append singular values to U
    bonds = len(S)

    Vd = Vd.reshape([bonds] + site_shape + [-1])
    U = U.reshape([-1] + site_shape + [bonds])

    # keep only chi bonds
    chi = min([bonds, max_bond_dim])
    U, Vd = U[..., :chi], Vd[:chi]
    return U, Vd


def decompose_dense(psi, n_wires, site_shape, max_bond_dim):
    """Decompose a dense state vector/gate matrix into MPS/MPO sites."""
    Ms = [[] for _ in range(n_wires)]
    site_len = np.prod(site_shape)
    psi = np.reshape(psi, (site_len, -1))  # split psi [2, 2, 2, 2...] to psi [site_len, -1]

    U, Vd = svd_split(
        psi, site_shape, max_bond_dim
    )  # psi [site_len, -1] -> U [site_len, mu] Vd [mu, (2x2x2x..)]

    Ms[0] = U.reshape(site_shape + [-1])
    bondL = Vd.shape[0]
    psi = Vd

    for i in range(1, n_wires - 1):
        psi = np.reshape(psi, (site_len * bondL, -1))  # reshape psi[site_len*bondL, -1]
        U, Vd = svd_split(
            psi, site_shape, max_bond_dim
        )  # psi [site_len*bondL, -1] -> U [site_len, mu] Vd [mu, (2x2x2x..)]
        Ms[i] = U

        psi = Vd
        bondL = Vd.shape[0]

    Ms[-1] = Vd.reshape([-1] + site_shape)

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
        # *,
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
        
        print("FDX == ", self._method)
        
        if self._method == "mps":
            LTensor = LightningTensorNetMPS(
            self._num_wires,
            self._method,
            self._c_dtype,
            self._max_bond_dim,
            self._cutoff,
            self._cutoff_mode,)
        if self._method == "exatn":
            LTensor = LightningTensorNetExactTN(
            self._num_wires,
            self._method,
            self._c_dtype,
            self._cutoff,
            self._cutoff_mode,)
            
        self._LTensor = LTensor

    def __getattr__(self, name):
        return getattr(self._LTensor, name)
