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
Utility functions for the ``quimb`` interface. These functions are used to convert PennyLane operators into `quimb` tensors and MPOs.
"""


import numpy as np
import quimb.tensor as qtn


def from_op_to_tensor(op) -> qtn.Tensor:
    """Returns the Quimb tensor corresponding to a PennyLane operator."""
    wires = tuple(op.wires)
    bra_inds = []
    for _, i in enumerate(wires):
        bra_inds.append(f"b{i}")
    bra_inds = tuple(bra_inds)
    ket_inds = []
    for _, i in enumerate(wires):
        ket_inds.append(f"k{i}")
    ket_inds = tuple(ket_inds)
    array = op.matrix()
    return qtn.Tensor(array.reshape([2] * int(np.log2(array.size))), inds=bra_inds + ket_inds)


def split_tensor(tensor, wires) -> list:
    """Returns the MPO factorization of a given tensor."""
    tensors = []
    v0 = tensor
    for c, i in enumerate(wires[0:-1]):
        inds = []
        for side in ["k", "b"]:
            inds.append(f"{side}{i}")
        if c > 0:
            inds.append(v0.inds[0])
        inds = tuple(inds)
        u0, v0 = v0.split(inds, cutoff=0.0)
        tensors.append(u0)
    tensors.append(v0)
    shift_tensor_indices(tensors)
    return tensors


def shift_tensor_indices(tensors) -> None:
    """Shifts the ``bra`` and ``ket`` indices to the right."""
    for t in tensors:
        for side in ["b", "k"]:
            for ind in t.inds:
                if ind[0] == side:
                    t.moveindex_(ind, -1)


def from_tensors_to_arrays(tensors, wires, n) -> list:
    """Converts a list of tensors into arrays that can be fed into ``MatrixProductOperator``."""
    arrays = []
    for _ in range(wires[0]):
        arrays.append(np.einsum("ij,kl->ijkl", np.eye(1, 1), np.eye(2, 2)))
    c = 0
    newaxes = 0
    for i in range(wires[0], wires[-1] + 1):
        if i in wires:
            arrays.append(np.expand_dims(tensors[c].data, axis=newaxes))
            c += 1
            newaxes = (1) if c == len(wires) - 1 else ()
        else:
            max_dim = np.max(arrays[-1].shape[0:2])
            arrays.append(np.einsum("ij,kl->ijkl", np.eye(max_dim, max_dim), np.eye(2, 2)))
    for _ in range(wires[-1] + 1, n):
        arrays.append(np.einsum("ij,kl->ijkl", np.eye(1, 1), np.eye(2, 2)))
    return arrays


def from_op_to_mpo(op, state) -> qtn.MatrixProductOperator:
    """Returns the MPO corresponding to the given operator."""
    wires = tuple(op.wires)
    tensor = from_op_to_tensor(op)
    tensors = split_tensor(tensor, wires)
    arrays = from_tensors_to_arrays(tensors, wires, state.L)
    return qtn.MatrixProductOperator(arrays, bond_name="x{}")
