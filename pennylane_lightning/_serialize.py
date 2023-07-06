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
Helper functions for serializing quantum tapes.
"""
from typing import List, Tuple

import numpy as np
from pennylane import (
    BasisState,
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    Identity,
    QubitStateVector,
    Rot,
)
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap
from warnings import warn

# Remove after the next release of PL
# Add from pennylane import matrix
import pennylane as qml

try:
    from .lightning_qubit_ops import (
        StateVectorC64,
        StateVectorC128,
        supporting_gates,
    )
    from .lightning_qubit_ops.adjoint_diff import (
        NamedObsC64,
        NamedObsC128,
        HermitianObsC64,
        HermitianObsC128,
        TensorProdObsC64,
        TensorProdObsC128,
        HamiltonianC64,
        HamiltonianC128,
        OpsStructC64,
        OpsStructC128,
        create_ops_list_C64,
        create_ops_list_C128,
        SparseHamiltonianC64,
        SparseHamiltonianC128,
    )
except ImportError:
    pass

pauli_name_map = {
    "I": "Identity",
    "X": "PauliX",
    "Y": "PauliY",
    "Z": "PauliZ",
}


def _serialize_named_obs(ob, wires_map: dict, use_csingle: bool):
    """Serializes a Named observable"""
    named_obs = NamedObsC64 if use_csingle else NamedObsC128
    wires = [wires_map[w] for w in ob.wires]
    if ob.name == "Identity":
        wires = wires[:1]
    return named_obs(ob.name, wires)


def _serialize_hermitian_ob(o, wires_map: dict, use_csingle: bool):
    """Serializes a Hermitian observable"""
    assert not isinstance(o, Tensor)

    if use_csingle:
        ctype = np.complex64
        hermitian_obs = HermitianObsC64
    else:
        ctype = np.complex128
        hermitian_obs = HermitianObsC128

    wires = [wires_map[w] for w in o.wires]
    return hermitian_obs(qml.matrix(o).ravel().astype(ctype), wires)


def _serialize_tensor_ob(ob, wires_map: dict, use_csingle: bool):
    """Serialize a tensor observable"""
    assert isinstance(ob, Tensor)

    if use_csingle:
        tensor_obs = TensorProdObsC64
    else:
        tensor_obs = TensorProdObsC128

    return tensor_obs([_serialize_ob(o, wires_map, use_csingle) for o in ob.obs])


def _serialize_hamiltonian(ob, wires_map: dict, use_csingle: bool):
    if use_csingle:
        rtype = np.float32
        hamiltonian_obs = HamiltonianC64
    else:
        rtype = np.float64
        hamiltonian_obs = HamiltonianC128

    coeffs = np.array(unwrap(ob.coeffs)).astype(rtype)
    terms = [_serialize_ob(t, wires_map, use_csingle) for t in ob.ops]
    return hamiltonian_obs(coeffs, terms)


def _serialize_sparse_hamiltonian(ob, wires_map: dict, use_csingle: bool):
    if use_csingle:
        rtype = np.float32
        ctype = np.complex64
        sparse_hamiltonian_obs = SparseHamiltonianC64
    else:
        rtype = np.float64
        ctype = np.complex128
        sparse_hamiltonian_obs = SparseHamiltonianC128

    CSR_SparseHamiltonian = ob.sparse_matrix().tocsr(
        copy=False
    )
    return sparse_hamiltonian_obs(
        CSR_SparseHamiltonian.indptr,
        CSR_SparseHamiltonian.indices,
        CSR_SparseHamiltonian.data.astype(ctype),
    )

def _serialize_pauli_word(ob, wires_map: dict, use_csingle: bool):
    """Serialize a :class:`pennylane.pauli.PauliWord` into a Named or Tensor observable."""
    if use_csingle:
        named_obs = NamedObsC64
        tensor_obs = TensorProdObsC64
    else:
        named_obs = NamedObsC128
        tensor_obs = TensorProdObsC128

    if len(ob) == 1:
        wire, pauli = list(ob.items())[0]
        return named_obs(pauli_name_map[pauli], [wires_map[wire]])

    return tensor_obs(
        [named_obs(pauli_name_map[pauli], [wires_map[wire]]) for wire, pauli in ob.items()]
    )


def _serialize_pauli_sentence(ob, wires_map: dict, use_csingle: bool):
    """Serialize a :class:`pennylane.pauli.PauliSentence` into a Hamiltonian."""
    if use_csingle:
        rtype = np.float32
        hamiltonian_obs = HamiltonianC64
    else:
        rtype = np.float64
        hamiltonian_obs = HamiltonianC128

    pwords, coeffs = zip(*ob.items())
    terms = [_serialize_pauli_word(pw, wires_map, use_csingle) for pw in pwords]
    coeffs = np.array(coeffs).astype(rtype)
    return hamiltonian_obs(coeffs, terms)


def _serialize_ob(ob, wires_map, use_csingle):
    if isinstance(ob, Tensor):
        return _serialize_tensor_ob(ob, wires_map, use_csingle)
    elif ob.name == "Hamiltonian":
        return _serialize_hamiltonian(ob, wires_map, use_csingle)
    elif ob.name == "SparseHamiltonian":
        return _serialize_sparse_hamiltonian(ob, wires_map, use_csingle)
    elif isinstance(ob, (PauliX, PauliY, PauliZ, Identity, Hadamard)):
        return _serialize_named_obs(ob, wires_map, use_csingle)
    elif ob._pauli_rep is not None:
        return _serialize_pauli_sentence(ob._pauli_rep, wires_map, use_csingle)
    else:
        return _serialize_hermitian_ob(ob, wires_map, use_csingle)


def _serialize_observables(tape: QuantumTape, wires_map: dict, use_csingle: bool = False) -> List:
    """Serializes the observables of an input tape.

    Args:
        tape (QuantumTape): the input quantum tape
        wires_map (dict): a dictionary mapping input wires to the device's backend wires
        use_csingle (bool): whether to use np.complex64 instead of np.complex128

    Returns:
        list(ObsStructC128 or ObsStructC64): A list of observable objects compatible with the C++ backend
    """

    return [_serialize_ob(ob, wires_map, use_csingle) for ob in tape.observables]


def _serialize_ops(tape: QuantumTape, wires_map: dict, use_csingle: bool = False):
    """Serializes the operations of an input tape.

    The state preparation operations are not included.

    Args:
        tape (QuantumTape): the input quantum tape
        wires_map (dict): a dictionary mapping input wires to the device's backend wires

    Returns:
        Tuple[list, list, list, list, list]: A serialization of the operations, containing a list
        of operation names, a list of operation parameters, a list of observable wires, a list of
        inverses, and a list of matrices for the operations that do not have a dedicated kernel.
    """
    names = []
    params = []
    wires = []
    mats = []

    if use_csingle:
        create_ops_list = create_ops_list_C64
    else:
        create_ops_list = create_ops_list_C128

    if use_csingle:
        rtype = np.float32
        ctype = np.complex64
    else:
        rtype = np.float64
        ctype = np.complex128

    trainable_op_idices = []
    param_idx = 0  # Parameter index for a tape
    lightning_ops_idx = 0
    record_tp_rows = []
    record_tp_idx = 0

    expanded_ops = []

    for op in tape.operations:
        if isinstance(op, Rot):
            op_list = op.expand().operations
        else:
            op_list = [op]
        expanded_ops.extend(op_list)

    # Transform a tape
    for op in expanded_ops:
        name = op.name
        wires_list = op.wires.tolist()

        if isinstance(op, (BasisState, QubitStateVector)):
            # We just ignore this
            pass
        elif name not in supporting_gates():
            if len(wires_list) == 1:
                name = "SingleQubitOp"
            elif len(wires_list) == 2:
                name = "TwoQubitOp"
            else:
                name = "MultiQubitOp"
            names.append(name)
            wires.append([wires_map[w] for w in wires_list])
            params.append([])
            mats.append(qml.matrix(op).astype(ctype))
            lightning_ops_idx += 1

            if op.num_params > 0 and param_idx in tape.trainable_params:
                warn(
                    "There is a gate with trainable parameters that lightning does not support natively. Even though you can use it, Lightning does not compute gradients of variables for those gates."
                )

        else:
            names.append(name)
            wires.append([wires_map[w] for w in wires_list])
            params.append(op.parameters)
            mats.append([])

            if op.num_params > 0 and param_idx in tape.trainable_params:
                trainable_op_idices.append(lightning_ops_idx)
                record_tp_rows.append(record_tp_idx)

            lightning_ops_idx += 1

        if op.num_params > 0 and param_idx in tape.trainable_params:
            # If the gradient of the current operator should be recored
            record_tp_idx += 1
        param_idx += op.num_params

    inverses = [False] * len(names)
    return (
        create_ops_list(names, params, wires, inverses, mats),
        trainable_op_idices,
        record_tp_rows,
    )
