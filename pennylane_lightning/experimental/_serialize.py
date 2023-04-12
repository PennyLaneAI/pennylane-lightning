# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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

from pennylane import matrix

try:
    from ..lightning_qubit_ops import (
        StateVectorC64,
        StateVectorC128,
    )
    from ..lightning_qubit_ops.adjoint_diff import (
        NamedObsC64,
        NamedObsC128,
        HermitianObsC64,
        HermitianObsC128,
        TensorProdObsC64,
        TensorProdObsC128,
        HamiltonianC64,
        HamiltonianC128,
    )
except ImportError:
    pass

pauli_name_map = {
    "I": "Identity",
    "X": "PauliX",
    "Y": "PauliY",
    "Z": "PauliZ",
}


def _serialize_named_obs(ob, use_csingle: bool):
    """Serializes a Named observable"""
    named_obs = NamedObsC64 if use_csingle else NamedObsC128
    wires = ob.wires.tolist()
    if ob.name == "Identity":
        wires = wires[:1]
    return named_obs(ob.name, wires)


def _serialize_hermitian_ob(ob, use_csingle: bool):
    """Serializes a Hermitian observable

    Args:
        ob: a named or Hermitian observable
        use_csingle (bool): whether to use np.complex64 instead of np.complex128

    Returns:
        ObsStructC128 or ObsStructC64: An observable object compatible with the C++ backend.
    """
    assert not isinstance(ob, Tensor)

    if use_csingle:
        ctype = np.complex64
        hermitian_obs = HermitianObsC64
    else:
        ctype = np.complex128
        hermitian_obs = HermitianObsC128

    return hermitian_obs(matrix(ob).ravel().astype(ctype), ob.wires.tolist())


def _serialize_tensor_ob(ob, use_csingle: bool):
    """Serializes a tensor product object.

    Args:
        ob: a tensor product object
        use_csingle (bool): whether to use np.complex64 instead of np.complex128

    Returns:
        ObsStructC128 or ObsStructC64: An observable object compatible with the C++ backend.
    """
    assert isinstance(ob, Tensor)

    if use_csingle:
        tensor_obs = TensorProdObsC64
    else:
        tensor_obs = TensorProdObsC128

    return tensor_obs([_serialize_ob(o, use_csingle) for o in ob.obs])


def _serialize_hamiltonian(ob, use_csingle: bool):
    """Serializes a Hamiltonian.

    Args:
        ob: a Hamiltonian
        use_csingle (bool): whether to use np.complex64 instead of np.complex128

    Returns:
        ObsStructC128 or ObsStructC64: An observable object compatible with the C++ backend.
    """
    if use_csingle:
        rtype = np.float32
        hamiltonian_obs = HamiltonianC64
    else:
        rtype = np.float64
        hamiltonian_obs = HamiltonianC128

    coeffs = np.array(unwrap(ob.coeffs)).astype(rtype)
    terms = [_serialize_ob(t, use_csingle) for t in ob.ops]
    return hamiltonian_obs(coeffs, terms)


def _serialize_pauli_word(ob, use_csingle: bool):
    """Serialize a PauliWord into a Named or Tensor."""
    if use_csingle:
        named_obs = NamedObsC64
        tensor_obs = TensorProdObsC64
    else:
        named_obs = NamedObsC128
        tensor_obs = TensorProdObsC128

    if len(ob) == 1:
        wire, pauli = list(ob.items())[0]
        return named_obs(pauli_name_map[pauli], [wire])

    return tensor_obs([named_obs(pauli_name_map[pauli], [wire]) for wire, pauli in ob.items()])


def _serialize_pauli_sentence(ob, use_csingle: bool):
    """Serialize a PauliSentence into a Hamiltonian."""
    if use_csingle:
        rtype = np.float32
        hamiltonian_obs = HamiltonianC64
    else:
        rtype = np.float64
        hamiltonian_obs = HamiltonianC128

    pwords, coeffs = zip(*ob.items())
    terms = [_serialize_pauli_word(pw, use_csingle) for pw in pwords]
    coeffs = np.array(coeffs).astype(rtype)
    return hamiltonian_obs(coeffs, terms)


def _serialize_ob(ob, use_csingle: bool = False):
    """Serializes an observable.

    Args:
        ob: a single observable
        use_csingle (bool): whether to use np.complex64 instead of np.complex128

    Returns:
        ObsStructC128 or ObsStructC64: An observable object compatible with the C++ backend.
    """
    if isinstance(ob, Tensor):
        return _serialize_tensor_ob(ob, use_csingle)
    elif ob.name == "Hamiltonian":
        return _serialize_hamiltonian(ob, use_csingle)
    elif isinstance(ob, (PauliX, PauliY, PauliZ, Identity, Hadamard)):
        return _serialize_named_obs(ob, use_csingle)
    elif ob._pauli_rep is not None:
        return _serialize_pauli_sentence(ob._pauli_rep, use_csingle)
    else:
        return _serialize_hermitian_ob(ob, use_csingle)


# def _serialize_observables(tape: QuantumTape, use_csingle: bool = False) -> List:
#     """Serializes the observables of an input tape.

#     Args:
#         tape (QuantumTape): the input quantum tape
#         use_csingle (bool): whether to use np.complex64 instead of np.complex128

#     Returns:
#         list(ObsStructC128 or ObsStructC64): A list of observable objects compatible with the C++ backend
#     """
#     return [_serialize_ob(ob, use_csingle) for ob in tape.observables]


# def _serialize_ops(
#     tape: QuantumTape,
# ) -> Tuple[List[List[str]], List[np.ndarray], List[List[int]], List[bool], List[np.ndarray]]:
#     """Serializes the operations of an input tape.

#     The state preparation operations are not included.

#     Args:
#         tape (QuantumTape): the input quantum tape

#     Returns:
#         Tuple[list, list, list, list, list]: A serialization of the operations, containing a list
#         of operation names, a list of operation parameters, a list of observable wires, a list of
#         inverses, and a list of matrices for the operations that do not have a dedicated kernel.
#     """
#     names = []
#     params = []
#     wires = []
#     mats = []

#     uses_stateprep = False

#     for o in tape.operations:
#         if isinstance(o, (BasisState, QubitStateVector)):
#             uses_stateprep = True
#             continue
#         elif isinstance(o, Rot):
#             op_list = o.expand().operations
#         else:
#             op_list = [o]

#         for single_op in op_list:
#             name = single_op.name
#             names.append(name)

#             if not hasattr(StateVectorC128, name):
#                 params.append([])
#                 mats.append(matrix(single_op))

#             else:
#                 params.append(single_op.parameters)
#                 mats.append([])

#             wires_list = single_op.wires.tolist()
#             wires.append(wires_list)

#     inverses = [False] * len(names)
#     return (names, params, wires, inverses, mats), uses_stateprep