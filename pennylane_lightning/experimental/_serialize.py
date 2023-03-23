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
    Hamiltonian,
    QubitStateVector,
    Rot,
)
from pennylane.grouping import is_pauli_word
from pennylane.operation import Observable, Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap

from pennylane import matrix

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


def _obs_has_kernel(ob: Observable) -> bool:
    """Returns True if the input observable has a supported kernel in the C++ backend.

    Args:
        ob (Observable): the input observable

    Returns:
        bool: indicating whether ``obs`` has a dedicated kernel in the backend
    """
    if is_pauli_word(ob):
        return True
    if isinstance(ob, (Hadamard)):
        return True
    if isinstance(ob, Hamiltonian):
        return all(_obs_has_kernel(o) for o in ob.ops)
    if isinstance(ob, Tensor):
        return all(_obs_has_kernel(o) for o in ob.obs)

    return False


def _serialize_named_hermitian_ob(ob, use_csingle: bool):
    """Serializes a named or Hermitian observable.

    Args:
        ob: a named or Hermitian observable
        use_csingle (bool): whether to use np.complex64 instead of np.complex128

    Returns:
        ObsStructC128 or ObsStructC64: An observable object compatible with the C++ backend.
    """
    assert not isinstance(ob, Tensor)

    if use_csingle:
        ctype = np.complex64
        named_obs = NamedObsC64
        hermitian_obs = HermitianObsC64
    else:
        ctype = np.complex128
        named_obs = NamedObsC128
        hermitian_obs = HermitianObsC128

    wires_list = ob.wires.tolist()
    if _obs_has_kernel(ob):
        return named_obs(ob.name, wires_list)
    return hermitian_obs(matrix(ob).ravel().astype(ctype), wires_list)


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
    else:
        return _serialize_named_hermitian_ob(ob, use_csingle)


def _serialize_observables(tape: QuantumTape, use_csingle: bool = False) -> List:
    """Serializes the observables of an input tape.

    Args:
        tape (QuantumTape): the input quantum tape
        use_csingle (bool): whether to use np.complex64 instead of np.complex128

    Returns:
        list(ObsStructC128 or ObsStructC64): A list of observable objects compatible with the C++ backend
    """

    return [_serialize_ob(ob, use_csingle) for ob in tape.observables]


def _serialize_ops(
    tape: QuantumTape,
) -> Tuple[List[List[str]], List[np.ndarray], List[List[int]], List[bool], List[np.ndarray]]:
    """Serializes the operations of an input tape.

    The state preparation operations are not included.

    Args:
        tape (QuantumTape): the input quantum tape

    Returns:
        Tuple[list, list, list, list, list]: A serialization of the operations, containing a list
        of operation names, a list of operation parameters, a list of observable wires, a list of
        inverses, and a list of matrices for the operations that do not have a dedicated kernel.
    """
    names = []
    params = []
    wires = []
    mats = []

    uses_stateprep = False

    for o in tape.operations:
        if isinstance(o, (BasisState, QubitStateVector)):
            uses_stateprep = True
            continue
        elif isinstance(o, Rot):
            op_list = o.expand().operations
        else:
            op_list = [o]

        for single_op in op_list:
            name = single_op.name
            names.append(name)

            if not hasattr(StateVectorC128, name):
                params.append([])
                mats.append(matrix(single_op))

            else:
                params.append(single_op.parameters)
                mats.append([])

            wires_list = single_op.wires.tolist()
            wires.append(wires_list)

    inverses = [False] * len(names)
    return (names, params, wires, inverses, mats), uses_stateprep
