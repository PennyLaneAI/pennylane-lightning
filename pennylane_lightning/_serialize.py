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
from typing import List

from pennylane import BasisState, Hadamard, Projector, QubitStateVector
from pennylane.grouping import is_pauli_word
from pennylane.operation import Observable, Tensor
from pennylane.tape import QuantumTape

try:
    from .lightning_qubit_ops import StateVectorC128, ObsStructC128, OpsStructC128
except ImportError:
    pass


def _obs_has_kernel(obs: Observable) -> bool:
    """Returns True if the input observable has a supported kernel in the C++ backend.

    Args:
        obs (Observable): the input observable

    Returns:
        bool: indicating whether ``obs`` has a dedicated kernel in the backend
    """
    if is_pauli_word(obs):
        return True
    if isinstance(obs, (Hadamard, Projector)):
        return True
    if isinstance(obs, Tensor):
        return all(_obs_has_kernel(o) for o in obs.obs)
    return False


def _serialize_obs(
    tape: QuantumTape, wires_map: dict
) -> List:
    """Serializes the observables of an input tape.

    Args:
        tape (QuantumTape): the input quantum tape
        wires_map (dict): a dictionary mapping input wires to the device's backend wires

    Returns:
        list(ObsStructC128): A list of observable objects compatible with the C++ backend
    """
    obs = []

    for o in tape.observables:
        is_tensor = isinstance(o, Tensor)

        wires_list = o.wires.tolist()
        wires = [wires_map[w] for w in wires_list]
        name = o.name if is_tensor else [o.name]

        params = []

        if not _obs_has_kernel(o):
            if is_tensor:
                for o_ in o.obs:
                    if not _obs_has_kernel(o_):
                        params.append(o_.matrix)
            else:
                params.append(o.matrix)

        ob = ObsStructC128(name, params, [wires])
        obs.append(ob)

    return obs


def _serialize_ops(
    tape: QuantumTape, wires_map: dict
):
    """Serializes the operations of an input tape.

    The state preparation operations are not included.

    Args:
        tape (QuantumTape): the input quantum tape
        wires_map (dict): a dictionary mapping input wires to the device's backend wires

    Returns:
        OpsStructC128: A C++-backend-compatible object containing a serialization of the input tape
        operations
    """
    names = []
    params = []
    wires = []
    inverses = []
    mats = []

    for o in tape.operations:
        if isinstance(o, (BasisState, QubitStateVector)):
            continue

        is_inverse = o.inverse

        name = o.name if not is_inverse else o.name[:-4]
        names.append(name)

        if getattr(StateVectorC128, name, None) is None:
            params.append([])
            mats.append(o.matrix)

            if is_inverse:
                is_inverse = False
        else:
            params.append(o.parameters)
            mats.append([])

        wires_list = o.wires.tolist()
        wires.append([wires_map[w] for w in wires_list])
        inverses.append(is_inverse)

    return OpsStructC128(names, params, wires, inverses, mats)
