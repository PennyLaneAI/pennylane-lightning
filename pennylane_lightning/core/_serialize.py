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

from pennylane import matrix, DeviceError

pauli_name_map = {
    "I": "Identity",
    "X": "PauliX",
    "Y": "PauliY",
    "Z": "PauliZ",
}


class QuantumScriptSerializer:
    """Serializer class for `pennylane.tape.QuantumScript` data.

    Args:
    device_name: device shortname.
    use_csingle (bool): whether to use np.complex64 instead of np.complex128

    """

    # pylint: disable=import-outside-toplevel, too-many-instance-attributes
    def __init__(self, device_name, use_csingle: bool = False):
        self.use_csingle = use_csingle
        if device_name == "lightning.qubit":
            try:
                import pennylane_lightning.lightning_qubit_ops as lightning_ops
            except ImportError as exception:
                raise ImportError(
                    f"Pre-compiled binaries for {device_name}"
                    " serialize functionality are not available."
                ) from exception
        elif device_name == "lightning.kokkos":
            try:
                import pennylane_lightning.lightning_kokkos_ops as lightning_ops
            except ImportError as exception:
                raise ImportError(
                    f"Pre-compiled binaries for {device_name}"
                    " serialize functionality are not available."
                ) from exception
        else:
            raise DeviceError(f'The device name "{device_name}" is not a valid option.')
        self.statevector_c128 = lightning_ops.StateVectorC128
        self.named_obs_c64 = lightning_ops.observables.NamedObsC64
        self.named_obs_c128 = lightning_ops.observables.NamedObsC128
        self.hermitian_obs_c64 = lightning_ops.observables.HermitianObsC64
        self.hermitian_obs_c128 = lightning_ops.observables.HermitianObsC128
        self.tensor_prod_obs_c64 = lightning_ops.observables.TensorProdObsC64
        self.tensor_prod_obs_c128 = lightning_ops.observables.TensorProdObsC128
        self.hamiltonian_c64 = lightning_ops.observables.HamiltonianC64
        self.hamiltonian_c128 = lightning_ops.observables.HamiltonianC128

    @property
    def ctype(self):
        """Complex type."""
        return np.complex64 if self.use_csingle else np.complex128

    @property
    def rtype(self):
        """Real type."""
        return np.float32 if self.use_csingle else np.float64

    @property
    def named_obs(self):
        """Named observable matching ``use_csingle`` precision."""
        return self.named_obs_c64 if self.use_csingle else self.named_obs_c128

    @property
    def hermitian_obs(self):
        """Hermitian observable matching ``use_csingle`` precision."""
        return self.hermitian_obs_c64 if self.use_csingle else self.hermitian_obs_c128

    @property
    def tensor_obs(self):
        """Tensor product observable matching ``use_csingle`` precision."""
        return self.tensor_prod_obs_c64 if self.use_csingle else self.tensor_prod_obs_c128

    @property
    def hamiltonian_obs(self):
        """Hamiltonian observable matching ``use_csingle`` precision."""
        return self.hamiltonian_c64 if self.use_csingle else self.hamiltonian_c128

    def _named_obs(self, observable, wires_map: dict):
        """Serializes a Named observable"""
        wires = [wires_map[w] for w in observable.wires]
        if observable.name == "Identity":
            wires = wires[:1]
        return self.named_obs(observable.name, wires)

    def _hermitian_ob(self, observable, wires_map: dict):
        """Serializes a Hermitian observable"""
        assert not isinstance(observable, Tensor)

        wires = [wires_map[w] for w in observable.wires]
        return self.hermitian_obs(matrix(observable).ravel().astype(self.ctype), wires)

    def _tensor_ob(self, observable, wires_map: dict):
        """Serialize a tensor observable"""
        assert isinstance(observable, Tensor)
        return self.tensor_obs([self._ob(obs, wires_map) for obs in observable.obs])

    def _hamiltonian(self, observable, wires_map: dict):
        coeffs = np.array(unwrap(observable.coeffs)).astype(self.rtype)
        terms = [self._ob(t, wires_map) for t in observable.ops]
        return self.hamiltonian_obs(coeffs, terms)

    def _pauli_word(self, observable, wires_map: dict):
        """Serialize a :class:`pennylane.pauli.PauliWord` into a Named or Tensor observable."""
        if len(observable) == 1:
            wire, pauli = list(observable.items())[0]
            return self.named_obs(pauli_name_map[pauli], [wires_map[wire]])

        return self.tensor_obs(
            [
                self.named_obs(pauli_name_map[pauli], [wires_map[wire]])
                for wire, pauli in observable.items()
            ]
        )

    def _pauli_sentence(self, observable, wires_map: dict):
        """Serialize a :class:`pennylane.pauli.PauliSentence` into a Hamiltonian."""
        pwords, coeffs = zip(*observable.items())
        terms = [self._pauli_word(pw, wires_map) for pw in pwords]
        coeffs = np.array(coeffs).astype(self.rtype)
        return self.hamiltonian_obs(coeffs, terms)

    # pylint: disable=protected-access
    def _ob(self, observable, wires_map):
        """Serialize a :class:`pennylane.operation.Observable` into an Observable."""
        if isinstance(observable, Tensor):
            return self._tensor_ob(observable, wires_map)
        if observable.name == "Hamiltonian":
            return self._hamiltonian(observable, wires_map)
        if isinstance(observable, (PauliX, PauliY, PauliZ, Identity, Hadamard)):
            return self._named_obs(observable, wires_map)
        if observable._pauli_rep is not None:
            return self._pauli_sentence(observable._pauli_rep, wires_map)
        return self._hermitian_ob(observable, wires_map)

    def serialize_observables(self, tape: QuantumTape, wires_map: dict) -> List:
        """Serializes the observables of an input tape.

        Args:
            tape (QuantumTape): the input quantum tape
            wires_map (dict): a dictionary mapping input wires to the device's backend wires

        Returns:
            list(ObsStructC128 or ObsStructC64): A list of observable objects compatible with
                the C++ backend
        """

        return [self._ob(observable, wires_map) for observable in tape.observables]

    def serialize_ops(
        self, tape: QuantumTape, wires_map: dict
    ) -> Tuple[List[List[str]], List[np.ndarray], List[List[int]], List[bool], List[np.ndarray]]:
        """Serializes the operations of an input tape.

        The state preparation operations are not included.

        Args:
            tape (QuantumTape): the input quantum tape
            wires_map (dict): a dictionary mapping input wires to the device's backend wires

        Returns:
            Tuple[list, list, list, list, list]: A serialization of the operations, containing a
            list of operation names, a list of operation parameters, a list of observable wires,
            a list of inverses, and a list of matrices for the operations that do not have a
            dedicated kernel.
        """
        names = []
        params = []
        wires = []
        mats = []

        uses_stateprep = False

        for operation in tape.operations:
            if isinstance(operation, (BasisState, QubitStateVector)):
                uses_stateprep = True
                continue
            if isinstance(operation, Rot):
                op_list = operation.expand().operations
            else:
                op_list = [operation]

            for single_op in op_list:
                name = single_op.name
                names.append(name)

                if not hasattr(self.statevector_c128, name):
                    params.append([])
                    mats.append(matrix(single_op))

                else:
                    params.append(single_op.parameters)
                    mats.append([])

                wires_list = single_op.wires.tolist()
                wires.append([wires_map[w] for w in wires_list])

        inverses = [False] * len(names)
        return (names, params, wires, inverses, mats), uses_stateprep
