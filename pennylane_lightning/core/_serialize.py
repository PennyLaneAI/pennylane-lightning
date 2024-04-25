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
from typing import List, Sequence, Tuple

import numpy as np
import pennylane as qml
from pennylane import (
    BasisState,
    DeviceError,
    Hadamard,
    Identity,
    PauliX,
    PauliY,
    PauliZ,
    QubitUnitary,
    Rot,
    SparseHamiltonian,
    StatePrep,
    matrix,
)
from pennylane.math import unwrap
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, LinearCombination, Prod, SProd, Sum
from pennylane.tape import QuantumTape

NAMED_OBS = (Identity, PauliX, PauliY, PauliZ, Hadamard)
OP_MATH_OBS = (Prod, SProd, Sum, LinearCombination)
PAULI_NAME_MAP = {
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
    use_mpi (bool, optional): If using MPI to accelerate calculation. Defaults to False.
    split_obs (bool, optional): If splitting the observables in a list. Defaults to False.

    """

    # pylint: disable=import-outside-toplevel, too-many-instance-attributes, c-extension-no-member
    def __init__(
        self, device_name, use_csingle: bool = False, use_mpi: bool = False, split_obs: bool = False
    ):
        self.use_csingle = use_csingle
        self.device_name = device_name
        self.split_obs = split_obs
        if device_name == "lightning.qubit":
            try:
                import pennylane_lightning.lightning_qubit_ops as lightning_ops
            except ImportError as exception:
                raise ImportError(
                    f"Pre-compiled binaries for {device_name} are not available."
                ) from exception
        elif device_name == "lightning.kokkos":
            try:
                import pennylane_lightning.lightning_kokkos_ops as lightning_ops
            except ImportError as exception:
                raise ImportError(
                    f"Pre-compiled binaries for {device_name} are not available."
                ) from exception
        elif device_name == "lightning.gpu":
            try:
                import pennylane_lightning.lightning_gpu_ops as lightning_ops
            except ImportError as exception:
                raise ImportError(
                    f"Pre-compiled binaries for {device_name} are not available."
                ) from exception
        else:
            raise DeviceError(f'The device name "{device_name}" is not a valid option.')
        self.statevector_c64 = lightning_ops.StateVectorC64
        self.statevector_c128 = lightning_ops.StateVectorC128
        self.named_obs_c64 = lightning_ops.observables.NamedObsC64
        self.named_obs_c128 = lightning_ops.observables.NamedObsC128
        self.hermitian_obs_c64 = lightning_ops.observables.HermitianObsC64
        self.hermitian_obs_c128 = lightning_ops.observables.HermitianObsC128
        self.tensor_prod_obs_c64 = lightning_ops.observables.TensorProdObsC64
        self.tensor_prod_obs_c128 = lightning_ops.observables.TensorProdObsC128
        self.hamiltonian_c64 = lightning_ops.observables.HamiltonianC64
        self.hamiltonian_c128 = lightning_ops.observables.HamiltonianC128
        self.sparse_hamiltonian_c64 = lightning_ops.observables.SparseHamiltonianC64
        self.sparse_hamiltonian_c128 = lightning_ops.observables.SparseHamiltonianC128

        self._use_mpi = use_mpi

        if self._use_mpi:
            self.statevector_mpi_c64 = lightning_ops.StateVectorMPIC64
            self.statevector_mpi_c128 = lightning_ops.StateVectorMPIC128
            self.named_obs_mpi_c64 = lightning_ops.observablesMPI.NamedObsMPIC64
            self.named_obs_mpi_c128 = lightning_ops.observablesMPI.NamedObsMPIC128
            self.hermitian_obs_mpi_c64 = lightning_ops.observablesMPI.HermitianObsMPIC64
            self.hermitian_obs_mpi_c128 = lightning_ops.observablesMPI.HermitianObsMPIC128
            self.tensor_prod_obs_mpi_c64 = lightning_ops.observablesMPI.TensorProdObsMPIC64
            self.tensor_prod_obs_mpi_c128 = lightning_ops.observablesMPI.TensorProdObsMPIC128
            self.hamiltonian_mpi_c64 = lightning_ops.observablesMPI.HamiltonianMPIC64
            self.hamiltonian_mpi_c128 = lightning_ops.observablesMPI.HamiltonianMPIC128
            self.sparse_hamiltonian_mpi_c64 = lightning_ops.observablesMPI.SparseHamiltonianMPIC64
            self.sparse_hamiltonian_mpi_c128 = lightning_ops.observablesMPI.SparseHamiltonianMPIC128

            self._mpi_manager = lightning_ops.MPIManager

    @property
    def ctype(self):
        """Complex type."""
        return np.complex64 if self.use_csingle else np.complex128

    @property
    def rtype(self):
        """Real type."""
        return np.float32 if self.use_csingle else np.float64

    @property
    def sv_type(self):
        """State vector matching ``use_csingle`` precision (and MPI if it is supported)."""
        if self._use_mpi:
            return self.statevector_mpi_c64 if self.use_csingle else self.statevector_mpi_c128
        return self.statevector_c64 if self.use_csingle else self.statevector_c128

    @property
    def named_obs(self):
        """Named observable matching ``use_csingle`` precision."""
        if self._use_mpi:
            return self.named_obs_mpi_c64 if self.use_csingle else self.named_obs_mpi_c128
        return self.named_obs_c64 if self.use_csingle else self.named_obs_c128

    @property
    def hermitian_obs(self):
        """Hermitian observable matching ``use_csingle`` precision."""
        if self._use_mpi:
            return self.hermitian_obs_mpi_c64 if self.use_csingle else self.hermitian_obs_mpi_c128
        return self.hermitian_obs_c64 if self.use_csingle else self.hermitian_obs_c128

    @property
    def tensor_obs(self):
        """Tensor product observable matching ``use_csingle`` precision."""
        if self._use_mpi:
            return (
                self.tensor_prod_obs_mpi_c64 if self.use_csingle else self.tensor_prod_obs_mpi_c128
            )
        return self.tensor_prod_obs_c64 if self.use_csingle else self.tensor_prod_obs_c128

    @property
    def hamiltonian_obs(self):
        """Hamiltonian observable matching ``use_csingle`` precision."""
        if self._use_mpi:
            return self.hamiltonian_mpi_c64 if self.use_csingle else self.hamiltonian_mpi_c128
        return self.hamiltonian_c64 if self.use_csingle else self.hamiltonian_c128

    @property
    def sparse_hamiltonian_obs(self):
        """SparseHamiltonian observable matching ``use_csingle`` precision."""
        if self._use_mpi:
            return (
                self.sparse_hamiltonian_mpi_c64
                if self.use_csingle
                else self.sparse_hamiltonian_mpi_c128
            )
        return self.sparse_hamiltonian_c64 if self.use_csingle else self.sparse_hamiltonian_c128

    def _named_obs(self, observable, wires_map: dict = None):
        """Serializes a Named observable"""
        wires = [wires_map[w] for w in observable.wires] if wires_map else observable.wires.tolist()
        if isinstance(observable, qml.Identity):
            wires = wires[:1]
        return self.named_obs(observable.name, wires)

    def _hermitian_ob(self, observable, wires_map: dict = None):
        """Serializes a Hermitian observable"""

        wires = [wires_map[w] for w in observable.wires] if wires_map else observable.wires.tolist()
        return self.hermitian_obs(matrix(observable).ravel().astype(self.ctype), wires)

    def _tensor_ob(self, observable, wires_map: dict = None):
        """Serialize a tensor observable"""
        obs = observable.obs if isinstance(observable, Tensor) else observable.operands
        return self.tensor_obs([self._ob(o, wires_map) for o in obs])

    def _hamiltonian(self, observable, wires_map: dict = None):
        coeffs, ops = observable.terms()
        coeffs = np.array(unwrap(coeffs)).astype(self.rtype)
        terms = [self._ob(t, wires_map) for t in ops]
        # TODO: This is in case `_hamiltonian` is called recursively which would cause a list
        # to be passed where `_ob` expects an observable.
        terms = [t[0] if isinstance(t, Sequence) and len(t) == 1 else t for t in terms]

        if self.split_obs:
            return [self.hamiltonian_obs([c], [t]) for (c, t) in zip(coeffs, terms)]

        return self.hamiltonian_obs(coeffs, terms)

    def _sparse_hamiltonian(self, observable, wires_map: dict = None):
        """Serialize an observable (Sparse Hamiltonian)

        Args:
            observable (Observable): the input observable (Sparse Hamiltonian)
            wire_map (dict): a dictionary mapping input wires to the device's backend wires

        Returns:
            sparse_hamiltonian_obs (SparseHamiltonianC64 or SparseHamiltonianC128): A Sparse Hamiltonian observable object compatible with the C++ backend
        """

        if self._use_mpi:
            Hmat = Hamiltonian([1.0], [Identity(0)]).sparse_matrix()
            H_sparse = SparseHamiltonian(Hmat, wires=range(1))
            spm = H_sparse.sparse_matrix()
            # Only root 0 needs the overall sparse matrix data
            if self._mpi_manager().getRank() == 0:
                spm = observable.sparse_matrix()
            self._mpi_manager().Barrier()
        else:
            spm = observable.sparse_matrix()
        data = np.array(spm.data).astype(self.ctype)
        indices = np.array(spm.indices).astype(np.int64)
        offsets = np.array(spm.indptr).astype(np.int64)

        wires = [wires_map[w] for w in observable.wires] if wires_map else observable.wires.tolist()

        return self.sparse_hamiltonian_obs(data, indices, offsets, wires)

    def _pauli_word(self, observable, wires_map: dict = None):
        """Serialize a :class:`pennylane.pauli.PauliWord` into a Named or Tensor observable."""

        def map_wire(wire: int):
            return wires_map[wire] if wires_map else wire

        if len(observable) == 1:
            wire, pauli = list(observable.items())[0]
            return self.named_obs(PAULI_NAME_MAP[pauli], [map_wire(wire)])

        return self.tensor_obs(
            [
                self.named_obs(PAULI_NAME_MAP[pauli], [map_wire(wire)])
                for wire, pauli in observable.items()
            ]
        )

    def _pauli_sentence(self, observable, wires_map: dict = None):
        """Serialize a :class:`pennylane.pauli.PauliSentence` into a Hamiltonian."""
        # Trivial Pauli sentences' items is empty, cannot unpack
        if not observable:
            return self.hamiltonian_obs(np.array([0.0]).astype(self.rtype), [self._ob(Identity(0))])
        pwords, coeffs = zip(*observable.items())
        terms = [self._pauli_word(pw, wires_map) for pw in pwords]
        coeffs = np.array(coeffs).astype(self.rtype)

        if len(terms) == 1 and coeffs[0] == 1.0:
            return terms[0]

        if self.split_obs:
            return [self.hamiltonian_obs([c], [t]) for (c, t) in zip(coeffs, terms)]
        return self.hamiltonian_obs(coeffs, terms)

    # pylint: disable=protected-access, too-many-return-statements
    def _ob(self, observable, wires_map: dict = None):
        """Serialize a :class:`pennylane.operation.Observable` into an Observable."""
        if isinstance(observable, NAMED_OBS):
            return self._named_obs(observable, wires_map)
        if isinstance(observable, Hamiltonian):
            return self._hamiltonian(observable, wires_map)
        if observable.pauli_rep is not None:
            return self._pauli_sentence(observable.pauli_rep, wires_map)
        if isinstance(observable, (Tensor, Prod)):
            if isinstance(observable, Prod) and observable.has_overlapping_wires:
                return self._hermitian_ob(observable, wires_map)
            return self._tensor_ob(observable, wires_map)
        if isinstance(observable, OP_MATH_OBS):
            return self._hamiltonian(observable, wires_map)
        if isinstance(observable, SparseHamiltonian):
            return self._sparse_hamiltonian(observable, wires_map)
        return self._hermitian_ob(observable, wires_map)

    def serialize_observables(self, tape: QuantumTape, wires_map: dict = None) -> List:
        """Serializes the observables of an input tape.

        Args:
            tape (QuantumTape): the input quantum tape
            wires_map (dict): a dictionary mapping input wires to the device's backend wires

        Returns:
            list(ObsStructC128 or ObsStructC64): A list of observable objects compatible with
                the C++ backend. For unsupported observables, the observable matrix is used
                to create a :class:`~pennylane.Hermitian` to be used for serialization.
        """

        serialized_obs = []
        offset_indices = [0]

        for observable in tape.observables:
            ser_ob = self._ob(observable, wires_map)
            if isinstance(ser_ob, list):
                serialized_obs.extend(ser_ob)
                offset_indices.append(offset_indices[-1] + len(ser_ob))
            else:
                serialized_obs.append(ser_ob)
                offset_indices.append(offset_indices[-1] + 1)
        return serialized_obs, offset_indices

    def serialize_ops(self, tape: QuantumTape, wires_map: dict = None) -> Tuple[
        List[List[str]],
        List[np.ndarray],
        List[List[int]],
        List[bool],
        List[np.ndarray],
        List[List[int]],
        List[List[bool]],
    ]:
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
        controlled_wires = []
        controlled_values = []
        wires = []
        mats = []

        uses_stateprep = False

        def get_wires(operation, single_op):
            if isinstance(operation, qml.ops.op_math.Controlled) and not isinstance(
                operation,
                (
                    qml.CNOT,
                    qml.CY,
                    qml.CZ,
                    qml.ControlledPhaseShift,
                    qml.CRX,
                    qml.CRY,
                    qml.CRZ,
                    qml.CRot,
                    qml.CSWAP,
                ),
            ):
                name = operation.base.name
                wires_list = list(operation.target_wires)
                controlled_wires_list = list(operation.control_wires)
                control_values_list = operation.control_values
                if not hasattr(self.sv_type, name):
                    single_op = QubitUnitary(matrix(single_op.base), single_op.base.wires)
                    name = single_op.name
            else:
                name = single_op.name
                wires_list = single_op.wires.tolist()
                controlled_wires_list = []
                control_values_list = []
            return single_op, name, list(wires_list), controlled_wires_list, control_values_list

        for operation in tape.operations:
            if isinstance(operation, (BasisState, StatePrep)):
                uses_stateprep = True
                continue
            if isinstance(operation, Rot):
                op_list = operation.expand().operations
            else:
                op_list = [operation]

            for single_op in op_list:
                (
                    single_op,
                    name,
                    wires_list,
                    controlled_wires_list,
                    controlled_values_list,
                ) = get_wires(operation, single_op)
                names.append(name)
                # QubitUnitary is a special case, it has a parameter which is not differentiable.
                # We thus pass a dummy 0.0 parameter which will not be referenced
                if isinstance(single_op, qml.QubitUnitary):
                    params.append([0.0])
                    mats.append(matrix(single_op))
                elif not hasattr(self.sv_type, name):
                    params.append([])
                    mats.append(matrix(single_op))
                else:
                    params.append(single_op.parameters)
                    mats.append([])

                controlled_values.append(controlled_values_list)
                controlled_wires.append(
                    [wires_map[w] for w in controlled_wires_list]
                    if wires_map
                    else list(controlled_wires_list)
                )
                wires.append([wires_map[w] for w in wires_list] if wires_map else wires_list)

        inverses = [False] * len(names)
        return (
            names,
            params,
            wires,
            inverses,
            mats,
            controlled_wires,
            controlled_values,
        ), uses_stateprep


def global_phase_diagonal(par, wires, controls, control_values):
    """Returns the diagonal of a C(GlobalPhase) operator."""
    diag = np.ones(2 ** len(wires), dtype=np.complex128)
    controls = np.array(controls)
    control_values = np.array(control_values)
    ind = np.argsort(controls)
    controls = controls[ind[-1::-1]]
    control_values = control_values[ind[-1::-1]]
    idx = np.arange(2 ** len(wires), dtype=np.int64).reshape([2 for _ in wires])
    for c, w in zip(control_values, controls):
        idx = np.take(idx, np.array(int(c)), w)
    diag[idx.ravel()] = np.exp(-1j * par)
    return diag
