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
Class implementation for state vector measurements.
"""

from __future__ import annotations

from warnings import warn

try:
    from pennylane_lightning.lightning_gpu_ops import MeasurementsC64, MeasurementsC128

    try:
        from pennylane_lightning.lightning_gpu_ops import MeasurementsMPIC64, MeasurementsMPIC128

        mpi_error = None
        MPI_SUPPORT = True
    except ImportError as ex_mpi:
        mpi_error = ex_mpi
        MPI_SUPPORT = False

except ImportError as error_import:
    warn(str(error_import), UserWarning)

import numpy as np
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.typing import TensorLike

# pylint: disable=ungrouped-imports
from pennylane_lightning.lightning_base._measurements import LightningBaseMeasurements
from pennylane_lightning.lightning_base._serialize import QuantumScriptSerializer


class LightningGPUMeasurements(LightningBaseMeasurements):  # pylint: disable=too-few-public-methods
    """Lightning GPU Measurements class

    Measures the state provided by the LightningGPUStateVector class.

    Args:
        qubit_state(LightningGPUStateVector): Lightning state-vector class containing the state vector to be measured.
    """

    def __init__(
        self,
        qubit_state: LightningGPUStateVector,  # pylint: disable=undefined-variable
    ) -> TensorLike:

        super().__init__(qubit_state)

        self._use_mpi = qubit_state._mpi_handler.use_mpi

        if self._use_mpi:
            self._mpi_handler = qubit_state._mpi_handler
            self._num_local_wires = qubit_state._mpi_handler.num_local_wires

        self._measurement_lightning = self._measurement_dtype()(qubit_state.state_vector)

    def _measurement_dtype(self):
        """Binding to Lightning GPU Measurements C++ class.

        Returns: the Measurements class
        """
        if self._use_mpi:
            if not MPI_SUPPORT:
                warn(str(mpi_error), UserWarning)

            return MeasurementsMPIC128 if self.dtype == np.complex128 else MeasurementsMPIC64

        # without MPI
        return MeasurementsC128 if self.dtype == np.complex128 else MeasurementsC64

    def expval(self, measurementprocess: MeasurementProcess):
        """Expectation value of the supplied observable contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the state

        Returns:
            Expectation value of the observable
        """

        if self._observable_is_sparse(measurementprocess.obs):
            # ensuring CSR sparse representation.

            if self._use_mpi:
                # Identity for CSR_SparseHamiltonian to pass to processes with rank != 0 to reduce
                # host(cpu) memory requirements
                CSR_SparseHamiltonian = qml.Identity(0).sparse_matrix()
                # CSR_SparseHamiltonian for rank == 0
                if self._mpi_handler.mpi_manager.getRank() == 0:
                    CSR_SparseHamiltonian = measurementprocess.obs.sparse_matrix().tocsr()
            else:
                CSR_SparseHamiltonian = measurementprocess.obs.sparse_matrix(
                    wire_order=list(range(self._qubit_state.num_wires))
                ).tocsr(copy=False)

            return self._measurement_lightning.expval(
                CSR_SparseHamiltonian.indptr,
                CSR_SparseHamiltonian.indices,
                CSR_SparseHamiltonian.data,
            )

        # use specialized function to compute expval(pauli_sentence)
        if measurementprocess.obs.pauli_rep is not None:
            return self._expval_pauli_sentence(measurementprocess)

        # use specialized functors to compute expval(Hermitian)
        if isinstance(measurementprocess.obs, qml.Hermitian):
            observable_wires = measurementprocess.obs.wires
            if self._use_mpi and len(observable_wires) > self._num_local_wires:
                raise RuntimeError(
                    "MPI backend does not support Hermitian with number of target wires larger than local wire number."
                )
            matrix = measurementprocess.obs.matrix()
            return self._measurement_lightning.expval(matrix, observable_wires)

        if measurementprocess.obs.arithmetic_depth:
            # pylint: disable=protected-access
            ob_serialized = QuantumScriptSerializer(
                self._qubit_state.device_name, self.dtype == np.complex64, self._use_mpi
            )._ob(measurementprocess.obs)
            return self._measurement_lightning.expval(ob_serialized)

        return self._measurement_lightning.expval(
            measurementprocess.obs.name, measurementprocess.obs.wires
        )

    def _expval_pauli_sentence(self, measurementprocess: MeasurementProcess):
        """Specialized method for computing the expectation value of a Pauli sentence.

        Args:
            measurementprocess (MeasurementProcess): Measurement process with pauli_rep.

        Returns:
            Expectation value.
        """
        pwords, coeffs = zip(*measurementprocess.obs.pauli_rep.items())
        pauli_words = [qml.pauli.pauli_word_to_string(p) for p in pwords]
        wires = [p.wires.tolist() for p in pwords]
        return self._measurement_lightning.expval(pauli_words, wires, coeffs)
