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

from typing import List

import numpy as np
import pennylane as qml
from pennylane.measurements import CountsMP, MeasurementProcess, SampleMeasurement, Shots
from pennylane.typing import TensorLike

# pylint: disable=ungrouped-imports
from pennylane_lightning.core._measurements_base import LightningBaseMeasurements
from pennylane_lightning.core._serialize import QuantumScriptSerializer


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

    def _measure_with_samples_diagonalizing_gates(
        self,
        mps: List[SampleMeasurement],
        shots: Shots,
    ) -> TensorLike:
        """
        Returns the samples of the measurement process performed on the given state,
        by rotating the state into the measurement basis using the diagonalizing gates
        given by the measurement process.

        Args:
            mps (~.measurements.SampleMeasurement): The sample measurements to perform
            shots (~.measurements.Shots): The number of samples to take

        Returns:
            TensorLike[Any]: Sample measurement results
        """
        # apply diagonalizing gates
        self._apply_diagonalizing_gates(mps)

        # Specific for LGPU:
        total_indices = self._qubit_state.num_wires
        wires = qml.wires.Wires(range(total_indices))

        def _process_single_shot(samples):
            processed = []
            for mp in mps:
                res = mp.process_samples(samples, wires)
                if not isinstance(mp, CountsMP):
                    res = qml.math.squeeze(res)

                processed.append(res)

            return tuple(processed)

        try:
            samples = self._measurement_lightning.generate_samples(
                len(wires), shots.total_shots
            ).astype(int, copy=False)

        except ValueError as ex:
            if str(ex) != "probabilities contain NaN":
                raise ex
            samples = qml.math.full((shots.total_shots, len(wires)), 0)

        self._apply_diagonalizing_gates(mps, adjoint=True)

        # if there is a shot vector, use the shots.bins generator to
        # split samples w.r.t. the shots
        processed_samples = []
        for lower, upper in shots.bins():
            result = _process_single_shot(samples[..., lower:upper, :])
            processed_samples.append(result)

        return (
            tuple(zip(*processed_samples)) if shots.has_partitioned_shots else processed_samples[0]
        )

    def expval(self, measurementprocess: MeasurementProcess):
        """Expectation value of the supplied observable contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the state

        Returns:
            Expectation value of the observable
        """

        if isinstance(measurementprocess.obs, qml.SparseHamiltonian):
            # ensuring CSR sparse representation.

            if self._use_mpi:
                # Identity for CSR_SparseHamiltonian to pass to processes with rank != 0 to reduce
                # host(cpu) memory requirements
                obs = qml.Identity(0)
                Hmat = qml.Hamiltonian([1.0], [obs]).sparse_matrix()
                H_sparse = qml.SparseHamiltonian(Hmat, wires=range(1))
                CSR_SparseHamiltonian = H_sparse.sparse_matrix().tocsr()
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

        # use specialized functors to compute expval(Hermitian)
        if isinstance(measurementprocess.obs, qml.Hermitian):
            observable_wires = measurementprocess.obs.wires
            if self._use_mpi and len(observable_wires) > self._num_local_wires:
                raise RuntimeError(
                    "MPI backend does not support Hermitian with number of target wires larger than local wire number."
                )
            matrix = measurementprocess.obs.matrix()
            return self._measurement_lightning.expval(matrix, observable_wires)

        if (
            isinstance(measurementprocess.obs, qml.ops.Hamiltonian)
            or (measurementprocess.obs.arithmetic_depth > 0)
            or isinstance(measurementprocess.obs.name, List)
        ):
            # pylint: disable=protected-access
            ob_serialized = QuantumScriptSerializer(
                self._qubit_state.device_name, self.dtype == np.complex64, self._use_mpi
            )._ob(measurementprocess.obs)
            return self._measurement_lightning.expval(ob_serialized)

        return self._measurement_lightning.expval(
            measurementprocess.obs.name, measurementprocess.obs.wires
        )
