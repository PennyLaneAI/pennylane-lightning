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

from warnings import warn

try:
    from pennylane_lightning.lightning_gpu_ops import MeasurementsC64, MeasurementsC128

    try:
        from pennylane_lightning.lightning_gpu_ops import MeasurementsMPIC64, MeasurementsMPIC128

        MPI_SUPPORT = True
    except ImportError as ex:
        warn(str(ex), UserWarning)

        MPI_SUPPORT = False

except ImportError as ex:
    warn(str(ex), UserWarning)

    pass

from typing import Any, List

import numpy as np
import pennylane as qml
from pennylane.measurements import CountsMP, MeasurementProcess, SampleMeasurement, Shots
from pennylane.typing import TensorLike

from pennylane_lightning.core._measurements_base import LightningBaseMeasurements

from ._mpi_handler import MPIHandler


class LightningGPUMeasurements(LightningBaseMeasurements):
    """Lightning GPU Measurements class

    Measures the state provided by the LightningGPUStateVector class.

    Args:
        qubit_state(LightningGPUStateVector): Lightning state-vector class containing the state vector to be measured.
        mpi_handler(MPIHandler): MPI handler for PennyLane Lightning GPU device.
            Provides functionality to run on multiple devices.

    """

    def __init__(
        self,
        lgpu_state,
        mpi_handler=None,
    ) -> TensorLike:

        super().__init__(lgpu_state)

        if mpi_handler is None:
            mpi_handler = MPIHandler(False, 0, None, lgpu_state.num_wires, lgpu_state.dtype)

        self._mpi_handler = mpi_handler

        self._measurement_lightning = self._measurement_dtype()(lgpu_state.state_vector)

    def _measurement_dtype(self):
        """Binding to Lightning GPU Measurements C++ class.

        Returns: the Measurements class
        """
        if self._mpi_handler.use_mpi:
            return MeasurementsMPIC128 if self.dtype == np.complex128 else MeasurementsMPIC64
        else:
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

    def _probs_retval_conversion(self, probs_results: Any) -> np.ndarray:
        """Convert the data structure from the C++ backend to a common structure through lightning devices.

        Args:
            probs_result (Any): Result provided by C++ backend.

        Returns:
            np.ndarray with probabilities of the supplied observable or wires.
        """

        # Device returns as col-major orderings, so perform transpose on data for bit-index shuffle for now.
        if len(probs_results) > 0:
            num_local_wires = len(probs_results).bit_length() - 1 if len(probs_results) > 0 else 0
            return probs_results.reshape([2] * num_local_wires).transpose().reshape(-1)

        return probs_results
