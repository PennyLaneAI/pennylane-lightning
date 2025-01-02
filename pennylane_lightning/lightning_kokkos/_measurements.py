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
    from pennylane_lightning.lightning_kokkos_ops import MeasurementsC64, MeasurementsC128
except ImportError as ex:
    warn(str(ex), UserWarning)

from typing import List

import numpy as np
import pennylane as qml
from pennylane.measurements import CountsMP, SampleMeasurement, Shots
from pennylane.typing import TensorLike

# pylint: disable=ungrouped-imports
from pennylane_lightning.core._measurements_base import LightningBaseMeasurements


class LightningKokkosMeasurements(
    LightningBaseMeasurements
):  # pylint: disable=too-few-public-methods
    """Lightning Kokkos Measurements class

    Measures the state provided by the LightningKokkosStateVector class.

    Args:
        qubit_state(LightningKokkosStateVector): Lightning state-vector class containing the state vector to be measured.
    """

    def __init__(
        self,
        kokkos_state: LightningKokkosStateVector,  # pylint: disable=undefined-variable
    ) -> None:
        super().__init__(kokkos_state)

        self._measurement_lightning = self._measurement_dtype()(kokkos_state.state_vector)

    def _measurement_dtype(self):
        """Binding to Lightning Kokkos Measurements C++ class.

        Returns: the Measurements class
        """
        return MeasurementsC64 if self.dtype == np.complex64 else MeasurementsC128

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

        # Specific for Kokkos:
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

        except ValueError as e:
            if str(e) != "probabilities contain NaN":
                raise e
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
