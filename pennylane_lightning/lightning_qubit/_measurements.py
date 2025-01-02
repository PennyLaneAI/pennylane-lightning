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

# pylint: disable=import-error, no-name-in-module, ungrouped-imports
from __future__ import annotations

from warnings import warn

try:
    from pennylane_lightning.lightning_qubit_ops import MeasurementsC64, MeasurementsC128
except ImportError as ex:
    warn(str(ex), UserWarning)

from functools import reduce
from typing import List

import numpy as np
import pennylane as qml
from pennylane.measurements import CountsMP, SampleMeasurement, Shots
from pennylane.typing import TensorLike

from pennylane_lightning.core._measurements_base import LightningBaseMeasurements


class LightningMeasurements(LightningBaseMeasurements):  # pylint: disable=too-few-public-methods
    """Lightning Qubit Measurements class

    Measures the state provided by the LightningStateVector class.

    Args:
        qubit_state(LightningStateVector): Lightning state-vector class containing the state vector to be measured.
        mcmc (bool): Determine whether to use the approximate Markov Chain Monte Carlo
            sampling method when generating samples.
        kernel_name (str): name of MCMC transition kernel. The current version supports
            two kernels: ``"Local"`` and ``"NonZeroRandom"``.
            The local kernel conducts a bit-flip local transition between states.
            The local kernel generates a random qubit site and then generates a random
            number to determine the new bit at that qubit site. The ``"NonZeroRandom"`` kernel
            randomly transits between states that have nonzero probability.
        num_burnin (int): number of MCMC steps that will be dropped. Increasing this value will
            result in a closer approximation but increased runtime.
    """

    def __init__(
        self,
        qubit_state: LightningStateVector,  # pylint: disable=undefined-variable
        mcmc: bool = None,
        kernel_name: str = None,
        num_burnin: int = None,
    ) -> None:
        super().__init__(qubit_state)

        self._mcmc = mcmc
        self._kernel_name = kernel_name
        self._num_burnin = num_burnin
        if self._mcmc and not self._kernel_name:
            self._kernel_name = "Local"
        if self._mcmc and not self._num_burnin:
            self._num_burnin = 100

        self._measurement_lightning = self._measurement_dtype()(qubit_state.state_vector)

    def _measurement_dtype(self):
        """Binding to Lightning Measurements C++ class.

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

        if self._mcmc:
            total_indices = self._qubit_state.num_wires
            wires = qml.wires.Wires(range(total_indices))
        else:
            wires = reduce(sum, (mp.wires for mp in mps))

        def _process_single_shot(samples):
            processed = []
            for mp in mps:
                res = mp.process_samples(samples, wires)
                if not isinstance(mp, CountsMP):
                    res = qml.math.squeeze(res)

                processed.append(res)

            return tuple(processed)

        try:
            if self._mcmc:
                samples = self._measurement_lightning.generate_mcmc_samples(
                    len(wires), self._kernel_name, self._num_burnin, shots.total_shots
                ).astype(int, copy=False)
            else:
                samples = self._measurement_lightning.generate_samples(
                    list(wires), shots.total_shots
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
