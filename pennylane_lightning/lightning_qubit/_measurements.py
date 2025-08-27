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

import numpy as np

from pennylane_lightning.lightning_base._measurements import LightningBaseMeasurements


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
        self._subsamples = True
        self._kernel_name = kernel_name
        self._num_burnin = num_burnin
        if self._mcmc and not self._kernel_name:
            self._kernel_name = "Local"
        if self._mcmc and not self._num_burnin:
            self._num_burnin = 100
        self._measurement_lightning = self._measurement_dtype()(qubit_state.state_vector)
        if qubit_state._rng:
            self._measurement_lightning.set_random_seed(qubit_state._rng.integers(0, 2**31 - 1))

    def _measurement_dtype(self):
        """Binding to Lightning Measurements C++ class.

        Returns: the Measurements class
        """
        return MeasurementsC64 if self.dtype == np.complex64 else MeasurementsC128
