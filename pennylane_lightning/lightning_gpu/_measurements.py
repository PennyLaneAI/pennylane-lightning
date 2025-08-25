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
        if qubit_state._rng:
            self._measurement_lightning.set_random_seed(qubit_state._rng.integers(0, 2**31 - 1))

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
