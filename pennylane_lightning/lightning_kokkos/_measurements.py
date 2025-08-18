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

    try:
        from pennylane_lightning.lightning_kokkos_ops import MeasurementsMPIC64, MeasurementsMPIC128

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

# pylint: disable=ungrouped-imports
from pennylane_lightning.lightning_base._measurements import LightningBaseMeasurements


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

        self._use_mpi = kokkos_state._mpi

        if self._use_mpi:
            self._num_local_wires = kokkos_state._qubit_state.getNumLocalWires()

        self._measurement_lightning = self._measurement_dtype()(kokkos_state.state_vector)
        if kokkos_state._rng:
            self._measurement_lightning.set_random_seed(kokkos_state._rng.integers(0, 2**31 - 1))

    def _measurement_dtype(self):
        """Binding to Lightning Kokkos Measurements C++ class.

        Returns: the Measurements class
        """
        if self._use_mpi:
            if not MPI_SUPPORT:
                warn(str(mpi_error), UserWarning)

            return MeasurementsMPIC64 if self.dtype == np.complex64 else MeasurementsMPIC128

        # without MPI
        return MeasurementsC64 if self.dtype == np.complex64 else MeasurementsC128

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
