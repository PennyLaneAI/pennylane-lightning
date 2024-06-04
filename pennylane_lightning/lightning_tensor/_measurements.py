# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Class implementation for state tensor measurements.
"""

# pylint: disable=import-error, no-name-in-module, ungrouped-imports
try:
    from pennylane_lightning.lightning_tensor_ops import MeasurementsC64, MeasurementsC128
except ImportError:
    pass

from typing import Callable

import numpy as np
import pennylane as qml
from pennylane.measurements import ExpectationMP, MeasurementProcess, StateMeasurement
from pennylane.tape import QuantumScript
from pennylane.typing import Result, TensorLike

from pennylane_lightning.core._serialize import QuantumScriptSerializer


class LightningMeasurements:
    """Lightning Measurements class

    Measures the state provided by the LightningStateTensor class.

    Args:
        tensor_state(LightningStateTensor): Lightning state-tensor class containing the state tensor to be measured.
    """

    def __init__(
        self,
        tensor_state,
    ) -> None:
        self._tensor_state = tensor_state
        self._dtype = tensor_state.dtype
        self._measurement_lightning = self._measurement_dtype()(tensor_state.state_tensor)

    @property
    def dtype(self):
        """Returns the simulation data type."""
        return self._dtype

    def _measurement_dtype(self):
        """Binding to Lightning Measurements C++ class.

        Returns: the Measurements class
        """
        return MeasurementsC64 if self.dtype == np.complex64 else MeasurementsC128

    # pylint: disable=protected-access
    def expval(self, measurementprocess: MeasurementProcess):
        """Expectation value of the supplied observable contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the state

        Returns:
            Expectation value of the observable
        """
        if isinstance(measurementprocess.obs, qml.SparseHamiltonian):
            raise NotImplementedError

        ob_serialized = QuantumScriptSerializer(
            self._tensor_state.device_name, self.dtype == np.complex64
        )._ob(measurementprocess.obs)
        return self._measurement_lightning.expval(ob_serialized)

    def get_measurement_function(
        self, measurementprocess: MeasurementProcess
    ) -> Callable[[MeasurementProcess, TensorLike], TensorLike]:
        """Get the appropriate method for performing a measurement.

        Args:
            measurementprocess (MeasurementProcess): measurement process to apply to the state

        Returns:
            Callable: function that returns the measurement result
        """
        if isinstance(measurementprocess, StateMeasurement):
            if isinstance(measurementprocess, ExpectationMP):
                return self.expval

        raise NotImplementedError

    def measurement(self, measurementprocess: MeasurementProcess) -> TensorLike:
        """Apply a measurement process to a state.

        Args:
            measurementprocess (MeasurementProcess): measurement process to apply to the state

        Returns:
            TensorLike: the result of the measurement
        """
        return self.get_measurement_function(measurementprocess)(measurementprocess)

    def measure_final_state(self, circuit: QuantumScript) -> Result:
        """
        Perform the measurements required by the circuit on the provided state.

        This is an internal function that will be called by the successor to ``lightning.tensor``.

        Args:
            circuit (QuantumScript): The single circuit to simulate

        Returns:
            Tuple[TensorLike]: The measurement results
        """

        if circuit.shots:
            raise NotImplementedError
        # analytic case
        if len(circuit.measurements) == 1:
            return self.measurement(circuit.measurements[0])

        return tuple(self.measurement(mp) for mp in circuit.measurements)
