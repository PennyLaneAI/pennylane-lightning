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

try:
    from pennylane_lightning.lightning_qubit_ops import (
        MeasurementsC64,
        MeasurementsC128,
    )
except ImportError:
    pass

import numpy as np
from typing import Callable, List

from pennylane.measurements import StateMeasurement, MeasurementProcess, ExpectationMP
from pennylane.typing import TensorLike, Result
from pennylane.tape import QuantumScript
from pennylane.wires import Wires

from typing import List

from ._serialize import QuantumScriptSerializer
from ._state_vector import LightningStateVector


class LightningMeasurements:
    """Lightning Measurements class

    Measures the state provided by the LightningStateVector class.

    Args:
        qubit_state(LightningStateVector)
    """

    def __init__(self, qubit_state: LightningStateVector) -> None:
        self._qubit_state = qubit_state
        self.state = qubit_state.state_vector
        self.dtype = qubit_state.dtype
        self.measurement_lightning = self._measurement_dtype()(self.state)

    def _measurement_dtype(self):
        """Binding to Lightning Managed state vector.

        Args:
            dtype (complex): Data complex type

        Returns: the state vector class
        """
        return MeasurementsC64 if self.dtype == np.complex64 else MeasurementsC128

    def state_diagonalizing_gates(self, measurementprocess: StateMeasurement) -> TensorLike:
        """Apply a measurement to state when the measurement process has an observable with diagonalizing gates.
            This method will is bypassing the measurement process to default.qubit implementation.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the state

        Returns:
            TensorLike: the result of the measurement
        """
        self._qubit_state.apply_operations(measurementprocess.diagonalizing_gates())

        state_array = self._qubit_state.state
        total_wires = int(np.log2(state_array.size))
        wires = Wires(range(total_wires))
        return measurementprocess.process_state(state_array, wires)

    def expval(self, measurementprocess: MeasurementProcess):
        """Expectation value of the supplied observable contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the state

        Returns:
            Expectation value of the observable
        """

        if measurementprocess.obs.name == "SparseHamiltonian":
            # ensuring CSR sparse representation.
            CSR_SparseHamiltonian = measurementprocess.obs.sparse_matrix(
                wire_order=list(range(self._qubit_state.num_wires))
            ).tocsr(copy=False)
            return self.measurement_lightning.expval(
                CSR_SparseHamiltonian.indptr,
                CSR_SparseHamiltonian.indices,
                CSR_SparseHamiltonian.data,
            )

        if (
            measurementprocess.obs.name in ["Hamiltonian", "Hermitian"]
            or (measurementprocess.obs.arithmetic_depth > 0)
            or isinstance(measurementprocess.obs.name, List)
        ):
            ob_serialized = QuantumScriptSerializer(
                self._qubit_state.name, self.dtype == np.complex64
            )._ob(measurementprocess.obs)
            return self.measurement_lightning.expval(ob_serialized)

        return self.measurement_lightning.expval(
            measurementprocess.obs.name, measurementprocess.obs.wires
        )

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
                if measurementprocess.obs.name in [
                    "Identity",
                    "Projector",
                ]:
                    return self.state_diagonalizing_gates
                return self.expval

            if measurementprocess.obs is None or measurementprocess.obs.has_diagonalizing_gates:
                return self.state_diagonalizing_gates

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

        This is an internal function that will be called by the successor to ``lightning.qubit``.

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
