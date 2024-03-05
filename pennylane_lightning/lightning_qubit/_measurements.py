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
try:
    from pennylane_lightning.lightning_qubit_ops import (
        MeasurementsC64,
        MeasurementsC128,
    )
except ImportError:
    pass

from typing import Callable, List

import numpy as np
import pennylane as qml
from pennylane.measurements import (
    ExpectationMP,
    ProbabilityMP,
    MeasurementProcess,
    StateMeasurement,
    VarianceMP,
)
from pennylane.tape import QuantumScript
from pennylane.typing import Result, TensorLike
from pennylane.wires import Wires

from pennylane_lightning.core._serialize import QuantumScriptSerializer

from ._state_vector import LightningStateVector


class LightningMeasurements:
    """Lightning Measurements class

    Measures the state provided by the LightningStateVector class.

    Args:
        qubit_state(LightningStateVector)
    """

    def __init__(self, qubit_state: LightningStateVector) -> None:
        self._qubit_state = qubit_state
        self._state = qubit_state.state_vector
        self._dtype = qubit_state.dtype
        self._measurement_lightning = self._measurement_dtype()(self.state)

    @property
    def qubit_state(self):
        """Returns a handle to the LightningStateVector class."""
        return self._qubit_state

    @property
    def state(self):
        """Returns a handle to the Lightning internal data class."""
        return self._state

    @property
    def dtype(self):
        """Returns the simulation data type."""
        return self._dtype

    def _measurement_dtype(self):
        """Binding to Lightning Measurements C++ class.

        Returns: the Measurements class
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
        wires = Wires(range(self._qubit_state.num_wires))
        return measurementprocess.process_state(state_array, wires)

    # pylint: disable=protected-access
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
            return self._measurement_lightning.expval(
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
                self._qubit_state.device_name, self.dtype == np.complex64
            )._ob(measurementprocess.obs)
            return self._measurement_lightning.expval(ob_serialized)

        return self._measurement_lightning.expval(
            measurementprocess.obs.name, measurementprocess.obs.wires
        )

    # pylint: disable=protected-access
    def probs(self, measurementprocess: MeasurementProcess):
        """Probabilities of the supplied observable or wires contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the state

        Returns:
            Probabilities of the supplied observable or wires
        """
        diagonalizing_gates = measurementprocess.diagonalizing_gates()
        if diagonalizing_gates:
            self._qubit_state.apply_operations(diagonalizing_gates)
        results = self._measurement_lightning.probs(measurementprocess.wires.tolist())
        if diagonalizing_gates:
            self._qubit_state.apply_operations(
                [qml.adjoint(g) for g in reversed(diagonalizing_gates)]
            )
        return results

    # pylint: disable=protected-access
    def var(self, measurementprocess: MeasurementProcess):
        """Variance of the supplied observable contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the state

        Returns:
            Variance of the observable
        """

        if measurementprocess.obs.name == "SparseHamiltonian":
            # ensuring CSR sparse representation.
            CSR_SparseHamiltonian = measurementprocess.obs.sparse_matrix(
                wire_order=list(range(self._qubit_state.num_wires))
            ).tocsr(copy=False)
            return self._measurement_lightning.var(
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
                self._qubit_state.device_name, self.dtype == np.complex64
            )._ob(measurementprocess.obs)
            return self._measurement_lightning.var(ob_serialized)

        return self._measurement_lightning.var(
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

<<<<<<< HEAD
<<<<<<< HEAD
            if isinstance(measurementprocess, ProbabilityMP):
                return self.probs

=======
>>>>>>> c1c03954 (Add qml.var support.)
=======
            if isinstance(measurementprocess, ProbabilityMP):
                return self.probs

>>>>>>> 383efdb1 (Add probs support.)
            if isinstance(measurementprocess, VarianceMP):
                if measurementprocess.obs.name in [
                    "Identity",
                    "Projector",
                ]:
                    return self.state_diagonalizing_gates
                return self.var

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
