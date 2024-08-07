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
Class implementation for tensornet measurements.
"""

# pylint: disable=import-error, no-name-in-module, ungrouped-imports
try:
    from pennylane_lightning.lightning_tensor_ops import MeasurementsC64, MeasurementsC128
except ImportError:
    pass

from typing import Callable

import numpy as np
import pennylane as qml
from pennylane.measurements import (
    ExpectationMP,
    MeasurementProcess,
    ProbabilityMP,
    StateMeasurement,
    VarianceMP,
)
from pennylane.tape import QuantumScript
from pennylane.typing import Result, TensorLike
from pennylane.wires import Wires

from pennylane_lightning.core._serialize import QuantumScriptSerializer


class LightningTensorMeasurements:
    """Lightning Tensor Measurements class

    Measures the tensor network provided by the LightningTensorNet class.

    Args:
        tensor_network(LightningTensorNet): Lightning tensornet class containing the tensor network to be measured.
    """

    def __init__(
        self,
        tensor_network,
    ) -> None:
        self._tensornet = tensor_network
        self._dtype = tensor_network.dtype
        self._measurement_lightning = self._measurement_dtype()(tensor_network.tensornet)

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
            This method is bypassing the measurement process to default.qubit implementation.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the state

        Returns:
            TensorLike: the result of the measurement
        """
        diagonalizing_gates = measurementprocess.diagonalizing_gates()
        self._tensornet.apply_operations(diagonalizing_gates)
        state_array = self._tensornet.state
        wires = Wires(range(self._tensornet.num_wires))
        result = measurementprocess.process_state(state_array, wires)
        self._tensornet.apply_operations([qml.adjoint(g) for g in reversed(diagonalizing_gates)])
        return result

    # pylint: disable=protected-access
    def expval(self, measurementprocess: MeasurementProcess):
        """Expectation value of the supplied observable contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the tensor network

        Returns:
            Expectation value of the observable
        """
        if isinstance(measurementprocess.obs, qml.SparseHamiltonian):
            raise NotImplementedError("Sparse Hamiltonians are not supported.")

        if isinstance(measurementprocess.obs, qml.Hermitian):
            if len(measurementprocess.obs.wires) > 1:
                raise ValueError("The number of Hermitian observables target wires should be 1.")

        ob_serialized = QuantumScriptSerializer(
            self._tensornet.device_name, self.dtype == np.complex64
        )._ob(measurementprocess.obs)
        return self._measurement_lightning.expval(ob_serialized)

    def probs(self, measurementprocess: MeasurementProcess):
        """Probabilities of the supplied observable or wires contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the state

        Returns:
            Probabilities of the supplied observable or wires
        """
        diagonalizing_gates = measurementprocess.diagonalizing_gates()
        if diagonalizing_gates:
            self._tensornet.apply_operations(diagonalizing_gates)
        results = self._measurement_lightning.probs(measurementprocess.wires.tolist())
        if diagonalizing_gates:
            self._tensornet.apply_operations(
                [qml.adjoint(g, lazy=False) for g in reversed(diagonalizing_gates)]
            )
        return results

    def var(self, measurementprocess: MeasurementProcess):
        """Variance of the supplied observable contained in the MeasurementProcess. Note that the variance is
        calculated as <obs**2> - <obs>**2. The current implementation only supports single-wire observables.
        Observables with more than 1 wire, projector and sparse-hamiltonian are not supported.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the state

        Returns:
            Variance of the observable
        """
        if isinstance(measurementprocess.obs, qml.SparseHamiltonian):
            raise NotImplementedError("Sparse Hamiltonian Observables are not supported.")

        if isinstance(measurementprocess.obs, qml.Hermitian):
            if len(measurementprocess.obs.wires) > 1:
                raise ValueError("The number of Hermitian observables target wires should be 1.")

        ob_serialized = QuantumScriptSerializer(
            self._tensornet.device_name, self.dtype == np.complex64
        )._ob(measurementprocess.obs)
        return self._measurement_lightning.var(ob_serialized)

    def get_measurement_function(
        self, measurementprocess: MeasurementProcess
    ) -> Callable[[MeasurementProcess, TensorLike], TensorLike]:
        """Get the appropriate method for performing a measurement.

        Args:
            measurementprocess (MeasurementProcess): measurement process to apply to the graph

        Returns:
            Callable: function that returns the measurement result
        """
        if isinstance(measurementprocess, StateMeasurement):
            if isinstance(measurementprocess, ExpectationMP):
                return self.expval

            if isinstance(measurementprocess, VarianceMP):
                return self.var

            if isinstance(measurementprocess, ProbabilityMP):
                return self.probs

            if measurementprocess.obs is None:
                return self.state_diagonalizing_gates

        raise NotImplementedError("Not supported measurement.")

    def measurement(self, measurementprocess: MeasurementProcess) -> TensorLike:
        """Apply a measurement process to a tensor network.

        Args:
            measurementprocess (MeasurementProcess): measurement process to apply to the graph

        Returns:
            TensorLike: the result of the measurement
        """
        return self.get_measurement_function(measurementprocess)(measurementprocess)

    def measure_tensor_network(self, circuit: QuantumScript) -> Result:
        """
        Perform the measurements required by the circuit on the provided tensor network.

        This is an internal function that will be called by the successor to ``lightning.tensor``.

        Args:
            circuit (QuantumScript): The single circuit to simulate

        Returns:
            Tuple[TensorLike]: The measurement results
        """

        if circuit.shots:
            raise NotImplementedError("Shots are not supported for tensor network simulations.")
        # analytic case
        if len(circuit.measurements) == 1:
            return self.measurement(circuit.measurements[0])

        return tuple(self.measurement(mp) for mp in circuit.measurements)
