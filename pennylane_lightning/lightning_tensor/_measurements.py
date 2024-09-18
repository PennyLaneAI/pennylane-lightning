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

from functools import reduce
from typing import Callable, List, Union

import numpy as np
import pennylane as qml
from pennylane.devices.qubit.sampling import _group_measurements
from pennylane.measurements import (
    ClassicalShadowMP,
    CountsMP,
    ExpectationMP,
    MeasurementProcess,
    ProbabilityMP,
    SampleMeasurement,
    ShadowExpvalMP,
    Shots,
    StateMeasurement,
    VarianceMP,
)
from pennylane.ops import Hamiltonian, SparseHamiltonian, Sum
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
        self._tensornet.appendMPSFinalState()
        state_array = self._tensornet.state
        wires = Wires(range(self._tensornet.num_wires))
        result = measurementprocess.process_state(state_array, wires)
        self._tensornet.apply_operations([qml.adjoint(g) for g in reversed(diagonalizing_gates)])
        self._tensornet.appendMPSFinalState()
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
            self._tensornet.appendMPSFinalState()
        results = self._measurement_lightning.probs(measurementprocess.wires.tolist())
        if diagonalizing_gates:
            self._tensornet.apply_operations(
                [qml.adjoint(g, lazy=False) for g in reversed(diagonalizing_gates)]
            )
            self._tensornet.appendMPSFinalState()
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
            raise NotImplementedError(
                "The var measurement does not support sparse Hamiltonian observables."
            )

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
                if isinstance(measurementprocess.obs, qml.Identity):
                    return self.state_diagonalizing_gates
                return self.expval

            if isinstance(measurementprocess, VarianceMP):
                if isinstance(measurementprocess.obs, qml.Identity):
                    return self.state_diagonalizing_gates
                return self.var

            if isinstance(measurementprocess, ProbabilityMP):
                return self.probs

            if measurementprocess.obs is None or measurementprocess.obs.has_diagonalizing_gates:
                return self.state_diagonalizing_gates

        raise NotImplementedError("Unsupported measurement type.")

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
            # finite-shot case
            results = self.measure_with_samples(
                circuit.measurements,
                shots=circuit.shots,
            )

            if len(circuit.measurements) == 1:
                if circuit.shots.has_partitioned_shots:
                    return tuple(res[0] for res in results)

                return results[0]

            return results
        # analytic case
        if len(circuit.measurements) == 1:
            return self.measurement(circuit.measurements[0])

        return tuple(self.measurement(mp) for mp in circuit.measurements)

    # pylint:disable = too-many-arguments
    def measure_with_samples(
        self,
        measurements: List[Union[SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP]],
        shots: Shots,
    ) -> List[TensorLike]:
        """
        Returns the samples of the measurement process performed on the given state.
        This function assumes that the user-defined wire labels in the measurement process
        have already been mapped to integer wires used in the device.

        Args:
            measurements (List[Union[SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP]]):
                The sample measurements to perform
            shots (Shots): The number of samples to take

        Returns:
            List[TensorLike[Any]]: Sample measurement results
        """
        mps = measurements
        groups, indices = _group_measurements(mps)

        all_res = []
        for group in groups:
            if isinstance(group[0], (ExpectationMP, VarianceMP)) and isinstance(
                group[0].obs, SparseHamiltonian
            ):
                raise TypeError(
                    "ExpectationMP/VarianceMP(SparseHamiltonian) cannot be computed with samples."
                )
            if isinstance(group[0], VarianceMP) and isinstance(group[0].obs, (Hamiltonian, Sum)):
                raise TypeError("VarianceMP(Hamiltonian/Sum) cannot be computed with samples.")
            if isinstance(group[0], (ClassicalShadowMP, ShadowExpvalMP)):
                raise TypeError(
                    "ExpectationMP(ClassicalShadowMP, ShadowExpvalMP) cannot be computed with samples."
                )
            if isinstance(group[0], ExpectationMP) and isinstance(group[0].obs, Sum):
                all_res.extend(self._measure_sum_with_samples(group, shots))
            else:
                all_res.extend(self._measure_with_samples_diagonalizing_gates(group, shots))

        # reorder results
        flat_indices = []
        for row in indices:
            flat_indices += row
        sorted_res = tuple(
            res for _, res in sorted(list(enumerate(all_res)), key=lambda r: flat_indices[r[0]])
        )

        # put the shot vector axis before the measurement axis
        if shots.has_partitioned_shots:
            sorted_res = tuple(zip(*sorted_res))

        return sorted_res

    def _apply_diagonalizing_gates(self, mps: List[SampleMeasurement], adjoint: bool = False):
        if len(mps) == 1:
            diagonalizing_gates = mps[0].diagonalizing_gates()
        elif all(mp.obs for mp in mps):
            diagonalizing_gates = qml.pauli.diagonalize_qwc_pauli_words([mp.obs for mp in mps])[0]
        else:
            diagonalizing_gates = []

        if adjoint:
            diagonalizing_gates = [
                qml.adjoint(g, lazy=False) for g in reversed(diagonalizing_gates)
            ]

        self._tensornet.apply_operations(diagonalizing_gates)
        self._tensornet.appendMPSFinalState()

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

    def _measure_sum_with_samples(
        self,
        mp: List[SampleMeasurement],
        shots: Shots,
    ):
        # the list contains only one element based on how we group measurements
        mp = mp[0]

        # if the measurement process involves a Sum, measure each
        # of the terms separately and sum
        def _sum_for_single_shot(s):
            results = self.measure_with_samples(
                [ExpectationMP(t) for t in mp.obs],
                s,
            )
            return sum(results)

        unsqueezed_results = tuple(_sum_for_single_shot(type(shots)(s)) for s in shots)
        return [unsqueezed_results] if shots.has_partitioned_shots else [unsqueezed_results[0]]
