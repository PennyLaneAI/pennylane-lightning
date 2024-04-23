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
    from pennylane_lightning.lightning_qubit_ops import MeasurementsC64, MeasurementsC128
except ImportError:
    pass

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


class LightningMeasurements:
    """Lightning Measurements class

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
        qubit_state,
        mcmc: bool = None,
        kernel_name: str = None,
        num_burnin: int = None,
    ) -> None:
        self._qubit_state = qubit_state
        self._dtype = qubit_state.dtype
        self._measurement_lightning = self._measurement_dtype()(qubit_state.state_vector)
        self._mcmc = mcmc
        self._kernel_name = kernel_name
        self._num_burnin = num_burnin
        if self._mcmc and not self._kernel_name:
            self._kernel_name = "Local"
        if self._mcmc and not self._num_burnin:
            self._num_burnin = 100

    @property
    def qubit_state(self):
        """Returns a handle to the LightningStateVector class."""
        return self._qubit_state

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
        self._qubit_state.apply_operations(diagonalizing_gates)
        state_array = self._qubit_state.state
        wires = Wires(range(self._qubit_state.num_wires))
        result = measurementprocess.process_state(state_array, wires)
        self._qubit_state.apply_operations([qml.adjoint(g) for g in reversed(diagonalizing_gates)])
        return result

    # pylint: disable=protected-access
    def expval(self, measurementprocess: MeasurementProcess):
        """Expectation value of the supplied observable contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the state

        Returns:
            Expectation value of the observable
        """

        if isinstance(measurementprocess.obs, qml.SparseHamiltonian):
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
            isinstance(measurementprocess.obs, (qml.ops.Hamiltonian, qml.Hermitian))
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
                [qml.adjoint(g, lazy=False) for g in reversed(diagonalizing_gates)]
            )
        return results

    def var(self, measurementprocess: MeasurementProcess):
        """Variance of the supplied observable contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the state

        Returns:
            Variance of the observable
        """

        if isinstance(measurementprocess.obs, qml.SparseHamiltonian):
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
            isinstance(measurementprocess.obs, (qml.ops.Hamiltonian, qml.Hermitian))
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
                if isinstance(measurementprocess.obs, (qml.Identity, qml.Projector)):
                    return self.state_diagonalizing_gates
                return self.expval

            if isinstance(measurementprocess, ProbabilityMP):
                return self.probs

            if isinstance(measurementprocess, VarianceMP):
                if isinstance(measurementprocess.obs, (qml.Identity, qml.Projector)):
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

    def measure_final_state(self, circuit: QuantumScript, mid_measurements=None) -> Result:
        """
        Perform the measurements required by the circuit on the provided state.

        This is an internal function that will be called by the successor to ``lightning.qubit``.

        Args:
            circuit (QuantumScript): The single circuit to simulate
            mid_measurements (None, dict): Dictionary of mid-circuit measurements

        Returns:
            Tuple[TensorLike]: The measurement results
        """

        if not circuit.shots:
            # analytic case
            if len(circuit.measurements) == 1:
                return self.measurement(circuit.measurements[0])

            return tuple(self.measurement(mp) for mp in circuit.measurements)

        # finite-shot case
        results = self.measure_with_samples(
            circuit.measurements,
            shots=circuit.shots,
            mid_measurements=mid_measurements,
        )

        if len(circuit.measurements) == 1:
            if circuit.shots.has_partitioned_shots:
                return tuple(res[0] for res in results)

            return results[0]

        return results

    # pylint:disable = too-many-arguments
    def measure_with_samples(
        self,
        measurements: List[Union[SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP]],
        shots: Shots,
        mid_measurements=None,
    ) -> List[TensorLike]:
        """
        Returns the samples of the measurement process performed on the given state.
        This function assumes that the user-defined wire labels in the measurement process
        have already been mapped to integer wires used in the device.

        Args:
            measurements (List[Union[SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP]]):
                The sample measurements to perform
            shots (Shots): The number of samples to take
            mid_measurements (None, dict): Dictionary of mid-circuit measurements

        Returns:
            List[TensorLike[Any]]: Sample measurement results
        """
        # last N measurements are sampling MCMs in ``dynamic_one_shot`` execution mode
        mps = measurements[0 : -len(mid_measurements)] if mid_measurements else measurements
        skip_measure = (
            any(v == -1 for v in mid_measurements.values()) if mid_measurements else False
        )

        groups, indices = _group_measurements(mps)

        all_res = []
        for group in groups:
            if skip_measure:
                all_res.extend([None] * len(group))
                continue
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
            if isinstance(group[0], ExpectationMP) and isinstance(group[0].obs, Hamiltonian):
                all_res.extend(self._measure_hamiltonian_with_samples(group, shots))
            elif isinstance(group[0], ExpectationMP) and isinstance(group[0].obs, Sum):
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

        # append MCM samples
        if mid_measurements:
            sorted_res += tuple(mid_measurements.values())

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

        self._qubit_state.apply_operations(diagonalizing_gates)

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

        # if there is a shot vector, build a list containing results for each shot entry
        if shots.has_partitioned_shots:
            processed_samples = []
            for s in shots:
                # currently we call sample_state for each shot entry, but it may be
                # better to call sample_state just once with total_shots, then use
                # the shot_range keyword argument
                try:
                    if self._mcmc:
                        samples = self._measurement_lightning.generate_mcmc_samples(
                            len(wires), self._kernel_name, self._num_burnin, s
                        ).astype(int, copy=False)
                    else:
                        samples = self._measurement_lightning.generate_samples(
                            len(wires), s
                        ).astype(int, copy=False)
                except ValueError as e:
                    if str(e) != "probabilities contain NaN":
                        raise e
                    samples = qml.math.full((s, len(wires)), 0)

                processed_samples.append(_process_single_shot(samples))
            self._apply_diagonalizing_gates(mps, adjoint=True)
            return tuple(zip(*processed_samples))

        try:
            if self._mcmc:
                samples = self._measurement_lightning.generate_mcmc_samples(
                    len(wires), self._kernel_name, self._num_burnin, shots.total_shots
                ).astype(int, copy=False)
            else:
                samples = self._measurement_lightning.generate_samples(
                    len(wires), shots.total_shots
                ).astype(int, copy=False)
        except ValueError as e:
            if str(e) != "probabilities contain NaN":
                raise e
            samples = qml.math.full((shots.total_shots, len(wires)), 0)

        self._apply_diagonalizing_gates(mps, adjoint=True)

        return _process_single_shot(samples)

    def _measure_hamiltonian_with_samples(
        self,
        mp: List[SampleMeasurement],
        shots: Shots,
    ):
        # the list contains only one element based on how we group measurements
        mp = mp[0]

        # if the measurement process involves a Hamiltonian, measure each
        # of the terms separately and sum
        def _sum_for_single_shot(s):
            results = self.measure_with_samples(
                [ExpectationMP(t) for t in mp.obs.terms()[1]],
                s,
            )
            return sum(c * res for c, res in zip(mp.obs.terms()[0], results))

        unsqueezed_results = tuple(_sum_for_single_shot(type(shots)(s)) for s in shots)
        return [unsqueezed_results] if shots.has_partitioned_shots else [unsqueezed_results[0]]

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
