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

from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Callable, List, Union

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
from pennylane.operation import Operator
from pennylane.ops import SparseHamiltonian, Sum
from pennylane.tape import QuantumScript
from pennylane.typing import Result, TensorLike
from pennylane.wires import Wires

from pennylane_lightning.lightning_base._serialize import QuantumScriptSerializer


class LightningBaseMeasurements(ABC):
    """Lightning [Device] Measurements class

    A class that serves as a base class for Lightning state-vector simulators.
    Measures the state provided by the Lightning [Device] StateVector class.

    Args:
        qubit_state(Lightning [Device] StateVector): Lightning state-vector class containing the state vector to be measured.
    """

    def __init__(
        self,
        qubit_state: Any,
    ) -> None:
        self._qubit_state = qubit_state

        # Declare whether to use the approximate Markov Chain Monte Carlo
        # sampling method when generating samples.
        # Currently, only Lightning-Qubit supports this feature.
        self._mcmc = False

        # Declare if generate_samples supports sampling of a subset of wires.
        # Currently, only Lightning-Qubit supports this feature.
        # TODO: implement this for other Lightning devices.
        self._subsamples = False

        # Declare if the device will use the MPI support. Currently,
        # only Lightning-GPU and Lightning-Kokkos supports this feature.
        self._use_mpi = False

        # Dummy for the C++ bindings
        self._measurement_lightning = None

    @property
    def qubit_state(self):
        """Returns a handle to the Lightning [Device] StateVector object."""
        return self._qubit_state

    @property
    def dtype(self):
        """Returns the simulation data type."""
        return self._qubit_state.dtype

    @staticmethod
    def _observable_is_sparse(obs: Operator) -> bool:
        """States if the required observable is sparse.

        Args:
            obs (Operator): PennyLane observable to check sparsity.

        Returns:
            True if the measurement process only uses the sparse data representation.
        """
        return isinstance(obs, SparseHamiltonian)

    @abstractmethod
    def _measurement_dtype(self):
        """Binding to Lightning [Device] Measurements C++ class.

        Returns: the Measurements class
        """

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

    def expval(self, measurementprocess: MeasurementProcess):
        """Expectation value of the supplied observable contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the state

        Returns:
            Expectation value of the observable
        """

        if self._observable_is_sparse(measurementprocess.obs):
            # We first ensure the CSR sparse representation.

            if self._use_mpi:
                if self._qubit_state.device_name == "lightning.kokkos":
                    raise NotImplementedError(
                        "Sparse Hamiltonian is not supported on the lightning.kokkos device with MPI."
                    )

                # Identity for CSR_SparseHamiltonian to pass to processes with rank != 0 to reduce
                # host(cpu) memory requirements
                CSR_SparseHamiltonian = qml.Identity(0).sparse_matrix()
                # CSR_SparseHamiltonian for rank == 0
                if self._mpi_handler.mpi_manager.getRank() == 0:
                    CSR_SparseHamiltonian = measurementprocess.obs.sparse_matrix().tocsr()
            else:
                CSR_SparseHamiltonian = measurementprocess.obs.sparse_matrix(
                    wire_order=list(range(self._qubit_state.num_wires))
                ).tocsr(copy=False)

            return self._measurement_lightning.expval(
                CSR_SparseHamiltonian.indptr,
                CSR_SparseHamiltonian.indices,
                CSR_SparseHamiltonian.data,
            )

        # use specialized function to compute expval(pauli_sentence)
        if measurementprocess.obs.pauli_rep is not None and hasattr(self, "_expval_pauli_sentence"):
            return self._expval_pauli_sentence(measurementprocess)

        # use specialized functors to compute expval(Hermitian)
        if isinstance(measurementprocess.obs, qml.Hermitian):
            observable_wires = measurementprocess.obs.wires
            if self._use_mpi and len(observable_wires) > self._num_local_wires:
                raise RuntimeError(
                    "MPI backend does not support Hermitian with number of target wires larger than local wire number."
                )
            matrix = measurementprocess.obs.matrix()
            return self._measurement_lightning.expval(matrix, observable_wires)

        if measurementprocess.obs.arithmetic_depth or isinstance(measurementprocess.obs.name, List):
            # pylint: disable=protected-access
            ob_serialized = QuantumScriptSerializer(
                self._qubit_state.device_name, self.dtype == np.complex64, self._use_mpi
            )._ob(measurementprocess.obs)
            return self._measurement_lightning.expval(ob_serialized)

        return self._measurement_lightning.expval(
            measurementprocess.obs.name, measurementprocess.obs.wires
        )

    def probs(self, measurementprocess: MeasurementProcess):
        """Probabilities of the supplied observable or wires contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the state.

        Returns:
            Probabilities of the supplied observable or wires.
        """
        diagonalizing_gates = measurementprocess.diagonalizing_gates()

        if diagonalizing_gates:
            self._qubit_state.apply_operations(diagonalizing_gates)

        if measurementprocess.wires == Wires([]):
            # For the case where no wires is specified for statevector
            # and measurement process, wires are determined here
            measurewires = self._qubit_state.wires
        else:
            measurewires = measurementprocess.wires.tolist()
        results = self._measurement_lightning.probs(measurewires)

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

        if self._observable_is_sparse(measurementprocess.obs):
            if self._qubit_state.device_name == "lightning.kokkos" and self._use_mpi:
                raise NotImplementedError(
                    "Sparse Hamiltonian is not supported on the lightning.kokkos device with MPI."
                )

            # Ensuring CSR sparse representation.
            CSR_SparseHamiltonian = measurementprocess.obs.sparse_matrix(
                wire_order=list(range(self._qubit_state.num_wires))
            ).tocsr(copy=False)
            return self._measurement_lightning.var(
                CSR_SparseHamiltonian.indptr,
                CSR_SparseHamiltonian.indices,
                CSR_SparseHamiltonian.data,
            )

        if (
            isinstance(measurementprocess.obs, qml.Hermitian)
            or (measurementprocess.obs.arithmetic_depth > 0)
            or isinstance(measurementprocess.obs.name, List)
        ):
            # pylint: disable=protected-access
            ob_serialized = QuantumScriptSerializer(
                self._qubit_state.device_name, self.dtype == np.complex64, self._use_mpi
            )._ob(measurementprocess.obs)
            return self._measurement_lightning.var(ob_serialized)

        return self._measurement_lightning.var(
            measurementprocess.obs.name, measurementprocess.obs.wires
        )

    def get_measurement_function(
        self, measurementprocess: MeasurementProcess
    ) -> Callable[[MeasurementProcess, TensorLike], TensorLike]:
        # pylint: disable=too-many-return-statements
        """Get the appropriate method for performing a measurement.

        Args:
            measurementprocess (MeasurementProcess): measurement process to apply to the state

        Returns:
            Callable: function that returns the measurement result
        """
        if isinstance(measurementprocess, StateMeasurement):
            if isinstance(measurementprocess, ExpectationMP) and measurementprocess.obs is not None:
                if self._use_mpi:
                    if isinstance(measurementprocess.obs, (qml.Projector)):
                        return self.state_diagonalizing_gates
                else:
                    if isinstance(measurementprocess.obs, (qml.Identity, qml.Projector)):
                        return self.state_diagonalizing_gates
                return self.expval

            if isinstance(measurementprocess, ProbabilityMP):
                return self.probs

            if isinstance(measurementprocess, VarianceMP) and measurementprocess.obs is not None:
                if self._use_mpi:
                    if isinstance(measurementprocess.obs, (qml.Projector)):
                        return self.state_diagonalizing_gates
                else:
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

        This is an internal function that will be called by the successor to ``lightning.[device]``.

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

    def measure_with_samples(
        self,
        measurements: List[Union[SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP]],
        shots: Shots,
        mid_measurements: dict = None,
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

        for measurement in mps:
            if measurement.wires == Wires([]):
                # This is required for the case where no wires is specific for the statevector
                # (i.e. dynamically determined from circuit), and no wires (and no observable)
                # is provided for the measurement (e.g. qml.probs() or qml.counts() or
                # qml.samples()). In the case where number of wires is provided for the statevector,
                # the same operation is performed in validate_device_wires during preprocess.

                # pylint:disable=protected-access
                measurement._wires = Wires(range(self._qubit_state.num_wires))

        groups, indices = _group_measurements(mps)

        all_res = []
        for group in groups:
            if isinstance(group[0], (ExpectationMP, VarianceMP)) and self._observable_is_sparse(
                group[0].obs
            ):
                raise TypeError(
                    "ExpectationMP/VarianceMP of sparse observables cannot be computed with samples."
                )
            if isinstance(group[0], VarianceMP) and isinstance(group[0].obs, Sum):
                raise TypeError("VarianceMP(Sum) cannot be computed with samples.")
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

        if not self._mcmc and self._subsamples:
            # To speed up the sampling process, we only need to sample a minimal
            # subset of wires when executing in finite-shot mode.
            wires = reduce(sum, (mp.wires for mp in mps))
        else:
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

        try:
            if self._mcmc:
                samples = self._measurement_lightning.generate_mcmc_samples(
                    len(wires), self._kernel_name, self._num_burnin, shots.total_shots
                ).astype(int, copy=False)
            elif self._subsamples:
                samples = self._measurement_lightning.generate_samples(
                    list(wires), shots.total_shots
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
