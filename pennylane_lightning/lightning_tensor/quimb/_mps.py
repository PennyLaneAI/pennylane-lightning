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
Class implementation for the Quimb MPS interface for simulating quantum circuits while keeping the state always in MPS form.
"""

from typing import Callable, Sequence, Union

import pennylane as qml
import quimb.tensor as qtn
from pennylane import numpy as np
from pennylane.devices import DefaultExecutionConfig, ExecutionConfig
from pennylane.measurements import ExpectationMP, MeasurementProcess, StateMeasurement, VarianceMP
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.typing import Result, ResultBatch, TensorLike
from pennylane.wires import Wires

from ._utils import from_op_to_mpo

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]

_operations = frozenset({})  # pragma: no cover
# The set of supported operations.

_observables = frozenset({})  # pragma: no cover
# The set of supported observables.


def stopping_condition(op: qml.operation.Operator) -> bool:
    """A function that determines if an operation is supported by ``lightning.tensor`` for this interface."""
    return op.name in _operations  # pragma: no cover


def accepted_observables(obs: qml.operation.Operator) -> bool:
    """A function that determines if an observable is supported by ``lightning.tensor`` for this interface."""
    return obs.name in _observables  # pragma: no cover


class QuimbMPS:
    """Quimb MPS class.

    Used internally by the `LightningTensor` device.
    Interfaces with `quimb` for MPS manipulation, and provides methods to execute quantum circuits.

    Args:
        num_wires (int): the number of wires in the circuit.
        interf_opts (dict): dictionary containing the interface options.
        dtype (np.dtype): the complex type used for the MPS.
    """

    def __init__(self, num_wires, interf_opts, dtype=np.complex128):

        if dtype not in [np.complex64, np.complex128]:  # pragma: no cover
            raise TypeError(f"Unsupported complex type: {dtype}")

        self._wires = Wires(range(num_wires))
        self._dtype = dtype

        self._init_state_ops = {
            "binary": "0" * max(1, len(self._wires)),
            "dtype": self._dtype.__name__,
            "tags": [str(l) for l in self._wires.labels],
        }

        self._gate_opts = {
            "contract": "swap+split",
            "parametrize": None,
            "cutoff": interf_opts["cutoff"],
            "max_bond": interf_opts["max_bond_dim"],
        }

        self._expval_opts = {
            "dtype": self._dtype.__name__,
            "simplify_sequence": "ADCRS",
            "simplify_atol": 0.0,
        }

        self._circuitMPS = qtn.CircuitMPS(psi0=self._initial_mps())

    @property
    def state(self):
        """Current MPS handled by the interface."""
        return self._circuitMPS.psi

    def state_to_array(self, digits: int = 5):
        """Contract the MPS into a dense array."""
        return self._circuitMPS.to_dense().round(digits)

    def _reset_state(self) -> None:
        """Reset the MPS."""
        self._circuitMPS = qtn.CircuitMPS(psi0=self._initial_mps())

    def _initial_mps(self) -> qtn.MatrixProductState:
        r"""
        Returns an initial state to :math:`\ket{0}`.

        Internally, it uses `quimb`'s `MPS_computational_state` method.

        Returns:
            MatrixProductState: The initial MPS of a circuit.
        """

        return qtn.MPS_computational_state(**self._init_state_ops)

    # pylint: disable=unused-argument
    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        """Execute a circuit or a batch of circuits and turn it into results.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the quantum circuits to be executed
            execution_config (ExecutionConfig): a datastructure with additional information required for execution

        Returns:
            TensorLike, tuple[TensorLike], tuple[tuple[TensorLike]]: A numeric result of the computation.
        """

        results = []
        for circuit in circuits:
            circuit = circuit.map_to_standard_wires()
            results.append(self.simulate(circuit))

        # TODO: add option for self._return_tn

        return tuple(results)

    def simulate(self, circuit: QuantumScript) -> Result:
        """Simulate a single quantum script. This function assumes that all operations provide matrices.

        Args:
            circuit (QuantumScript): The single circuit to simulate.

        Returns:
            Tuple[TensorLike]: The results of the simulation.
        """

        self._reset_state()

        ##############################################################
        ### PART 1: Applying operations
        ##############################################################

        for op in circuit.operations:
            self._apply_operation(op)

        ##############################################################
        ### PART 2: Measurements
        ##############################################################

        if not circuit.shots:
            if len(circuit.measurements) == 1:
                return self.measurement(circuit.measurements[0])
            return tuple(self.measurement(mp) for mp in circuit.measurements)

        raise NotImplementedError

    def _apply_operation(self, op: qml.operation.Operator) -> None:
        """Apply a single operator to the circuit, keeping the state always in a MPS form.

        Internally it uses `quimb`'s `apply_gate` method for one and two-qubit gates.
        For operations that act on more than two wires, it converts the operation to a MPO and applies it to the MPS.

        Args:
            op (Operator): The operation to apply.
        """

        if len(op.wires) <= 2:

            self._circuitMPS.apply_gate(op.matrix(), *op.wires, **self._gate_opts)

        else:

            # If the operation is not a one or two-qubit gate, we need to explicitly convert it to a MPO
            # and apply it to the MPS of the circuit.

            mat_prod_op = from_op_to_mpo(op, self._circuitMPS.psi)
            new_state = mat_prod_op.apply(self._circuitMPS.psi, compress=True)
            self._circuitMPS._psi.__dict__.update(new_state.__dict__)

            gate = qtn.circuit.parse_to_gate(op.matrix(), tuple(op.wires))
            self._circuitMPS.gates.append(gate)

    def measurement(self, measurementprocess: MeasurementProcess) -> TensorLike:
        """Measure the measurement required by the circuit over the MPS.

        Args:
            measurementprocess (MeasurementProcess): measurement to apply to the state.

        Returns:
            TensorLike: the result of the measurement.
        """

        return self._get_measurement_function(measurementprocess)(measurementprocess)

    def _get_measurement_function(
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

            if isinstance(measurementprocess, VarianceMP):
                return self.var

        raise NotImplementedError

    def expval(self, measurementprocess: MeasurementProcess) -> float:
        """Expectation value of the supplied observable contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the MPS.

        Returns:
            Expectation value of the observable.
        """

        obs = measurementprocess.obs

        return self._local_expectation(obs.matrix(), tuple(obs.wires))

    def var(self, measurementprocess: MeasurementProcess) -> float:
        """Variance of the supplied observable contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the MPS.

        Returns:
            Variance of the observable.
        """

        obs = measurementprocess.obs

        obs_mat = obs.matrix()
        expect_squar_op = self._local_expectation(np.dot(obs_mat, obs_mat), tuple(obs.wires))
        expect_op = self._local_expectation(obs_mat, tuple(obs.wires))

        return expect_squar_op - np.square(expect_op)

    def _local_expectation(self, matrix, wires) -> float:
        """Compute the local expectation value of a matrix on the MPS.

        Internally, it uses `quimb`'s `local_expectation` method.

        Args:
            matrix (array): the matrix to compute the expectation value of.
            wires (tuple[int]): the wires the matrix acts on.

        Returns:
            Local expectation value of the matrix on the MPS.
        """

        return np.real(
            self._circuitMPS.local_expectation(
                matrix,
                wires,
                **self._expval_opts,
            )
        )
