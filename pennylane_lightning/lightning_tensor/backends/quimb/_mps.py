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
import copy
from typing import Callable, Sequence, Union

import pennylane as qml
import quimb.tensor as qtn
from pennylane import numpy as np
from pennylane.devices import DefaultExecutionConfig, ExecutionConfig
from pennylane.devices.preprocess import (
    decompose,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from pennylane.measurements import ExpectationMP, MeasurementProcess, StateMeasurement, VarianceMP
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms.core import TransformProgram
from pennylane.typing import Result, ResultBatch, TensorLike
from pennylane.wires import Wires

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]

_operations = frozenset(
    {
        "Identity",
        "QubitUnitary",
        "ControlledQubitUnitary",
        "MultiControlledX",
        "DiagonalQubitUnitary",
        "PauliX",
        "PauliY",
        "PauliZ",
        "MultiRZ",
        "GlobalPhase",
        "Hadamard",
        "S",
        "Adjoint(S)",
        "T",
        "Adjoint(T)",
        "SX",
        "Adjoint(SX)",
        "CNOT",
        "SWAP",
        "ISWAP",
        "PSWAP",
        "Adjoint(ISWAP)",
        "SISWAP",
        "Adjoint(SISWAP)",
        "SQISW",
        "CSWAP",
        "Toffoli",
        "CY",
        "CZ",
        "PhaseShift",
        "ControlledPhaseShift",
        "CPhase",
        "RX",
        "RY",
        "RZ",
        "Rot",
        "CRX",
        "CRY",
        "CRZ",
        "C(PauliX)",
        "C(PauliY)",
        "C(PauliZ)",
        "C(Hadamard)",
        "C(S)",
        "C(T)",
        "C(PhaseShift)",
        "C(RX)",
        "C(RY)",
        "C(RZ)",
        "C(Rot)",
        "C(SWAP)",
        "C(IsingXX)",
        "C(IsingXY)",
        "C(IsingYY)",
        "C(IsingZZ)",
        "C(SingleExcitation)",
        "C(SingleExcitationMinus)",
        "C(SingleExcitationPlus)",
        "C(DoubleExcitation)",
        "C(MultiRZ)",
        "C(GlobalPhase)",
        "CRot",
        "IsingXX",
        "IsingYY",
        "IsingZZ",
        "IsingXY",
        "SingleExcitation",
        "SingleExcitationPlus",
        "SingleExcitationMinus",
        "DoubleExcitation",
        "QubitCarry",
        "QubitSum",
        "OrbitalRotation",
        "QFT",
        "ECR",
        "BlockEncode",
    }
)
# The set of supported operations.


_observables = frozenset(
    {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "Hermitian",
        "Identity",
        "Projector",
        "SparseHamiltonian",
        "Hamiltonian",
        "LinearCombination",
        "Sum",
        "SProd",
        "Prod",
        "Exp",
    }
)
# The set of supported observables.


def stopping_condition(op: qml.operation.Operator) -> bool:
    """A function that determines if an operation is supported by ``lightning.tensor`` for this interface."""
    return op.name in _operations  # pragma: no cover


def accepted_observables(obs: qml.operation.Operator) -> bool:
    """A function that determines if an observable is supported by ``lightning.tensor`` for this interface."""
    return obs.name in _observables  # pragma: no cover


def decompose_recursive(op: qml.operation.Operator) -> list:
    """Decompose a Pennylane operator into a list of operators with at most 2 wires.

    Args:
        op (Operator): the operator to decompose.

    Returns:
        list[Operator]: a list of operators with at most 2 wires.
    """

    if len(op.wires) <= 2:
        return [op]

    decomposed_ops = []
    for sub_op in op.decomposition():
        decomposed_ops.extend(decompose_recursive(sub_op))

    return decomposed_ops


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
    def name_interf(self) -> str:
        """The name of this interface."""
        return "QuimbMPS interface"

    @property
    def state(self) -> qtn.MatrixProductState:
        """Return the current MPS handled by the interface."""
        return self._circuitMPS.psi

    def state_to_array(self) -> np.ndarray:
        """Contract the MPS into a dense array."""
        return self._circuitMPS.to_dense()

    def _reset_state(self) -> None:
        """Reset the MPS."""
        self._circuitMPS = qtn.CircuitMPS(psi0=self._initial_mps())

    def _initial_mps(self) -> qtn.MatrixProductState:
        r"""
        Return an initial state to :math:`\ket{0}`.

        Internally, it uses `quimb`'s `MPS_computational_state` method.

        Returns:
            MatrixProductState: The initial MPS of a circuit.
        """
        return qtn.MPS_computational_state(**self._init_state_ops)

    def preprocess(self) -> TransformProgram:
        """This function defines the device transform program to be applied for this interface.

        Returns:
            TransformProgram: A transform program that when called returns :class:`~.QuantumTape`'s that the
            device can natively execute as well as a postprocessing function to be called after execution.

        This interface:

        * Supports any one or two-qubit operations that provide a matrix.
        * Supports any three or four-qubit operations that provide a decomposition method.
        * Currently does not support finite shots.
        """

        program = TransformProgram()

        program.add_transform(validate_measurements, name=self.name_interf)
        program.add_transform(validate_observables, accepted_observables, name=self.name_interf)
        program.add_transform(validate_device_wires, self._wires, name=self.name_interf)
        program.add_transform(
            decompose,
            stopping_condition=stopping_condition,
            skip_initial_state_prep=True,
            name=self.name_interf,
        )
        program.add_transform(qml.transforms.broadcast_expand)

        return program

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

        return tuple(results)

    def simulate(self, circuit: QuantumScript) -> Result:
        """Simulate a single quantum script. This function assumes that all operations provide matrices.

        Args:
            circuit (QuantumScript): The single circuit to simulate.

        Returns:
            Tuple[TensorLike]: The results of the simulation.
        """

        self._reset_state()

        for op in circuit.operations:
            self._apply_operation(op)

        if not circuit.shots:
            if len(circuit.measurements) == 1:
                return self.measurement(circuit.measurements[0])
            return tuple(self.measurement(mp) for mp in circuit.measurements)

        raise NotImplementedError

    def _apply_operation(self, op: qml.operation.Operator) -> None:
        """Apply a single operator to the circuit, keeping the state always in a MPS form.

        Internally it uses `quimb`'s `apply_gate` method. For operations that act on more than two wires,
        it decomposes them first into operations that act on at most two wires.

        Args:
            op (Operator): The operation to apply.
        """

        if len(op.wires) <= 2:
            self._circuitMPS.apply_gate(op.matrix(), *op.wires, **self._gate_opts)
        else:
            decom_ops = decompose_recursive(op)
            for o in decom_ops:
                self._circuitMPS.apply_gate(o.matrix(), *o.wires, **self._gate_opts)

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

        result = self._local_expectation(obs.matrix(), tuple(obs.wires))

        return result

    def var(self, measurementprocess: MeasurementProcess) -> float:
        """Variance of the supplied observable contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the MPS.

        Returns:
            Variance of the observable.
        """

        obs = measurementprocess.obs

        obs_mat = obs.matrix()
        expect_squar_op = self._local_expectation(obs_mat @ obs_mat.conj().T, tuple(obs.wires))
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

        # We need to copy the MPS to avoid modifying the original state
        qc = copy.deepcopy(self._circuitMPS)

        exp_val = qc.local_expectation(
            matrix,
            wires,
            **self._expval_opts,
        )

        return float(np.real(exp_val))
