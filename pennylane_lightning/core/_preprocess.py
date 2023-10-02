# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains functions for preprocessing `QuantumTape` objects to ensure
that they are supported for execution by a device."""
# pylint: disable=protected-access
from dataclasses import replace
from typing import Generator, Callable, Tuple, Union, Sequence
from copy import copy
import warnings
from functools import partial

import pennylane as qml

from pennylane.operation import Tensor, StatePrepBase, Operation
from pennylane.measurements import (
    MidMeasureMP,
    StateMeasurement,
    SampleMeasurement,
    ExpectationMP,
    ClassicalShadowMP,
    ShadowExpvalMP,
)
from pennylane.typing import ResultBatch, Result
from pennylane import DeviceError
from pennylane.transforms.core import transform, TransformProgram
from pennylane.wires import WireError

from pennylane.devices import ExecutionConfig, DefaultExecutionConfig

PostprocessingFn = Callable[[ResultBatch], Union[Result, ResultBatch]]

# Lightning default list of supported observables.
_observables = {
    "PauliX",
    "PauliY",
    "PauliZ",
    "Hadamard",
    "Hermitian",
    "Identity",
    "Projector",
    "SparseHamiltonian",
    "Hamiltonian",
    "Sum",
    "SProd",
    "Prod",
    "Exp",
}

# Lightning default list of supported operations.
_operations = {
    "Identity",
    # "BasisState",
    "QubitStateVector",
    # "StatePrep",
    "QubitUnitary",
    "ControlledQubitUnitary",
    "MultiControlledX",
    "DiagonalQubitUnitary",
    "PauliX",
    "PauliY",
    "PauliZ",
    "MultiRZ",
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
    "CRot",
    "IsingXX",
    "IsingYY",
    "IsingZZ",
    "IsingXY",
    "SingleExcitation",
    "SingleExcitationPlus",
    "SingleExcitationMinus",
    "DoubleExcitation",
    "DoubleExcitationPlus",
    "DoubleExcitationMinus",
    "QubitCarry",
    "QubitSum",
    "OrbitalRotation",
    "QFT",
    "ECR",
}

### UTILITY FUNCTIONS FOR EXPANDING UNSUPPORTED OPERATIONS ###

def _accepted_operator(op: qml.operation.Operator, dev_operations_list=_operations) -> bool:
    """Specify whether or not an Operator object is accepted to be decomposed by the device."""
    # print("op", op, len(op.wires))
    if op.name == "QFT":
        return len(op.wires) < 10
    if op.name == "GroverOperator":
        return len(op.wires) < 13
    return (op.name in dev_operations_list) or op.has_matrix


def _accepted_adjoint_operator(op: qml.operation.Operator) -> bool:
    """Specify whether or not an Operator is supported by adjoint differentiation."""
    return op.num_params == 0 or op.num_params == 1 and op.has_generator


def _operator_decomposition_gen(
    op: qml.operation.Operator, acceptance_function: Callable[[qml.operation.Operator], bool]
) -> Generator[qml.operation.Operator, None, None]:
    """A generator that yields the next operation that is accepted by Lightning."""
    # print("--> op_gen", op)
    if acceptance_function(op):
        yield op
    else:
        try:
            decomp = op.decomposition()
        except qml.operation.DecompositionUndefinedError as e:
            raise DeviceError(
                f"Operator {op} not supported on Lightning. Must provide either a matrix or a decomposition."
            ) from e

        for sub_op in decomp:
            yield from _operator_decomposition_gen(sub_op, acceptance_function)


@transform
def validate_device_wires(
    tape: qml.tape.QuantumTape, device
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Validates the device wires.

    Args:
        tape (QuantumTape): a quantum circuit.
        device (pennylane.devices.Device): The device to be checked.

    Returns:
        pennylane.QNode or qfunc or Tuple[List[.QuantumTape], Callable]: If a QNode is passed,
        it returns a QNode with the transform added to its transform program.
        If a tape is passed, returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions.
    """
    if device.wires:
        if extra_wires := set(tape.wires) - set(device.wires):
            raise WireError(
                f"Cannot run circuit(s) on {device.name} as they contain wires "
                f"not found on the device: {extra_wires}"
            )
        measurements = tape.measurements.copy()
        modified = False
        for m_idx, mp in enumerate(measurements):
            if not mp.obs and not mp.wires:
                modified = True
                new_mp = copy(mp)
                new_mp._wires = device.wires  # pylint:disable=protected-access
                measurements[m_idx] = new_mp
        if modified:
            tape = type(tape)(tape.operations, measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocessing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [tape], null_postprocessing


@transform
def validate_and_expand_adjoint(
    tape: qml.tape.QuantumTape,
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Function for validating that the operations and observables present in the input tape
    are valid for adjoint differentiation.

    Args:
        tape(.QuantumTape): the tape to validate

    Returns:
        pennylane.QNode or qfunc or Tuple[List[.QuantumTape], Callable]: If a QNode is passed,
        it returns a QNode with the transform added to its transform program.
        If a tape is passed, returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions.
    """
    try:
        new_ops = [
            final_op
            for op in tape.operations[tape.num_preps :]
            for final_op in _operator_decomposition_gen(op, _accepted_adjoint_operator)
        ]
    except RecursionError as e:
        raise DeviceError(
            "Reached recursion limit trying to decompose operations. "
            "Operator decomposition may have entered an infinite loop."
        ) from e

    for k in tape.trainable_params:
        if hasattr(tape._par_info[k]["op"], "return_type"):
            warnings.warn(
                "Differentiating with respect to the input parameters of "
                f"{tape._par_info[k]['op'].name} is not supported with the "
                "adjoint differentiation method. Gradients are computed "
                "only with regards to the trainable parameters of the circuit.\n\n Mark "
                "the parameters of the measured observables as non-trainable "
                "to silence this warning.",
                UserWarning,
            )

    # Check validity of measurements
    measurements = []
    for m in tape.measurements:
        if not isinstance(m, ExpectationMP):
            raise DeviceError(
                "Adjoint differentiation method does not support "
                f"measurement {m.__class__.__name__}."
            )

        if m.obs.name == "Hamiltonian":
            if not all(tuple(t.has_matrix for t in m.obs.ops)):
                raise DeviceError(
                    f"Adjoint differentiation method does not support some of the Hamiltonian terms."
                )
        elif not m.obs.has_matrix:
            raise DeviceError(
                f"Adjoint differentiation method does not support observable {m.obs.name}."
            )

        measurements.append(m)

    new_ops = tape.operations[: tape.num_preps] + new_ops
    new_tape = qml.tape.QuantumScript(new_ops, measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocessing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing


@transform
def validate_measurements(
    tape: qml.tape.QuantumTape, execution_config: ExecutionConfig = DefaultExecutionConfig
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Check that the circuit contains a valid set of measurements. A valid
    set of measurements is defined as:

    1. If circuit.shots is None (i.e., the execution is analytic), then
       the circuit must only contain ``StateMeasurements``.
    2. If circuit.shots is not None, then the circuit must only contain
       ``SampleMeasurements``.

    If the circuit has an invalid set of measurements, then an error is raised.

    Args:
        tape (.QuantumTape): the circuit to validate
        execution_config (.ExecutionConfig): execution configuration with configurable
            options for the execution.

    Returns:
        pennylane.QNode or qfunc or Tuple[List[.QuantumTape], Callable]: If a QNode is passed,
        it returns a QNode with the transform added to its transform program.
        If a tape is passed, returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions.
    """
    if not tape.shots:
        for m in tape.measurements:
            if not isinstance(m, StateMeasurement):
                raise DeviceError(f"Analytic circuits must only contain StateMeasurements; got {m}")
    else:
        # check if an analytic diff method is used with finite shots
        if execution_config.gradient_method in ["adjoint", "backprop"]:
            raise DeviceError(
                f"Circuits with finite shots must be executed with non-analytic "
                f"gradient methods; got {execution_config.gradient_method}"
            )

        # for m in circuit.measurements:
        #     if not isinstance(m, (SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP)):
        #         raise DeviceError(
        #             f"Circuits with finite shots must only contain SampleMeasurements, ClassicalShadowMP, or ShadowExpvalMP; got {m}"
        #         )

    def null_postprocessing(results):
        """A postprocessing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [tape], null_postprocessing


@transform
def expand_fn(
    tape: qml.tape.QuantumTape, observables_list=_observables
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Method for expanding or decomposing an input tape.

    This method expands the tape if:

    - mid-circuit measurements are present,
    - any operations are not supported on the device.

    Args:
        tape (.QuantumTape): the tape to expand.

    Returns:
        pennylane.QNode or qfunc or Tuple[List[.QuantumTape], Callable]: If a QNode is passed,
        it returns a QNode with the transform added to its transform program.
        If a tape is passed, returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions.
    """

    if any(isinstance(o, MidMeasureMP) for o in tape.operations):
        tapes, _ = qml.defer_measurements(tape)
        tape = tapes[0]

    # for op in tape.operations:
    #     print("====>", op, _accepted_operator(op))

    if not all(_accepted_operator(op) for op in tape.operations):
        # print("====> not all accepted")
        try:
            # don't decompose initial operations if its StatePrepBase
            prep_op = [tape[0]] if isinstance(tape[0], StatePrepBase) else []

            new_ops = [
                final_op
                for op in tape.operations[bool(prep_op) :]
                for final_op in _operator_decomposition_gen(op, _accepted_operator)
            ]
        except RecursionError as e:
            raise DeviceError(
                "Reached recursion limit trying to decompose operations. "
                "Operator decomposition may have entered an infinite loop."
            ) from e
        tape = qml.tape.QuantumScript(prep_op + new_ops, tape.measurements, shots=tape.shots)

    for observable in tape.observables:
        if isinstance(observable, Tensor):
            if any(o.name not in observables_list for o in observable.obs):
                raise DeviceError(f"Observable {observable} not supported on Lightning")
        elif observable.name not in observables_list:
            raise DeviceError(f"Observable {observable} not supported on Lightning")

    def null_postprocessing(results):
        """A postprocessing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [tape], null_postprocessing


# def batch_transform(
#     circuit: qml.tape.QuantumScript, execution_config: ExecutionConfig = DefaultExecutionConfig
# ) -> Tuple[Tuple[qml.tape.QuantumScript], PostprocessingFn]:
#     """Apply a differentiable batch transform for preprocessing a circuit
#     prior to execution.

#     By default, this method contains logic for generating multiple
#     circuits, one per term, of a circuit that terminates in ``expval(Sum)``.

#     .. warning::

#         This method will be tracked by auto-differentiation libraries,
#         such as Autograd, JAX, TensorFlow, and Torch. Please make sure
#         to use ``qml.math`` for autodiff-agnostic tensor processing
#         if required.

#     Args:
#         circuit (.QuantumTape): the circuit to preprocess
#         execution_config (.ExecutionConfig): execution configuration with configurable
#             options for the execution.

#     Returns:
#         tuple[Sequence[.QuantumTape], callable]: Returns a tuple containing
#         the sequence of circuits to be executed, and a post-processing function
#         to be applied to the list of evaluated circuit results.
#     """
#     # Check whether the circuit was broadcasted or if the diff method is anything other than adjoint
#     if circuit.batch_size is None or execution_config.gradient_method != "adjoint":
#         # If the circuit wasn't broadcasted, or if built-in PennyLane broadcasting
#         # can be used, then no action required
#         circuits = [circuit]

#         def batch_fn(res: ResultBatch) -> Result:
#             """A post-processing function to convert the results of a batch of
#             executions into the result of a single execution."""
#             return res[0]

#         return circuits, batch_fn

#     # Expand each of the broadcasted circuits
#     tapes, batch_fn = qml.transforms.broadcast_expand(circuit)

#     return tapes, batch_fn


def _update_config(config: ExecutionConfig) -> ExecutionConfig:
    """Choose the "best" options for the configuration if they are left unspecified.

    Args:
        config (ExecutionConfig): the initial execution config

    Returns:
        ExecutionConfig: a new config with the best choices selected.
    """
    updated_values = {}
    if config.gradient_method == "best":
        updated_values["gradient_method"] = "adjoint"
    if config.use_device_gradient is None:
        updated_values["use_device_gradient"] = config.gradient_method in {
            "best",
            "adjoint",
        }
    if config.grad_on_execution is None:
        updated_values["grad_on_execution"] = config.gradient_method == "adjoint"
    return replace(config, **updated_values)


def preprocess(
    execution_config: ExecutionConfig = DefaultExecutionConfig,
) -> Tuple[Tuple[qml.tape.QuantumScript], PostprocessingFn, ExecutionConfig]:
    """Preprocess a batch of :class:`~.QuantumTape` objects to make them ready for execution.

    This function validates a batch of :class:`~.QuantumTape` objects by transforming and expanding
    them to ensure all operators and measurements are supported by the execution device.

    Args:
        execution_config (ExecutionConfig): execution configuration with configurable
            options for the execution.

    Returns:
        TransformProgram, ExecutionConfig: A transform program and a configuration with originally unset specifications
        filled in.
    """
    transform_program = TransformProgram()

    # Validate measurement
    transform_program.add_transform(validate_measurements, execution_config)

    # Circuit expand
    transform_program.add_transform(expand_fn)

    if execution_config.gradient_method == "adjoint":
        # Adjoint expand
        transform_program.add_transform(validate_and_expand_adjoint)
        ### Broadcast expand
        transform_program.add_transform(qml.transforms.broadcast_expand)

    return transform_program, _update_config(execution_config)
