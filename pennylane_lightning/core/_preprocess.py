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
# from dataclasses import replace
# from typing import Tuple, Sequence
from typing import Callable, Union

# import warnings

import pennylane as qml

# from pennylane.operation import Tensor, StatePrepBase
from pennylane.measurements import (
    # MidMeasureMP,
    ExpectationMP,
)

# from pennylane.devices.preprocess import (
#     validate_measurements,
#     _accepted_adjoint_operator,
#     _operator_decomposition_gen,
# )
from pennylane.typing import ResultBatch, Result

from pennylane import DeviceError
# from pennylane.transforms.core import transform
from pennylane.transforms.core import TransformProgram

# from pennylane.devices import ExecutionConfig, DefaultExecutionConfig

from pennylane.devices.preprocess import (
    decompose,
    validate_observables,
    validate_measurements,
    # validate_device_wires,
    warn_about_trainable_observables,
    no_sampling,
)

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
    "QubitStateVector",
    "QubitUnitary",
    "ControlledQubitUnitary",
    "MultiControlledX",
    "DiagonalQubitUnitary",
    "PauliX",
    "Adjoint(PauliX)",
    "PauliY",
    "Adjoint(PauliY)",
    "PauliZ",
    "Adjoint(PauliZ)",
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


def observable_stopping_condition(obs: qml.operation.Operator) -> bool:
    """Specifies whether or not an observable is accepted by Lightning."""
    return obs.name in _observables


def stopping_condition(op: qml.operation.Operator, dev_operations_list=_operations) -> bool:
    """Specify whether or not an Operator object is directly supported by Lightning."""
    if op.name == "QFT":
        return len(op.wires) < 10
    if op.name == "GroverOperator":
        return len(op.wires) < 13
    return op.name in dev_operations_list


def accepted_sample_measurement(m: qml.measurements.MeasurementProcess) -> bool:
    """Specifies whether or not a measurement is accepted when sampling."""
    return isinstance(
        m,
        (
            qml.measurements.SampleMeasurement,
            qml.measurements.ClassicalShadowMP,
            qml.measurements.ShadowExpvalMP,
        ),
    )



def _add_adjoint_transforms(program: TransformProgram) -> None:
    """Private helper function for ``preprocess`` that adds the transforms specific
    for adjoint differentiation.
    Args:
        program (TransformProgram): where we will add the adjoint differentiation transforms
    Side Effects:
        Adds transforms to the input program.
    """

    # def adjoint_ops(op: qml.operation.Operator) -> bool:
    #     """Specify whether or not an Operator is supported by adjoint differentiation."""
    #     return op.num_params == 0 or op.num_params == 1 and op.has_generator

    def adjoint_observables(obs: qml.operation.Operator) -> bool:
        """Specifies whether or not an observable is compatible with adjoint differentiation on Lightning."""
        # return obs.has_matrix
        return obs.name in _operations

    def accepted_adjoint_measurement(m: qml.measurements.MeasurementProcess) -> bool:
        if not isinstance(m, ExpectationMP):
            # raise DeviceError(
            #     "Adjoint differentiation method does not support "
            #     f"measurement {m.__class__.__name__}."
            # )
            return False
        if m.obs.name == "Hamiltonian":
            if not all(tuple(t.has_matrix for t in m.obs.ops)):
                # raise DeviceError(
                #     f"Adjoint differentiation method does not support some of the Hamiltonian terms."
                # )
                return False
        elif not m.obs.has_matrix:
            # raise DeviceError(
            #     f"Adjoint differentiation method does not support observable {m.obs.name}."
            # )
            return False
        return True

    name = "adjoint + Lightning"
    program.add_transform(no_sampling, name=name)
    # program.add_transform(
    #     decompose,
    #     stopping_condition=adjoint_ops,
    #     name=name,
    # )
    program.add_transform(
        decompose,
        stopping_condition=stopping_condition,
        name=name,
    )
    program.add_transform(validate_observables, observable_stopping_condition, name=name)
    program.add_transform(
        validate_measurements,
        analytic_measurements=accepted_adjoint_measurement,
        name=name,
    )
    program.add_transform(qml.transforms.broadcast_expand)
    program.add_transform(warn_about_trainable_observables)


# @transform
# def validate_and_expand_adjoint(
#     tape: qml.tape.QuantumTape,
# ) -> (Sequence[qml.tape.QuantumTape], Callable):
#     """Function for validating that the operations and observables present in the input tape
#     are valid for adjoint differentiation.

#     Args:
#         tape(.QuantumTape): the tape to validate

#     Returns:
#         pennylane.QNode or qfunc or Tuple[List[.QuantumTape], Callable]: If a QNode is passed,
#         it returns a QNode with the transform added to its transform program.
#         If a tape is passed, returns a tuple containing a list of
#         quantum tapes to be evaluated, and a function to be applied to these
#         tape executions.
#     """
#     try:
#         new_ops = [
#             final_op
#             for op in tape.operations[tape.num_preps :]
#             for final_op in _operator_decomposition_gen(op, _accepted_adjoint_operator)
#         ]
#     except RecursionError as e:
#         raise DeviceError(
#             "Reached recursion limit trying to decompose operations. "
#             "Operator decomposition may have entered an infinite loop."
#         ) from e

#     for k in tape.trainable_params:
#         if hasattr(tape._par_info[k]["op"], "return_type"):
#             warnings.warn(
#                 "Differentiating with respect to the input parameters of "
#                 f"{tape._par_info[k]['op'].name} is not supported with the "
#                 "adjoint differentiation method. Gradients are computed "
#                 "only with regards to the trainable parameters of the circuit.\n\n Mark "
#                 "the parameters of the measured observables as non-trainable "
#                 "to silence this warning.",
#                 UserWarning,
#             )

#     # Check validity of measurements
#     measurements = []
#     for m in tape.measurements:
#         if not isinstance(m, ExpectationMP):
#             raise DeviceError(
#                 "Adjoint differentiation method does not support "
#                 f"measurement {m.__class__.__name__}."
#             )

#         if m.obs.name == "Hamiltonian":
#             if not all(tuple(t.has_matrix for t in m.obs.ops)):
#                 raise DeviceError(
#                     f"Adjoint differentiation method does not support some of the Hamiltonian terms."
#                 )
#         elif not m.obs.has_matrix:
#             raise DeviceError(
#                 f"Adjoint differentiation method does not support observable {m.obs.name}."
#             )

#         measurements.append(m)

#     new_ops = tape.operations[: tape.num_preps] + new_ops
#     new_tape = qml.tape.QuantumScript(new_ops, measurements, shots=tape.shots)

#     def null_postprocessing(results):
#         """A postprocessing function returned by a transform that only converts the batch of results
#         into a result for a single ``QuantumTape``.
#         """
#         return results[0]

#     return [new_tape], null_postprocessing


# @transform
# def expand_fn(
#     tape: qml.tape.QuantumTape, observables_list=_observables
# ) -> (Sequence[qml.tape.QuantumTape], Callable):
#     """Method for expanding or decomposing an input tape.

#     This method expands the tape if:

#     - mid-circuit measurements are present,
#     - any operations are not supported on the device.

#     Args:
#         tape (.QuantumTape): the tape to expand.

#     Returns:
#         pennylane.QNode or qfunc or Tuple[List[.QuantumTape], Callable]: If a QNode is passed,
#         it returns a QNode with the transform added to its transform program.
#         If a tape is passed, returns a tuple containing a list of
#         quantum tapes to be evaluated, and a function to be applied to these
#         tape executions.
#     """

#     if any(isinstance(o, MidMeasureMP) for o in tape.operations):
#         tapes, _ = qml.defer_measurements(tape)
#         tape = tapes[0]

#     if not all(_accepted_operator(op) for op in tape.operations):
#         try:
#             # don't decompose initial operations if its StatePrepBase
#             prep_op = [tape[0]] if isinstance(tape[0], StatePrepBase) else []

#             new_ops = [
#                 final_op
#                 for op in tape.operations[bool(prep_op) :]
#                 for final_op in _operator_decomposition_gen(op, _accepted_operator)
#             ]
#         except RecursionError as e:
#             raise DeviceError(
#                 "Reached recursion limit trying to decompose operations. "
#                 "Operator decomposition may have entered an infinite loop."
#             ) from e
#         tape = qml.tape.QuantumScript(prep_op + new_ops, tape.measurements, shots=tape.shots)

#     for observable in tape.observables:
#         if isinstance(observable, Tensor):
#             if any(o.name not in observables_list for o in observable.obs):
#                 raise DeviceError(f"Observable {observable} not supported on Lightning")
#         elif observable.name not in observables_list:
#             raise DeviceError(f"Observable {observable} not supported on Lightning")

#     def null_postprocessing(results):
#         """A postprocessing function returned by a transform that only converts the batch of results
#         into a result for a single ``QuantumTape``.
#         """
#         return results[0]

#     return [tape], null_postprocessing


# def _update_config(config: ExecutionConfig) -> ExecutionConfig:
#     """Choose the "best" options for the configuration if they are left unspecified.

#     Args:
#         config (ExecutionConfig): the initial execution config

#     Returns:
#         ExecutionConfig: a new config with the best choices selected.
#     """
#     updated_values = {}
#     if config.gradient_method == "best":
#         updated_values["gradient_method"] = "adjoint"
#     if config.use_device_gradient is None:
#         updated_values["use_device_gradient"] = config.gradient_method in {
#             "best",
#             "adjoint",
#         }
#     if config.grad_on_execution is None:
#         updated_values["grad_on_execution"] = config.gradient_method == "adjoint"
#     return replace(config, **updated_values)


# def preprocess(
#     execution_config: ExecutionConfig = DefaultExecutionConfig,
# ) -> Tuple[Tuple[qml.tape.QuantumScript], PostprocessingFn, ExecutionConfig]:
#     """Preprocess a batch of :class:`~.QuantumTape` objects to make them ready for execution.

#     This function validates a batch of :class:`~.QuantumTape` objects by transforming and expanding
#     them to ensure all operators and measurements are supported by the execution device.

#     Args:
#         execution_config (ExecutionConfig): execution configuration with configurable
#             options for the execution.

#     Returns:
#         TransformProgram, ExecutionConfig: A transform program and a configuration with originally unset specifications
#         filled in.
#     """
#     transform_program = TransformProgram()

#     # Validate measurement
#     transform_program.add_transform(validate_measurements, execution_config)

#     # Circuit expand
#     transform_program.add_transform(expand_fn)

#     if execution_config.gradient_method == "adjoint":
#         # Adjoint expand
#         transform_program.add_transform(validate_and_expand_adjoint)
#         ### Broadcast expand
#         transform_program.add_transform(qml.transforms.broadcast_expand)

#     return transform_program, _update_config(execution_config)
