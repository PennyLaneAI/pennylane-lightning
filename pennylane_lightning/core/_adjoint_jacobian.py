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
r"""
Internal methods for adjoint Jacobian differentiation method.
"""
from typing import List

import numpy as np
import pennylane as qml
from pennylane import (
    BasisState,
    StatePrep,
    Projector,
    Rot,
    QuantumFunctionError,
)
from pennylane.operation import Tensor, Operation
from pennylane.measurements import (
    MeasurementProcess,
    Expectation,
    State,
    ExpectationMP,
)
from pennylane.tape import QuantumTape
from pennylane.devices.qubit.initialize_state import create_initial_state

from pennylane_lightning.core._serialize import QuantumScriptSerializer

from pennylane import DeviceError

from pennylane_lightning.lightning_qubit._simulate import asarray

from typing import Generator, Callable, Tuple, Union, Sequence
from pennylane.transforms.core import transform, TransformProgram
import warnings
from ._preprocess import _operator_decomposition_gen


class AdjointJacobian:
    """Check and execute the adjoint Jacobian differentiation method.

    Args:
    device_name: device shortname.

    """

    # pylint: disable=import-outside-toplevel, too-many-instance-attributes
    def __init__(self, device_name):
        if device_name == "lightning.qubit":
            try:
                from pennylane_lightning.lightning_qubit_ops import (
                    StateVectorC64,
                    StateVectorC128,
                )

                from pennylane_lightning.lightning_qubit_ops.algorithms import (
                    AdjointJacobianC64,
                    AdjointJacobianC128,
                    create_ops_listC64,
                    create_ops_listC128,
                )

                from pennylane_lightning.lightning_qubit._apply_operations import (
                    apply_operations,
                )

            except ImportError as exception:
                raise ImportError(
                    f"Pre-compiled binaries for {device_name}"
                    " adjoint Jacobian functionality are not available."
                ) from exception
        elif device_name == "lightning.kokkos":
            try:
                from pennylane_lightning.lightning_kokkos_ops import (
                    StateVectorC64,
                    StateVectorC128,
                )

                from pennylane_lightning.lightning_kokkos_ops.algorithms import (
                    AdjointJacobianC64,
                    AdjointJacobianC128,
                    create_ops_listC64,
                    create_ops_listC128,
                )
            except ImportError as exception:
                raise ImportError(
                    f"Pre-compiled binaries for {device_name}"
                    " adjoint Jacobian functionality are not available."
                ) from exception
        else:
            raise DeviceError(f'The device name "{device_name}" is not a valid option.')

        self.device_name = device_name

        self.statevector_c64 = StateVectorC64
        self.statevector_c128 = StateVectorC128

        self.adjointJacobian_c64 = AdjointJacobianC64
        self.adjointJacobian_c128 = AdjointJacobianC128
        self.create_ops_list_c64 = create_ops_listC64
        self.create_ops_list_c128 = create_ops_listC128

        self.apply_operations = apply_operations

    def _check_supported_measurements(self, measurements: List[MeasurementProcess]):
        """Check whether given list of measurement is supported by the adjoint jacobian differentiation method.

        Args:
            measurements (List[MeasurementProcess]): a list of measurement processes to check.

        Raises:
            QuantumFunctionError: if adjoint method is not supported for any measurement
        """
        if len(measurements) == 1 and measurements[0].return_type is State:
            raise QuantumFunctionError(
                "This method does not support state vector return type. "
                "Use vjp method instead for this purpose."
            )

        # Now the return_type of measurement processes must be expectation
        if not all([m.return_type is Expectation for m in measurements]):
            raise QuantumFunctionError(
                "Adjoint differentiation method does not support expectation return type "
                "mixed with other return types"
            )

        for m in measurements:
            if not isinstance(m.obs, Tensor):
                if isinstance(m.obs, Projector):
                    raise QuantumFunctionError(
                        "Adjoint differentiation method does not support the Projector observable"
                    )
            else:
                if any([isinstance(o, Projector) for o in m.obs.non_identity_obs]):
                    raise QuantumFunctionError(
                        "Adjoint differentiation method does not support the Projector observable"
                    )

    def _check_supported_operations(self, operations):
        """Check Lightning adjoint differentiation method support for a tape.

        Args:
            operations (List): list with operations

        Raises:
            QuantumFunctionError:  if ``tape`` contains observables or operations,
        not supported by the Lightning adjoint differentiation method.
        """

        for op in operations:
            if op.num_params > 1 and not isinstance(op, Rot):
                raise QuantumFunctionError(
                    f"The {op.name} operation is not supported using "
                    'the "adjoint" differentiation method'
                )

    def _check_adjoint_method_supported(self, tape: QuantumTape):
        """Check measurement and operation lists for adjoint Jacobian support in Lightning.

        Args:
            tape (QuantumTape): A quantum tape recording a variational quantum program.
        """
        self._check_supported_measurements(tape.measurements)
        self._check_supported_operations(tape.operations)

    def _process_jacobian_tape(self, tape, state):
        """Process a Jacobian tape before calculation.

        Args:
            tape (QuantumTape): A quantum tape recording a variational quantum program.
            state(np.array): unravelled initial state (1D)

        Returns:
            Set: A set with a tape processed for Lightning.
        """
        # print("tape")
        # print(tape)
        use_csingle = True if state.dtype == np.complex64 else False
        # To support np.complex64 based on the type of self._state
        create_ops_list = self.create_ops_list_c64 if use_csingle else self.create_ops_list_c128

        Serializer = QuantumScriptSerializer(self.device_name, state.dtype == np.complex64)
        obs_serialized = Serializer.serialize_observables(tape)
        ops_serialized, use_sp = Serializer.serialize_ops(tape)

        ops_serialized = create_ops_list(*ops_serialized)

        # We need to filter out indices in trainable_params which do not
        # correspond to operators.
        # print("tape.trainable_params")
        # print(tape.trainable_params)

        trainable_params = sorted(tape.trainable_params)
        if len(trainable_params) == 0:
            return None

        # print("trainable_params = sorted(tape.trainable_params)")
        # print(trainable_params)
        tp_shift = []
        record_tp_rows = []
        all_params = 0

        # for op_idx, tp in enumerate(trainable_params):
        #     op, _, _ = tape.get_operation(
        #         op_idx
        #     )  # get op_idx-th operator among differentiable operators
        #     if isinstance(op, Operation) and not isinstance(op, (BasisState, QubitStateVector)):
        #         # We now just ignore non-op or state preps
        #         tp_shift.append(tp)
        #         record_tp_rows.append(all_params)
        #     all_params += 1

        for op_idx, trainable_param in enumerate(trainable_params):
            # get op_idx-th operator among differentiable operators
            operation, _, _ = tape.get_operation(op_idx)
            if isinstance(operation, Operation) and not isinstance(
                operation, (BasisState, StatePrep)
            ):
                # We now just ignore non-op or state preps
                tp_shift.append(trainable_param)
                record_tp_rows.append(all_params)
            all_params += 1

        if use_sp:
            # When the first element of the tape is state preparation. Still, I am not sure
            # whether there must be only one state preparation...
            tp_shift = [i - 1 for i in tp_shift]

        state_vector = self.statevector_c64(state) if use_csingle else self.statevector_c128(state)
        return {
            "state_vector": state_vector,
            "obs_serialized": obs_serialized,
            "ops_serialized": ops_serialized,
            "tp_shift": tp_shift,
            "record_tp_rows": record_tp_rows,
            "all_params": all_params,
        }

    def _adjoint_jacobian_processing(self, jac):
        """
        Post-process the Jacobian matrix returned by ``adjoint_jacobian`` for
        the new return type system.
        """
        jac = np.squeeze(jac)

        if jac.ndim == 0:
            return np.array(jac)

        if jac.ndim == 1:
            return tuple(np.array(j) for j in jac)

        # must be 2-dimensional
        return tuple(tuple(np.array(j_) for j_ in j) for j in jac)

    # def _accepted_adjoint_operator(self, op: qml.operation.Operator) -> bool:
    #     """Specify whether or not an Operator is supported by adjoint differentiation."""
    #     print("_accepted_adjoint_operator")
    #     return op.num_params == 0 or op.num_params == 1 and op.has_generator

    # @transform
    # def validate_and_expand_adjoint(
    #     self,
    #     tape: qml.tape.QuantumTape
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
    #     print("validate_and_expand_adjoint")
    #     try:
    #         new_ops = [
    #             final_op
    #             for op in tape.operations[tape.num_preps :]
    #             for final_op in _operator_decomposition_gen(op, self._accepted_adjoint_operator)
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
    #             return DeviceError(
    #                 "Adjoint differentiation method does not support "
    #                 f"measurement {m.__class__.__name__}."
    #             )

    #         if m.obs.name == "Hamiltonian":
    #             if not all(tuple(t.has_matrix for t in m.obs.ops)):
    #                 return DeviceError(
    #                     f"Adjoint differentiation method does not support some of the Hamiltonian terms."
    #                 )
    #         elif not m.obs.has_matrix:
    #             return DeviceError(
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

    #     return "lala"
    #     # return [new_tape], null_postprocessing

    # # def validate_and_expand_adjoint(
    # #     self,
    # #     tape: qml.tape.QuantumTape
    # # ) -> (Sequence[qml.tape.QuantumTape]):
    # #     """Function for validating that the operations and observables present in the input tape
    # #     are valid for adjoint differentiation.

    # #     Args:
    # #         tape(.QuantumTape): the tape to validate

    # #     Returns:
    # #         pennylane.QNode or qfunc or Tuple[List[.QuantumTape], Callable]: If a QNode is passed,
    # #         it returns a QNode with the transform added to its transform program.
    # #         If a tape is passed, returns a tuple containing a list of
    # #         quantum tapes to be evaluated, and a function to be applied to these
    # #         tape executions.
    # #     """
    # #     print("=====>    validate_and_expand_adjoint")
    # #     try:
    # #         new_ops = [
    # #             final_op
    # #             for op in tape.operations[tape.num_preps :]
    # #             for final_op in _operator_decomposition_gen(op, self._accepted_adjoint_operator)
    # #         ]
    # #     except RecursionError as e:
    # #         raise DeviceError(
    # #             "Reached recursion limit trying to decompose operations. "
    # #             "Operator decomposition may have entered an infinite loop."
    # #         ) from e

    # #     for k in tape.trainable_params:
    # #         if hasattr(tape._par_info[k]["op"], "return_type"):
    # #             warnings.warn(
    # #                 "Differentiating with respect to the input parameters of "
    # #                 f"{tape._par_info[k]['op'].name} is not supported with the "
    # #                 "adjoint differentiation method. Gradients are computed "
    # #                 "only with regards to the trainable parameters of the circuit.\n\n Mark "
    # #                 "the parameters of the measured observables as non-trainable "
    # #                 "to silence this warning.",
    # #                 UserWarning,
    # #             )

    # #     # Check validity of measurements
    # #     measurements = []
    # #     for m in tape.measurements:
    # #         if not isinstance(m, ExpectationMP):
    # #             return DeviceError(
    # #                 "Adjoint differentiation method does not support "
    # #                 f"measurement {m.__class__.__name__}."
    # #             )

    # #         if m.obs.name == "Hamiltonian":
    # #             if not all(tuple(t.has_matrix for t in m.obs.ops)):
    # #                 return DeviceError(
    # #                     f"Adjoint differentiation method does not support some of the Hamiltonian terms."
    # #                 )
    # #         elif not m.obs.has_matrix:
    # #             return DeviceError(
    # #                 f"Adjoint differentiation method does not support observable {m.obs.name}."
    # #             )

    # #         measurements.append(m)

    # #     new_ops = tape.operations[: tape.num_preps] + new_ops
    # #     new_tape = qml.tape.QuantumScript(new_ops, measurements, shots=tape.shots)

    # #     return new_tape

    def calculate_adjoint_jacobian(self, tape, c_dtype=np.complex128):
        """Calculates the Adjoint Jacobian for a given tape.

        Args:
            tape (QuantumTape): A quantum tape recording a variational quantum program.
            c_dtype (Complex data type, Optional): Default to ``np.complex128``.
            starting_state (np.array, Optional): unravelled initial state (1D). Default to None.

        Returns:
            np.array: An array results.
        """

        state = create_initial_state(tape.wires)
        state = np.ravel(asarray(state, c_dtype))
        state = self.apply_operations(tape._ops, state)

        if len(tape.measurements) == 0:  # the tape does not have measurements
            return np.array([], dtype=state.dtype)

        processed_data = self._process_jacobian_tape(tape, state)

        if not processed_data:  # training_params is empty
            return np.array([], dtype=state.dtype)

        trainable_params = processed_data["tp_shift"]
        # print("trainable_params", trainable_params)

        use_csingle = True if state.dtype == np.complex64 else False
        adjoint_jacobian = (
            self.adjointJacobian_c64() if use_csingle else self.adjointJacobian_c128()
        )

        jac = adjoint_jacobian(
            processed_data["state_vector"],
            processed_data["obs_serialized"],
            processed_data["ops_serialized"],
            trainable_params,
        )
        jac = np.array(jac)
        jac = jac.reshape(-1, len(trainable_params))
        jac_r = np.zeros((jac.shape[0], processed_data["all_params"]))
        jac_r[:, processed_data["record_tp_rows"]] = jac

        if hasattr(qml, "active_return"):  # pragma: no cover
            return self._adjoint_jacobian_processing(jac_r) if qml.active_return() else jac_r
        return self._adjoint_jacobian_processing(jac_r)
