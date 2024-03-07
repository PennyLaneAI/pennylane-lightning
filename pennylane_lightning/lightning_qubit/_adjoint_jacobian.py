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
from os import getenv
from typing import List
from warnings import warn

import numpy as np
import pennylane as qml
from pennylane import BasisState, Projector, QuantumFunctionError, Rot, StatePrep
from pennylane.measurements import Expectation, MeasurementProcess, State
from pennylane.operation import Operation, Tensor
from pennylane.tape import QuantumTape

from pennylane_lightning.core._serialize import QuantumScriptSerializer
from pennylane_lightning.core.lightning_base import _chunk_iterable

# pylint: disable=import-error, no-name-in-module, ungrouped-imports
try:
    from pennylane_lightning.lightning_qubit_ops.algorithms import (
        AdjointJacobianC64,
        AdjointJacobianC128,
        create_ops_listC64,
        create_ops_listC128,
    )
except ImportError:
    pass

from ._state_vector import LightningStateVector


class LightningAdjointJacobian:
    """Check and execute the adjoint Jacobian differentiation method.

    Args:
        qubit_state(LightningStateVector): State Vector to calculate the adjoint Jacobian with.
        batch_obs(bool): If serialized tape is to be batched or not.
    """

    def __init__(self, qubit_state: LightningStateVector, batch_obs: bool = False) -> None:
        self._qubit_state = qubit_state
        self._state = qubit_state.state_vector
        self._dtype = qubit_state.dtype
        self._jacobian_lightning = (
            AdjointJacobianC64() if self._dtype == np.complex64 else AdjointJacobianC128()
        )
        self._create_ops_list_lightning = (
            create_ops_listC64 if self._dtype == np.complex64 else create_ops_listC128
        )
        self._batch_obs = batch_obs

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

    @staticmethod
    def _check_adjdiff_supported_measurements(
        measurements: List[MeasurementProcess],
    ):
        """Check whether given list of measurement is supported by adjoint_differentiation.

        Args:
            measurements (List[MeasurementProcess]): a list of measurement processes to check.

        Returns:
            Expectation or State: a common return type of measurements.
        """
        if not measurements:
            return None

        if len(measurements) == 1 and measurements[0].return_type is State:
            return State

        # Now the return_type of measurement processes must be expectation
        if any(measurement.return_type is not Expectation for measurement in measurements):
            raise QuantumFunctionError(
                "Adjoint differentiation method does not support expectation return type "
                "mixed with other return types"
            )

        for measurement in measurements:
            if isinstance(measurement.obs, Tensor):
                if any(isinstance(obs, Projector) for obs in measurement.obs.non_identity_obs):
                    raise QuantumFunctionError(
                        "Adjoint differentiation method does "
                        "not support the Projector observable"
                    )
            elif isinstance(measurement.obs, Projector):
                raise QuantumFunctionError(
                    "Adjoint differentiation method does not support the Projector observable"
                )
        return Expectation

    @staticmethod
    def _check_adjdiff_supported_operations(operations):
        """Check Lightning adjoint differentiation method support for a tape.

        Raise ``QuantumFunctionError`` if ``tape`` contains not supported measurements,
        observables, or operations by the Lightning adjoint differentiation method.

        Args:
            tape (.QuantumTape): quantum tape to differentiate.
        """
        for operation in operations:
            if operation.num_params > 1 and not isinstance(operation, Rot):
                raise QuantumFunctionError(
                    f"The {operation.name} operation is not supported using "
                    'the "adjoint" differentiation method'
                )

    # pylint: disable=too-many-function-args, assignment-from-no-return, too-many-arguments
    def _process_jacobian_tape(
        self, tape: QuantumTape, use_mpi: bool = False, split_obs: bool = False
    ):
        use_csingle = self._dtype == np.complex64

        obs_serialized, obs_idx_offsets = QuantumScriptSerializer(
            self._qubit_state.device_name, use_csingle, use_mpi, split_obs
        ).serialize_observables(tape)

        ops_serialized, use_sp = QuantumScriptSerializer(
            self._qubit_state.device_name, use_csingle, use_mpi, split_obs
        ).serialize_ops(tape)

        ops_serialized = self._create_ops_list_lightning(*ops_serialized)

        # We need to filter out indices in trainable_params which do not
        # correspond to operators.
        trainable_params = sorted(tape.trainable_params)
        if len(trainable_params) == 0:
            return None

        tp_shift = []
        record_tp_rows = []
        all_params = 0

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

        return {
            "state_vector": self.state,
            "obs_serialized": obs_serialized,
            "ops_serialized": ops_serialized,
            "tp_shift": tp_shift,
            "record_tp_rows": record_tp_rows,
            "all_params": all_params,
            "obs_idx_offsets": obs_idx_offsets,
        }

    @staticmethod
    def _adjoint_jacobian_processing(jac):
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

    def calculate_jacobian(self, tape: QuantumTape):
        """Computes and returns the Jacobian with the adjoint method."""
        if tape.shots:
            warn(
                "Requested adjoint differentiation to be computed with finite shots. "
                "The derivative is always exact when using the adjoint "
                "differentiation method.",
                UserWarning,
            )

        tape_return_type = self._check_adjdiff_supported_measurements(tape.measurements)

        if not tape_return_type:  # the tape does not have measurements
            return np.array([], dtype=self._dtype)

        if tape_return_type is State:
            raise QuantumFunctionError(
                "This method does not support statevector return type. "
                "Use vjp method instead for this purpose."
            )

        self._check_adjdiff_supported_operations(tape.operations)

        processed_data = self._process_jacobian_tape(tape)

        if not processed_data:  # training_params is empty
            return np.array([], dtype=self._dtype)

        trainable_params = processed_data["tp_shift"]

        # If requested batching over observables, chunk into OMP_NUM_THREADS sized chunks.
        # This will allow use of Lightning with adjoint for large-qubit numbers AND large
        # numbers of observables, enabling choice between compute time and memory use.
        requested_threads = int(getenv("OMP_NUM_THREADS", "1"))

        if self._batch_obs and requested_threads > 1:
            obs_partitions = _chunk_iterable(processed_data["obs_serialized"], requested_threads)
            jac = []
            for obs_chunk in obs_partitions:
                jac_local = self._jacobian_lightning(
                    processed_data["state_vector"],
                    obs_chunk,
                    processed_data["ops_serialized"],
                    trainable_params,
                )
                jac.extend(jac_local)
        else:
            jac = self._jacobian_lightning(
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

    # pylint: disable=line-too-long, inconsistent-return-statements
    def calculate_vjp(self, tape: QuantumTape, grad_vec):
        """Generate the processing function required to compute the vector-Jacobian products
        of a tape.

        .. code-block:: python

            vjp_f = dev.vjp([qml.state()], grad_vec)
            vjp = vjp_f(tape)

        computes :math:`w = (w_1,\\cdots,w_m)` where

        .. math::

            w_k = \\langle v| \\frac{\\partial}{\\partial \\theta_k} | \\psi_{\\pmb{\\theta}} \\rangle.

        Here, :math:`m` is the total number of trainable parameters, :math:`\\pmb{\\theta}`
        is the vector of trainable parameters and :math:`\\psi_{\\pmb{\\theta}}`
        is the output quantum state.

        Args:
            measurements (list): List of measurement processes for vector-Jacobian product.
                Now it must be expectation values or a quantum state.
            grad_vec (tensor_like): Gradient-output vector. Must have shape matching the output
                shape of the corresponding tape, i.e. number of measurements if
                the return type is expectation or :math:`2^N` if the return type is statevector

        Returns:
            The processing function required to compute the vector-Jacobian products of a tape.
        """
        if tape.shots is not None:
            warn(
                "Requested adjoint differentiation to be computed with finite shots. "
                "The derivative is always exact when using the adjoint differentiation "
                "method.",
                UserWarning,
            )

        measurements = tape.measurements
        tape_return_type = self._check_adjdiff_supported_measurements(measurements)

        if qml.math.allclose(grad_vec, 0) or tape_return_type is None:
            return qml.math.convert_like(np.zeros(len(tape.trainable_params)), grad_vec)

        if tape_return_type is Expectation:
            if len(grad_vec) != len(measurements):
                raise ValueError(
                    "Number of observables in the tape must be the same as the "
                    "length of grad_vec in the vjp method"
                )

            if np.iscomplexobj(grad_vec):
                raise ValueError(
                    "The vjp method only works with a real-valued grad_vec when the "
                    "tape is returning an expectation value"
                )

            ham = qml.Hamiltonian(grad_vec, [m.obs for m in measurements])

            # pylint: disable=protected-access
            def processing_fn_expval(tape):
                nonlocal ham
                num_params = len(tape.trainable_params)

                if num_params == 0:
                    return np.array([], dtype=self.qubit_state.dtype)

                new_tape = tape.copy()
                new_tape._measurements = [qml.expval(ham)]

                return self.calculate_jacobian(new_tape)

            return processing_fn_expval(tape)

        if tape_return_type is State:
            raise QuantumFunctionError(
                "Adjoint differentiation does not support State measurements."
            )
