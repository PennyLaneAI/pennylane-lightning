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
r"""
Internal methods for adjoint Jacobian differentiation method.
"""

from __future__ import annotations

from warnings import warn

try:
    from pennylane_lightning.lightning_gpu_ops import DevPool
    from pennylane_lightning.lightning_gpu_ops.algorithms import (
        AdjointJacobianC64,
        AdjointJacobianC128,
        create_ops_listC64,
        create_ops_listC128,
    )

    try:
        from pennylane_lightning.lightning_gpu_ops.algorithmsMPI import (
            AdjointJacobianMPIC64,
            AdjointJacobianMPIC128,
            create_ops_listMPIC64,
            create_ops_listMPIC128,
        )

        mpi_error = None
        MPI_SUPPORT = True
    except ImportError as ex_mpi:
        mpi_error = ex_mpi
        MPI_SUPPORT = False

except ImportError as ex:
    warn(str(ex), UserWarning)


import numpy as np
from pennylane import BasisState, StatePrep
from pennylane.operation import Operation
from pennylane.tape import QuantumTape
from scipy.sparse import csr_matrix

# pylint: disable=ungrouped-imports
from pennylane_lightning.core._adjoint_jacobian_base import LightningBaseAdjointJacobian
from pennylane_lightning.core._serialize import QuantumScriptSerializer


class LightningGPUAdjointJacobian(LightningBaseAdjointJacobian):
    """Check and execute the adjoint Jacobian differentiation method.

    Args:
        qubit_state(LightningGPUStateVector): State Vector to calculate the adjoint Jacobian with.
        batch_obs(bool): If serialized tape is to be batched or not.
            For Lightning GPU, distribute the observations across GPUs in the same node. Defaults to False.
            For Lightning GPU-MPI, if `batch_obs=False` the computation requires more memory and is faster,
            while `batch_obs=True` allows a larger number of qubits simulation
            at the expense of high computational cost. Defaults to False.
    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        qubit_state: LightningGPUStateVector,  # pylint: disable=undefined-variable
        batch_obs: bool = False,
    ) -> None:

        super().__init__(qubit_state, batch_obs)

        self._dp = DevPool()

        self._use_mpi = qubit_state._mpi_handler.use_mpi

        if self._use_mpi:
            self._mpi_handler = qubit_state._mpi_handler

        # Initialize the C++ binds
        self._jacobian_lightning, self._create_ops_list_lightning = self._adjoint_jacobian_dtype()

        # Warning about performance with MPI and batch observation
        if self._use_mpi and not self._batch_obs:
            warn(
                "Using LightningGPU with `batch_obs=False` and `use_mpi=True` has the limitation of requiring more memory. If you want to allocate larger number of qubits use the option `batch_obs=True`"
                "For more information Check out the section `Parallel adjoint differentiation support` in our website https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html for more details.",
                RuntimeWarning,
            )

    def _adjoint_jacobian_dtype(self):
        """Binding to Lightning GPU Adjoint Jacobian C++ class.

        Returns: the AdjointJacobian class
        """
        if self._use_mpi:
            if not MPI_SUPPORT:
                warn(str(mpi_error), UserWarning)

            jacobian_lightning = (
                AdjointJacobianMPIC64() if self.dtype == np.complex64 else AdjointJacobianMPIC128()
            )
            create_ops_list_lightning = (
                create_ops_listMPIC64 if self.dtype == np.complex64 else create_ops_listMPIC128
            )
            return jacobian_lightning, create_ops_list_lightning

        # without MPI
        jacobian_lightning = (
            AdjointJacobianC64() if self.dtype == np.complex64 else AdjointJacobianC128()
        )
        create_ops_list_lightning = (
            create_ops_listC64 if self.dtype == np.complex64 else create_ops_listC128
        )
        return jacobian_lightning, create_ops_list_lightning

    def _process_jacobian_tape(
        self, tape: QuantumTape, split_obs: bool = False, use_mpi: bool = False
    ):
        """Process a tape, serializing and building a dictionary proper for
        the adjoint Jacobian calculation in the C++ layer.

        Args:
            tape (QuantumTape): Operations and measurements that represent instructions for execution on Lightning.
            split_obs (bool, optional): If splitting the observables in a list. Defaults to False.
            use_mpi (bool, optional): If distributing computation with MPI. Defaults to False.

        Returns:
            dictionary: dictionary providing serialized data for Jacobian calculation.
        """
        use_csingle = self._qubit_state.dtype == np.complex64

        obs_serialized, obs_indices = QuantumScriptSerializer(
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
            "obs_indices": obs_indices,
        }

    def calculate_jacobian(self, tape: QuantumTape):
        """Computes the Jacobian with the adjoint method.

        .. code-block:: python

            statevector = LightningGPUStateVector(num_wires=num_wires)
            statevector = statevector.get_final_state(tape)
            jacobian = LightningGPUAdjointJacobian(statevector).calculate_jacobian(tape)

        Args:
            tape (QuantumTape): Operations and measurements that represent instructions for execution on Lightning.

        Returns:
            The Jacobian of a tape.
        """

        empty_array = self._handle_raises(tape, is_jacobian=True)

        if empty_array:
            return np.array([], dtype=self.dtype)

        if self._use_mpi:
            split_obs = False  # with MPI batched means compute Jacobian one observables at a time, no point splitting linear combinations
        else:
            split_obs = self._dp.getTotalDevices() if self._batch_obs else False

        processed_data = self._process_jacobian_tape(tape, split_obs, self._use_mpi)

        if not processed_data:  # training_params is empty
            return np.array([], dtype=self.dtype)

        trainable_params = processed_data["tp_shift"]

        if self._batch_obs:  # Batching of Measurements
            jac = self._jacobian_lightning.batched(
                processed_data["state_vector"],
                processed_data["obs_serialized"],
                processed_data["ops_serialized"],
                trainable_params,
            )
        else:
            jac = self._jacobian_lightning(
                processed_data["state_vector"],
                processed_data["obs_serialized"],
                processed_data["ops_serialized"],
                trainable_params,
            )

        jac = np.array(jac)
        has_shape0 = bool(len(jac))

        num_obs = len(np.unique(processed_data["obs_indices"]))
        rows = processed_data["obs_indices"]
        cols = np.arange(len(rows), dtype=int)
        data = np.ones(len(rows))
        red_mat = csr_matrix((data, (rows, cols)), shape=(num_obs, len(rows)))
        jac = red_mat @ jac.reshape((len(rows), -1))
        jac = jac.reshape(-1, len(trainable_params)) if has_shape0 else jac
        jac_r = np.zeros((jac.shape[0], processed_data["all_params"]))
        jac_r[:, processed_data["record_tp_rows"]] = jac
        return self._adjoint_jacobian_processing(jac_r)
