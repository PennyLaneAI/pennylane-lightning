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
from pennylane.tape import QuantumTape
from scipy.sparse import csr_matrix

# pylint: disable=ungrouped-imports
from pennylane_lightning.lightning_base._adjoint_jacobian import LightningBaseAdjointJacobian


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

        self._dp = DevPool()

        self._use_mpi = qubit_state._mpi_handler.use_mpi

        super().__init__(qubit_state, batch_obs)

        if self._use_mpi:
            self._mpi_handler = qubit_state._mpi_handler

        # Warning about performance with MPI and batch observation
        if self._use_mpi and not self._batch_obs:
            warn(
                "Using LightningGPU with `batch_obs=False` and `use_mpi=True` has the limitation of requiring more memory. If you want to allocate larger number of qubits use the option `batch_obs=True`"
                "For more information Check out the section `Parallel adjoint differentiation support` in our website https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html for more details.",
                RuntimeWarning,
            )

    def _adjoint_jacobian_dtype(self):
        """Binding to Lightning GPU Adjoint Jacobian C++ class.

        Returns: A pair of the AdjointJacobian class and the create_ops_list function. Default is None.
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
