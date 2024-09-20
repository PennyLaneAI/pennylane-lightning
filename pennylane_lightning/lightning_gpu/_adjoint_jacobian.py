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

from warnings import warn

try:
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

        MPI_SUPPORT = True
    except ImportError as ex:
        warn(str(ex), UserWarning)
        MPI_SUPPORT = False

except ImportError as ex:
    warn(str(ex), UserWarning)
    pass

from typing import Optional
import numpy as np
from pennylane.tape import QuantumTape

# pylint: disable=ungrouped-imports
from pennylane_lightning.core._adjoint_jacobian_base import LightningBaseAdjointJacobian

from ._state_vector import LightningGPUStateVector


class LightningGPUAdjointJacobian(LightningBaseAdjointJacobian):
    """Check and execute the adjoint Jacobian differentiation method.

    Args:
        qubit_state(LightningGPUStateVector): State Vector to calculate the adjoint Jacobian with.
        batch_obs(bool): If serialized tape is to be batched or not.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, 
                 qubit_state: LightningGPUStateVector, 
                 batch_obs: bool = False,
                use_mpi: Optional[bool] = False,
                 ) -> None:
        
        super().__init__(qubit_state, batch_obs)
        
        self._use_mpi = use_mpi
        
        # Initialize the C++ binds
        self._jacobian_lightning, self._create_ops_list_lightning = self._adjoint_jacobian_dtype()

    def _adjoint_jacobian_dtype(self):
        """Binding to Lightning GPU Adjoint Jacobian C++ class.

        Returns: the AdjointJacobian class
        """
        if self._use_mpi:
            if self.dtype == np.complex64:
                return AdjointJacobianMPIC64, create_ops_listMPIC64
            else:
                return AdjointJacobianMPIC128, create_ops_listMPIC128
        else:
            if self.dtype == np.complex64:
                return AdjointJacobianC64, create_ops_listC64
            else:
                return AdjointJacobianC128, create_ops_listC128
            
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


        processed_data = self._process_jacobian_tape(tape,split_obs,self._use_mpi)

        if not processed_data:  # training_params is empty
            return np.array([], dtype=self.dtype)

        trainable_params = processed_data["tp_shift"]
        print(processed_data["state_vector"])
        print(processed_data["obs_serialized"])
        print(processed_data["ops_serialized"])
        print(trainable_params)
        jac = self._jacobian_lightning(
            processed_data["state_vector"],
            processed_data["obs_serialized"],
            processed_data["ops_serialized"],
            trainable_params,
        )
        jac = np.array(jac)
        jac = jac.reshape(-1, len(trainable_params)) if len(jac) else jac
        jac_r = np.zeros((jac.shape[0], processed_data["all_params"]))
        jac_r[:, processed_data["record_tp_rows"]] = jac

        return self._adjoint_jacobian_processing(jac_r)
