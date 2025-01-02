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
    from pennylane_lightning.lightning_kokkos_ops.algorithms import (
        AdjointJacobianC64,
        AdjointJacobianC128,
        create_ops_listC64,
        create_ops_listC128,
    )
except ImportError as ex:
    warn(str(ex), UserWarning)

import numpy as np
from pennylane.tape import QuantumTape

# pylint: disable=ungrouped-imports
from pennylane_lightning.core._adjoint_jacobian_base import LightningBaseAdjointJacobian


class LightningKokkosAdjointJacobian(LightningBaseAdjointJacobian):
    """Check and execute the adjoint Jacobian differentiation method.

    Args:
        qubit_state(LightningKokkosStateVector): State Vector to calculate the adjoint Jacobian with.
        batch_obs(bool): If serialized tape is to be batched or not.
    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        qubit_state: LightningKokkosStateVector,  # pylint: disable=undefined-variable
        batch_obs: bool = False,
    ) -> None:
        super().__init__(qubit_state, batch_obs)

        # Initialize the C++ binds
        self._jacobian_lightning, self._create_ops_list_lightning = self._adjoint_jacobian_dtype()

    def _adjoint_jacobian_dtype(self):
        """Binding to Lightning Kokkos Adjoint Jacobian C++ class.

        Returns: the AdjointJacobian class
        """
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

            statevector = LightningKokkosStateVector(num_wires=num_wires)
            statevector = statevector.get_final_state(tape)
            jacobian = LightningKokkosAdjointJacobian(statevector).calculate_jacobian(tape)

        Args:
            tape (QuantumTape): Operations and measurements that represent instructions for execution on Lightning.

        Returns:
            The Jacobian of a tape.
        """

        empty_array = self._handle_raises(tape, is_jacobian=True)

        if empty_array:
            return np.array([], dtype=self.dtype)

        processed_data = self._process_jacobian_tape(tape)

        if not processed_data:  # training_params is empty
            return np.array([], dtype=self.dtype)

        trainable_params = processed_data["tp_shift"]
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
