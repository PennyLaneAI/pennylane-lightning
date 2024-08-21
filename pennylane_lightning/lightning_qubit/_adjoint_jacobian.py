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

try:
    from pennylane_lightning.lightning_qubit_ops.algorithms import (
        AdjointJacobianC64,
        AdjointJacobianC128,
        create_ops_listC64,
        create_ops_listC128,
    )
except ImportError:
    pass

import numpy as np

from ._state_vector import LightningStateVector
from pennylane_lightning.core._adjoint_jacobian_base import LightningBaseAdjointJacobian

class LightningAdjointJacobian(LightningBaseAdjointJacobian):
    
    def __init__(self, qubit_state: LightningStateVector, batch_obs: bool = False) -> None:
        super().__init__(qubit_state, batch_obs)
        
        # Initialize the C++ bind
        self._set_jacobian_lightning()
        
    def _set_jacobian_lightning(self):
        self._jacobian_lightning = (
            AdjointJacobianC64() if self._dtype == np.complex64 else AdjointJacobianC128()
        )
        self._create_ops_list_lightning = (
            create_ops_listC64 if self._dtype == np.complex64 else create_ops_listC128
        )
