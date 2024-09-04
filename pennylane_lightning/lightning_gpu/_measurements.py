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
Class implementation for state vector measurements.
"""

import numpy as np
import pennylane as qml
from pennylane.typing import TensorLike

from pennylane_lightning.core._measurements_base import LightningBaseMeasurements


class LightningGPUMeasurements(LightningBaseMeasurements):
    """Lightning GPU Measurements class

    Measures the state provided by the LightningGPUStateVector class.

    Args:
        qubit_state(LightningGPUStateVector): Lightning state-vector class containing the state vector to be measured.
    """

    def __init__(
        self,
        lgpu_state,
    ) -> TensorLike:

        super().__init__(lgpu_state)
