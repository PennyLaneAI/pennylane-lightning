# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Class implementation for tensornet measurements.
"""

# pylint: disable=import-error, no-name-in-module, ungrouped-imports

from functools import reduce
from typing import Callable, List, Union

import numpy as np
import pennylane as qml
from pennylane.devices.qubit.sampling import _group_measurements
from pennylane.measurements import (
    ClassicalShadowMP,
    CountsMP,
    ExpectationMP,
    MeasurementProcess,
    ProbabilityMP,
    SampleMeasurement,
    ShadowExpvalMP,
    Shots,
    StateMeasurement,
    VarianceMP,
)
from pennylane.ops import SparseHamiltonian, Sum
from pennylane.tape import QuantumScript
from pennylane.typing import Result, TensorLike
from pennylane.wires import Wires

from pennylane_lightning.core._serialize import QuantumScriptSerializer

from ._measurements_MPS import LightningTensorMeasurementsMPS
from ._measurements_ExactTN import LightningTensorMeasurementsExactTN


class LightningTensorMeasurements:
    """Lightning Tensor Measurements class

    Measures the tensor network provided by the LightningTensorNet class.

    Args:
        tensor_network(LightningTensorNet): Lightning tensornet class containing the tensor network to be measured.
    """

    def __init__(
        self,
        tensor_network,
    ) -> None:
        self._tensornet = tensor_network
        self._method = tensor_network._method
        self._dtype = tensor_network.dtype
        
        
        # self._measurement_lightning = self._measurement_dtype()(tensor_network.tensornet)
        if self._method == "mps":
            LTensorMeasurement = LightningTensorMeasurementsMPS(self._tensornet)
        if self._method == "exatn":
            LTensorMeasurement = LightningTensorMeasurementsExactTN(self._tensornet)
            
        self._LTensorMeasurement = LTensorMeasurement

    def __getattr__(self, name):
        return getattr(self._LTensorMeasurement, name)

  