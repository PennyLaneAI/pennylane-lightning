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
This module contains the LightningTensor class that inherits from the new device interface.
"""
from dataclasses import replace
from numbers import Number
from typing import Callable, Optional, Sequence, Tuple, Union


import numpy as np
import pennylane as qml
from pennylane.devices import DefaultExecutionConfig, Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.typing import Result, ResultBatch

from ._mps import QuimbMPS

supported_backends = ["quimb", "cutensornet"]
supported_methods = ["mps", "tn"]

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]


# TODO: add class docs
@simulator_tracking
@single_tape_support
class LightningTensor(Device):
    """PennyLane Lightning Tensor device.

    A device to perform fast linear algebra and tensor network calculations.
    """

    _device_options = ("backend", "c_dtype", "method", "max_bond_dim")

    _new_API = True

    # TODO: add `max_bond_dim` parameter
    def __init__(
        self,
        *,
        wires=None,
        backend="quimb",
        method="mps",
        c_dtype=np.complex128,
        shots=None,
    ):

        if backend not in supported_backends:
            raise ValueError(f"Unsupported backend: {backend}")

        if method not in supported_methods:
            raise ValueError(f"Unsupported method: {method}")

        if shots is not None:
            raise ValueError("LightningTensor does not support the `shots` parameter.")

        super().__init__(wires=wires, shots=shots)

        self._backend = backend
        self._method = method
        self._c_dtype = c_dtype
        self._num_wires = len(self.wires) if self.wires else 0
        self._statetensor = None

        if backend == "quimb" and method == "mps":
            self._statetensor = QuimbMPS(num_wires=self.num_wires, dtype=self._c_dtype)

    @property
    def name(self):
        """The name of the device."""
        return "lightning.tensor"

    @property
    def backend(self):
        """Supported backend."""
        return self._backend

    @property
    def method(self):
        """Supported method."""
        return self._method

    @property
    def c_dtype(self):
        """State vector complex data type."""
        return self._c_dtype

    @property
    def num_wires(self):
        """Number of wires addressed on this device."""
        return self._num_wires

    dtype = c_dtype

    # should `backend` and `method` be inserted here?
    def _setup_execution_config(self, config):
        pass

    def preprocess(self, execution_config: ExecutionConfig = DefaultExecutionConfig):
        pass

    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        pass

    def supports_derivatives(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[qml.tape.QuantumTape] = None,
    ) -> bool:
        pass

    def compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        pass

    def execute_and_compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        pass

    def supports_vjp(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        pass

    def compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        pass

    def execute_and_compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        pass
