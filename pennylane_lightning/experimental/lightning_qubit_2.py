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
"""
This module contains the next generation successor to LightningQubit
"""

import numpy as np
from typing import Union, Callable, Tuple, Optional, Sequence
from warnings import warn

from pennylane.devices.experimental import Device
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.devices.experimental.execution_config import ExecutionConfig, DefaultExecutionConfig
from pennylane.devices.qubit.simulate import simulate
from pennylane.devices.qubit.preprocess import preprocess

QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]

try:
    from ..lightning_qubit_ops import StateVectorC64, StateVectorC128

    CPP_BINARY_AVAILABLE = True
except ModuleNotFoundError:
    CPP_BINARY_AVAILABLE = False

if CPP_BINARY_AVAILABLE:
    DeviceExecutionConfig = DefaultExecutionConfig

    class LightningQubit2(Device):
        """PennyLane Lightning device.

        A device that interfaces with C++ to perform fast linear algebra calculations.

        Use of this device requires pre-built binaries or compilation from source. Check out the
        :doc:`/installation` guide for more details.

        Args:
            c_dtype: Datatypes for statevector representation. Must be one of ``np.complex64`` or ``np.complex128``.
        """

        def __init__(self, c_dtype=np.complex128):
            self.C_DTYPE = c_dtype
            if self.C_DTYPE not in [np.complex64, np.complex128]:
                raise TypeError(f"Unsupported complex Type: {c_dtype}")

            super().__init__()

        @property
        def name(self):
            """The name of the device."""
            return "lightning.qubit.2"

        def preprocess(
            self,
            circuits: QuantumTape_or_Batch,
            execution_config: ExecutionConfig = DeviceExecutionConfig,
        ) -> Tuple[QuantumTapeBatch, Callable]:
            """Converts an arbitrary circuit or batch of circuits into a batch natively executable by the :meth:`~.execute` method.

            Args:
                circuits (Union[QuantumTape, Sequence[QuantumTape]]): The circuit or a batch of circuits to preprocess
                    before execution on the device
                execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure describing the parameters needed to fully describe
                    the execution. Includes such information as shots.

            Returns:
                Sequence[QuantumTape], Callable: QuantumTapes that the device can natively execute
                and a postprocessing function to be called after execution.

            This device:

            * Supports any qubit operations that provide a matrix
            * Currently does not support finite shots
            * Currently does not intrinsically support parameter broadcasting

            """
            is_single_circuit = False
            if isinstance(circuits, QuantumScript):
                circuits = [circuits]
                is_single_circuit = True

            batch, post_processing_fn = preprocess(circuits, execution_config=execution_config)

            if is_single_circuit:

                def convert_batch_to_single_output(results):
                    """Unwraps a dimension so that executing the batch of circuits looks like executing a single circuit."""
                    return post_processing_fn(results)[0]

                return batch, convert_batch_to_single_output

            return batch, post_processing_fn

        def execute(
            self,
            circuits: QuantumTape_or_Batch,
            execution_config: ExecutionConfig = DeviceExecutionConfig,
        ):
            """Execute a circuit or a batch of circuits and turn it into results.

            Args:
                circuits (Union[QuantumTape, Sequence[QuantumTape]]): the quantum circuits to be executed
                execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): a data structure with additional information required for execution

            Returns:
                TensorLike, tuple[TensorLike], tuple[tuple[TensorLike]]: A numeric result of the computation.
            """
            is_single_circuit = False
            if isinstance(circuits, QuantumScript):
                is_single_circuit = True
                circuits = [circuits]

            if self.tracker.active:
                self.tracker.update(batches=1, executions=len(circuits))
                self.tracker.record()

            results = tuple(simulate(c) for c in circuits)
            return results[0] if is_single_circuit else results

        def supports_derivatives(
            self,
            execution_config: ExecutionConfig,
            circuit: Optional[QuantumTape] = None,
        ) -> bool:
            """Check whether or not derivatives are available for a given configuration and circuit.

            ``LightningQubit2`` supports adjoint differentiation method.

            Args:
                execution_config (ExecutionConfig): The configuration of the desired derivative calculation
                circuit (QuantumTape): An optional circuit to check derivatives support for.

            Returns:
                Bool: Whether or not a derivative can be calculated provided the given information
            """
            if (
                execution_config.gradient_method != "adjoint"
                or execution_config.derivative_order != 1
            ):
                return False
            return True

else:
    from pennylane.devices.experimental import DefaultQubit2

    class LightningQubit2(DefaultQubit2):  # pragma: no cover
        def __init__(self, c_dtype=np.complex128):
            warn(
                "Pre-compiled binaries for lightning.qubit are not available. Falling back to "
                "using the Python-based default.qubit implementation. To manually compile from "
                "source, follow the instructions at "
                "https://pennylane-lightning.readthedocs.io/en/latest/installation.html.",
                UserWarning,
            )
            self.C_DTYPE = c_dtype
            if self.C_DTYPE not in [np.complex64, np.complex128]:
                raise TypeError(f"Unsupported complex Type: {c_dtype}")

            super().__init__()
