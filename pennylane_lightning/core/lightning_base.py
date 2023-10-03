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
This module contains the base class for all PennyLane Lightning simulator devices,
and interfaces with C++ for improved performance.
"""
import abc
import numpy as np
from typing import Union, Callable, Tuple, Optional, Sequence
from warnings import warn
from pennylane.devices import Device
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.typing import Result, ResultBatch
from pennylane.transforms.core import TransformProgram
from pennylane.devices.execution_config import ExecutionConfig, DefaultExecutionConfig

from pennylane.devices.qubit.preprocess import (
    validate_device_wires,
)

from ._preprocess import preprocess, validate_and_expand_adjoint

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
# always a function from a resultbatch to either a result or a result batch
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]

from pennylane.devices import DefaultQubit

from ._version import __version__

from pennylane_lightning.core._adjoint_jacobian import AdjointJacobian


class LightningBase(Device):
    """PennyLane Lightning device.

    A device that interfaces with C++ to perform fast linear algebra calculations.

    Use of this device requires pre-built binaries or compilation from source. Check out the
    :doc:`/installation` guide for more details.

    Args:
        c_dtype: Datatypes for state vector representation. Must be one of ``np.complex64`` or ``np.complex128``.
    """

    pennylane_requires = ">=0.30"
    version = __version__
    author = "Xanadu Inc."
    _CPP_BINARY_AVAILABLE = True

    # pylint:disable = too-many-arguments
    def __init__(
        self,
        wires=None,
        shots=None,
        c_dtype=np.complex128,
    ) -> None:
        self.C_DTYPE = c_dtype
        if self.C_DTYPE not in [np.complex64, np.complex128]:
            raise TypeError(f"Unsupported complex Type: {c_dtype}")
        super().__init__(wires=wires, shots=shots)

    @property
    def name(self):
        """The name of the device."""
        return "lightning.base"

    def _check_adjoint_method_supported(self, tape: QuantumTape):
        """Check measurement and operation lists for adjoint Jacobian support in Lightning.

        Args:
            tape (QuantumTape): A quantum tape recording a variational quantum program.
        """
        AdjointJacobian(self.name)._check_adjoint_method_supported(tape)

    def preprocess(
        self,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Tuple[QuantumTapeBatch, PostprocessingFn, ExecutionConfig]:
        """This function defines the device transform program to be applied and an updated device configuration.

        Args:
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure describing the
                parameters needed to fully describe the execution.

        Returns:
            TransformProgram, ExecutionConfig: A transform program that when called returns QuantumTapes that the device
            can natively execute as well as a postprocessing function to be called after execution, and a configuration with
            unset specifications filled in.

        This device:

        * Supports any qubit operations that provide a matrix
        * Currently does not support finite shots
        * Currently does not intrinsically support parameter broadcasting

        """
        transform_program = TransformProgram()
        # Validate device wires
        transform_program.add_transform(validate_device_wires, self)

        # General preprocessing (Validate measurement, expand, adjoint expand, broadcast expand)
        transform_program_preprocess, config = preprocess(execution_config=execution_config)
        transform_program = transform_program + transform_program_preprocess
        return transform_program, config
        # is_single_circuit = False
        # if isinstance(circuits, QuantumScript):
        #     circuits = [circuits]
        #     is_single_circuit = True

        # if execution_config.gradient_method == "adjoint":
        #     for c in circuits:
        #         self._check_adjoint_method_supported(c)

        # batch, post_processing_fn, config = preprocess(circuits, execution_config=execution_config)

        # if is_single_circuit:

        #     def convert_batch_to_single_output(results: ResultBatch) -> Result:
        #         """Unwraps a dimension so that executing the batch of circuits looks like executing a single circuit."""
        #         return post_processing_fn(results)[0]

        #     return batch, convert_batch_to_single_output, config

        # return batch, post_processing_fn, config

    @abc.abstractmethod
    def simulate(
        self, circuit: QuantumScript, c_dtype=np.complex128, rng=None, debugger=None
    ) -> Result:
        """Simulate a single quantum script.

        Args:
            circuit (QuantumTape): The single circuit to simulate
            rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
                seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
                If no value is provided, a default RNG will be used.
            debugger (_Debugger): The debugger to use

        Returns:
            tuple(TensorLike): The results of the simulation

        Note that this function can return measurements for non-commuting observables simultaneously.

        This function assumes that all operations provide matrices.

        """

    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
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

        results = tuple(self.simulate(c, self.C_DTYPE) for c in circuits)
        return results[0] if is_single_circuit else results

    def supports_derivatives(
        self,
        execution_config: ExecutionConfig,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """Check whether or not derivatives are available for a given configuration and circuit.

        ``LightningQubit`` supports adjoint differentiation method.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            Bool: Whether or not a derivative can be calculated provided the given information
        """
        # # if execution_config.gradient_method != "adjoint" or execution_config.derivative_order != 1:
        # #     return False
        # # return True
        # if execution_config.gradient_method == "adjoint" and execution_config.derivative_order == 1:
        #     if circuit is None:
        #         return True
        #     print("circuit", circuit)
        #     print(validate_and_expand_adjoint(tape=circuit))
        #     tape, _ = validate_and_expand_adjoint(tape=circuit)
        #     return isinstance(tape[0][0], QuantumScript)

        # return False
        if execution_config.gradient_method == "adjoint" and execution_config.use_device_gradient:
            if self.shots is not None:
                warn(
                    "Requested adjoint differentiation to be computed with finite shots. "
                    "The derivative is always exact when using the adjoint "
                    "differentiation method.",
                    UserWarning,
                )
            if circuit is None:
                return True
            return self._check_adjoint_method_supported(circuit) and isinstance(
                validate_and_expand_adjoint(circuit)[0][0], QuantumScript
            )

        return False

    def adjoint_jacobian(self, tape, c_dtype=np.complex128):
        """Calculates the Adjoint Jacobian for a given tape.

        Args:
            tape (QuantumTape): A quantum tape recording a variational quantum program.
            c_dtype (Complex data type, Optional): Default to ``np.complex128``.
            starting_state (np.array, Optional): unravelled initial state (1D). Default to None.

        Returns:
            np.array: An array results.
        """
        return AdjointJacobian(self.name).calculate_adjoint_jacobian(tape, c_dtype)

    def compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig,
    ):
        """Calculate the jacobian of either a single or a batch of circuits on the device.
        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuits to calculate derivatives for
            execution_config (ExecutionConfig): a data structure with all additional information required for execution
        Returns:
            Tuple: The jacobian for each trainable parameter
        .. seealso:: :meth:`~.supports_derivatives` and :meth:`~.execute_and_compute_derivatives`.
        **Execution Config:**
        The execution config has ``gradient_method`` and ``order`` property that describes the order of differentiation requested. If the requested
        method or order of gradient is not provided, the device should raise a ``NotImplementedError``. The :meth:`~.supports_derivatives`
        method can pre-validate supported orders and gradient methods.
        **Return Shape:**
        If a batch of quantum scripts is provided, this method should return a tuple with each entry being the gradient of
        each individual quantum script. If the batch is of length 1, then the return tuple should still be of length 1, not squeezed.
        """
        if execution_config.gradient_method == "adjoint":
            is_single_circuit = False
            if isinstance(circuits, QuantumScript):
                is_single_circuit = True
                circuits = [circuits]

            if self.tracker.active:
                for c in circuits:
                    self.tracker.update(resources=c.specs["resources"])
                self.tracker.update(
                    execute_and_derivative_batches=1,
                    executions=len(circuits),
                    derivatives=len(circuits),
                )
                self.tracker.record()

            results = tuple(self.adjoint_jacobian(c, self.C_DTYPE) for c in circuits)
            return results[0] if is_single_circuit else results

        raise NotImplementedError


class LightningBaseFallBack(DefaultQubit):  # pragma: no cover
    # pylint: disable=missing-class-docstring
    pennylane_requires = ">=0.30"
    version = __version__
    author = "Xanadu Inc."
    _CPP_BINARY_AVAILABLE = False

    def __init__(
        self,
        wires=None,
        shots=None,
        c_dtype=np.complex128,
    ) -> None:
        self.C_DTYPE = c_dtype
        if self.C_DTYPE not in [np.complex64, np.complex128]:
            raise TypeError(f"Unsupported complex Type: {c_dtype}")

        super().__init__(wires=wires, shots=shots)
