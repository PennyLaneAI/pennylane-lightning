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
from dataclasses import replace
import abc
import numpy as np
from typing import Union, Callable, Tuple, Optional, Sequence
from warnings import warn

import pennylane as qml
from pennylane.devices import Device, DefaultQubit
from pennylane.devices.qubit.sampling import get_num_shots_and_executions
from pennylane.devices.execution_config import ExecutionConfig, DefaultExecutionConfig
from pennylane.devices.preprocess import (
    decompose,
    validate_observables,
    validate_measurements,
    validate_device_wires,
    # warn_about_trainable_observables,
    # no_sampling,
)

from pennylane.tape import QuantumTape, QuantumScript
from pennylane.typing import Result, ResultBatch
from pennylane.transforms.core import TransformProgram

# from ._preprocess import preprocess, validate_and_expand_adjoint
from ._preprocess import (
    _add_adjoint_transforms,
    stopping_condition,
    accepted_sample_measurement,
    observable_stopping_condition,
)

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
# always a function from a resultbatch to either a result or a result batch
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]

from ._version import __version__

from pennylane_lightning.core._adjoint_jacobian import AdjointJacobian

AdjointExecutionConfig = ExecutionConfig(use_device_gradient=True, gradient_method="adjoint")


class LightningBase(Device):
    """PennyLane Lightning device.

    A device that interfaces with C++ to perform fast linear algebra calculations.

    Use of this device requires pre-built binaries or compilation from source. Check out the
    :doc:`/installation` guide for more details.

    Args:
        shots (int, Sequence[int], Sequence[Union[int, Sequence[int]]]): The default number of shots to use in executions involving
            this device.
        c_dtype: Datatypes for state vector representation. Must be one of ``np.complex64`` or ``np.complex128``.
        batch_obs (bool): Determine whether we process observables in parallel when
                computing the jacobian. This value is only relevant when the lightning
                qubit is built with OpenMP.
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
        seed="global",
        batch_obs=False,
    ) -> None:
        self.C_DTYPE = c_dtype
        self._batch_obs = batch_obs
        if self.C_DTYPE not in [np.complex64, np.complex128]:
            raise TypeError(f"Unsupported complex Type: {c_dtype}")
        super().__init__(wires=wires, shots=shots)
        seed = np.random.randint(0, high=10000000) if seed == "global" else seed
        self._rng = np.random.default_rng(seed)
        self._debugger = None

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

    def _setup_execution_config(self, config: ExecutionConfig) -> ExecutionConfig:
        """Choose the "best" options for the configuration if they are left unspecified.

        Args:
            config (ExecutionConfig): the initial execution config

        Returns:
            ExecutionConfig: a new config with the best choices selected.
        """
        updated_values = {}
        if config.gradient_method == "best":
            updated_values["gradient_method"] = "adjoint"
        if config.use_device_gradient is None:
            updated_values["use_device_gradient"] = config.gradient_method in {
                "best",
                "adjoint",
            }
        if config.grad_on_execution is None:
            updated_values["grad_on_execution"] = config.gradient_method == "adjoint"

        updated_values["device_options"] = dict(config.device_options)
        if "rng" not in updated_values["device_options"]:
            updated_values["device_options"]["rng"] = self._rng

        return replace(config, **updated_values)

    def preprocess(
        self,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Tuple[TransformProgram, ExecutionConfig]:
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
        config = self._setup_execution_config(execution_config)
        transform_program = TransformProgram()

        transform_program.add_transform(qml.defer_measurements)
        transform_program.add_transform(validate_device_wires, self.wires, name=self.name)
        transform_program.add_transform(
            decompose, stopping_condition=stopping_condition, name=self.name
        )
        transform_program.add_transform(
            validate_measurements, sample_measurements=accepted_sample_measurement, name=self.name
        )
        transform_program.add_transform(
            validate_observables, stopping_condition=observable_stopping_condition, name=self.name
        )

        if config.gradient_method == "adjoint":
            _add_adjoint_transforms(transform_program)

        return transform_program, config

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
            self.tracker.update(batches=1)
            self.tracker.record()
            for c in circuits:
                qpu_executions, shots = get_num_shots_and_executions(c)
                if c.shots:
                    self.tracker.update(
                        simulations=1,
                        executions=qpu_executions,
                        shots=shots,
                        resources=c.specs["resources"],
                    )
                else:
                    self.tracker.update(
                        simulations=1, executions=qpu_executions, resources=c.specs["resources"]
                    )
                self.tracker.record()

        results = tuple(
            self.simulate(c, self.C_DTYPE, rng=self._rng, debugger=self._debugger) for c in circuits
        )
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
        if (
            execution_config.gradient_method == "adjoint"
            and execution_config.use_device_gradient is not False
        ):
            if circuit is None:
                return True
            # if self.shots.total_shots is not None:
            #     warn(
            #         "Requested adjoint differentiation to be computed with finite shots. "
            #         "The derivative is always exact when using the adjoint "
            #         "differentiation method.",
            #         UserWarning,
            #     )
            # if circuit is None:
            #     return True
            # return self._check_adjoint_method_supported(circuit) and isinstance(
            #     validate_and_expand_adjoint(circuit)[0][0], QuantumScript
            # )
            prog = TransformProgram()
            _add_adjoint_transforms(prog)

            try:
                prog((circuit,))
            except (qml.operation.DecompositionUndefinedError, qml.DeviceError):
                return False
            return self._check_adjoint_method_supported(circuit)

        return False

    def adjoint_jacobian(self, tape, c_dtype=np.complex128, starting_state=None):
        """Calculates the Adjoint Jacobian for a given tape.

        Args:
            tape (QuantumTape): A quantum tape recording a variational quantum program.
            c_dtype (Complex data type, Optional): Default to ``np.complex128``.
            starting_state (np.array, Optional): unravelled initial state (1D). Default to None.

        Returns:
            np.array: An array results.
        """
        return AdjointJacobian(self.name).calculate_adjoint_jacobian(
            tape, c_dtype, starting_state, self._batch_obs
        )

    def compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = AdjointExecutionConfig,
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
                self.tracker.update(derivative_batches=1, derivatives=len(circuits))
                self.tracker.record()

            results = tuple(self.adjoint_jacobian(c, self.C_DTYPE) for c in circuits)
            return results[0] if is_single_circuit else results

        raise NotImplementedError

    @abc.abstractmethod
    def simulate_and_adjoint(
        self, circuit: QuantumScript, c_dtype=np.complex128, rng=None, debugger=None
    ) -> Result:
        """Simulate a single quantum script and calculates the state gradient with the adjoint Jacobian method.

        Args:
            circuit (QuantumTape): The single circuit to simulate
            rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
                seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
                If no value is provided, a default RNG will be used.
            debugger (_Debugger): The debugger to use

        Returns:
            Results of the simulation and circuit gradient
        """

    def execute_and_compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
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

            results = tuple(
                self.simulate_and_adjoint(c, rng=self._rng, debugger=self._debugger)
                for c in circuits
            )
            results, jacs = tuple(zip(*results))
            return (results[0], jacs[0]) if is_single_circuit else (results, jacs)
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
