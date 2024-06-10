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
This module contains the LightningTensor class that inherits from the new device interface.
It is a device to perform tensor network simulations of quantum circuits using `cutensornet`. 
"""
from dataclasses import replace
from numbers import Number
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pennylane as qml
from pennylane.devices import DefaultExecutionConfig, Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import (
    decompose,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from pennylane.operation import Operator
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms.core import TransformProgram
from pennylane.typing import Result, ResultBatch

from ._measurements import LightningTensorMeasurements
from ._state_tensor import LightningStateTensor

try:
    # pylint: disable=import-error, unused-import
    from pennylane_lightning.lightning_tensor_ops import backend_info

    LT_CPP_BINARY_AVAILABLE = True
except ImportError:
    LT_CPP_BINARY_AVAILABLE = False

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]


_backends = frozenset({"cutensornet"})
# The set of supported backends.

_methods = frozenset({"mps"})
# The set of supported methods.

_operations = frozenset(
    {
        "Identity",
        "BasisState",
        "QubitUnitary",
        "ControlledQubitUnitary",
        "MultiControlledX",
        "DiagonalQubitUnitary",
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "S",
        "Adjoint(S)",
        "T",
        "Adjoint(T)",
        "SX",
        "Adjoint(SX)",
        "CNOT",
        "SWAP",
        "ISWAP",
        "Adjoint(ISWAP)",
        "PSWAP",
        "Adjoint(SISWAP)",
        "SISWAP",
        "SQISW",
        "CSWAP",
        "Toffoli",
        "CY",
        "CZ",
        "PhaseShift",
        "ControlledPhaseShift",
        "RX",
        "RY",
        "RZ",
        "Rot",
        "CRX",
        "CRY",
        "CRZ",
        "CRot",
        "IsingXX",
        "IsingYY",
        "IsingZZ",
        "IsingXY",
        "SingleExcitation",
        "SingleExcitationPlus",
        "SingleExcitationMinus",
        "DoubleExcitation",
        "QubitCarry",
        "QubitSum",
        "OrbitalRotation",
        "QFT",
        "ECR",
        "BlockEncode",
        "C(BlockEncode)",
    }
)

_observables = frozenset(
    {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "Hermitian",
        "Identity",
        "Hamiltonian",
        "LinearCombination",
        "Sum",
        "SProd",
        "Prod",
        "Exp",
    }
)
# The set of supported observables.


def stopping_condition(op: Operator) -> bool:
    """A function that determines whether or not an operation is supported by the ``mps`` method of ``lightning.tensor``."""
    # These thresholds are adapted from `lightning_base.py`
    # To avoid building matrices beyond the given thresholds.
    # This should reduce runtime overheads for larger systems.
    return op.has_matrix and len(op.wires) <= 2 or isinstance(op, qml.GlobalPhase)


def simulate(circuit: QuantumScript, state: LightningStateTensor) -> Result:
    """Simulate a single quantum script.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        state (LightningStateTensor): handle to Lightning state tensor

    Returns:
        Tuple[TensorLike]: The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.
    """
    state.reset_state()
    state.get_final_state(circuit)
    return LightningTensorMeasurements(state).measure_final_state(circuit)


def accepted_observables(obs: Operator) -> bool:
    """A function that determines whether or not an observable is supported by ``lightning.tensor``."""
    return obs.name in _observables


def accepted_backends(backend: str) -> bool:
    """A function that determines whether or not a backend is supported by ``lightning.tensor``."""
    return backend in _backends


def accepted_methods(method: str) -> bool:
    """A function that determines whether or not a method is supported by ``lightning.tensor``."""
    return method in _methods


@simulator_tracking
@single_tape_support
class LightningTensor(Device):
    """PennyLane Lightning Tensor device.

    A device to perform tensor network operations on a quantum circuit.

    This device is designed to simulate large-scale quantum circuits using tensor network methods. For
    samll circuits, other devices like ``lightning.qubit``, ``lightning.gpu``or ``lightning.kokkos``  are
    recommended.

    Currently, only the Matrix Product State (MPS) method is supported, based on ``cutensornet`` backends.

    Args:
        wires (int): The number of wires to initialize the device with.
            Defaults to ``None`` if not specified.
        max_bond_dim (int): The maximum bond dimension to be used in the MPS simulation.
        backend (str): Supported backend. Currently, only ``cutensornet`` is supported.
        method (str): Supported method. Currently, only ``mps`` is supported.
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Currently, it can only be ``None``, so that computation of
            statistics like expectation values and variances is performed analytically.
        c_dtype: Datatypes for the tensor representation. Must be one of
            ``np.complex64`` or ``np.complex128``.
        **kwargs: keyword arguments.
    """

    # pylint: disable=too-many-instance-attributes

    # So far we just consider the options for MPS simulator
    _device_options = ("backend", "c_dtype")
    _CPP_BINARY_AVAILABLE = LT_CPP_BINARY_AVAILABLE
    _new_API = True

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        wires=None,
        max_bond_dim=128,
        backend="cutensornet",
        method="mps",
        shots=None,
        c_dtype=np.complex128,
        **kwargs,
    ):
        if not self._CPP_BINARY_AVAILABLE:
            raise ImportError("Pre-compiled binaries for lightning.tensor are not available. ")

        if not accepted_backends(backend):
            raise ValueError(f"Unsupported backend: {backend}")

        if not accepted_methods(method):
            raise ValueError(f"Unsupported method: {method}")

        if shots is not None:
            raise ValueError("lightning.tensor does not support finite shots.")

        if c_dtype not in [np.complex64, np.complex128]:  # pragma: no cover
            raise TypeError(f"Unsupported complex type: {c_dtype}")

        if wires is None:
            raise ValueError("The number of wires must be specified.")

        if not isinstance(max_bond_dim, int) or max_bond_dim < 1:
            raise ValueError("The maximum bond dimension must be an integer greater than 0.")

        for arg in kwargs:
            if arg not in self._device_options:
                raise TypeError(
                    f"Unexpected argument: {arg} during initialization of the lightning.tensor device."
                )

        super().__init__(wires=wires, shots=shots)

        if isinstance(wires, int):
            self._wire_map = None  # should just use wires as is
        else:
            self._wire_map = {w: i for i, w in enumerate(self.wires)}

        self._num_wires = len(self.wires) if self.wires else 0
        self._max_bond_dim = max_bond_dim
        self._backend = backend
        self._method = method
        self._c_dtype = c_dtype

    @property
    def name(self):
        """The name of the device."""
        return "lightning.tensor"

    @property
    def num_wires(self):
        """Number of wires addressed on this device."""
        return self._num_wires

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
        """Tensor complex data type."""
        return self._c_dtype

    def _state_tensor(self):
        """Return the state tensor object."""
        return LightningStateTensor(self._num_wires, self._max_bond_dim, self._c_dtype)

    dtype = c_dtype

    def _setup_execution_config(
        self, config: Optional[ExecutionConfig] = DefaultExecutionConfig
    ) -> ExecutionConfig:
        """
        Update the execution config with choices for how the device should be used and the device options.
        """
        # TODO: add options for gradients next quarter

        updated_values = {}

        new_device_options = dict(config.device_options)
        for option in self._device_options:
            if option not in new_device_options:
                new_device_options[option] = getattr(self, f"_{option}", None)

        return replace(config, **updated_values, device_options=new_device_options)

    def preprocess(
        self,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """This function defines the device transform program to be applied and an updated device configuration.

        Args:
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure describing the
                parameters needed to fully describe the execution.

        Returns:
            TransformProgram, ExecutionConfig: A transform program that when called returns :class:`~.QuantumTape`'s that the
            device can natively execute as well as a postprocessing function to be called after execution, and a configuration
            with unset specifications filled in.

        This device currently:

        * Does not support finite shots.
        * Does not support derivatives.
        * Does not support vector-Jacobian products.
        """

        # TODO: remove comments when cuTensorNet MPS backend is available as a prototype
        config = self._setup_execution_config(execution_config)

        program = TransformProgram()

        program.add_transform(validate_measurements, name=self.name)
        program.add_transform(validate_observables, accepted_observables, name=self.name)
        program.add_transform(validate_device_wires, self._wires, name=self.name)
        program.add_transform(
            decompose,
            stopping_condition=stopping_condition,
            skip_initial_state_prep=True,
            name=self.name,
        )
        return program, config

    # pylint: disable=unused-argument
    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        """Execute a circuit or a batch of circuits and turn it into results.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the quantum circuits to be executed.
            execution_config (ExecutionConfig): a data structure with additional information required for execution.

        Returns:
            TensorLike, tuple[TensorLike], tuple[tuple[TensorLike]]: A numeric result of the computation.
        """

        results = []

        for circuit in circuits:
            if self._wire_map is not None:
                [circuit], _ = qml.map_wires(circuit, self._wire_map)
            results.append(simulate(circuit, self._state_tensor()))

        return tuple(results)

    # pylint: disable=unused-argument
    def supports_derivatives(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[qml.tape.QuantumTape] = None,
    ) -> bool:
        """Check whether or not derivatives are available for a given configuration and circuit.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation.
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            Bool: Whether or not a derivative can be calculated provided the given information.

        """
        return False

    def compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """Calculate the Jacobian of either a single or a batch of circuits on the device.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuits to calculate derivatives for.
            execution_config (ExecutionConfig): a data structure with all additional information required for execution.

        Returns:
            Tuple: The Jacobian for each trainable parameter.
        """
        raise NotImplementedError(
            "The computation of derivatives has yet to be implemented for the lightning.tensor device."
        )

    def execute_and_compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """Compute the results and Jacobians of circuits at the same time.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuits or batch of circuits.
            execution_config (ExecutionConfig): a data structure with all additional information required for execution.

        Returns:
            tuple: A numeric result of the computation and the gradient.
        """
        raise NotImplementedError(
            "The computation of derivatives has yet to be implemented for the lightning.tensor device."
        )

    # pylint: disable=unused-argument
    def supports_vjp(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """Whether or not this device defines a custom vector-Jacobian product.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation.
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            Bool: Whether or not a derivative can be calculated provided the given information.
        """
        return False

    def compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        r"""The vector-Jacobian product used in reverse-mode differentiation.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuit or batch of circuits.
            cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must have shape matching the output shape of the
                corresponding circuit. If the circuit has a single output, ``cotangents`` may be a single number, not an iterable
                of numbers.
            execution_config (ExecutionConfig): a data structure with all additional information required for execution.

        Returns:
            tensor-like: A numeric result of computing the vector-Jacobian product.
        """
        raise NotImplementedError(
            "The computation of vector-Jacobian product has yet to be implemented for the lightning.tensor device."
        )

    def execute_and_compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """Calculate both the results and the vector-Jacobian product used in reverse-mode differentiation.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuit or batch of circuits to be executed.
            cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must have shape matching the output shape of the
                corresponding circuit.
            execution_config (ExecutionConfig): a data structure with all additional information required for execution.

        Returns:
            Tuple, Tuple: the result of executing the scripts and the numeric result of computing the vector-Jacobian product
        """
        raise NotImplementedError(
            "The computation of vector-Jacobian product has yet to be implemented for the lightning.tensor device."
        )
