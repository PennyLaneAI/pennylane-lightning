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
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch

from .quimb._mps import QuimbMPS

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]


_backends = frozenset({"quimb"})
# The set of supported backends.

_methods = frozenset({"mps"})
# The set of supported methods.


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

    Args:
        wires (int): The number of wires to initialize the device with.
            Defaults to ``None`` if not specified.
        backend (str): Supported backend. Must be one of ``quimb`` or ``cutensornet``.
        method (str): Supported method. Must be one of ``mps`` or ``tn``.
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Currently, it can only be ``None``, so that computation of
            statistics like expectation values and variances is performed analytically.
        c_dtype: Datatypes for statevector representation. Must be one of
            ``np.complex64`` or ``np.complex128``.
        **kwargs: keyword arguments.
    """

    # TODO: decide whether to move some of the attributes in interfaces classes
    # pylint: disable=too-many-instance-attributes

    # So far we just insert the options for MPS simulator
    _device_options = (
        "apply_reverse_lightcone",
        "backend",
        "c_dtype",
        "cutoff",
        "method",
        "max_bond_dim",
        "measure_algorithm",
        "return_tn",
        "rehearse",
    )

    _new_API = True

    # TODO: decide if `backend` and `method` should be keyword args as well
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        wires=None,
        backend="quimb",
        method="mps",
        shots=None,
        c_dtype=np.complex128,
        **kwargs,
    ):

        if not accepted_backends(backend):
            raise ValueError(f"Unsupported backend: {backend}")

        if not accepted_methods(method):
            raise ValueError(f"Unsupported method: {method}")

        if shots is not None:
            raise ValueError("LightningTensor does not support the `shots` parameter.")

        super().__init__(wires=wires, shots=shots)

        self._num_wires = len(self.wires) if self.wires else 0
        self._backend = backend
        self._method = method
        self._c_dtype = c_dtype

        # options for MPS
        self._max_bond_dim = kwargs.get("max_bond_dim", None)
        self._cutoff = kwargs.get("cutoff", 1e-16)
        self._measure_algorithm = kwargs.get("measure_algorithm", None)

        # common options (MPS and TN)
        self._apply_reverse_lightcone = kwargs.get("apply_reverse_lightcone", None)
        self._return_tn = kwargs.get("return_tn", None)
        self._rehearse = kwargs.get("rehearse", None)

        self._interface = None

        # TODO: implement the remaining combs of `backend` and `interface`
        if self.backend == "quimb" and self.method == "mps":
            self._interface = QuimbMPS(self._num_wires, self._c_dtype)

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
        """State vector complex data type."""
        return self._c_dtype

    dtype = c_dtype

    def _setup_execution_config(self, config):
        """
        Update the execution config with choices for how the device should be used and the device options.
        """
        updated_values = {}
        if config.use_device_gradient is None:
            updated_values["use_device_gradient"] = config.gradient_method in (
                "best",
                "adjoint",
            )
        if config.grad_on_execution is None:
            updated_values["grad_on_execution"] = True

        new_device_options = dict(config.device_options)
        for option in self._device_options:
            if option not in new_device_options:
                new_device_options[option] = getattr(self, f"_{option}", None)

        return replace(config, **updated_values, device_options=new_device_options)

    def preprocess(
        self,
        circuits: QuantumTape_or_Batch,  # pylint: disable=unused-argument
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """This function defines the device transform program to be applied and an updated device configuration.

        Args:
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure describing the
                parameters needed to fully describe the execution.

        Returns:
            # TODO: decide
        """

        config = self._setup_execution_config(execution_config)

        return config

    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        """Execute a circuit or a batch of circuits and turn it into results.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the quantum circuits to be executed.
            execution_config (ExecutionConfig): a datastructure with additional information required for execution.

        Returns:
            TensorLike, tuple[TensorLike], tuple[tuple[TensorLike]]: A numeric result of the computation.
        """
        # TODO: call the function implemented in the appropriate interface

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
        # TODO: call the function implemented in the appropriate interface

    def compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """Calculate the jacobian of either a single or a batch of circuits on the device.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuits to calculate derivatives for.
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution.

        Returns:
            Tuple: The jacobian for each trainable parameter.
        """
        # TODO: call the function implemented in the appropriate interface

    def execute_and_compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """Compute the results and jacobians of circuits at the same time.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuits or batch of circuits.
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution.

        Returns:
            tuple: A numeric result of the computation and the gradient.
        """
        # TODO: call the function implemented in the appropriate interface

    def supports_vjp(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """Whether or not this device defines a custom vector jacobian product.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation.
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            Bool: Whether or not a derivative can be calculated provided the given information.
        """
        # TODO: call the function implemented in the appropriate interface

    def compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        r"""The vector jacobian product used in reverse-mode differentiation.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuit or batch of circuits.
            cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must have shape matching the output shape of the
                corresponding circuit. If the circuit has a single output, ``cotangents`` may be a single number, not an iterable
                of numbers.
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution.

        Returns:
            tensor-like: A numeric result of computing the vector jacobian product.
        """
        # TODO: call the function implemented in the appropriate interface

    def execute_and_compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """Calculate both the results and the vector jacobian product used in reverse-mode differentiation.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuit or batch of circuits to be executed.
            cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must have shape matching the output shape of the
                corresponding circuit.
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution.

        Returns:
            Tuple, Tuple: the result of executing the scripts and the numeric result of computing the vector jacobian product
        """
        # TODO: call the function implemented in the appropriate interface
