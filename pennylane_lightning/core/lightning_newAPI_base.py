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
This module contains the :class:`~.LightningBase` class, that serves as a base class for Lightning simulator devices that
interfaces with C++ for fast linear algebra calculations.
"""
from abc import abstractmethod
from numbers import Number
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pennylane as qml
from pennylane.devices import DefaultExecutionConfig, Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.typing import Result, ResultBatch

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]


@simulator_tracking
@single_tape_support
class LightningBase(Device):
    """PennyLane Lightning Base device.

    A class that serves as a base class for Lightning state-vector simulators.

    Args:
        wires (int or list): number or list of wires to initialize the device with
        sync (bool): immediately sync with host-sv after applying operations
        c_dtype: Datatypes for statevector representation. Must be one of
            ``np.complex64`` or ``np.complex128``.
        shots (int or list): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically.
    """

    # pylint: disable=too-many-instance-attributes

    # General device options
    _new_API = True

    def __init__(  # pylint: disable=too-many-arguments
        self,
        wires: Union[int, List],
        *,
        c_dtype: Union[np.complex64, np.complex128],
        shots: Union[int, List],
        batch_obs: bool,
    ):
        super().__init__(wires=wires, shots=shots)

        self._c_dtype = c_dtype
        self._batch_obs = batch_obs

        if isinstance(wires, int):
            self._wire_map = None  # should just use wires as is
        else:
            self._wire_map = {w: i for i, w in enumerate(self.wires)}

        # Dummy for LightningStateVector, LightningMeasurements, LightningAdjointJacobian
        self.LightningStateVector: Callable = None
        self.LightningMeasurements: Callable = None
        self.LightningAdjointJacobian: Callable = None

    @property
    def c_dtype(self):
        """State vector complex data type."""
        return self._c_dtype

    dtype = c_dtype

    @abstractmethod
    def _set_lightning_classes(self):
        """Load the LightningStateVector, LightningMeasurements, LightningAdjointJacobian as class attribute"""

    @abstractmethod
    def _setup_execution_config(self, config):
        """
        Update the execution config with choices for how the device should be used and the device options.
        """

    @abstractmethod
    def preprocess(self, execution_config: ExecutionConfig = DefaultExecutionConfig):
        """This function defines the device transform program to be applied and an updated device configuration.

        Args:
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure describing the
                parameters needed to fully describe the execution.

        Returns:
            TransformProgram, ExecutionConfig: A transform program that when called returns :class:`~.QuantumTape`'s that the
            device can natively execute as well as a postprocessing function to be called after execution, and a configuration
            with unset specifications filled in.

        This device:

        * Supports any qubit operations that provide a matrix
        * Currently does not support finite shots
        * Currently does not intrinsically support parameter broadcasting

        """

    @abstractmethod
    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        """Execute a circuit or a batch of circuits and turn it into results.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the quantum circuits to be executed
            execution_config (ExecutionConfig): a datastructure with additional information required for execution

        Returns:
            TensorLike, tuple[TensorLike], tuple[tuple[TensorLike]]: A numeric result of the computation.
        """

    @abstractmethod
    def simulate(
        self,
        circuit: QuantumScript,
        state,  # Lightning [Device] StateVector
        postselect_mode: str = None,
    ) -> Result:
        """Simulate a single quantum script.

        Args:
            circuit (QuantumTape): The single circuit to simulate
            state (Lightning [Device] StateVector): handle to Lightning state vector
            postselect_mode (str): Configuration for handling shots with mid-circuit measurement
                postselection. Use ``"hw-like"`` to discard invalid shots and ``"fill-shots"`` to
                keep the same number of shots. Default is ``None``.

        Returns:
            Tuple[TensorLike]: The results of the simulation

        Note that this function can return measurements for non-commuting observables simultaneously.
        """

    @abstractmethod
    def supports_derivatives(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[qml.tape.QuantumTape] = None,
    ) -> bool:
        """Check whether or not derivatives are available for a given configuration and circuit.

        ``LightningGPU`` supports adjoint differentiation with analytic results.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            Bool: Whether or not a derivative can be calculated provided the given information

        """

    def jacobian(
        self,
        circuit: QuantumTape,
        state,  # Lightning [Device] StateVector
        batch_obs: bool = False,
        wire_map: dict = None,
    ):
        """Compute the Jacobian for a single quantum script.

        Args:
            circuit (QuantumTape): The single circuit to simulate
            state (Lightning [Device] StateVector): handle to the Lightning state vector
            batch_obs (bool): Determine whether we process observables in parallel when
                computing the jacobian. Default is False.
            wire_map (Optional[dict]): a map from wire labels to simulation indices

        Returns:
            TensorLike: The Jacobian of the quantum script
        """
        if wire_map is not None:
            [circuit], _ = qml.map_wires(circuit, wire_map)
        state.reset_state()
        final_state = state.get_final_state(circuit)
        # pylint: disable=not-callable
        return self.LightningAdjointJacobian(final_state, batch_obs=batch_obs).calculate_jacobian(
            circuit
        )

    def simulate_and_jacobian(
        self,
        circuit: QuantumTape,
        state,  # Lightning [Device] StateVector
        batch_obs: bool = False,
        wire_map: dict = None,
    ) -> Tuple:
        """Simulate a single quantum script and compute its Jacobian.

        Args:
            circuit (QuantumTape): The single circuit to simulate
            state (Lightning [Device] StateVector): handle to the Lightning state vector
            batch_obs (bool): Determine whether we process observables in parallel when
                computing the jacobian. Default is False.
            wire_map (Optional[dict]): a map from wire labels to simulation indices

        Returns:
            Tuple[TensorLike]: The results of the simulation and the calculated Jacobian

        Note that this function can return measurements for non-commuting observables simultaneously.
        """
        if wire_map is not None:
            [circuit], _ = qml.map_wires(circuit, wire_map)
        res = self.simulate(circuit, state)
        # pylint: disable=not-callable
        jac = self.LightningAdjointJacobian(state, batch_obs=batch_obs).calculate_jacobian(circuit)
        return res, jac

    def vjp(  # pylint: disable=too-many-arguments
        self,
        circuit: QuantumTape,
        cotangents: Tuple[Number],
        state,  # Lightning [Device] StateVector
        batch_obs: bool = False,
        wire_map: dict = None,
    ):
        """Compute the Vector-Jacobian Product (VJP) for a single quantum script.
        Args:
            circuit (QuantumTape): The single circuit to simulate
            cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must
                have shape matching the output shape of the corresponding circuit. If
                the circuit has a single output, ``cotangents`` may be a single number,
                not an iterable of numbers.
            state (Lightning [Device] StateVector): handle to the Lightning state vector
            batch_obs (bool): Determine whether we process observables in parallel when
                computing the VJP.
            wire_map (Optional[dict]): a map from wire labels to simulation indices

        Returns:
            TensorLike: The VJP of the quantum script
        """
        if wire_map is not None:
            [circuit], _ = qml.map_wires(circuit, wire_map)
        state.reset_state()
        final_state = state.get_final_state(circuit)
        # pylint: disable=not-callable
        return self.LightningAdjointJacobian(final_state, batch_obs=batch_obs).calculate_vjp(
            circuit, cotangents
        )

    def simulate_and_vjp(  # pylint: disable=too-many-arguments
        self,
        circuit: QuantumTape,
        cotangents: Tuple[Number],
        state,
        batch_obs: bool = False,
        wire_map: dict = None,
    ) -> Tuple:
        """Simulate a single quantum script and compute its Vector-Jacobian Product (VJP).
        Args:
            circuit (QuantumTape): The single circuit to simulate
            cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must
                have shape matching the output shape of the corresponding circuit. If
                the circuit has a single output, ``cotangents`` may be a single number,
                not an iterable of numbers.
            state (Lightning [Device] StateVector): handle to the Lightning state vector
            batch_obs (bool): Determine whether we process observables in parallel when
                computing the jacobian.
            wire_map (Optional[dict]): a map from wire labels to simulation indices

        Returns:
            Tuple[TensorLike]: The results of the simulation and the calculated VJP
        Note that this function can return measurements for non-commuting observables simultaneously.
        """
        if wire_map is not None:
            [circuit], _ = qml.map_wires(circuit, wire_map)
        res = self.simulate(circuit, state)
        # pylint: disable=not-callable
        _vjp = self.LightningAdjointJacobian(state, batch_obs=batch_obs).calculate_vjp(
            circuit, cotangents
        )
        return res, _vjp

    def compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """Calculate the jacobian of either a single or a batch of circuits on the device.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuits to calculate derivatives for
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            Tuple: The jacobian for each trainable parameter
        """
        batch_obs = execution_config.device_options.get("batch_obs", self._batch_obs)

        return tuple(
            self.jacobian(circuit, self._statevector, batch_obs=batch_obs, wire_map=self._wire_map)
            for circuit in circuits
        )

    def execute_and_compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Tuple:
        """Compute the results and jacobians of circuits at the same time.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuits or batch of circuits
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            Tuple: A numeric result of the computation and the gradient.
        """
        batch_obs = execution_config.device_options.get("batch_obs", self._batch_obs)
        results = tuple(
            self.simulate_and_jacobian(
                c, self._statevector, batch_obs=batch_obs, wire_map=self._wire_map
            )
            for c in circuits
        )
        return tuple(zip(*results))

    def supports_vjp(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """Whether or not this device defines a custom vector jacobian product.
        ``Lightning[Device]`` will check if supports adjoint differentiation with analytic results.
        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation
            circuit (QuantumTape): An optional circuit to check derivatives support for.
        Returns:
            Bool: Whether or not a derivative can be calculated provided the given information
        """
        return self.supports_derivatives(execution_config, circuit)

    def compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Tuple:
        r"""The vector jacobian product used in reverse-mode differentiation. ``Lightning[Device]`` uses the
        adjoint differentiation method to compute the VJP.
        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuit or batch of circuits
            cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must have shape matching the output shape of the
                corresponding circuit. If the circuit has a single output, ``cotangents`` may be a single number, not an iterable
                of numbers.
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution
        Returns:
            tensor-like: A numeric result of computing the vector jacobian product
        **Definition of vjp:**
        If we have a function with jacobian:
        .. math::
            \vec{y} = f(\vec{x}) \qquad J_{i,j} = \frac{\partial y_i}{\partial x_j}
        The vector jacobian product is the inner product of the derivatives of the output ``y`` with the
        Jacobian matrix. The derivatives of the output vector are sometimes called the **cotangents**.
        .. math::
            \text{d}x_i = \Sigma_{i} \text{d}y_i J_{i,j}
        **Shape of cotangents:**
        The value provided to ``cotangents`` should match the output of :meth:`~.execute`. For computing the full Jacobian,
        the cotangents can be batched to vectorize the computation. In this case, the cotangents can have the following
        shapes. ``batch_size`` below refers to the number of entries in the Jacobian:
        * For a state measurement, the cotangents must have shape ``(batch_size, 2 ** n_wires)``
        * For ``n`` expectation values, the cotangents must have shape ``(n, batch_size)``. If ``n = 1``,
          then the shape must be ``(batch_size,)``.
        """
        batch_obs = execution_config.device_options.get("batch_obs", self._batch_obs)
        return tuple(
            self.vjp(circuit, cots, self._statevector, batch_obs=batch_obs, wire_map=self._wire_map)
            for circuit, cots in zip(circuits, cotangents)
        )

    def execute_and_compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Tuple:
        """Calculate both the results and the vector jacobian product used in reverse-mode differentiation.
        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuit or batch of circuits to be executed
            cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must have shape matching the output shape of the
                corresponding circuit. If the circuit has a single output, ``cotangents`` may be a single number, not an iterable
                of numbers.
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution
        Returns:
            Tuple, Tuple: the result of executing the scripts and the numeric result of computing the vector jacobian product
        """
        batch_obs = execution_config.device_options.get("batch_obs", self._batch_obs)
        results = tuple(
            self.simulate_and_vjp(
                circuit, cots, self._statevector, batch_obs=batch_obs, wire_map=self._wire_map
            )
            for circuit, cots in zip(circuits, cotangents)
        )
        return tuple(zip(*results))
