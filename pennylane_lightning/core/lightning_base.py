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
from functools import partial
from numbers import Number
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pennylane as qml
from pennylane.devices import DefaultExecutionConfig, Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.typing import Result, ResultBatch, TensorLike

from ._measurements_base import LightningBaseMeasurements
from ._version import __version__

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]


@simulator_tracking
@single_tape_support
class LightningBase(Device):
    """PennyLane Lightning Base device.

    A class that serves as a base class for Lightning simulators.

    Args:
        wires (Optional[int, list]): number or list of wires to initialize the device with. Defaults to ``None`` if not specified, and the device will allocate the number of wires depending on the circuit to execute.
        sync (bool): immediately sync with host after applying operations on the device
        c_dtype: Datatypes for statevector representation. Must be one of
            ``np.complex64`` or ``np.complex128``.
        shots (int or list): How many times the circuit should be evaluated (or sampled) to estimate
            stochastic return values. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically.
        batch_obs (bool): Determine whether we process observables in parallel when
            computing the jacobian.
    """

    # pylint: disable=too-many-instance-attributes
    pennylane_requires = ">=0.41"
    version = __version__

    def __init__(  # pylint: disable=too-many-arguments
        self,
        wires: Union[int, List] = None,
        *,
        c_dtype: Union[np.complex64, np.complex128],
        shots: Union[int, List],
        batch_obs: bool,
    ):
        super().__init__(wires=wires, shots=shots)

        self._c_dtype = c_dtype
        self._batch_obs = batch_obs

        # State-vector is dynamically allocated just before execution
        self._statevector = None
        self._sv_init_kwargs = {}

        if isinstance(wires, int) or wires is None:
            self._wire_map = None  # should just use wires as is
        else:
            self._wire_map = {w: i for i, w in enumerate(self.wires)}

        # Dummy for LightningStateVector, LightningMeasurements, LightningAdjointJacobian
        self.LightningStateVector: Callable = None
        self.LightningMeasurements: type[LightningBaseMeasurements] = None
        self.LightningAdjointJacobian: Callable = None

    @property
    def c_dtype(self):
        """State vector complex data type."""
        return self._c_dtype

    @abstractmethod
    def _set_lightning_classes(self):
        """Load the LightningStateVector, LightningMeasurements, LightningAdjointJacobian as class attribute"""

    @abstractmethod
    def _setup_execution_config(self, config: ExecutionConfig):
        """
        Update the execution config with choices for how the device should be used and the device options.

        Args:
            config (ExecutionConfig): A data structure describing the parameters needed to fully describe the execution.

        Returns:
            ExecutionConfig: An updated execution config with device options set.

        """

    def dynamic_wires_from_circuit(self, circuit):
        """Allocate the underlying quantum state from the pre-defined wires or a given circuit if applicable. Circuit wires will be mapped to Pennylane ``default.qubit`` standard wire order.

        Args:
            circuit (QuantumTape): The circuit to execute.

        Returns:
            QuantumTape: The updated circuit with the wires mapped to the standard wire order.
        """

        if self.wires is None:
            num_wires = circuit.num_wires
            # Map to follow the standard wire order for dynamic wires
            circuit = circuit.map_to_standard_wires()
        else:
            num_wires = len(self.wires)

        if (self._statevector is None) or (self._statevector.num_wires != num_wires):
            self._statevector = self.LightningStateVector(
                num_wires=num_wires, dtype=self._c_dtype, **self._sv_init_kwargs
            )
        else:
            self._statevector.reset_state()

        return circuit

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
            execution_config (ExecutionConfig): a data structure with additional information required for execution

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
            state (Lightning [Device] StateVector): A handle to the underlying Lightning state
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

        Args:
            execution_config (ExecutionConfig): An optional configuration of the desired derivative calculation. Default is ``None``.
            circuit (QuantumTape): An optional circuit to check derivatives support for. Default is ``None``.

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
            state (Lightning [Device] StateVector): A handle to the underlying Lightning state
            batch_obs (bool): Determine whether we process observables in parallel when
                computing the jacobian. Default is ``False``.
            wire_map (Optional[dict]): an optional map from wire labels to simulation indices. Default is ``None``.

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
            state (Lightning [Device] StateVector): A handle to the underlying Lightning state
            batch_obs (bool): Determine whether we process observables in parallel when
                computing the jacobian. Default is ``False``.
            wire_map (Optional[dict]): an optional map from wire labels to simulation indices. Default is ``None``.

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

    def vjp(  # pylint: disable=too-many-arguments, too-many-positional-arguments
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
            state: A handle to the underlying Lightning state
            batch_obs (bool): Determine whether we process observables in parallel when
                computing the VJP. Default is ``False``.
            wire_map (Optional[dict]): an optional map from wire labels to simulation indices. Default is ``None``.

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

    def simulate_and_vjp(  # pylint: disable=too-many-arguments, too-many-positional-arguments
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
            state: A handle to the underlying Lightning state
            batch_obs (bool): Determine whether we process observables in parallel when
                computing the jacobian. Default is ``False``.
            wire_map (Optional[dict]): an optional map from wire labels to simulation indices. Default is ``None``.

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
    ) -> Tuple:
        """Calculate the jacobian of either a single or a batch of circuits on the device.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuits to calculate derivatives for
            execution_config (ExecutionConfig): a data structure with all additional information required for execution. Default is ``DefaultExecutionConfig``.

        Returns:
            Tuple: The jacobian for each trainable parameter
        """
        batch_obs = execution_config.device_options.get("batch_obs", self._batch_obs)

        return tuple(
            self.jacobian(
                self.dynamic_wires_from_circuit(circuit),
                self._statevector,
                batch_obs=batch_obs,
                wire_map=self._wire_map,
            )
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
            execution_config (ExecutionConfig): a data structure with all additional information required for execution. Default is ``DefaultExecutionConfig``.

        Returns:
            Tuple: A numeric result of the computation and the gradient.
        """
        batch_obs = execution_config.device_options.get("batch_obs", self._batch_obs)
        results = tuple(
            self.simulate_and_jacobian(
                self.dynamic_wires_from_circuit(circuit),
                self._statevector,
                batch_obs=batch_obs,
                wire_map=self._wire_map,
            )
            for circuit in circuits
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
            self.vjp(
                self.dynamic_wires_from_circuit(circuit),
                cots,
                self._statevector,
                batch_obs=batch_obs,
                wire_map=self._wire_map,
            )
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
                self.dynamic_wires_from_circuit(circuit),
                cots,
                self._statevector,
                batch_obs=batch_obs,
                wire_map=self._wire_map,
            )
            for circuit, cots in zip(circuits, cotangents)
        )
        return tuple(zip(*results))

    # pylint: disable=import-outside-toplevel, unused-argument
    def eval_jaxpr(
        self,
        jaxpr: "jax.core.Jaxpr",
        consts: list[TensorLike],
        *args: TensorLike,
        execution_config: Optional[ExecutionConfig] = None,
    ) -> list[TensorLike]:
        """Execute pennylane variant jaxpr using C++ simulation tools.

        Args:
            jaxpr (jax.core.Jaxpr): jaxpr containing quantum operations
            consts (list[TensorLike]): List of constants for the jaxpr closure variables
            *args (TensorLike): The arguments to the jaxpr.

        Keyword Args:
            execution_config (Optional[ExecutionConfig]): a datastructure with additional
                information required for execution

        Returns:
            list(TensorLike): the results of the execution

        .. code-block:: python

            import pennylane as qml
            import jax
            qml.capture.enable()

            def f(x):
                @qml.for_loop(3)
                def loop(i, y):
                    qml.RX(y, i)
                    return y + 0.5
                loop(x)
                return [qml.expval(qml.Z(i)) for i in range(3)]

            jaxpr = jax.make_jaxpr(f)(0.5)

            dev = qml.device('lightning.qubit', wires=3)
            dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.0)

        .. code-block::

            [1.0, 0.8775825618903728, 0.5403023058681395]

        """
        # jax is still an optional dependency for pennylane, but mandatory for program capture
        # jax imports cannot be placed in the standard import path
        import jax
        from pennylane.capture.primitives import AbstractMeasurement

        from .lightning_interpreter import LightningInterpreter

        # pylint: disable=no-member
        dtype_map = {
            float: (jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32),
            int: jax.numpy.int64 if jax.config.jax_enable_x64 else jax.numpy.int32,
            complex: (jax.numpy.complex128 if jax.config.jax_enable_x64 else jax.numpy.complex64),
        }

        if self.wires is None:
            raise NotImplementedError("Wires must be specified for integration with plxpr capture.")

        self._statevector = self.LightningStateVector(
            num_wires=len(self.wires), dtype=self._c_dtype
        )

        interpreter = LightningInterpreter(
            self._statevector, self.LightningMeasurements, shots=self.shots
        )
        evaluator = partial(interpreter.eval, jaxpr)

        def shape(var):
            if isinstance(var.aval, AbstractMeasurement):
                shots = self.shots.total_shots
                s, dtype = var.aval.abstract_eval(num_device_wires=len(self.wires), shots=shots)
                return jax.core.ShapedArray(s, dtype_map[dtype])
            return var.aval

        shapes = [shape(var) for var in jaxpr.outvars]
        return jax.pure_callback(evaluator, shapes, consts, *args, vectorized=False)

    def jaxpr_jvp(
        self,
        jaxpr: "jax.core.Jaxpr",
        args: Sequence[TensorLike],
        tangents: Sequence[TensorLike],
        execution_config: Optional[ExecutionConfig] = None,
    ) -> tuple[Sequence[TensorLike], Sequence[TensorLike]]:
        """
        An **experimental** method for computing the results and jvp for PLXPR with LightningBase devices.

        Args:
            jaxpr (jax.core.Jaxpr): Pennylane variant jaxpr containing quantum operations
                and measurements
            args (Sequence[TensorLike]): the arguments to the ``jaxpr``. Should contain ``consts`` followed
                by non-constant arguments
            tangents (Sequence[TensorLike]): the tangents corresponding to ``args``.
                May contain ``jax.interpreters.ad.Zero``.

        Keyword Args:
            execution_config (Optional[ExecutionConfig]): a data structure with additional information required for execution

        Returns:
            Sequence[TensorLike], Sequence[TensorLike]: the results and Jacobian vector products


        .. note::

            For LightningBase devices, the current implementation of this method is based on the conversion of the jaxpr to a PennyLane tape.
            This has strict limitations. The ``args`` should contain the concatenation of ``jaxpr.constvars`` and ``jaxpr.invars``,
            which are assumed to represent the trainable parameters of the circuit.
            The method will raise an error if ``args`` do not match exactly the parameters of the tape created from the jaxpr.
            Finally, only the adjoint method is supported for gradient calculation on LightningBase devices.

        """

        # pylint: disable=import-outside-toplevel
        import jax

        from pennylane_lightning.core.jaxpr_jvp import (
            convert_jaxpr_to_tape,
            get_output_shapes,
            validate_args_tangents,
        )

        if self.wires is None:
            raise NotImplementedError("Wires must be specified for integration with plxpr capture.")

        gradient_method = getattr(execution_config, "gradient_method", "adjoint")

        if gradient_method != "adjoint":
            raise NotImplementedError(
                f"LightningQubit does not support gradient_method={gradient_method} for jaxpr_jvp."
            )

        if self.shots.total_shots is not None:
            raise NotImplementedError(
                "LightningBase does not support finite shots for ``jaxpr_jvp``. Please use shots=None."
            )

        tangents = validate_args_tangents(args, tangents)

        self._statevector = self.LightningStateVector(
            num_wires=len(self.wires), dtype=self._c_dtype
        )

        def wrapper(*args):
            """
            Evaluate a jaxpr by converting it into a quantum tape, ensuring the provided
            parameters match the tape's parameters, and then simulating the tape to compute
            both the result and the Jacobian.

            The *args should contain the concatenation of jaxpr.constvars and jaxpr.invars,
            which are assumed to represent the trainable parameters.
            """
            tape = convert_jaxpr_to_tape(jaxpr, args)
            return self.simulate_and_jacobian(tape, state=self._statevector)

        shapes_res, shapes_jac = get_output_shapes(jaxpr, len(self.wires))
        results, jacobians = jax.pure_callback(wrapper, (shapes_res, shapes_jac), *args)

        if len(tangents) == 1 and not jax.numpy.isscalar(tangents[0]):
            tangents = tangents[0]

        jvps = (
            [qml.gradients.compute_jvp_single(tangents, jacobians)]
            if len(jaxpr.outvars) == 1
            else qml.gradients.compute_jvp_multi(tangents, jacobians)
        )

        return results, jvps
