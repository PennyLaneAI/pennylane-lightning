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

import os
import sys
from abc import abstractmethod
from dataclasses import replace
from functools import partial
from numbers import Number
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np
import pennylane as qml
from numpy.random import BitGenerator, Generator, SeedSequence
from numpy.typing import ArrayLike
from pennylane.devices import Device, ExecutionConfig, MCMConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.exceptions import DeviceError
from pennylane.measurements import MidMeasureMP, Shots, ShotsLike
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.typing import Result, ResultBatch, TensorLike

from pennylane_lightning.core import __version__
from pennylane_lightning.lightning_base._measurements import LightningBaseMeasurements
from pennylane_lightning.lightning_base._mid_circuit_measure_tree_traversal import (
    mcm_tree_traversal,
)

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
        seed (Union[str, None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``, or
            a request to seed from numpy's global random number generator.
            The default, ``seed="global"`` pulls a seed from NumPy's global generator. ``seed=None``
            will pull a seed from the OS entropy.
        batch_obs (bool): Determine whether we process observables in parallel when
            computing the jacobian.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(  # pylint: disable=too-many-arguments
        self,
        wires: Union[int, List] = None,
        *,
        c_dtype: Union[np.complex64, np.complex128],
        shots: Union[int, List],
        seed: Union[str, None, int, ArrayLike, SeedSequence, BitGenerator, Generator],
        batch_obs: bool,
    ):
        super().__init__(wires=wires, shots=shots)

        self._c_dtype = c_dtype
        self._batch_obs = batch_obs
        self._rng = np.random.default_rng(
            np.random.randint(2**31 - 1) if seed == "global" else seed
        )
        self._mpi = False
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

        # Lightning device name and library name for get_c_interface
        self._lightning_device_name: str = None
        self._lightning_lib_name: str = None

    @property
    def c_dtype(self):
        """State vector complex data type."""
        return self._c_dtype

    @abstractmethod
    def _set_lightning_classes(self):
        """Load the LightningStateVector, LightningMeasurements, LightningAdjointJacobian as class attribute"""

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
                num_wires=num_wires, dtype=self._c_dtype, rng=self._rng, **self._sv_init_kwargs
            )
        else:
            self._statevector.reset_state()

        return circuit

    @abstractmethod
    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig | None = None,
    ) -> Result_or_ResultBatch:
        """Execute a circuit or a batch of circuits and turn it into results.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the quantum circuits to be executed
            execution_config (ExecutionConfig): a data structure with additional information required for execution

        Returns:
            TensorLike, tuple[TensorLike], tuple[tuple[TensorLike]]: A numeric result of the computation.
        """

    def simulate(  # pylint: disable=too-many-arguments
        self,
        circuit: QuantumScript,
        state,  # Lightning [Device] StateVector
        *,
        postselect_mode: str = None,
        mcmc: dict = None,
        mcm_method: str = None,
    ) -> Result:
        """Simulate a single quantum script.

        Args:
            circuit (QuantumTape): The single circuit to simulate
            state (Lightning [Device] StateVector): A handle to the underlying Lightning state
            postselect_mode (str): Configuration for handling shots with mid-circuit measurement
                postselection. Use ``"hw-like"`` to discard invalid shots and ``"fill-shots"`` to
                keep the same number of shots. Default is ``None``.
            mcmc (dict): Dictionary containing the Markov Chain Monte Carlo
                parameters: mcmc, kernel_name, num_burnin. Currently only supported for
                ``lightning.qubit``, more detail can be found in :class:`~.LightningQubit`.
            mcm_method (str): The method to use for mid-circuit measurements. Default is ``"one-shot"`` if ``circuit.shots`` is set, otherwise it defaults to ``"deferred"``.

        Returns:
            Tuple[TensorLike]: The results of the simulation

        Note that this function can return measurements for non-commuting observables simultaneously.
        """
        if mcmc is None:
            mcmc = {}

        if any(isinstance(op, MidMeasureMP) for op in circuit.operations):
            if self._mpi:
                raise DeviceError(
                    "Lightning Device with MPI does not support Mid-circuit measurements."
                )

        # Simulate with Mid Circuit Measurements
        if any(isinstance(op, MidMeasureMP) for op in circuit.operations):
            # If mcm_method is not specified and the circuit does not have shots, default to "deferred".
            # It is not listed here because all mid-circuit measurements are replaced with additional wires.

            if mcm_method == "tree-traversal":
                # Using the tree traversal MCM method.
                return mcm_tree_traversal(
                    circuit, state, self.LightningMeasurements, postselect_mode
                )

            if mcm_method == "one-shot" or (mcm_method is None and circuit.shots):
                # Using the one-shot MCM method.
                results = []
                aux_circ = qml.tape.QuantumScript(
                    circuit.operations,
                    circuit.measurements,
                    shots=[1],
                    trainable_params=circuit.trainable_params,
                )
                for _ in range(circuit.shots.total_shots):
                    state.reset_state()
                    mid_measurements = {}
                    final_state = state.get_final_state(
                        aux_circ, mid_measurements=mid_measurements, postselect_mode=postselect_mode
                    )
                    results.append(
                        self.LightningMeasurements(final_state, **mcmc).measure_final_state(
                            aux_circ, mid_measurements=mid_measurements
                        )
                    )
                return tuple(results)

        final_state = state.get_final_state(circuit)
        return self.LightningMeasurements(final_state, **mcmc).measure_final_state(circuit)

    @abstractmethod
    def supports_derivatives(
        self,
        execution_config: ExecutionConfig | None = None,
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
        execution_config: ExecutionConfig | None = None,
    ) -> Tuple:
        """Calculate the jacobian of either a single or a batch of circuits on the device.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuits to calculate derivatives for
            execution_config (ExecutionConfig): a data structure with all additional information required for execution. Default is ``None``, which sets the execution config to the default setup.

        Returns:
            Tuple: The jacobian for each trainable parameter
        """
        if execution_config is None:
            execution_config = ExecutionConfig(gradient_method="adjoint")

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
        execution_config: ExecutionConfig | None = None,
    ) -> Tuple:
        """Compute the results and jacobians of circuits at the same time.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuits or batch of circuits
            execution_config (ExecutionConfig): a data structure with all additional information required for execution. Default is ``None``, which sets the execution config to the default setup.

        Returns:
            Tuple: A numeric result of the computation and the gradient.
        """
        if execution_config is None:
            execution_config = ExecutionConfig(gradient_method="adjoint")

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
        execution_config: ExecutionConfig | None = None,
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
        execution_config: ExecutionConfig | None = None,
    ) -> Tuple:
        r"""The vector jacobian product used in reverse-mode differentiation. ``Lightning[Device]`` uses the
        adjoint differentiation method to compute the VJP.
        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuit or batch of circuits
            cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must have shape matching the output shape of the
                corresponding circuit. If the circuit has a single output, ``cotangents`` may be a single number, not an iterable
                of numbers.
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution.
                Default is ``None``, which sets the execution config to the default setup.
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
        if execution_config is None:
            execution_config = ExecutionConfig(gradient_method="adjoint")

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
        execution_config: ExecutionConfig | None = None,
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
        if execution_config is None:
            execution_config = ExecutionConfig(gradient_method="adjoint")

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
        jaxpr: "jax.extend.core.Jaxpr",
        consts: list[TensorLike],
        *args: TensorLike,
        execution_config: ExecutionConfig | None = None,
        shots: ShotsLike = None,
    ) -> list[TensorLike]:
        """Execute pennylane variant jaxpr using C++ simulation tools.

        Args:
            jaxpr (jax.extend.core.Jaxpr): jaxpr containing quantum operations
            consts (list[TensorLike]): List of constants for the jaxpr closure variables
            *args (TensorLike): The arguments to the jaxpr.

        Keyword Args:
            execution_config (ExecutionConfig | None): a datastructure with additional
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
            num_wires=len(self.wires), dtype=self._c_dtype, rng=self._rng
        )

        shots = Shots(shots)
        interpreter = LightningInterpreter(
            self._statevector, self.LightningMeasurements, shots=shots
        )
        evaluator = partial(interpreter.eval, jaxpr)

        def shape(var, shots):
            if isinstance(var.aval, AbstractMeasurement):
                shots = shots.total_shots
                s, dtype = var.aval.abstract_eval(num_device_wires=len(self.wires), shots=shots)
                return jax.core.ShapedArray(s, dtype_map[dtype])
            return var.aval

        shapes = [shape(var, shots) for var in jaxpr.outvars]
        return jax.pure_callback(evaluator, shapes, consts, *args, vmap_method="sequential")

    def jaxpr_jvp(
        self,
        jaxpr: "jax.extend.core.Jaxpr",
        args: Sequence[TensorLike],
        tangents: Sequence[TensorLike],
        execution_config: ExecutionConfig | None = None,
    ) -> tuple[Sequence[TensorLike], Sequence[TensorLike]]:
        """
        An **experimental** method for computing the results and jvp for PLXPR with LightningBase devices.

        Args:
            jaxpr (jax.extend.core.Jaxpr): Pennylane variant jaxpr containing quantum operations
                and measurements
            args (Sequence[TensorLike]): the arguments to the ``jaxpr``. Should contain ``consts`` followed
                by non-constant arguments
            tangents (Sequence[TensorLike]): the tangents corresponding to ``args``.
                May contain ``jax.interpreters.ad.Zero``.

        Keyword Args:
            execution_config (ExecutionConfig | None): a data structure with additional information required for execution

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

        from pennylane_lightning.lightning_base.jaxpr_jvp import (
            convert_jaxpr_to_tape,
            get_output_shapes,
            validate_args_tangents,
        )

        if self.wires is None:
            raise NotImplementedError("Wires must be specified for integration with plxpr capture.")

        if execution_config is None:
            execution_config = ExecutionConfig(gradient_method="adjoint")
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
            num_wires=len(self.wires), dtype=self._c_dtype, rng=self._rng
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
        results, jacobians = jax.pure_callback(
            wrapper, (shapes_res, shapes_jac), *args, vmap_method="sequential"
        )

        if len(tangents) == 1 and not jax.numpy.isscalar(tangents[0]):
            tangents = tangents[0]

        jvps = (
            [qml.gradients.compute_jvp_single(tangents, jacobians)]
            if len(jaxpr.outvars) == 1
            else qml.gradients.compute_jvp_multi(tangents, jacobians)
        )

        return results, jvps

    @staticmethod
    def get_c_interface_impl(
        lightning_device_name: str, lightning_lib_name: str
    ) -> Tuple[str, str]:
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """

        # The shared object file extension varies depending on the underlying operating system
        file_extension = ""
        OS = sys.platform
        if OS == "linux":
            file_extension = ".so"
        elif OS == "darwin":
            file_extension = ".dylib"
        else:
            raise RuntimeError(
                f"'{lightning_device_name}' shared library not available for '{OS}' platform"
            )

        lib_name = "lib" + lightning_lib_name + "_catalyst" + file_extension
        package_root = Path(__file__).parent

        # The absolute path of the plugin shared object varies according to the installation mode.

        # Wheel mode:
        # Fixed location at the root of the project
        wheel_mode_location = package_root.parent / lib_name
        if wheel_mode_location.is_file():
            return lightning_device_name, wheel_mode_location.as_posix()

        # Editable mode:
        # The build directory contains a folder which varies according to the platform:
        #   lib.<system>-<architecture>-<python-id>"
        # To avoid mismatching the folder name, we search for the shared object instead.
        # TODO: locate where the naming convention of the folder is decided and replicate it here.
        build_lightning_dir = "build_" + lightning_lib_name
        editable_mode_path = package_root.parent.parent / build_lightning_dir
        for path, _, files in os.walk(editable_mode_path):
            if lib_name in files:
                lib_location = (Path(path) / lib_name).as_posix()
                return lightning_device_name, lib_location

        raise RuntimeError(f"'{lightning_device_name}' shared library not found")


def resolve_mcm_method(mcm_config: MCMConfig, tape: QuantumScript | None, device_name: str):
    """Resolve the mcm config for the Lightning device."""

    mcm_supported_methods = (
        ("device", "deferred", "tree-traversal", "one-shot", None)
        if not qml.capture.enabled()
        else ("deferred", "single-branch-statistics", None)
    )

    if (mcm_method := mcm_config.mcm_method) not in mcm_supported_methods:
        raise DeviceError(f"mcm_method='{mcm_method}' is not supported with {device_name}.")

    final_mcm_method = mcm_config.mcm_method
    if mcm_config.mcm_method is None:
        final_mcm_method = "one-shot" if getattr(tape, "shots", None) else "deferred"
    elif mcm_config.mcm_method == "device":
        final_mcm_method = "tree-traversal"

    # TODO: Update this condition when postselection is natively supported in Lightning [sc-82462]
    if mcm_config.postselect_mode == "fill-shots" and final_mcm_method != "deferred":
        raise DeviceError("Using postselect_mode='fill-shots' is not supported.")

    mcm_config = replace(mcm_config, mcm_method=final_mcm_method)

    if qml.capture.enabled():
        mcm_updated_values = {}

        if mcm_method == "single-branch-statistics" and mcm_config.postselect_mode is not None:
            warn(
                "Setting 'postselect_mode' is not supported with mcm_method='single-branch-"
                "statistics'. 'postselect_mode' will be ignored.",
                UserWarning,
            )
            mcm_updated_values["postselect_mode"] = None
        elif mcm_method is None:
            mcm_updated_values["mcm_method"] = "deferred"

        mcm_config = replace(mcm_config, **mcm_updated_values)

    return mcm_config
