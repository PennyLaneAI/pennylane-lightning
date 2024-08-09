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
This module contains the :class:`~.LightningKokkos` class, a PennyLane simulator device that
interfaces with C++ for fast linear algebra calculations.
"""
from dataclasses import replace
from numbers import Number
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pennylane as qml
from pennylane.devices import DefaultExecutionConfig, Device, ExecutionConfig
from pennylane.devices.default_qubit import adjoint_ops
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import (
    decompose,
    mid_circuit_measurements,
    no_sampling,
    validate_adjoint_trainable_params,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from pennylane.measurements import MidMeasureMP
from pennylane.operation import DecompositionUndefinedError, Operator, Tensor
from pennylane.ops import Prod, SProd, Sum
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms.core import TransformProgram
from pennylane.typing import Result, ResultBatch

from ._adjoint_jacobian import LightningKokkosAdjointJacobian
from ._measurements import LightningKokkosMeasurements
from ._state_vector import LightningKokkosStateVector

try:
    # pylint: disable=import-error, no-name-in-module
    from pennylane_lightning.lightning_kokkos_ops import backend_info, print_configuration

    LK_CPP_BINARY_AVAILABLE = True
except ImportError:
    LK_CPP_BINARY_AVAILABLE = False
    backend_info = None

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]


def simulate(
    circuit: QuantumScript,
    state: LightningKokkosStateVector,
    postselect_mode: str = None,
) -> Result:
    """Simulate a single quantum script.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        state (LightningStateVector): handle to Lightning state vector
        postselect_mode (str): Configuration for handling shots with mid-circuit measurement
            postselection. Use ``"hw-like"`` to discard invalid shots and ``"fill-shots"`` to
            keep the same number of shots. Default is ``None``.

    Returns:
        Tuple[TensorLike]: The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.
    """
    has_mcm = any(isinstance(op, MidMeasureMP) for op in circuit.operations)
    if circuit.shots and has_mcm:
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
                LightningKokkosMeasurements(final_state).measure_final_state(
                    aux_circ, mid_measurements=mid_measurements
                )
            )
        return tuple(results)
    state.reset_state()
    final_state = state.get_final_state(circuit)
    return LightningKokkosMeasurements(final_state).measure_final_state(circuit)


def jacobian(
    circuit: QuantumTape, state: LightningKokkosStateVector, batch_obs=False, wire_map=None
):
    """Compute the Jacobian for a single quantum script.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        state (LightningKokkosStateVector): handle to the Lightning state vector
        batch_obs (bool): Determine whether we process observables in parallel when
            computing the jacobian. This value is only relevant when the lightning
            kokkos is built with OpenMP. Default is False.
        wire_map (Optional[dict]): a map from wire labels to simulation indices

    Returns:
        TensorLike: The Jacobian of the quantum script
    """
    if wire_map is not None:
        [circuit], _ = qml.map_wires(circuit, wire_map)
    state.reset_state()
    final_state = state.get_final_state(circuit)
    return LightningKokkosAdjointJacobian(final_state, batch_obs=batch_obs).calculate_jacobian(
        circuit
    )


def simulate_and_jacobian(
    circuit: QuantumTape, state: LightningKokkosStateVector, batch_obs=False, wire_map=None
):
    """Simulate a single quantum script and compute its Jacobian.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        state (LightningKokkosStateVector): handle to the Lightning state vector
        batch_obs (bool): Determine whether we process observables in parallel when
            computing the jacobian. This value is only relevant when the lightning
            kokkos is built with OpenMP. Default is False.
        wire_map (Optional[dict]): a map from wire labels to simulation indices

    Returns:
        Tuple[TensorLike]: The results of the simulation and the calculated Jacobian

    Note that this function can return measurements for non-commuting observables simultaneously.
    """
    if wire_map is not None:
        [circuit], _ = qml.map_wires(circuit, wire_map)
    res = simulate(circuit, state)
    jac = LightningKokkosAdjointJacobian(state, batch_obs=batch_obs).calculate_jacobian(circuit)
    return res, jac


def vjp(
    circuit: QuantumTape,
    cotangents: Tuple[Number],
    state: LightningKokkosStateVector,
    batch_obs=False,
    wire_map=None,
):
    """Compute the Vector-Jacobian Product (VJP) for a single quantum script.
    Args:
        circuit (QuantumTape): The single circuit to simulate
        cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must
            have shape matching the output shape of the corresponding circuit. If
            the circuit has a single output, ``cotangents`` may be a single number,
            not an iterable of numbers.
        state (LightningKokkosStateVector): handle to the Lightning state vector
        batch_obs (bool): Determine whether we process observables in parallel when
            computing the VJP. This value is only relevant when the lightning
            kokkos is built with OpenMP.
        wire_map (Optional[dict]): a map from wire labels to simulation indices

    Returns:
        TensorLike: The VJP of the quantum script
    """
    if wire_map is not None:
        [circuit], _ = qml.map_wires(circuit, wire_map)
    state.reset_state()
    final_state = state.get_final_state(circuit)
    return LightningKokkosAdjointJacobian(final_state, batch_obs=batch_obs).calculate_vjp(
        circuit, cotangents
    )


def simulate_and_vjp(
    circuit: QuantumTape,
    cotangents: Tuple[Number],
    state: LightningKokkosStateVector,
    batch_obs=False,
    wire_map=None,
):
    """Simulate a single quantum script and compute its Vector-Jacobian Product (VJP).
    Args:
        circuit (QuantumTape): The single circuit to simulate
        cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must
            have shape matching the output shape of the corresponding circuit. If
            the circuit has a single output, ``cotangents`` may be a single number,
            not an iterable of numbers.
        state (LightningKokkosStateVector): handle to the Lightning state vector
        batch_obs (bool): Determine whether we process observables in parallel when
            computing the jacobian. This value is only relevant when the lightning
            kokkos is built with OpenMP.
        wire_map (Optional[dict]): a map from wire labels to simulation indices

    Returns:
        Tuple[TensorLike]: The results of the simulation and the calculated VJP
    Note that this function can return measurements for non-commuting observables simultaneously.
    """
    if wire_map is not None:
        [circuit], _ = qml.map_wires(circuit, wire_map)
    res = simulate(circuit, state)
    _vjp = LightningKokkosAdjointJacobian(state, batch_obs=batch_obs).calculate_vjp(
        circuit, cotangents
    )
    return res, _vjp


_operations = frozenset(
    {
        "Identity",
        "QubitStateVector",
        "QubitUnitary",
        "ControlledQubitUnitary",
        "MultiControlledX",
        "DiagonalQubitUnitary",
        "PauliX",
        "PauliY",
        "PauliZ",
        "MultiRZ",
        "GlobalPhase",
        "C(GlobalPhase)",
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
        "PSWAP",
        "Adjoint(ISWAP)",
        "SISWAP",
        "Adjoint(SISWAP)",
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
        "DoubleExcitationPlus",
        "DoubleExcitationMinus",
        "QubitCarry",
        "QubitSum",
        "OrbitalRotation",
        "QFT",
        "ECR",
        "BlockEncode",
        "C(BlockEncode)",
    }
)
# The set of supported operations.

_observables = frozenset(
    {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "Hermitian",
        "Identity",
        "Projector",
        "SparseHamiltonian",
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
    """A function that determines whether or not an operation is supported by ``lightning.kokkos``."""
    # These thresholds are adapted from `lightning_base.py`
    # To avoid building matrices beyond the given thresholds.
    # This should reduce runtime overheads for larger systems.
    if isinstance(op, qml.QFT):
        return len(op.wires) < 10
    if isinstance(op, qml.GroverOperator):
        return len(op.wires) < 13

    # As ControlledQubitUnitary == C(QubitUnitrary),
    # it can be removed from `_operations` to keep
    # consistency with `lightning_qubit.toml`
    if isinstance(op, qml.ControlledQubitUnitary):
        return True

    return op.name in _operations


def stopping_condition_shots(op: Operator) -> bool:
    """A function that determines whether or not an operation is supported by ``lightning.kokkos``
    with finite shots."""
    return stopping_condition(op) or isinstance(op, (MidMeasureMP, qml.ops.op_math.Conditional))


def accepted_observables(obs: Operator) -> bool:
    """A function that determines whether or not an observable is supported by ``lightning.kokkos``."""
    return obs.name in _observables


def adjoint_observables(obs: Operator) -> bool:
    """A function that determines whether or not an observable is supported by ``lightning.kokkos``
    when using the adjoint differentiation method."""
    if isinstance(obs, qml.Projector):
        return False

    if isinstance(obs, Tensor):
        if any(isinstance(o, qml.Projector) for o in obs.non_identity_obs):
            return False
        return True

    if isinstance(obs, SProd):
        return adjoint_observables(obs.base)

    if isinstance(obs, (Sum, Prod)):
        return all(adjoint_observables(o) for o in obs)

    return obs.name in _observables


def adjoint_measurements(mp: qml.measurements.MeasurementProcess) -> bool:
    """Specifies whether or not an observable is compatible with adjoint differentiation on DefaultQubit."""
    return isinstance(mp, qml.measurements.ExpectationMP)


def _supports_adjoint(circuit):
    if circuit is None:
        return True

    prog = TransformProgram()
    _add_adjoint_transforms(prog)

    try:
        prog((circuit,))
    except (DecompositionUndefinedError, qml.DeviceError, AttributeError):
        return False
    return True


def _add_adjoint_transforms(program: TransformProgram) -> None:
    """Private helper function for ``preprocess`` that adds the transforms specific
    for adjoint differentiation.

    Args:
        program (TransformProgram): where we will add the adjoint differentiation transforms

    Side Effects:
        Adds transforms to the input program.

    """

    name = "adjoint + lightning.kokkos"
    program.add_transform(no_sampling, name=name)
    program.add_transform(
        decompose,
        stopping_condition=adjoint_ops,
        stopping_condition_shots=stopping_condition_shots,
        name=name,
        skip_initial_state_prep=False,
    )
    program.add_transform(validate_observables, accepted_observables, name=name)
    program.add_transform(
        validate_measurements, analytic_measurements=adjoint_measurements, name=name
    )
    program.add_transform(qml.transforms.broadcast_expand)
    program.add_transform(validate_adjoint_trainable_params)


def _kokkos_configuration():
    return print_configuration()


@simulator_tracking
@single_tape_support
class LightningKokkos(Device):
    """PennyLane Lightning Kokkos device.

    A device that interfaces with C++ to perform fast linear algebra calculations.

    Use of this device requires pre-built binaries or compilation from source. Check out the
    :doc:`/lightning_kokkos/installation` guide for more details.

    Args:
        wires (int): the number of wires to initialize the device with
        sync (bool): immediately sync with host-sv after applying operations
        c_dtype: Datatypes for statevector representation. Must be one of
            ``np.complex64`` or ``np.complex128``.
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically.
        kokkos_args (InitializationSettings): binding for Kokkos::InitializationSettings
            (threading parameters).
    """

    # pylint: disable=too-many-instance-attributes

    # General device options
    _device_options = ("c_dtype", "batch_obs")
    _new_API = True

    # Device specific options
    _CPP_BINARY_AVAILABLE = LK_CPP_BINARY_AVAILABLE
    _backend_info = backend_info if LK_CPP_BINARY_AVAILABLE else None
    kokkos_config = {}

    # This `config` is used in Catalyst-Frontend
    config = Path(__file__).parent / "lightning_kokkos.toml"

    # TODO: Move supported ops/obs to TOML file
    operations = _operations
    # The names of the supported operations.

    observables = _observables
    # The names of the supported observables.

    def __init__(  # pylint: disable=too-many-arguments
        self,
        wires,
        *,
        c_dtype=np.complex128,
        shots=None,
        batch_obs=False,
        # Kokkos arguments
        sync=True,
        kokkos_args=None,
    ):
        if not self._CPP_BINARY_AVAILABLE:
            raise ImportError(
                "Pre-compiled binaries for lightning.kokkos are not available. "
                "To manually compile from source, follow the instructions at "
                "https://pennylane-lightning.readthedocs.io/en/latest/installation.html."
            )

        super().__init__(wires=wires, shots=shots)

        if isinstance(wires, int):
            self._wire_map = None  # should just use wires as is
        else:
            self._wire_map = {w: i for i, w in enumerate(self.wires)}

        self._c_dtype = c_dtype
        self._batch_obs = batch_obs

        # Kokkos specific options
        self._kokkos_args = kokkos_args
        self._sync = sync
        if not LightningKokkos.kokkos_config:
            LightningKokkos.kokkos_config = _kokkos_configuration()

        self._statevector = LightningKokkosStateVector(
            num_wires=len(self.wires), dtype=c_dtype, kokkos_args=kokkos_args, sync=sync
        )

    @property
    def name(self):
        """The name of the device."""
        return "lightning.kokkos"

    @property
    def c_dtype(self):
        """State vector complex data type."""
        return self._c_dtype

    @property
    def dtype(self):
        """State vector complex data type."""
        return self._c_dtype

    def _setup_execution_config(self, config):
        """
        Update the execution config with choices for how the device should be used and the device options.
        """
        updated_values = {}
        if config.gradient_method == "best":
            updated_values["gradient_method"] = "adjoint"
        if config.use_device_gradient is None:
            updated_values["use_device_gradient"] = config.gradient_method in ("best", "adjoint")
        if config.grad_on_execution is None:
            updated_values["grad_on_execution"] = True

        new_device_options = dict(config.device_options)
        for option in self._device_options:
            if option not in new_device_options:
                new_device_options[option] = getattr(self, f"_{option}", None)

        # It is necessary to set the mcmc default configuration to complete the requirements of ExecuteConfig
        mcmc_default = {"mcmc": False, "kernel_name": None, "num_burnin": 0}
        new_device_options.update(mcmc_default)

        return replace(config, **updated_values, device_options=new_device_options)

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
        exec_config = self._setup_execution_config(execution_config)
        program = TransformProgram()

        program.add_transform(validate_measurements, name=self.name)
        program.add_transform(validate_observables, accepted_observables, name=self.name)
        program.add_transform(validate_device_wires, self.wires, name=self.name)
        program.add_transform(
            mid_circuit_measurements, device=self, mcm_config=exec_config.mcm_config
        )
        program.add_transform(
            decompose,
            stopping_condition=stopping_condition,
            stopping_condition_shots=stopping_condition_shots,
            skip_initial_state_prep=True,
            name=self.name,
        )
        program.add_transform(qml.transforms.broadcast_expand)

        if exec_config.gradient_method == "adjoint":
            _add_adjoint_transforms(program)
        return program, exec_config

    # pylint: disable=unused-argument
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
        results = []
        for circuit in circuits:
            if self._wire_map is not None:
                [circuit], _ = qml.map_wires(circuit, self._wire_map)
            results.append(
                simulate(
                    circuit,
                    self._statevector,
                    postselect_mode=execution_config.mcm_config.postselect_mode,
                )
            )

        return tuple(results)

    def supports_derivatives(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[qml.tape.QuantumTape] = None,
    ) -> bool:
        """Check whether or not derivatives are available for a given configuration and circuit.

        ``LightningKokkos`` supports adjoint differentiation with analytic results.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            Bool: Whether or not a derivative can be calculated provided the given information

        """
        if execution_config is None and circuit is None:
            return True
        if execution_config.gradient_method not in {"adjoint", "best"}:
            return False
        if circuit is None:
            return True
        return _supports_adjoint(circuit=circuit)

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
            jacobian(circuit, self._statevector, batch_obs=batch_obs, wire_map=self._wire_map)
            for circuit in circuits
        )

    def execute_and_compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """Compute the results and jacobians of circuits at the same time.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuits or batch of circuits
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            tuple: A numeric result of the computation and the gradient.
        """
        batch_obs = execution_config.device_options.get("batch_obs", self._batch_obs)
        results = tuple(
            simulate_and_jacobian(
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
        ``LightningKokkos`` supports adjoint differentiation with analytic results.
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
    ):
        r"""The vector jacobian product used in reverse-mode differentiation. ``LightningKokkos`` uses the
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
            vjp(circuit, cots, self._statevector, batch_obs=batch_obs, wire_map=self._wire_map)
            for circuit, cots in zip(circuits, cotangents)
        )

    def execute_and_compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
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
            simulate_and_vjp(
                circuit, cots, self._statevector, batch_obs=batch_obs, wire_map=self._wire_map
            )
            for circuit, cots in zip(circuits, cotangents)
        )
        return tuple(zip(*results))
