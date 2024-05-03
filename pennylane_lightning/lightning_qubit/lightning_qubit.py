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
This module contains the LightningQubit class that inherits from the new device interface.
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

from ._adjoint_jacobian import LightningAdjointJacobian
from ._measurements import LightningMeasurements
from ._state_vector import LightningStateVector

try:
    # pylint: disable=import-error, unused-import
    from pennylane_lightning.lightning_qubit_ops import backend_info

    LQ_CPP_BINARY_AVAILABLE = True
except ImportError:
    LQ_CPP_BINARY_AVAILABLE = False

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]


def simulate(circuit: QuantumScript, state: LightningStateVector, mcmc: dict = None) -> Result:
    """Simulate a single quantum script.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        state (LightningStateVector): handle to Lightning state vector
        mcmc (dict): Dictionary containing the Markov Chain Monte Carlo
            parameters: mcmc, kernel_name, num_burnin. Descriptions of
            these fields are found in :class:`~.LightningQubit`.

    Returns:
        Tuple[TensorLike]: The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.
    """
    if mcmc is None:
        mcmc = {}
    state.reset_state()
    has_mcm = any(isinstance(op, MidMeasureMP) for op in circuit.operations)
    if circuit.shots and has_mcm:
        mid_measurements = {}
        final_state = state.get_final_state(circuit, mid_measurements=mid_measurements)
        return LightningMeasurements(final_state, **mcmc).measure_final_state(
            circuit, mid_measurements=mid_measurements
        )
    final_state = state.get_final_state(circuit)
    return LightningMeasurements(final_state, **mcmc).measure_final_state(circuit)


def jacobian(circuit: QuantumTape, state: LightningStateVector, batch_obs=False, wire_map=None):
    """Compute the Jacobian for a single quantum script.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        state (LightningStateVector): handle to Lightning state vector
        batch_obs (bool): Determine whether we process observables in parallel when
            computing the jacobian. This value is only relevant when the lightning
            qubit is built with OpenMP. Default is False.
        wire_map (Optional[dict]): a map from wire labels to simulation indices

    Returns:
        TensorLike: The Jacobian of the quantum script
    """
    if wire_map is not None:
        [circuit], _ = qml.map_wires(circuit, wire_map)
    state.reset_state()
    final_state = state.get_final_state(circuit)
    return LightningAdjointJacobian(final_state, batch_obs=batch_obs).calculate_jacobian(circuit)


def simulate_and_jacobian(
    circuit: QuantumTape, state: LightningStateVector, batch_obs=False, wire_map=None
):
    """Simulate a single quantum script and compute its Jacobian.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        state (LightningStateVector): handle to Lightning state vector
        batch_obs (bool): Determine whether we process observables in parallel when
            computing the jacobian. This value is only relevant when the lightning
            qubit is built with OpenMP. Default is False.
        wire_map (Optional[dict]): a map from wire labels to simulation indices

    Returns:
        Tuple[TensorLike]: The results of the simulation and the calculated Jacobian

    Note that this function can return measurements for non-commuting observables simultaneously.
    """
    if wire_map is not None:
        [circuit], _ = qml.map_wires(circuit, wire_map)
    res = simulate(circuit, state)
    jac = LightningAdjointJacobian(state, batch_obs=batch_obs).calculate_jacobian(circuit)
    return res, jac


def vjp(
    circuit: QuantumTape,
    cotangents: Tuple[Number],
    state: LightningStateVector,
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
        state (LightningStateVector): handle to Lightning state vector
        batch_obs (bool): Determine whether we process observables in parallel when
            computing the VJP. This value is only relevant when the lightning
            qubit is built with OpenMP.
        wire_map (Optional[dict]): a map from wire labels to simulation indices

    Returns:
        TensorLike: The VJP of the quantum script
    """
    if wire_map is not None:
        [circuit], _ = qml.map_wires(circuit, wire_map)
    state.reset_state()
    final_state = state.get_final_state(circuit)
    return LightningAdjointJacobian(final_state, batch_obs=batch_obs).calculate_vjp(
        circuit, cotangents
    )


def simulate_and_vjp(
    circuit: QuantumTape,
    cotangents: Tuple[Number],
    state: LightningStateVector,
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
        state (LightningStateVector): handle to Lightning state vector
        batch_obs (bool): Determine whether we process observables in parallel when
            computing the jacobian. This value is only relevant when the lightning
            qubit is built with OpenMP.
        wire_map (Optional[dict]): a map from wire labels to simulation indices

    Returns:
        Tuple[TensorLike]: The results of the simulation and the calculated VJP
    Note that this function can return measurements for non-commuting observables simultaneously.
    """
    if wire_map is not None:
        [circuit], _ = qml.map_wires(circuit, wire_map)
    res = simulate(circuit, state)
    _vjp = LightningAdjointJacobian(state, batch_obs=batch_obs).calculate_vjp(circuit, cotangents)
    return res, _vjp


_operations = frozenset(
    {
        "Identity",
        "QubitUnitary",
        "ControlledQubitUnitary",
        "MultiControlledX",
        "DiagonalQubitUnitary",
        "PauliX",
        "PauliY",
        "PauliZ",
        "MultiRZ",
        "GlobalPhase",
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
        "CPhase",
        "RX",
        "RY",
        "RZ",
        "Rot",
        "CRX",
        "CRY",
        "CRZ",
        "C(PauliX)",
        "C(PauliY)",
        "C(PauliZ)",
        "C(Hadamard)",
        "C(S)",
        "C(T)",
        "C(PhaseShift)",
        "C(RX)",
        "C(RY)",
        "C(RZ)",
        "C(Rot)",
        "C(SWAP)",
        "C(IsingXX)",
        "C(IsingXY)",
        "C(IsingYY)",
        "C(IsingZZ)",
        "C(SingleExcitation)",
        "C(SingleExcitationMinus)",
        "C(SingleExcitationPlus)",
        "C(DoubleExcitation)",
        "C(DoubleExcitationMinus)",
        "C(DoubleExcitationPlus)",
        "C(MultiRZ)",
        "C(GlobalPhase)",
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
    """A function that determines whether or not an operation is supported by ``lightning.qubit``."""
    # These thresholds are adapted from `lightning_base.py`
    # To avoid building matrices beyond the given thresholds.
    # This should reduce runtime overheads for larger systems.
    if isinstance(op, qml.QFT):
        return len(op.wires) < 10
    if isinstance(op, qml.GroverOperator):
        return len(op.wires) < 13
    return op.name in _operations


def stopping_condition_shots(op: Operator) -> bool:
    """A function that determines whether or not an operation is supported by ``lightning.qubit``
    with finite shots."""
    return stopping_condition(op) or isinstance(op, (MidMeasureMP, qml.ops.op_math.Conditional))


def accepted_observables(obs: Operator) -> bool:
    """A function that determines whether or not an observable is supported by ``lightning.qubit``."""
    return obs.name in _observables


def adjoint_observables(obs: Operator) -> bool:
    """A function that determines whether or not an observable is supported by ``lightning.qubit``
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

    name = "adjoint + lightning.qubit"
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


@simulator_tracking
@single_tape_support
class LightningQubit(Device):
    """PennyLane Lightning Qubit device.

    A device that interfaces with C++ to perform fast linear algebra calculations.

    Use of this device requires pre-built binaries or compilation from source. Check out the
    :doc:`/lightning_qubit/installation` guide for more details.

    Args:
        wires (int): the number of wires to initialize the device with
        c_dtype: Datatypes for statevector representation. Must be one of
            ``np.complex64`` or ``np.complex128``.
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically.
        seed (Union[str, None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``, or
            a request to seed from numpy's global random number generator.
            The default, ``seed="global"`` pulls a seed from NumPy's global generator. ``seed=None``
            will pull a seed from the OS entropy.
        mcmc (bool): Determine whether to use the approximate Markov Chain Monte Carlo
            sampling method when generating samples.
        kernel_name (str): name of transition MCMC kernel. The current version supports
            two kernels: ``"Local"`` and ``"NonZeroRandom"``.
            The local kernel conducts a bit-flip local transition between states.
            The local kernel generates a random qubit site and then generates a random
            number to determine the new bit at that qubit site. The ``"NonZeroRandom"`` kernel
            randomly transits between states that have nonzero probability.
        num_burnin (int): number of MCMC steps that will be dropped. Increasing this value will
            result in a closer approximation but increased runtime.
        batch_obs (bool): Determine whether we process observables in parallel when
            computing the jacobian. This value is only relevant when the lightning
            qubit is built with OpenMP.
    """

    # pylint: disable=too-many-instance-attributes

    _device_options = ("rng", "c_dtype", "batch_obs", "mcmc", "kernel_name", "num_burnin")
    _CPP_BINARY_AVAILABLE = LQ_CPP_BINARY_AVAILABLE
    _new_API = True
    _backend_info = backend_info if LQ_CPP_BINARY_AVAILABLE else None

    # This `config` is used in Catalyst-Frontend
    config = Path(__file__).parent / "lightning_qubit.toml"

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
        seed="global",
        mcmc=False,
        kernel_name="Local",
        num_burnin=100,
        batch_obs=False,
    ):
        if not self._CPP_BINARY_AVAILABLE:
            raise ImportError(
                "Pre-compiled binaries for lightning.qubit are not available. "
                "To manually compile from source, follow the instructions at "
                "https://pennylane-lightning.readthedocs.io/en/latest/installation.html."
            )

        super().__init__(wires=wires, shots=shots)

        if isinstance(wires, int):
            self._wire_map = None  # should just use wires as is
        else:
            self._wire_map = {w: i for i, w in enumerate(self.wires)}

        self._statevector = LightningStateVector(num_wires=len(self.wires), dtype=c_dtype)

        # TODO: Investigate usefulness of creating numpy random generator
        seed = np.random.randint(0, high=10000000) if seed == "global" else seed
        self._rng = np.random.default_rng(seed)

        self._c_dtype = c_dtype
        self._batch_obs = batch_obs
        self._mcmc = mcmc
        if self._mcmc:
            if kernel_name not in [
                "Local",
                "NonZeroRandom",
            ]:
                raise NotImplementedError(
                    f"The {kernel_name} is not supported and currently "
                    "only 'Local' and 'NonZeroRandom' kernels are supported."
                )
            shots = shots if isinstance(shots, Sequence) else [shots]
            if any(num_burnin >= s for s in shots):
                raise ValueError("Shots should be greater than num_burnin.")
            self._kernel_name = kernel_name
            self._num_burnin = num_burnin
        else:
            self._kernel_name = None
            self._num_burnin = 0

    @property
    def name(self):
        """The name of the device."""
        return "lightning.qubit"

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
        program.add_transform(mid_circuit_measurements, device=self)
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
        mcmc = {
            "mcmc": self._mcmc,
            "kernel_name": self._kernel_name,
            "num_burnin": self._num_burnin,
        }
        results = []
        for circuit in circuits:
            if self._wire_map is not None:
                [circuit], _ = qml.map_wires(circuit, self._wire_map)
            results.append(simulate(circuit, self._statevector, mcmc=mcmc))

        return tuple(results)

    def supports_derivatives(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[qml.tape.QuantumTape] = None,
    ) -> bool:
        """Check whether or not derivatives are available for a given configuration and circuit.

        ``LightningQubit`` supports adjoint differentiation with analytic results.

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
        ``LightningQubit`` supports adjoint differentiation with analytic results.
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
        r"""The vector jacobian product used in reverse-mode differentiation. ``LightningQubit`` uses the
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
