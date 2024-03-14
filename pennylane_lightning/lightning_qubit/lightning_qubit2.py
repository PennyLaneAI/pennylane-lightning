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
This module contains the LightningQubit2 class that inherits from the new device interface.
"""
from typing import Optional, Union, Sequence, Callable
from dataclasses import replace
import numpy as np

import pennylane as qml
from pennylane.devices import Device, ExecutionConfig, DefaultExecutionConfig
from pennylane.devices.default_qubit import adjoint_ops
from pennylane.devices.modifiers import single_tape_support, simulator_tracking
from pennylane.devices.preprocess import (
    decompose,
    validate_adjoint_trainable_params,
    validate_device_wires,
    validate_measurements,
    validate_observables,
    no_sampling,
)
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms.core import TransformProgram
from pennylane.typing import Result, ResultBatch

from ._adjoint_jacobian import LightningAdjointJacobian
from ._state_vector import LightningStateVector
from ._measurements import LightningMeasurements

try:
    # pylint: disable=import-error, unused-import
    import pennylane_lightning.lightning_qubit_ops

    LQ_CPP_BINARY_AVAILABLE = True
except ImportError:
    LQ_CPP_BINARY_AVAILABLE = False

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]


def simulate(circuit: QuantumScript, state: LightningStateVector) -> Result:
    """Simulate a single quantum script.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        state (LightningStateVector): handle to Lightning state vector

    Returns:
        Tuple[TensorLike]: The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.
    """
    circuit = circuit.map_to_standard_wires()
    state.reset_state()
    final_state = state.get_final_state(circuit)
    return LightningMeasurements(final_state).measure_final_state(circuit)


def jacobian(circuit: QuantumTape, state: LightningStateVector, batch_obs=False):
    """Compute the Jacobian for a single quantum script.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        state (LightningStateVector): handle to Lightning state vector
        batch_obs (bool): Determine whether we process observables in parallel when
            computing the jacobian. This value is only relevant when the lightning
            qubit is built with OpenMP.

    Returns:
        TensorLike: The Jacobian of the quantum script
    """
    circuit = circuit.map_to_standard_wires()
    state.reset_state()
    final_state = state.get_final_state(circuit)
    return LightningAdjointJacobian(final_state, batch_obs=batch_obs).calculate_jacobian(circuit)


def simulate_and_jacobian(circuit: QuantumTape, state: LightningStateVector, batch_obs=False):
    """Simulate a single quantum script and compute its Jacobian.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        state (LightningStateVector): handle to Lightning state vector
        batch_obs (bool): Determine whether we process observables in parallel when
            computing the jacobian. This value is only relevant when the lightning
            qubit is built with OpenMP.

    Returns:
        Tuple[TensorLike]: The results of the simulation and the calculated Jacobian

    Note that this function can return measurements for non-commuting observables simultaneously.
    """
    circuit = circuit.map_to_standard_wires()
    res = simulate(circuit, state)
    jac = LightningAdjointJacobian(state, batch_obs=batch_obs).calculate_jacobian(circuit)
    return res, jac


_operations = frozenset(
    {
        "Identity",
        "BasisState",
        "QubitStateVector",
        "StatePrep",
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
"""The set of supported operations."""


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
        "Sum",
        "SProd",
        "Prod",
        "Exp",
    }
)
"""The set of supported observables."""


def stopping_condition(op: qml.operation.Operator) -> bool:
    """A function that determines whether or not an operation is supported by ``lightning.qubit``."""
    return op.name in _operations


def accepted_observables(obs: qml.operation.Operator) -> bool:
    """A function that determines whether or not an observable is supported by ``lightning.qubit``."""
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
    except (qml.operation.DecompositionUndefinedError, qml.DeviceError):
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
    program.add_transform(decompose, stopping_condition=adjoint_ops, name=name)
    program.add_transform(validate_observables, accepted_observables, name=name)
    program.add_transform(
        validate_measurements, analytic_measurements=adjoint_measurements, name=name
    )
    program.add_transform(qml.transforms.broadcast_expand)
    program.add_transform(validate_adjoint_trainable_params)


@simulator_tracking
@single_tape_support
class LightningQubit2(Device):
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
        kernel_name (str): name of transition kernel. The current version supports
            two kernels: ``"Local"`` and ``"NonZeroRandom"``.
            The local kernel conducts a bit-flip local transition between states.
            The local kernel generates a random qubit site and then generates a random
            number to determine the new bit at that qubit site. The ``"NonZeroRandom"`` kernel
            randomly transits between states that have nonzero probability.
        num_burnin (int): number of steps that will be dropped. Increasing this value will
            result in a closer approximation but increased runtime.
        batch_obs (bool): Determine whether we process observables in parallel when
            computing the jacobian. This value is only relevant when the lightning
            qubit is built with OpenMP.
    """

    _device_options = ("rng", "c_dtype", "batch_obs", "mcmc", "kernel_name", "num_burnin")
    _CPP_BINARY_AVAILABLE = LQ_CPP_BINARY_AVAILABLE
    _new_API = True

    operations = _operations
    """The names of the supported operations."""

    observables = _observables
    """The names of the supported observables."""

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
        if not LQ_CPP_BINARY_AVAILABLE:
            raise ImportError(
                "Pre-compiled binaries for lightning.qubit are not available. "
                "To manually compile from source, follow the instructions at "
                "https://pennylane-lightning.readthedocs.io/en/latest/installation.html."
            )

        super().__init__(wires=wires, shots=shots)

        self._statevector = LightningStateVector(num_wires=len(self.wires), dtype=c_dtype)
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
            if num_burnin >= shots:
                raise ValueError("Shots should be greater than num_burnin.")
            self._kernel_name = kernel_name
            self._num_burnin = num_burnin
        else:
            self._kernel_name = None
            self._num_burnin = None

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
        config = self._setup_execution_config(execution_config)
        program = TransformProgram()

        program.add_transform(validate_measurements, name=self.name)
        # TODO: Remove no_sampling from preprocess after shots support is added
        program.add_transform(no_sampling)
        program.add_transform(validate_observables, accepted_observables, name=self.name)
        program.add_transform(validate_device_wires, self.wires, name=self.name)
        program.add_transform(qml.defer_measurements, device=self)
        program.add_transform(decompose, stopping_condition=stopping_condition, name=self.name)
        program.add_transform(qml.transforms.broadcast_expand)

        if config.gradient_method == "adjoint":
            _add_adjoint_transforms(program)
        return program, config

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
            circuit = circuit.map_to_standard_wires()
            results.append(simulate(circuit, self._statevector))

        return tuple(results)

    def supports_derivatives(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[qml.tape.QuantumTape] = None,
    ) -> bool:
        """Check whether or not derivatives are available for a given configuration and circuit.

        ``LightningQubit2`` supports adjoint differentiation with analytic results.

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
            jacobian(circuit, self._statevector, batch_obs=batch_obs) for circuit in circuits
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
            simulate_and_jacobian(c, self._statevector, batch_obs=batch_obs) for c in circuits
        )
        return tuple(zip(*results))
