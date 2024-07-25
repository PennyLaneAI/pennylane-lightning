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
from numbers import Number
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pennylane as qml
from pennylane.devices import DefaultExecutionConfig, Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.measurements import MidMeasureMP
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.typing import Result, ResultBatch

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
    state.reset_state()
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

    final_state = state.get_final_state(circuit)
    return LightningKokkosMeasurements(final_state).measure_final_state(circuit)


def jacobian(  # pylint: disable=unused-argument
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
    return 0


def simulate_and_jacobian(  # pylint: disable=unused-argument
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
    return 0


def vjp(  # pylint: disable=unused-argument
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
    return 0


def simulate_and_vjp(  # pylint: disable=unused-argument
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
    return 0


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
        kokkos_args (InitializationSettings): binding for Kokkos::InitializationSettings
            (threading parameters).
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically.
    """

    # pylint: disable=too-many-instance-attributes

    _device_options = ("rng", "c_dtype", "batch_obs", "kernel_name")
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

        self._statevector = LightningKokkosStateVector(num_wires=len(self.wires), dtype=c_dtype)

        self._c_dtype = c_dtype
        self._batch_obs = batch_obs

        # Kokkos specific options
        self._kokkos_args = kokkos_args
        self._sync = sync
        if not LightningKokkos.kokkos_config:
            LightningKokkos.kokkos_config = _kokkos_configuration()

    @property
    def name(self):
        """The name of the device."""
        return "lightning.kokkos"

    @property
    def c_dtype(self):
        """State vector complex data type."""
        return self._c_dtype

    dtype = c_dtype

    def _setup_execution_config(self, config):
        """
        Update the execution config with choices for how the device should be used and the device options.
        """
        return 0

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
        return 0

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
        return 0

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
        return 0

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

        return 0

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
        return 0

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
        return 0

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
        return 0

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
        return 0
