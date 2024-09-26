# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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
This module contains the :class:`~.LightningGPU` class, a PennyLane simulator device that
interfaces with the NVIDIA cuQuantum cuStateVec simulator library for GPU-enabled calculations.
"""

from ctypes.util import find_library
from dataclasses import replace
from importlib import util as imp_util
from numbers import Number
from pathlib import Path
from typing import List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pennylane as qml
from pennylane.devices import DefaultExecutionConfig, ExecutionConfig
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
from pennylane.typing import Result

from pennylane_lightning.core.lightning_newAPI_base import (
    LightningBase,
    QuantumTape_or_Batch,
    Result_or_ResultBatch,
)

try:
    from pennylane_lightning.lightning_gpu_ops import (
        DevPool,
        backend_info,
        get_gpu_arch,
        is_gpu_supported,
    )

    LGPU_CPP_BINARY_AVAILABLE = True

except (ImportError, ValueError) as ex:
    warn(str(ex), UserWarning)
    LGPU_CPP_BINARY_AVAILABLE = False
    backend_info = None

from ._adjoint_jacobian import LightningGPUAdjointJacobian
from ._measurements import LightningGPUMeasurements
from ._mpi_handler import MPIHandler
from ._state_vector import LightningGPUStateVector

# The set of supported operations.
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
# End the set of supported operations.

# The set of supported observables.
_observables = frozenset(
    {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "SparseHamiltonian",
        "Hamiltonian",
        "LinearCombination",
        "Hermitian",
        "Identity",
        "Projector",
        "Sum",
        "Prod",
        "SProd",
    }
)


def stopping_condition(op: Operator) -> bool:
    """A function that determines whether or not an operation is supported by ``lightning.gpu``."""
    # To avoid building matrices beyond the given thresholds.
    # This should reduce runtime overheads for larger systems.
    if isinstance(op, qml.QFT):
        return len(op.wires) < 10
    if isinstance(op, qml.GroverOperator):
        return len(op.wires) < 13
    if isinstance(op, qml.PauliRot):
        return False

    return op.name in _operations


def stopping_condition_shots(op: Operator) -> bool:
    """A function that determines whether or not an operation is supported by ``lightning.gpu``
    with finite shots."""
    if isinstance(op, (MidMeasureMP, qml.ops.op_math.Conditional)):
        # LightningGPU does not support Mid-circuit measurements.
        return False
    return stopping_condition(op)


def accepted_observables(obs: Operator) -> bool:
    """A function that determines whether or not an observable is supported by ``lightning.gpu``."""
    return obs.name in _observables


def adjoint_observables(obs: Operator) -> bool:
    """A function that determines whether or not an observable is supported by ``lightning.gpu``
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


def _adjoint_ops(op: qml.operation.Operator) -> bool:
    """Specify whether or not an Operator is supported by adjoint differentiation."""
    return not isinstance(op, qml.PauliRot) and adjoint_ops(op)


def _add_adjoint_transforms(program: TransformProgram) -> None:
    """Private helper function for ``preprocess`` that adds the transforms specific
    for adjoint differentiation.

    Args:
        program (TransformProgram): where we will add the adjoint differentiation transforms

    Side Effects:
        Adds transforms to the input program.

    """

    name = "adjoint + lightning.gpu"
    program.add_transform(no_sampling, name=name)
    program.add_transform(
        decompose,
        stopping_condition=_adjoint_ops,
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


# LightningGPU specific methods
def check_gpu_resources() -> None:
    """Check the available resources of each Nvidia GPU"""
    if find_library("custatevec") is None and not imp_util.find_spec("cuquantum"):

        raise ImportError(
            "cuStateVec libraries not found. Please pip install the appropriate cuStateVec library in a virtual environment."
        )

    if not DevPool.getTotalDevices():
        raise ValueError("No supported CUDA-capable device found")

    if not is_gpu_supported():
        raise ValueError(f"CUDA device is an unsupported version: {get_gpu_arch()}")


@simulator_tracking
@single_tape_support
class LightningGPU(LightningBase):
    """PennyLane Lightning GPU device.

    A device that interfaces with C++ to perform fast linear algebra calculations.

    Use of this device requires pre-built binaries or compilation from source. Check out the
    :doc:`/lightning_gpu/installation` guide for more details.

    Args:
        wires (int): the number of wires to initialize the device with
        c_dtype: Datatypes for statevector representation. Must be one of
            ``np.complex64`` or ``np.complex128``.
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically.
        batch_obs (bool): Determine whether we process observables in parallel when
            computing the jacobian. This value is only relevant when the lightning.gpu
            is built with MPI. Default is False.
        mpi (bool): declare if the device will use the MPI support.
        mpi_buf_size (int): size of GPU memory (in MiB) set for MPI operation and its default value is 64 MiB.
        sync (bool): immediately sync with host-sv after applying operation.
    """

    # General device options
    _device_options = ("c_dtype", "batch_obs")

    # Device specific options
    _CPP_BINARY_AVAILABLE = LGPU_CPP_BINARY_AVAILABLE
    _backend_info = backend_info if LGPU_CPP_BINARY_AVAILABLE else None

    # This `config` is used in Catalyst-Frontend
    config = Path(__file__).parent / "lightning_gpu.toml"

    # TODO: Move supported ops/obs to TOML file
    operations = _operations
    # The names of the supported operations.

    observables = _observables
    # The names of the supported observables.

    def __init__(  # pylint: disable=too-many-arguments
        self,
        wires: Union[int, List],
        *,
        c_dtype: Union[np.complex128, np.complex64] = np.complex128,
        shots: Union[int, List] = None,
        batch_obs: bool = False,
        # GPU and MPI arguments
        mpi: bool = False,
        mpi_buf_size: int = 0,
        sync: bool = False,
    ):
        if not self._CPP_BINARY_AVAILABLE:
            raise ImportError(
                "Pre-compiled binaries for lightning.gpu are not available. "
                "To manually compile from source, follow the instructions at "
                "https://docs.pennylane.ai/projects/lightning/en/stable/dev/installation.html."
            )

        check_gpu_resources()

        super().__init__(
            wires=wires,
            c_dtype=c_dtype,
            shots=shots,
            batch_obs=batch_obs,
        )

        # Set the attributes to call the LightningGPU classes
        self._set_lightning_classes()

        # GPU specific options
        self._dp = DevPool()
        self._sync = sync

        # Creating the state vector
        self._mpi_handler = MPIHandler(mpi, mpi_buf_size, len(self.wires), c_dtype)

        self._statevector = self.LightningStateVector(
            num_wires=len(self.wires), dtype=c_dtype, mpi_handler=self._mpi_handler, sync=self._sync
        )

    @property
    def name(self):
        """The name of the device."""
        return "lightning.gpu"

    def _set_lightning_classes(self):
        """Load the LightningStateVector, LightningMeasurements, LightningAdjointJacobian as class attribute"""
        self.LightningStateVector = LightningGPUStateVector
        self.LightningMeasurements = LightningGPUMeasurements
        self.LightningAdjointJacobian = LightningGPUAdjointJacobian

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
        mcmc_default = {"mcmc": False, "kernel_name": None, "num_burnin": 0, "rng": None}
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
                self.simulate(
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

        ``LightningGPU`` supports adjoint differentiation with analytic results.

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

    def simulate(
        self,
        circuit: QuantumScript,
        state: LightningGPUStateVector,
        postselect_mode: str = None,
    ) -> Result:
        """Simulate a single quantum script.

        Args:
            circuit (QuantumTape): The single circuit to simulate
            state (LightningGPUStateVector): handle to Lightning state vector
            postselect_mode (str): Configuration for handling shots with mid-circuit measurement
                postselection. Use ``"hw-like"`` to discard invalid shots and ``"fill-shots"`` to
                keep the same number of shots. Default is ``None``.

        Returns:
            Tuple[TensorLike]: The results of the simulation

        Note that this function can return measurements for non-commuting observables simultaneously.
        """
        if circuit.shots and (any(isinstance(op, MidMeasureMP) for op in circuit.operations)):
            raise qml.DeviceError("LightningGPU does not support Mid-circuit measurements.")

        state.reset_state(sync=False)
        final_state = state.get_final_state(circuit)
        return LightningGPUMeasurements(
            final_state, self._mpi_handler.use_mpi, self._mpi_handler
        ).measure_final_state(circuit)

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
        state.reset_state(self._sync)
        final_state = state.get_final_state(circuit)
        return self.LightningAdjointJacobian(
            final_state, batch_obs, self._mpi_handler.use_mpi, self._mpi_handler
        ).calculate_jacobian(circuit)

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
        jac = self.LightningAdjointJacobian(
            state, batch_obs, self._mpi_handler.use_mpi, self._mpi_handler
        ).calculate_jacobian(circuit)
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
        state.reset_state(self._sync)
        final_state = state.get_final_state(circuit)
        return self.LightningAdjointJacobian(
            final_state, batch_obs, self._mpi_handler.use_mpi, self._mpi_handler
        ).calculate_vjp(circuit, cotangents)

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
        _vjp = self.LightningAdjointJacobian(
            state, batch_obs, self._mpi_handler.use_mpi, self._mpi_handler
        ).calculate_vjp(circuit, cotangents)
        return res, _vjp
