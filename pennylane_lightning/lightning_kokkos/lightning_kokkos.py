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
import os
import sys
from dataclasses import replace
from functools import reduce
from pathlib import Path
from typing import List, Optional, Union
from warnings import warn

import numpy as np
import pennylane as qml
from pennylane.devices import DefaultExecutionConfig, ExecutionConfig
from pennylane.devices.capabilities import OperatorProperties
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
from pennylane.operation import DecompositionUndefinedError, Operator
from pennylane.ops import Conditional, PauliRot, Prod, SProd, Sum
from pennylane.tape import QuantumScript
from pennylane.transforms.core import TransformProgram
from pennylane.typing import Result

from pennylane_lightning.core.lightning_newAPI_base import (
    LightningBase,
    QuantumTape_or_Batch,
    Result_or_ResultBatch,
)

try:
    from pennylane_lightning.lightning_kokkos_ops import backend_info, print_configuration

    LK_CPP_BINARY_AVAILABLE = True
except ImportError as ex:
    warn(str(ex), UserWarning)
    LK_CPP_BINARY_AVAILABLE = False
    backend_info = None

from ._adjoint_jacobian import LightningKokkosAdjointJacobian
from ._measurements import LightningKokkosMeasurements
from ._state_vector import LightningKokkosStateVector

_to_matrix_ops = {
    "BlockEncode": OperatorProperties(),
    "DiagonalQubitUnitary": OperatorProperties(),
    "ECR": OperatorProperties(),
    "ISWAP": OperatorProperties(),
    "OrbitalRotation": OperatorProperties(),
    "PSWAP": OperatorProperties(),
    "QubitCarry": OperatorProperties(),
    "QubitSum": OperatorProperties(),
    "SISWAP": OperatorProperties(),
    "SQISW": OperatorProperties(),
}


def stopping_condition(op: Operator) -> bool:
    """A function that determines whether or not an operation is supported by ``lightning.kokkos``."""
    if isinstance(op, qml.PauliRot):
        word = op._hyperparameters["pauli_word"]  # pylint: disable=protected-access
        # decomposes to IsingXX, etc. for n <= 2
        return reduce(lambda x, y: x + (y != "I"), word, 0) > 2
    return _supports_operation(op.name)


def stopping_condition_shots(op: Operator) -> bool:
    """A function that determines whether or not an operation is supported by ``lightning.kokkos``
    with finite shots."""
    return stopping_condition(op) or isinstance(op, (MidMeasureMP, qml.ops.op_math.Conditional))


def accepted_observables(obs: Operator) -> bool:
    """A function that determines whether or not an observable is supported by ``lightning.kokkos``."""
    return _supports_observable(obs.name)


def adjoint_observables(obs: Operator) -> bool:
    """A function that determines whether or not an observable is supported by ``lightning.kokkos``
    when using the adjoint differentiation method."""
    if isinstance(obs, qml.Projector):
        return False

    if isinstance(obs, SProd):
        return adjoint_observables(obs.base)

    if isinstance(obs, (Sum, Prod)):
        return all(adjoint_observables(o) for o in obs)

    return _supports_observable(obs.name)


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

    return not isinstance(op, (Conditional, MidMeasureMP, PauliRot)) and (
        not qml.operation.is_trainable(op) or (op.num_params == 1 and op.has_generator)
    )


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


# Kokkos specific methods
def _kokkos_configuration():
    return print_configuration()


@simulator_tracking
@single_tape_support
class LightningKokkos(LightningBase):
    """PennyLane Lightning Kokkos device.

    A device that interfaces with C++ to perform fast linear algebra calculations on CPUs or GPUs using `Kokkos`.

    Use of this device requires pre-built binaries or compilation from source. Check out the
    :doc:`/lightning_kokkos/installation` guide for more details.

    Args:
        wires (int): the number of wires to initialize the device with
        c_dtype: Datatypes for statevector representation. Must be one of
            ``np.complex64`` or ``np.complex128``.
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically.
        sync (bool): immediately sync with host-sv after applying operations
        kokkos_args (InitializationSettings): binding for Kokkos::InitializationSettings
            (threading parameters).
    """

    # General device options
    _device_options = ("c_dtype", "batch_obs")

    # Device specific options
    _CPP_BINARY_AVAILABLE = LK_CPP_BINARY_AVAILABLE
    _backend_info = backend_info if LK_CPP_BINARY_AVAILABLE else None
    kokkos_config = {}

    # The configuration file declares the capabilities of the device
    config_filepath = Path(__file__).parent / "lightning_kokkos.toml"

    # TODO: This is to communicate to Catalyst in qjit-compiled workflows that these operations
    #       should be converted to QubitUnitary instead of their original decompositions. Remove
    #       this when customizable multiple decomposition pathways are implemented
    _to_matrix_ops = _to_matrix_ops

    def __init__(  # pylint: disable=too-many-arguments
        self,
        wires: Union[int, List] = None,
        *,
        c_dtype: Union[np.complex128, np.complex64] = np.complex128,
        shots: Union[int, List] = None,
        batch_obs: bool = False,
        # Kokkos arguments
        kokkos_args=None,
    ):
        if not self._CPP_BINARY_AVAILABLE:
            raise ImportError(
                "Pre-compiled binaries for lightning.kokkos are not available. "
                "To manually compile from source, follow the instructions at "
                "https://docs.pennylane.ai/projects/lightning/en/stable/dev/installation.html."
            )

        super().__init__(
            wires=wires,
            c_dtype=c_dtype,
            shots=shots,
            batch_obs=batch_obs,
        )

        # Set the attributes to call the Lightning classes
        self._set_lightning_classes()

        # Kokkos specific options
        self._kokkos_args = kokkos_args

        # Creating the state vector
        self._statevector = (
            self.LightningStateVector(
                num_wires=len(self.wires), dtype=c_dtype, kokkos_args=kokkos_args
            )
            if wires is not None
            else None
        )

        if not LightningKokkos.kokkos_config:
            LightningKokkos.kokkos_config = _kokkos_configuration()

    @property
    def name(self):
        """The name of the device."""
        return "lightning.kokkos"

    def _set_lightning_classes(self):
        """Load the LightningStateVector, LightningMeasurements, LightningAdjointJacobian as class attribute"""
        self.LightningStateVector = LightningKokkosStateVector
        self.LightningMeasurements = LightningKokkosMeasurements
        self.LightningAdjointJacobian = LightningKokkosAdjointJacobian

    def _setup_execution_config(self, config):
        """
        Update the execution config with choices for how the device should be used and the device options.
        """
        updated_values = {}
        if config.gradient_method == "best":
            updated_values["gradient_method"] = "adjoint"
        if config.use_device_gradient is None:
            updated_values["use_device_gradient"] = config.gradient_method in ("best", "adjoint")
        if (
            config.use_device_gradient
            or updated_values.get("use_device_gradient", False)
            and config.grad_on_execution is None
        ):
            updated_values["grad_on_execution"] = True

        new_device_options = dict(config.device_options)
        for option in self._device_options:
            if option not in new_device_options:
                new_device_options[option] = getattr(self, f"_{option}", None)

        # It is necessary to set the mcmc default configuration to complete the requirements of ExecuteConfig
        mcmc_default = {"mcmc": False, "kernel_name": None, "num_burnin": 0, "rng": None}
        new_device_options.update(mcmc_default)

        return replace(config, **updated_values, device_options=new_device_options)

    def dynamic_wire_alloc(self, circuit):
        """(DUMMY IMPLEMENTATION) Allocate a new statevector with number of wires for a given circuit.

        Args:
            circuit (QuantumTape): The circuit to execute.

        Returns:
            QuantumTape: The updated circuit with the wires mapped to the standard wire order.
        """

        return circuit

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
            if self.wires is None:  # Dynamic wires allocation
                circuit = self.update_dynamic_wires(circuit)

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

    def simulate(
        self,
        circuit: QuantumScript,
        state: LightningKokkosStateVector,
        postselect_mode: str = None,
    ) -> Result:
        """Simulate a single quantum script.

        Args:
            circuit (QuantumTape): The single circuit to simulate
            state (LightningKokkosStateVector): handle to Lightning state vector
            postselect_mode (str): Configuration for handling shots with mid-circuit measurement
                postselection. Use ``"hw-like"`` to discard invalid shots and ``"fill-shots"`` to
                keep the same number of shots. Default is ``None``.

        Returns:
            Tuple[TensorLike]: The results of the simulation

        Note that this function can return measurements for non-commuting observables simultaneously.
        """
        if circuit.shots and (any(isinstance(op, MidMeasureMP) for op in circuit.operations)):
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
                    self.LightningMeasurements(final_state).measure_final_state(
                        aux_circ, mid_measurements=mid_measurements
                    )
                )
            return tuple(results)

        state.reset_state()
        final_state = state.get_final_state(circuit)
        return self.LightningMeasurements(final_state).measure_final_state(circuit)

    @staticmethod
    def get_c_interface():
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
                f"'LightningKokkosSimulator' shared library not available for '{OS}' platform"
            )

        lib_name = "liblightning_kokkos_catalyst" + file_extension
        package_root = Path(__file__).parent

        # The absolute path of the plugin shared object varies according to the installation mode.

        # Wheel mode:
        # Fixed location at the root of the project
        wheel_mode_location = package_root.parent / lib_name
        if wheel_mode_location.is_file():
            return "LightningKokkosSimulator", wheel_mode_location.as_posix()

        # Editable mode:
        # The build directory contains a folder which varies according to the platform:
        #   lib.<system>-<architecture>-<python-id>"
        # To avoid mismatching the folder name, we search for the shared object instead.
        # TODO: locate where the naming convention of the folder is decided and replicate it here.
        editable_mode_path = package_root.parent.parent / "build_lightning_kokkos"
        for path, _, files in os.walk(editable_mode_path):
            if lib_name in files:
                lib_location = (Path(path) / lib_name).as_posix()
                return "LightningKokkosSimulator", lib_location

        raise RuntimeError("'LightningKokkosSimulator' shared library not found")


_supports_operation = LightningKokkos.capabilities.supports_operation
_supports_observable = LightningKokkos.capabilities.supports_observable
