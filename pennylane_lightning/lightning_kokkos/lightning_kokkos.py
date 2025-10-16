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
from functools import partial, reduce
from pathlib import Path
from typing import List, Optional, Union
from warnings import warn

import numpy as np
import pennylane as qml
from numpy.random import BitGenerator, Generator, SeedSequence
from numpy.typing import ArrayLike
from pennylane.devices import ExecutionConfig
from pennylane.devices.capabilities import OperatorProperties
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import (
    decompose,
    device_resolve_dynamic_wires,
    no_sampling,
    validate_adjoint_trainable_params,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from pennylane.exceptions import DecompositionUndefinedError, DeviceError
from pennylane.measurements import MidMeasureMP
from pennylane.operation import Operator
from pennylane.ops import Conditional, PauliRot, Prod, SProd, Sum
from pennylane.transforms import defer_measurements, dynamic_one_shot
from pennylane.transforms.core import TransformProgram

from pennylane_lightning.lightning_base.lightning_base import (
    LightningBase,
    QuantumTape_or_Batch,
    Result_or_ResultBatch,
    resolve_mcm_method,
)

try:
    from pennylane_lightning.lightning_kokkos_ops import backend_info

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
    "QubitCarry": OperatorProperties(),
    "QubitSum": OperatorProperties(),
    "SISWAP": OperatorProperties(),
    "SQISW": OperatorProperties(),
}


def stopping_condition(op: Operator, allow_mcms: bool = True) -> bool:
    """A function that determines whether or not an operation is supported by ``lightning.kokkos``."""
    if isinstance(op, qml.PauliRot):
        word = op._hyperparameters["pauli_word"]  # pylint: disable=protected-access
        # decomposes to IsingXX, etc. for n <= 2
        return reduce(lambda x, y: x + (y != "I"), word, 0) > 2

    if isinstance(op, MidMeasureMP):
        return allow_mcms

    return _supports_operation(op.name)


# need to create these once so we can compare in tests
allow_mcms_stopping_condition = partial(stopping_condition, allow_mcms=True)
no_mcms_stopping_condition = partial(stopping_condition, allow_mcms=False)


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


def _supports_adjoint(circuit, device_wires=None):
    if circuit is None:
        return True

    prog = TransformProgram()
    _add_adjoint_transforms(prog, device_wires=device_wires)

    try:
        prog((circuit,))
    except (DecompositionUndefinedError, DeviceError, AttributeError):
        return False
    return True


def _adjoint_ops(op: qml.operation.Operator) -> bool:
    """Specify whether or not an Operator is supported by adjoint differentiation."""

    return not isinstance(op, (Conditional, MidMeasureMP, PauliRot)) and (
        not any(qml.math.requires_grad(d) for d in op.data)
        or (op.num_params == 1 and op.has_generator)
    )


def _add_adjoint_transforms(program: TransformProgram, device_wires=None) -> None:
    """Private helper function for ``preprocess`` that adds the transforms specific
    for adjoint differentiation.

    Args:
        program (TransformProgram): where we will add the adjoint differentiation transforms

    Side Effects:
        Adds transforms to the input program.

    """

    name = "adjoint + lightning.kokkos"
    program.add_transform(no_sampling, name=name)
    program.add_transform(qml.transforms.broadcast_expand)
    program.add_transform(
        decompose,
        stopping_condition=_adjoint_ops,
        name=name,
        skip_initial_state_prep=False,
        device_wires=device_wires,
        target_gates=LightningKokkos.capabilities.operations.keys(),
    )
    program.add_transform(validate_observables, accepted_observables, name=name)
    program.add_transform(
        validate_measurements, analytic_measurements=adjoint_measurements, name=name
    )
    program.add_transform(validate_adjoint_trainable_params)


@simulator_tracking
@single_tape_support
class LightningKokkos(LightningBase):
    """PennyLane Lightning Kokkos device.

    A device that interfaces with C++ to perform fast linear algebra calculations on CPUs or GPUs using `Kokkos`.

    Use of this device requires pre-built binaries or compilation from source. Check out the
    :doc:`/lightning_kokkos/installation` guide for more details.

    Args:
        wires (Optional[int, list]): the number of wires to initialize the device with. Defaults to ``None`` if not specified, and the device will allocate the number of wires depending on the circuit to execute.
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
        mpi  (bool): Use MPI to distribute statevector across multiple processes.
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
        seed: Union[str, None, int, ArrayLike, SeedSequence, BitGenerator, Generator] = "global",
        # Kokkos arguments
        mpi: bool = False,
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
            seed=seed,
            batch_obs=batch_obs,
        )

        # Set the attributes to call the Lightning classes
        self._set_lightning_classes()

        self._mpi = mpi
        if mpi:
            if wires is None:
                raise DeviceError("Lightning-Kokkos-MPI does not support dynamic wires allocation.")
            self._statevector = self.LightningStateVector(
                num_wires=len(self.wires),
                dtype=self.c_dtype,
                kokkos_args=kokkos_args,
                mpi=True,
                rng=self._rng,
            )
        else:
            self._statevector = None
            self._sv_init_kwargs = {"kokkos_args": kokkos_args}

    @property
    def name(self):
        """The name of the device."""
        return "lightning.kokkos"

    def _set_lightning_classes(self):
        """Load the LightningStateVector, LightningMeasurements, LightningAdjointJacobian as class attribute"""
        self.LightningStateVector = LightningKokkosStateVector
        self.LightningMeasurements = LightningKokkosMeasurements
        self.LightningAdjointJacobian = LightningKokkosAdjointJacobian

    def setup_execution_config(
        self, config: ExecutionConfig | None = None, circuit: qml.tape.QuantumScript | None = None
    ) -> ExecutionConfig:
        """
        Update the execution config with choices for how the device should be used and the device options.
        """
        if config is None:
            config = ExecutionConfig()
        updated_values = {}

        # It is necessary to set the mcmc default configuration to complete the requirements of ExecuteConfig
        mcmc_default = {"mcmc": False, "kernel_name": None, "num_burnin": 0, "rng": None}

        for option, _ in config.device_options.items():
            if option not in self._device_options and option not in mcmc_default:
                raise DeviceError(f"device option {option} not present on {self}")

        if config.gradient_method == "best":
            updated_values["gradient_method"] = "adjoint"
        if config.use_device_jacobian_product is None:
            updated_values["use_device_jacobian_product"] = config.gradient_method in (
                "best",
                "adjoint",
            )
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

        new_device_options.update(mcmc_default)

        mcm_config = resolve_mcm_method(config.mcm_config, circuit, "lightning.kokkos")
        updated_values["mcm_config"] = mcm_config
        return replace(config, **updated_values, device_options=new_device_options)

    def preprocess_transforms(
        self, execution_config: ExecutionConfig | None = None
    ) -> TransformProgram:
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
        if execution_config is None:
            execution_config = ExecutionConfig()
        exec_config = execution_config
        program = TransformProgram()

        if qml.capture.enabled():
            if exec_config.mcm_config.mcm_method == "deferred":
                program.add_transform(qml.defer_measurements, num_wires=len(self.wires))
            program.add_transform(qml.transforms.decompose, gate_set=no_mcms_stopping_condition)
            return program

        program.add_transform(validate_measurements, name=self.name)
        program.add_transform(validate_observables, accepted_observables, name=self.name)
        if exec_config.mcm_config.mcm_method == "deferred":
            program.add_transform(defer_measurements, allow_postselect=False)
            _stopping_condition = no_mcms_stopping_condition
        else:
            _stopping_condition = allow_mcms_stopping_condition

        program.add_transform(
            decompose,
            stopping_condition=_stopping_condition,
            skip_initial_state_prep=True,
            name=self.name,
            device_wires=self.wires,
            target_gates=self.capabilities.operations.keys(),
        )

        _allow_resets = exec_config.mcm_config.mcm_method != "deferred"
        program.add_transform(
            device_resolve_dynamic_wires, wires=self.wires, allow_resets=_allow_resets
        )
        program.add_transform(validate_device_wires, self.wires, name=self.name)
        if exec_config.mcm_config.mcm_method == "one-shot":
            program.add_transform(
                dynamic_one_shot, postselect_mode=exec_config.mcm_config.postselect_mode
            )

        program.add_transform(qml.transforms.broadcast_expand)

        if exec_config.gradient_method == "adjoint":
            _add_adjoint_transforms(program, device_wires=self.wires)
        return program

    # pylint: disable=unused-argument
    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig | None = None,
    ) -> Result_or_ResultBatch:
        """Execute a circuit or a batch of circuits and turn it into results.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the quantum circuits to be executed
            execution_config (ExecutionConfig): a datastructure with additional information required for execution

        Returns:
            TensorLike, tuple[TensorLike], tuple[tuple[TensorLike]]: A numeric result of the computation.
        """
        if execution_config is None:
            execution_config = ExecutionConfig()
        results = []
        for circuit in circuits:
            if self._wire_map is not None:
                [circuit], _ = qml.map_wires(circuit, self._wire_map)
            results.append(
                self.simulate(
                    self.dynamic_wires_from_circuit(circuit),
                    self._statevector,
                    postselect_mode=execution_config.mcm_config.postselect_mode,
                    mcm_method=execution_config.mcm_config.mcm_method,
                )
            )

        return tuple(results)

    def supports_derivatives(
        self,
        execution_config: ExecutionConfig | None = None,
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

        if execution_config and execution_config.gradient_method in {"adjoint", "best"}:
            if circuit is None:
                return True
            return _supports_adjoint(circuit=circuit, device_wires=self.wires)

        return False

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """

        return LightningBase.get_c_interface_impl("LightningKokkosSimulator", "lightning_kokkos")


_supports_operation = LightningKokkos.capabilities.supports_operation
_supports_observable = LightningKokkos.capabilities.supports_observable
