# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
This module contains the LightningQubit class, a PennyLane simulator device that
interfaces with C++ for fast linear algebra calculations.
"""
from dataclasses import replace
from functools import reduce
from pathlib import Path
from typing import List, Optional, Sequence, Union
from warnings import warn

import numpy as np
import pennylane as qml
from numpy.random import BitGenerator, Generator, SeedSequence
from numpy.typing import ArrayLike
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
from pennylane.exceptions import DeviceError
from pennylane.measurements import MidMeasureMP
from pennylane.operation import DecompositionUndefinedError, Operator
from pennylane.ops import Conditional, PauliRot, Prod, SProd, Sum
from pennylane.transforms.core import TransformProgram

from pennylane_lightning.lightning_base.lightning_base import (
    LightningBase,
    QuantumTape_or_Batch,
    Result_or_ResultBatch,
)

try:
    from pennylane_lightning.lightning_qubit_ops import backend_info

    LQ_CPP_BINARY_AVAILABLE = True
except ImportError as ex:
    warn(str(ex), UserWarning)
    LQ_CPP_BINARY_AVAILABLE = False

from ._adjoint_jacobian import LightningAdjointJacobian
from ._measurements import LightningMeasurements
from ._state_vector import LightningStateVector

_to_matrix_ops = {
    "BlockEncode": OperatorProperties(controllable=True),
    "DiagonalQubitUnitary": OperatorProperties(),
    "ECR": OperatorProperties(),
    "ISWAP": OperatorProperties(),
    "OrbitalRotation": OperatorProperties(),
    "QubitCarry": OperatorProperties(),
    "QubitSum": OperatorProperties(),
    "SISWAP": OperatorProperties(),
    "SQISW": OperatorProperties(),
}


def stopping_condition(op: Operator) -> bool:
    """A function that determines whether or not an operation is supported by ``lightning.qubit``."""
    # As ControlledQubitUnitary == C(QubitUnitrary),
    # it can be removed from `_operations` to keep
    # consistency with `lightning_qubit.toml`
    if isinstance(op, qml.ControlledQubitUnitary):
        return True
    if isinstance(op, qml.PauliRot):
        word = op._hyperparameters["pauli_word"]  # pylint: disable=protected-access
        # decomposes to IsingXX, etc. for n <= 2
        return reduce(lambda x, y: x + (y != "I"), word, 0) > 2
    if op.name in ("C(SProd)", "C(Exp)"):
        return True

    if (isinstance(op, Conditional) and stopping_condition(op.base)) or isinstance(
        op, MidMeasureMP
    ):
        # Conditional and MidMeasureMP should not be decomposed
        return True

    return _supports_operation(op.name)


def stopping_condition_shots(op: Operator) -> bool:
    """A function that determines whether or not an operation is supported by ``lightning.qubit``
    with finite shots."""
    return stopping_condition(op) or isinstance(op, (MidMeasureMP, qml.ops.op_math.Conditional))


def accepted_observables(obs: Operator) -> bool:
    """A function that determines whether or not an observable is supported by ``lightning.qubit``."""
    return _supports_observable(obs.name)


def adjoint_observables(obs: Operator) -> bool:
    """A function that determines whether or not an observable is supported by ``lightning.qubit``
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
    except (DecompositionUndefinedError, DeviceError, AttributeError):
        return False
    return True


def _adjoint_ops(op: qml.operation.Operator) -> bool:
    """Specify whether or not an Operator is supported by adjoint differentiation."""

    return not isinstance(op, (Conditional, MidMeasureMP, PauliRot)) and (
        not any(qml.math.requires_grad(d) for d in op.data)
        or (op.num_params == 1 and op.has_generator)
    )


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
    program.add_transform(qml.transforms.broadcast_expand)
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
    program.add_transform(validate_adjoint_trainable_params)


@simulator_tracking
@single_tape_support
class LightningQubit(LightningBase):
    """PennyLane Lightning Qubit device.

    A device that interfaces with C++ to perform fast linear algebra calculations.

    Use of this device requires pre-built binaries or compilation from source. Check out the
    :doc:`/lightning_qubit/installation` guide for more details.

    Args:
        wires (Optional[int, list]): the number of wires to initialize the device with. Defaults to ``None`` if not specified, and the device will allocate the number of wires depending on the circuit to execute.
        c_dtype: Datatypes for statevector representation. Must be one of
            ``np.complex64`` or ``np.complex128``.
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically.
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
        seed (Union[str, None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``, or
            a request to seed from numpy's global random number generator.
            The default, ``seed="global"`` pulls a seed from NumPy's global generator. ``seed=None``
            will pull a seed from the OS entropy.
    """

    # pylint: disable=too-many-instance-attributes

    # General device options
    _device_options = ("rng", "c_dtype", "batch_obs", "mcmc", "kernel_name", "num_burnin")
    # Device specific options
    _CPP_BINARY_AVAILABLE = LQ_CPP_BINARY_AVAILABLE
    _backend_info = backend_info if LQ_CPP_BINARY_AVAILABLE else None

    # This configuration file declares the device capabilities
    config_filepath = Path(__file__).parent / "lightning_qubit.toml"

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
        # Markov Chain Monte Carlo (MCMC) sampling method arguments
        mcmc: bool = False,
        kernel_name: str = "Local",
        num_burnin: int = 100,
    ):
        if not self._CPP_BINARY_AVAILABLE:
            raise ImportError(
                "Pre-compiled binaries for lightning.qubit are not available. "
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

        # Markov Chain Monte Carlo (MCMC) sampling method specific options
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

        self.device_kwargs = {
            "mcmc": self._mcmc,
            "num_burnin": self._num_burnin,
            "kernel_name": self._kernel_name,
        }

        self._statevector = None
        self._sv_init_kwargs = {}

    @property
    def name(self):
        """The name of the device."""
        return "lightning.qubit"

    def _set_lightning_classes(self):
        """Load the LightningStateVector, LightningMeasurements, LightningAdjointJacobian as class attribute"""
        self.LightningStateVector = LightningStateVector
        self.LightningMeasurements = LightningMeasurements
        self.LightningAdjointJacobian = LightningAdjointJacobian

    def _setup_execution_config(self, config):
        """
        Update the execution config with choices for how the device should be used and the device options.
        """
        updated_values = {}

        for option, _ in config.device_options.items():
            if option not in self._device_options:
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

        mcm_supported_methods = (
            ("deferred", "tree-traversal", "one-shot", None)
            if not qml.capture.enabled()
            else ("deferred", "single-branch-statistics", None)
        )

        mcm_config = config.mcm_config

        if (mcm_method := mcm_config.mcm_method) not in mcm_supported_methods:
            raise DeviceError(f"mcm_method='{mcm_method}' is not supported with lightning.qubit")

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

            updated_values["mcm_config"] = replace(mcm_config, **mcm_updated_values)

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

        if qml.capture.enabled():

            if exec_config.mcm_config.mcm_method == "deferred":
                program.add_transform(qml.defer_measurements, num_wires=len(self.wires))
            # Using stopping_condition_shots because we don't want to decompose Conditionals or MCMs
            program.add_transform(qml.transforms.decompose, gate_set=stopping_condition_shots)
            return program, exec_config

        program.add_transform(validate_measurements, name=self.name)
        program.add_transform(validate_observables, accepted_observables, name=self.name)
        program.add_transform(
            mid_circuit_measurements, device=self, mcm_config=exec_config.mcm_config
        )
        program.add_transform(validate_device_wires, self.wires, name=self.name)

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
            results.append(
                self.simulate(
                    self.dynamic_wires_from_circuit(circuit),
                    self._statevector,
                    mcmc=mcmc,
                    postselect_mode=execution_config.mcm_config.postselect_mode,
                    mcm_method=execution_config.mcm_config.mcm_method,
                )
            )

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

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """

        return LightningBase.get_c_interface_impl("LightningSimulator", "lightning_qubit")


_supports_operation = LightningQubit.capabilities.supports_operation
_supports_observable = LightningQubit.capabilities.supports_observable
