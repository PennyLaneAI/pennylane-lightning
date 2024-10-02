# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This module contains the LightningTensor class that inherits from the new device interface.
It is a device to perform tensor network simulations of quantum circuits using `cutensornet`. 
"""
from dataclasses import replace
from numbers import Number
from typing import Callable, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np
import pennylane as qml
from pennylane.devices import DefaultExecutionConfig, Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import (
    decompose,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from pennylane.operation import Operator
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms.core import TransformProgram
from pennylane.typing import Result, ResultBatch

from ._measurements import LightningTensorMeasurements
from ._tensornet import LightningTensorNet

try:
    # pylint: disable=import-error, unused-import
    from pennylane_lightning.lightning_tensor_ops import (
        backend_info,
        get_gpu_arch,
        is_gpu_supported,
    )

    if not is_gpu_supported():  # pragma: no cover
        raise ValueError(f"CUDA device is an unsupported version: {get_gpu_arch()}")

    LT_CPP_BINARY_AVAILABLE = True

except ImportError as ex:
    warn(str(ex), UserWarning)
    LT_CPP_BINARY_AVAILABLE = False

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]


_backends = frozenset({"cutensornet"})
# The set of supported backends.

_methods = frozenset({"mps"})
# The set of supported methods.

_operations = frozenset(
    {
        "Identity",
        "BasisState",
        "QubitUnitary",
        "ControlledQubitUnitary",
        "DiagonalQubitUnitary",
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "GlobalPhase",
        "S",
        "Adjoint(S)",
        "T",
        "Adjoint(T)",
        "SX",
        "Adjoint(SX)",
        "CNOT",
        "SWAP",
        "ISWAP",
        "Adjoint(ISWAP)",
        "PSWAP",
        "Adjoint(SISWAP)",
        "SISWAP",
        "SQISW",
        "CSWAP",
        "Toffoli",
        "CY",
        "CZ",
        "PhaseShift",
        "ControlledPhaseShift",
        "C(Hadamard)",
        "C(S)",
        "C(T)",
        "C(PhaseShift)",
        "C(RX)",
        "C(RY)",
        "C(RZ)",
        "C(Rot)",
        "C(IsingXX)",
        "C(IsingYY)",
        "C(IsingZZ)",
        "C(IsingXY)",
        "C(SingleExcitation)",
        "C(SingleExcitationPlus)",
        "C(SingleExcitationMinus)",
        "C(DoubleExcitation)",
        "C(DoubleExcitationMinus)",
        "C(DoubleExcitationPlus)",
        "C(GlobalPhase)",
        "C(MultiRZ)",
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
        "ECR",
        "BlockEncode",
        "C(BlockEncode)",
    }
)

_observables = frozenset(
    {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "Hermitian",
        "Identity",
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
    """A function that determines whether or not an operation is supported by the ``mps`` method of ``lightning.tensor``."""
    if isinstance(op, qml.ControlledQubitUnitary):
        return True

    return op.has_matrix and op.name in _operations


def simulate(circuit: QuantumScript, tensornet: LightningTensorNet) -> Result:
    """Simulate a single quantum script.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        tensornet (LightningTensorNet): handle to Lightning tensor network

    Returns:
        Tuple[TensorLike]: The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.
    """
    tensornet.reset_state()
    tensornet.set_tensor_network(circuit)
    return LightningTensorMeasurements(tensornet).measure_tensor_network(circuit)


def accepted_observables(obs: Operator) -> bool:
    """A function that determines whether or not an observable is supported by ``lightning.tensor``."""
    return obs.name in _observables


def accepted_backends(backend: str) -> bool:
    """A function that determines whether or not a backend is supported by ``lightning.tensor``."""
    return backend in _backends


def accepted_methods(method: str) -> bool:
    """A function that determines whether or not a method is supported by ``lightning.tensor``."""
    return method in _methods


@simulator_tracking
@single_tape_support
class LightningTensor(Device):
    """PennyLane Lightning Tensor device.

    A device to perform tensor network operations on a quantum circuit.

    This device is designed to simulate large-scale quantum circuits using tensor network methods. For
    small circuits, other devices like ``lightning.qubit``, ``lightning.gpu``or ``lightning.kokkos``  are
    recommended.

    Currently, only the Matrix Product State (MPS) method as implemented in the ``cutensornet`` backend is supported.

    Args:
        wires (int): The number of wires to initialize the device with.
            Defaults to ``None`` if not specified.
        shots (int):  Measurements are performed drawing ``shots`` times from a discrete random variable distribution associated with a state vector and an observable. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically.
        method (str): Supported method. Currently, only ``mps`` is supported.
        c_dtype: Datatypes for the tensor representation. Must be one of
            ``numpy.complex64`` or ``numpy.complex128``. Default is ``numpy.complex128``.
    Keyword Args:
        max_bond_dim (int): The maximum bond dimension to be used in the MPS simulation. Default is 128.
            The accuracy of the wavefunction representation comes with a memory tradeoff which can be
            tuned with `max_bond_dim`. The larger the internal bond dimension, the more entanglement can
            be described but the larger the memory requirements. Note that GPUs are ill-suited (i.e. less
            competitive compared with CPUs) for simulating circuits with low bond dimensions and/or circuit
            layers with a single or few gates because the arithmetic intensity is lower.
        cutoff (float): The threshold used to truncate the singular values of the MPS tensors. The default is 0.
        cutoff_mode (str): Singular value truncation mode. The options are ``"rel"`` and ``"abs"``. The default is ``"abs"``.
        backend (str): Supported backend. Currently, only ``cutensornet`` is supported.

    **Example**

    .. code-block:: python

        import pennylane as qml

        num_qubits = 100

        dev = qml.device("lightning.tensor", wires=num_qubits)

        @qml.qnode(dev)
        def circuit(num_qubits):
            for qubit in range(0, num_qubits - 1):
                qml.CZ(wires=[qubit, qubit + 1])
                qml.X(wires=[qubit])
                qml.Z(wires=[qubit + 1])
            return qml.expval(qml.Z(0))

    >>> print(circuit(num_qubits))
    -1.0
    """

    # pylint: disable=too-many-instance-attributes

    # So far we just consider the options for MPS simulator
    _device_options = ("backend", "max_bond_dim", "cutoff", "cutoff_mode")
    _CPP_BINARY_AVAILABLE = LT_CPP_BINARY_AVAILABLE
    _new_API = True

    # TODO: Move supported ops/obs to TOML file
    operations = _operations
    # The names of the supported operations.

    observables = _observables
    # The names of the supported observables.

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        wires=None,
        shots=None,
        method: str = "mps",
        c_dtype=np.complex128,
        **kwargs,
    ):
        if not self._CPP_BINARY_AVAILABLE:
            raise ImportError("Pre-compiled binaries for lightning.tensor are not available. ")

        if not accepted_methods(method):
            raise ValueError(f"Unsupported method: {method}")

        if c_dtype not in [np.complex64, np.complex128]:  # pragma: no cover
            raise TypeError(f"Unsupported complex type: {c_dtype}")

        if wires is None:
            raise ValueError("The number of wires must be specified.")

        super().__init__(wires=wires, shots=shots)

        if isinstance(wires, int):
            self._wire_map = None  # should just use wires as is
        else:
            self._wire_map = {w: i for i, w in enumerate(self.wires)}

        self._num_wires = len(self.wires) if self.wires else 0
        self._method = method
        self._c_dtype = c_dtype

        self._max_bond_dim = kwargs.get("max_bond_dim", 128)
        self._cutoff = kwargs.get("cutoff", 0)
        self._cutoff_mode = kwargs.get("cutoff_mode", "abs")
        self._backend = kwargs.get("backend", "cutensornet")

        for arg in kwargs:
            if arg not in self._device_options:
                raise TypeError(
                    f"Unexpected argument: {arg} during initialization of the lightning.tensor device."
                )

        if not accepted_backends(self._backend):
            raise ValueError(f"Unsupported backend: {self._backend}")

        if self._cutoff_mode not in ["rel", "abs"]:
            raise ValueError(f"Unsupported cutoff mode: {self._cutoff_mode}")

        if not isinstance(self._max_bond_dim, int) or self._max_bond_dim < 1:
            raise ValueError("The maximum bond dimension must be an integer greater than 0.")

    @property
    def name(self):
        """The name of the device."""
        return "lightning.tensor"

    @property
    def num_wires(self):
        """Number of wires addressed on this device."""
        return self._num_wires

    @property
    def backend(self):
        """Supported backend."""
        return self._backend

    @property
    def method(self):
        """Supported method."""
        return self._method

    @property
    def c_dtype(self):
        """Tensor complex data type."""
        return self._c_dtype

    def _tensornet(self):
        """Return the tensornet object."""
        return LightningTensorNet(
            self._num_wires,
            self._method,
            self._c_dtype,
            self._max_bond_dim,
            self._cutoff,
            self._cutoff_mode,
        )

    dtype = c_dtype

    def _setup_execution_config(
        self, config: Optional[ExecutionConfig] = DefaultExecutionConfig
    ) -> ExecutionConfig:
        """
        Update the execution config with choices for how the device should be used and the device options.
        """
        # TODO: add options for gradients next quarter

        updated_values = {}

        new_device_options = dict(config.device_options)
        for option in self._device_options:
            if option not in new_device_options:
                new_device_options[option] = getattr(self, f"_{option}", None)

        return replace(config, **updated_values, device_options=new_device_options)

    def preprocess(
        self,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """This function defines the device transform program to be applied and an updated device configuration.

        Args:
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure describing the
                parameters needed to fully describe the execution.

        Returns:
            TransformProgram, ExecutionConfig: A transform program that when called returns :class:`~.QuantumTape`'s that the
            device can natively execute as well as a postprocessing function to be called after execution, and a configuration
            with unset specifications filled in.

        This device currently:

        * Does not support derivatives.
        * Does not support vector-Jacobian products.
        """

        config = self._setup_execution_config(execution_config)

        program = TransformProgram()

        program.add_transform(validate_measurements, name=self.name)
        program.add_transform(validate_observables, accepted_observables, name=self.name)
        program.add_transform(validate_device_wires, self._wires, name=self.name)
        program.add_transform(
            decompose,
            stopping_condition=stopping_condition,
            stopping_condition_shots=stopping_condition,
            skip_initial_state_prep=True,
            name=self.name,
        )
        return program, config

    # pylint: disable=unused-argument
    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        """Execute a circuit or a batch of circuits and turn it into results.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the quantum circuits to be executed.
            execution_config (ExecutionConfig): a data structure with additional information required for execution.

        Returns:
            TensorLike, tuple[TensorLike], tuple[tuple[TensorLike]]: A numeric result of the computation.
        """

        results = []

        for circuit in circuits:
            if self._wire_map is not None:
                [circuit], _ = qml.map_wires(circuit, self._wire_map)
            results.append(simulate(circuit, self._tensornet()))

        return tuple(results)

    # pylint: disable=unused-argument
    def supports_derivatives(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[qml.tape.QuantumTape] = None,
    ) -> bool:
        """Check whether or not derivatives are available for a given configuration and circuit.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation.
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            Bool: Whether or not a derivative can be calculated provided the given information.

        """
        return False

    def compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """Calculate the Jacobian of either a single or a batch of circuits on the device.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuits to calculate derivatives for.
            execution_config (ExecutionConfig): a data structure with all additional information required for execution.

        Returns:
            Tuple: The Jacobian for each trainable parameter.
        """
        raise NotImplementedError(
            "The computation of derivatives has yet to be implemented for the lightning.tensor device."
        )

    def execute_and_compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """Compute the results and Jacobians of circuits at the same time.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuits or batch of circuits.
            execution_config (ExecutionConfig): a data structure with all additional information required for execution.

        Returns:
            tuple: A numeric result of the computation and the gradient.
        """
        raise NotImplementedError(
            "The computation of derivatives has yet to be implemented for the lightning.tensor device."
        )

    # pylint: disable=unused-argument
    def supports_vjp(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """Whether or not this device defines a custom vector-Jacobian product.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation.
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            Bool: Whether or not a derivative can be calculated provided the given information.
        """
        return False

    def compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        r"""The vector-Jacobian product used in reverse-mode differentiation.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuit or batch of circuits.
            cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must have shape matching the output shape of the
                corresponding circuit. If the circuit has a single output, ``cotangents`` may be a single number, not an iterable
                of numbers.
            execution_config (ExecutionConfig): a data structure with all additional information required for execution.

        Returns:
            tensor-like: A numeric result of computing the vector-Jacobian product.
        """
        raise NotImplementedError(
            "The computation of vector-Jacobian product has yet to be implemented for the lightning.tensor device."
        )

    def execute_and_compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """Calculate both the results and the vector-Jacobian product used in reverse-mode differentiation.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuit or batch of circuits to be executed.
            cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must have shape matching the output shape of the
                corresponding circuit.
            execution_config (ExecutionConfig): a data structure with all additional information required for execution.

        Returns:
            Tuple, Tuple: the result of executing the scripts and the numeric result of computing the vector-Jacobian product
        """
        raise NotImplementedError(
            "The computation of vector-Jacobian product has yet to be implemented for the lightning.tensor device."
        )
