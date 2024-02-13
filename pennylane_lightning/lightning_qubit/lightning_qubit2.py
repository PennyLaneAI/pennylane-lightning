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
from typing import Union, Sequence, Optional
from dataclasses import replace
import numpy as np

import pennylane as qml
from pennylane.devices import Device, ExecutionConfig, DefaultExecutionConfig
from pennylane.devices.preprocess import (
    decompose,
    validate_device_wires,
    decompose,
    validate_measurements,
    validate_observables,
    no_sampling,
)
from pennylane.devices.qubit.sampling import get_num_shots_and_executions
from pennylane.tape import QuantumScript
from pennylane.transforms.core import TransformProgram
from pennylane.typing import Result, ResultBatch

from ._state_vector import LightningStateVector
from ._measurements import LightningMeasurements


try:
    # pylint: disable=import-error, no-name-in-module
    from pennylane_lightning.lightning_qubit_ops import (
        StateVectorC64,
        StateVectorC128,
    )

    LQ_CPP_BINARY_AVAILABLE = True
except ImportError:
    LQ_CPP_BINARY_AVAILABLE = False


def simulate(circuit: QuantumScript, dtype=np.complex128, debugger=None) -> Result:
    """Simulate a single quantum script.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        dtype: Datatypes for state-vector representation. Must be one of
            ``np.complex64`` or ``np.complex128``.
        debugger (_Debugger): The debugger to use

    Returns:
        tuple(TensorLike): The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.

    """
    state, is_state_batched = LightningStateVector(
        num_wires=circuit.num_wires, dtype=dtype
    ).get_final_state(circuit, debugger=debugger)
    return LightningMeasurements(state).measure_final_state(circuit, is_state_batched)


def dummy_jacobian(circuit: QuantumScript):
    return np.array(0.0)


def simulate_and_jacobian(circuit: QuantumScript):
    return np.array(0.0), np.array(0.0)


Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[qml.tape.QuantumTape]
QuantumTape_or_Batch = Union[qml.tape.QuantumTape, QuantumTapeBatch]


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


def stopping_condition(op: qml.operation.Operator) -> bool:
    return op.name in _operations


def accepted_observables(obs: qml.operation.Operator) -> bool:
    return obs.name in _observables


def accepted_measurements(m: qml.measurements.MeasurementProcess) -> bool:
    return isinstance(m, (qml.measurements.ExpectationMP))


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
        seed (str, int, rng)
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

    name = "lightning.qubit2"
    _CPP_BINARY_AVAILABLE = LQ_CPP_BINARY_AVAILABLE

    _device_options = ["rng", "c_dtype", "batch_obs", "mcmc", "kernel_name", "num_burnin"]

    def __init__(  # pylint: disable=too-many-arguments
        self,
        wires=None,  # the number of wires are always needed for Lightning. Do we have a better way to get it
        *,
        c_dtype=np.complex128,
        shots=None,
        seed="global",
        mcmc=False,
        kernel_name="Local",
        num_burnin=100,
        batch_obs=False,
    ):
        super().__init__(wires=wires, shots=shots)
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

    @property
    def operation(self) -> frozenset[str]:
        """The names of supported operations."""
        return _operations

    @property
    def observables(self) -> frozenset[str]:
        """The names of supported observables."""
        return _observables

    def _setup_execution_config(self, config):
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

    def supports_derivatives(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[qml.tape.QuantumTape] = None,
    ) -> bool:
        if execution_config is None and circuit is None:
            return True
        if execution_config.gradient_method not in {"adjoint", "best"}:
            return False
        if circuit is None:
            return True
        return (
            all(isinstance(m, qml.measurements.ExpectationMP) for m in circuit.measurements)
            and not circuit.shots
        )

    def preprocess(self, execution_config: ExecutionConfig = DefaultExecutionConfig):
        program = TransformProgram()
        program.add_transform(
            validate_measurements, analytic_measurements=accepted_measurements, name=self.name
        )
        program.add_transform(no_sampling)
        program.add_transform(validate_observables, accepted_observables, name=self.name)
        program.add_transform(validate_device_wires, self.wires, name=self.name)
        program.add_transform(qml.defer_measurements, device=self)
        program.add_transform(decompose, stopping_condition=stopping_condition, name=self.name)
        program.add_transform(qml.transforms.broadcast_expand)
        return program, self._setup_execution_config(execution_config)

    def _execute_tracking(self, circuits):
        self.tracker.update(batches=1)
        self.tracker.record()
        for c in circuits:
            qpu_executions, shots = get_num_shots_and_executions(c)
            if c.shots:
                self.tracker.update(
                    simulations=1,
                    executions=qpu_executions,
                    shots=shots,
                )
            else:
                self.tracker.update(
                    simulations=1,
                    executions=qpu_executions,
                )
            self.tracker.record()

    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = (circuits,)

        if self.tracker.active:
            self._execute_tracking(circuits)

        results = []
        for circuit in circuits:
            circuit = circuit.map_to_standard_wires()
            results.append(simulate(circuit, **execution_config.device_options))

        return results[0] if is_single_circuit else tuple(results)

    def compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]

        if self.tracker.active:
            self.tracker.update(derivative_batches=1, derivatives=len(circuits))
            self.tracker.record()
        res = tuple(dummy_jacobian(circuit) for circuit in circuits)

        return res[0] if is_single_circuit else res

    def execute_and_compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]

        if self.tracker.active:
            for c in circuits:
                self.tracker.update(resources=c.specs["resources"])
            self.tracker.update(
                execute_and_derivative_batches=1,
                executions=len(circuits),
                derivatives=len(circuits),
            )
            self.tracker.record()

        results = tuple(simulate_and_jacobian(c) for c in circuits)
        results, jacs = tuple(zip(*results))
        return (results[0], jacs[0]) if is_single_circuit else (results, jacs)
