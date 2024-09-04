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
from importlib import util as imp_util
from numbers import Number
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pennylane as qml
from pennylane.devices import DefaultExecutionConfig, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.operation import Operator
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms.core import TransformProgram
from pennylane.typing import Result

from pennylane_lightning.core.lightning_newAPI_base import (
    LightningBase,
    QuantumTape_or_Batch,
    Result_or_ResultBatch,
)

from ._adjoint_jacobian import LightningGPUAdjointJacobian
from ._measurements import LightningGPUMeasurements
from ._state_vector import LightningGPUStateVector

try:

    from pennylane_lightning.lightning_gpu_ops import (
        DevPool,
        backend_info,
        get_gpu_arch,
        is_gpu_supported,
    )

    LGPU_CPP_BINARY_AVAILABLE = True

    try:
        # pylint: disable=no-name-in-module
        from pennylane_lightning.lightning_gpu_ops import (
            DevTag,
            MPIManager,
        )            
        MPI_SUPPORT = True
    except ImportError:
        MPI_SUPPORT = False

except (ImportError, ValueError) as e:
    backend_info = None
    LGPU_CPP_BINARY_AVAILABLE = False


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
        "Sum",
        "Prod",
        "SProd",
    }
)

gate_cache_needs_hash = (
    qml.BlockEncode,
    qml.ControlledQubitUnitary,
    qml.DiagonalQubitUnitary,
    qml.MultiControlledX,
    qml.OrbitalRotation,
    qml.PSWAP,
    qml.QubitUnitary,
)

# MPI options
class LightningGPU_MPIHandler():
    """MPI handler for PennyLane Lightning GPU device  
    
    MPI handler to use a GPU-backed Lightning device using NVIDIA cuQuantum SDK with parallel capabilities.
    
    Use the MPI library is necessary to initialize different variables and methods to handle the data across  nodes and perform checks for memory allocation on each device. 
    
    Args:
        mpi (bool): declare if the device will use the MPI support.
        mpi_buf_size (int): size of GPU memory (in MiB) set for MPI operation and its default value is 64 MiB.
        dev_pool (Callable): Method to handle the GPU devices available.
        num_wires (int): the number of wires to initialize the device wit.h 
        c_dtype (np.complex64, np.complex128): Datatypes for statevector representation
        
    """

    def __init__(self, 
                 mpi: bool, 
                 mpi_buf_size: int, 
                 dev_pool: Callable, 
                 num_wires: int, 
                 c_dtype: Union[np.complex64, np.complex128]) -> None:
        
        self.use_mpi = mpi
        self.mpi_but_size = mpi_buf_size
        self._dp = dev_pool
        
        if self.use_mpi: 
            
            if not MPI_SUPPORT:
                raise ImportError("MPI related APIs are not found.")

            if mpi_buf_size < 0:
                raise TypeError(f"Unsupported mpi_buf_size value: {mpi_buf_size}, should be >= 0")

            if (mpi_buf_size > 0 
                and (mpi_buf_size & (mpi_buf_size - 1))):

                raise ValueError(f"Unsupported mpi_buf_size value: {mpi_buf_size}. mpi_buf_size should be power of 2.")
            
            # After check if all MPI parameter are ok
            self.mpi_manager, self.devtag = self._mpi_init_helper(num_wires)

            # set the number of global and local wires
            commSize = self._mpi_manager.getSize()
            self.num_global_wires = commSize.bit_length() - 1
            self.num_local_wires = num_wires - self._num_global_wires
            
            # Memory size in bytes
            sv_memsize = np.dtype(c_dtype).itemsize * (1 << self.num_local_wires)
            if self._mebibytesToBytes(mpi_buf_size) > sv_memsize:
                raise ValueError("The MPI buffer size is larger than the local state vector size.")

        if not self.use_mpi: 
            self.num_local_wires = num_wires

    def _mebibytesToBytes(mebibytes):
        return mebibytes * 1024 * 1024
    
    def _mpi_init_helper(self, num_wires):
        """Set up MPI checks and initializations."""
        
        # initialize MPIManager and config check in the MPIManager ctor
        mpi_manager = MPIManager()
        
        # check if number of GPUs per node is larger than number of processes per node
        numDevices = self._dp.getTotalDevices()
        numProcsNode = mpi_manager.getSizeNode()
        
        if numDevices < numProcsNode:
            raise ValueError(
                "Number of devices should be larger than or equal to the number of processes on each node."
            )
        
        # check if the process number is larger than number of statevector elements
        if mpi_manager.getSize() > (1 << (num_wires - 1)):
            raise ValueError(
                "Number of processes should be smaller than the number of statevector elements."
            )
        
        # set GPU device
        rank = self._mpi_manager.getRank()
        deviceid = rank % numProcsNode
        self._dp.setDeviceID(deviceid)
        devtag = DevTag(deviceid)
        
        return (mpi_manager, devtag)


def check_gpu_resources() -> None:
    """ Check the available resources of each Nvidia GPU """
    if (find_library("custatevec") is None 
        and not imp_util.find_spec("cuquantum")):
        
        raise ImportError(
            "custatevec libraries not found. Please pip install the appropriate custatevec library in a virtual environment."
        )
        
    if not DevPool.getTotalDevices():
        raise ValueError("No supported CUDA-capable device found")

    if not is_gpu_supported():
        raise ValueError(f"CUDA device is an unsupported version: {get_gpu_arch()}")
    

def stopping_condition(op: Operator) -> bool:
    """A function that determines whether or not an operation is supported by ``lightning.gpu``."""
    # To avoid building matrices beyond the given thresholds.
    # This should reduce runtime overheads for larger systems.
    return 0


def stopping_condition_shots(op: Operator) -> bool:
    """A function that determines whether or not an operation is supported by ``lightning.gpu``
    with finite shots."""
    return 0


def accepted_observables(obs: Operator) -> bool:
    """A function that determines whether or not an observable is supported by ``lightning.gpu``."""
    return 0


def adjoint_observables(obs: Operator) -> bool:
    """A function that determines whether or not an observable is supported by ``lightning.gpu``
    when using the adjoint differentiation method."""
    return 0


def adjoint_measurements(mp: qml.measurements.MeasurementProcess) -> bool:
    """Specifies whether or not an observable is compatible with adjoint differentiation on DefaultQubit."""
    return 0


def _supports_adjoint(circuit):
    return 0


def _adjoint_ops(op: qml.operation.Operator) -> bool:
    """Specify whether or not an Operator is supported by adjoint differentiation."""
    return 0


def _add_adjoint_transforms(program: TransformProgram) -> None:
    """Private helper function for ``preprocess`` that adds the transforms specific
    for adjoint differentiation.

    Args:
        program (TransformProgram): where we will add the adjoint differentiation transforms

    Side Effects:
        Adds transforms to the input program.

    """

    name = "adjoint + lightning.gpu"
    return 0


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
        sync (bool): immediately sync with host-sv after applying operation
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
        wires,
        *,
        c_dtype=np.complex128,
        shots=None,
        batch_obs=False,
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

        # GPU specific options

        # Creating the state vector
        
        self._dp = DevPool()
        self._c_dtype = c_dtype
        self._batch_obs = batch_obs
        self._sync = sync
        
        if isinstance(wires, int):
            self._wire_map = None  # should just use wires as is
        else:
            self._wire_map = {w: i for i, w in enumerate(self.wires)}

        self._mpi_handler = LightningGPU_MPIHandler(mpi, mpi_buf_size, self._dp, self.num_wires, c_dtype)
        
        self._num_local_wires = self._mpi_handler.num_local_wires

        self._statevector = LightningGPUStateVector(self.num_wires, dtype=c_dtype, mpi_handler=self._mpi_handler, sync=self._sync)


    @property
    def name(self):
        """The name of the device."""
        return "lightning.gpu"

    def _set_Lightning_classes(self):
        """Load the LightningStateVector, LightningMeasurements, LightningAdjointJacobian as class attribute"""
        return 0

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

        ``LightningGPU`` supports adjoint differentiation with analytic results.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            Bool: Whether or not a derivative can be calculated provided the given information

        """
        return 0

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
        return 0
