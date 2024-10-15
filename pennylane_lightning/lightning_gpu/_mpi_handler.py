# Copyright 2022-2024 Xanadu Quantum Technologies Inc.

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
This module contains the :class:`~.LightningGPU_MPIHandler` class, a MPI handler to use LightningGPU device with multi-GPU on multi-node system.
"""

try:
    # pylint: disable=no-name-in-module
    from pennylane_lightning.lightning_gpu_ops import DevPool, DevTag, MPIManager

    MPI_SUPPORT = True
except ImportError:
    MPI_SUPPORT = False

from typing import Union

import numpy as np


# MPI options
class MPIHandler:  # pylint: disable=too-few-public-methods
    """MPI handler for PennyLane Lightning GPU device.

    MPI handler to use a GPU-backed Lightning device using NVIDIA cuQuantum SDK with parallel capabilities.

    Use the MPI library is necessary to initialize different variables and methods to handle the data across nodes and perform checks for memory allocation on each device.

    Args:
        mpi (bool): declare if the device will use the MPI support.
        mpi_buf_size (int): size of GPU memory (in MiB) set for MPI operation and its default value is 64 MiB.
        num_wires (int): the number of wires to initialize the device with.
        c_dtype (np.complex64, np.complex128): Datatypes for statevector representation.
    """

    def __init__(
        self,
        mpi: bool,
        mpi_buf_size: int,
        num_wires: int,
        c_dtype: Union[np.complex64, np.complex128],
    ) -> None:

        self.use_mpi = mpi
        self.mpi_buf_size = mpi_buf_size

        self._dp = DevPool()

        if self.use_mpi:

            if not MPI_SUPPORT:
                raise ImportError(
                    "Pre-compiled binaries for lightning.gpu with MPI support are not available. "
                    "To manually compile from source, follow the instructions at "
                    "https://docs.pennylane.ai/projects/lightning/en/stable/dev/installation.html."
                )

            if mpi_buf_size < 0:
                raise ValueError(f"Unsupported mpi_buf_size value: {mpi_buf_size}, should be >= 0")

            if mpi_buf_size > 0 and (mpi_buf_size & (mpi_buf_size - 1)):
                raise ValueError(
                    f"Unsupported mpi_buf_size value: {mpi_buf_size}. mpi_buf_size should be power of 2."
                )

            # After check if all MPI parameters are ok
            self.mpi_manager, self.devtag = self._mpi_init_helper(num_wires)

            # set the number of global and local wires
            commSize = self.mpi_manager.getSize()
            self.num_global_wires = commSize.bit_length() - 1
            self.num_local_wires = num_wires - self.num_global_wires

            self._check_memory_size(c_dtype, mpi_buf_size)

        if not self.use_mpi:
            self.num_local_wires = num_wires
            self.num_global_wires = num_wires

    def _mebibytesToBytes(self, mebibytes):
        return mebibytes * 1024 * 1024

    def _check_memory_size(self, c_dtype, mpi_buf_size):
        # Memory size in bytes
        sv_memsize = np.dtype(c_dtype).itemsize * (1 << self.num_local_wires)
        if self._mebibytesToBytes(mpi_buf_size) > sv_memsize:
            raise RuntimeError("The MPI buffer size is larger than the local state vector size.")

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
        rank = mpi_manager.getRank()
        deviceid = rank % numProcsNode
        self._dp.setDeviceID(deviceid)
        devtag = DevTag(deviceid)

        return (mpi_manager, devtag)
