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
"""
Unit tests for Lightning devices creation.
"""
# pylint: disable=protected-access,unused-variable,missing-function-docstring,c-extension-no-member

import pennylane as qml
import pytest
from conftest import LightningDevice as ld
from conftest import device_name
from mpi4py import MPI
from pennylane.exceptions import DeviceError
from pennylane.tape import QuantumScript

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


def test_create_device():
    if MPI.COMM_WORLD.Get_size() > 2:
        with pytest.raises(
            ValueError,
            match="Number of devices should be larger than or equal to the number of processes on each node.",
        ):
            dev = qml.device(device_name, mpi=True, wires=4)
    else:
        dev = qml.device(device_name, mpi=True, wires=4)


def test_unsupported_dynamic_wires():
    with pytest.raises(
        DeviceError,
        match="does not support dynamic wires allocation.",
    ):
        dev = qml.device(device_name, mpi=True)


@pytest.mark.parametrize(
    "circuit_in, n_wires, wires_list",
    [
        (
            QuantumScript(
                [
                    qml.RX(0.1, 0),
                    qml.CNOT([1, 0]),
                    qml.RZ(0.1, 1),
                    qml.CNOT([2, 1]),
                ],
                [qml.expval(qml.Z(0))],
            ),
            3,
            [0, 1, 2],
        ),
        (
            QuantumScript(
                [
                    qml.RX(0.1, 0),
                    qml.CNOT([1, 4]),
                    qml.RZ(0.1, 4),
                    qml.CNOT([2, 1]),
                ],
                [qml.expval(qml.Z(6))],
            ),
            7,
            [0, 1, 4, 2, 6],
        ),
    ],
)
def test_dynamic_wires_from_circuit_fixed_wires(circuit_in, n_wires, wires_list):
    """Test that dynamic_wires_from_circuit creates correct statevector and circuit."""
    dev = qml.device(device_name, mpi=True, wires=n_wires)
    circuit_out = dev.dynamic_wires_from_circuit(circuit_in)

    assert circuit_out.num_wires == circuit_in.num_wires
    assert circuit_out.wires == qml.wires.Wires(wires_list)
    assert circuit_out.operations == circuit_in.operations
    assert circuit_out.measurements == circuit_in.measurements

    if device_name == "lightning.gpu":
        assert dev._statevector._mpi_handler.use_mpi
        assert (
            dev._statevector._mpi_handler.num_local_wires
            + dev._statevector._mpi_handler.num_global_wires
        ) == n_wires
    elif device_name == "lightning.kokkos":
        assert dev._statevector._mpi

@pytest.mark.skipif(device_name != "lightning.gpu", reason="Only for LGPU")
def test_unsupported_mpi_buf_size():
    with pytest.raises(ValueError, match="Unsupported mpi_buf_size value"):
        dev = qml.device(device_name, mpi=True, wires=4, mpi_buf_size=-1)
    with pytest.raises(ValueError, match="Unsupported mpi_buf_size value"):
        dev = qml.device(device_name, mpi=True, wires=4, mpi_buf_size=3)
    with pytest.raises(
        RuntimeError,
        match="The MPI buffer size is larger than the local state vector size.",
    ):
        dev = qml.device(device_name, mpi=True, wires=4, mpi_buf_size=2**4)
    with pytest.raises(
        ValueError,
        match="Number of processes should be smaller than the number of statevector elements",
    ):
        dev = qml.device(device_name, mpi=True, wires=1)


@pytest.mark.skipif(device_name != "lightning.gpu", reason="Only for LGPU")
def test_unsupported_gate():
    comm = MPI.COMM_WORLD
    dev = qml.device(device_name, mpi=True, wires=4)
    op = qml.ctrl(qml.GlobalPhase(0.1, wires=[1, 2, 3]), [0], control_values=[True])
    tape = QuantumScript([op])
    with pytest.raises(
        DeviceError, match="Lightning-GPU-MPI does not support Controlled GlobalPhase gates"
    ):
        dev.execute(tape)
        comm.Barrier()
