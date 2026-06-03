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
Unit tests for device creation on :mod:`pennylane_lightning` MPI-enabled devices.
"""

# pylint: disable=protected-access,unused-variable,missing-function-docstring,c-extension-no-member

import pennylane as qp
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
            dev = qp.device(device_name, mpi=True, wires=4)
    else:
        dev = qp.device(device_name, mpi=True, wires=4)


def test_unsupported_dynamic_wires():
    with pytest.raises(
        DeviceError,
        match="does not support dynamic wires allocation.",
    ):
        dev = qp.device(device_name, mpi=True)


@pytest.mark.parametrize(
    "circuit_in, n_wires, wires_list",
    [
        (
            QuantumScript(
                [
                    qp.RX(0.1, 0),
                    qp.CNOT([1, 0]),
                    qp.RZ(0.1, 1),
                    qp.CNOT([2, 1]),
                ],
                [qp.expval(qp.Z(0))],
            ),
            3,
            [0, 1, 2],
        ),
        (
            QuantumScript(
                [
                    qp.RX(0.1, 0),
                    qp.CNOT([1, 4]),
                    qp.RZ(0.1, 4),
                    qp.CNOT([2, 1]),
                ],
                [qp.expval(qp.Z(6))],
            ),
            7,
            [0, 1, 4, 2, 6],
        ),
    ],
)
def test_dynamic_wires_from_circuit_fixed_wires(circuit_in, n_wires, wires_list):
    """Test that dynamic_wires_from_circuit creates correct statevector and circuit."""
    dev = qp.device(device_name, mpi=True, wires=n_wires)
    circuit_out = dev.dynamic_wires_from_circuit(circuit_in)

    assert circuit_out.num_wires == circuit_in.num_wires
    assert circuit_out.wires == qp.wires.Wires(wires_list)
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
        assert (
            dev._statevector._qubit_state.getNumLocalWires()
            + dev._statevector._qubit_state.getNumGlobalWires()
        ) == n_wires


@pytest.mark.skipif(device_name != "lightning.gpu", reason="Only for LGPU")
def test_unsupported_mpi_buf_size():
    with pytest.raises(ValueError, match="Unsupported mpi_buf_size value"):
        dev = qp.device(device_name, mpi=True, wires=4, mpi_buf_size=-1)
    with pytest.raises(ValueError, match="Unsupported mpi_buf_size value"):
        dev = qp.device(device_name, mpi=True, wires=4, mpi_buf_size=3)
    with pytest.raises(
        RuntimeError,
        match="The MPI buffer size is larger than the local state vector size.",
    ):
        dev = qp.device(device_name, mpi=True, wires=4, mpi_buf_size=2**4)
    with pytest.raises(
        ValueError,
        match="Number of processes should be smaller than the number of statevector elements",
    ):
        dev = qp.device(device_name, mpi=True, wires=1)


@pytest.mark.skipif(device_name != "lightning.gpu", reason="Only for LGPU")
def test_unsupported_gate():
    comm = MPI.COMM_WORLD
    dev = qp.device(device_name, mpi=True, wires=4)
    op = qp.ctrl(qp.GlobalPhase(0.1, wires=[1, 2, 3]), [0], control_values=[True])
    tape = QuantumScript([op])
    with pytest.raises(
        DeviceError, match="Lightning-GPU-MPI does not support Controlled GlobalPhase gates"
    ):
        dev.execute(tape)
        comm.Barrier()


@pytest.mark.skipif(device_name != "lightning.kokkos", reason="Only for Lightning-Kokkos")
@pytest.mark.parametrize("comm_buffer_ratio", [1, 2, 3, 4, 5])
def test_comm_buffer_ratio(comm_buffer_ratio):
    """For ``mpi=True``, every comm_buffer_ratio yields the correct result."""
    n_wires = 7

    def circuit():
        qp.Hadamard(0)
        qp.CNOT([0, 6])
        qp.RX(0.37, wires=2)
        qp.CNOT([4, 1])
        qp.RY(0.21, wires=5)
        return qp.expval(qp.PauliZ(0) @ qp.PauliZ(6))

    dev = qp.device(device_name, wires=n_wires, mpi=True, comm_buffer_ratio=comm_buffer_ratio)
    dev_ref = qp.device("lightning.qubit", wires=n_wires)
    res = qp.QNode(circuit, dev)()
    res_ref = qp.QNode(circuit, dev_ref)()
    assert res == pytest.approx(res_ref)


@pytest.mark.skipif(device_name != "lightning.kokkos", reason="Only for Lightning-Kokkos")
def test_comm_buffer_ratio_requires_mpi():
    """Setting comm_buffer_ratio without mpi=True raises a clear error."""
    with pytest.raises(DeviceError, match="comm_buffer_ratio requires mpi=True"):
        qp.device(device_name, wires=4, comm_buffer_ratio=2)


@pytest.mark.skipif(device_name != "lightning.kokkos", reason="Only for Lightning-Kokkos")
@pytest.mark.parametrize("bad_ratio", [0, -1, 2.5, "4"])
def test_comm_buffer_ratio_invalid_value(bad_ratio):
    """A non-positive-integer comm_buffer_ratio raises a DeviceError."""
    with pytest.raises(DeviceError, match="comm_buffer_ratio must be a positive integer"):
        qp.device(device_name, wires=4, mpi=True, comm_buffer_ratio=bad_ratio)
