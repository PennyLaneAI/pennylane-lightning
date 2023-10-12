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
Unit tests for the :mod:`pennylane_lightning_gpu.LightningGPU` device (MPI).
"""
# pylint: disable=protected-access,cell-var-from-loop
from mpi4py import MPI
import numpy as np
import pennylane as qml
import pytest

'''
try:
    from pennylane_lightning.lightning_gpu import LGPU_CPP_BINARY_AVAILABLE
    if not LGPU_CPP_BINARY_AVAILABLE:
        raise ImportError("PennyLane-Lightning-GPU binary is not found on this platform")
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "PennyLane-Lightning-GPU binary is not found on this platform. Skipping.",
        allow_module_level=True,
    )
'''
numQubits = 8
SAMPLE_TOL = 0.005

def create_random_init_state(numWires, seed_value=48):
    np.random.seed(seed_value)
    num_elements = 1 << numWires
    init_state = np.random.rand(num_elements) + 1j * np.random.rand(num_elements)
    scale_sum = np.sqrt(np.sum(np.abs(init_state) ** 2))
    init_state = init_state / scale_sum
    return init_state


def apply_operation_gates_qnode_param(operation, par, Wires):
    num_wires = numQubits
    comm = MPI.COMM_WORLD
    commSize = comm.Get_size()
    num_global_wires = commSize.bit_length() - 1
    num_local_wires = num_wires - num_global_wires

    expected_output_cpu = np.zeros(1 << num_wires).astype(np.complex128)
    local_state_vector = np.zeros(1 << num_local_wires).astype(np.complex128)
    local_expected_output_cpu = np.zeros(1 << num_local_wires).astype(np.complex128)

    state_vector = create_random_init_state(num_wires)

    comm.Scatter(state_vector, local_state_vector, root=0)
    comm.Bcast(state_vector, root=0)
    dev_cpu = qml.device("default.qubit", wires=num_wires)

    dev_gpumpi = qml.device(
        "lightning.gpu",
        wires=num_wires,
        mpi = True,
        c_dtype=np.complex128
    )

    def circuit(*params):
        qml.StatePrep(state_vector, wires=range(num_wires))
        operation(*params, wires=Wires)
        return qml.state()

    cpu_qnode = qml.QNode(circuit, dev_cpu)
    expected_output_cpu = cpu_qnode(*par)
    comm.Scatter(expected_output_cpu, local_expected_output_cpu, root=0)

    gpumpi_qnode = qml.QNode(circuit, dev_gpumpi)
    local_state_vector = gpumpi_qnode(*par)

    assert np.allclose(local_expected_output_cpu, local_state_vector, atol=1e-4, rtol=0)

class TestApply:
    @pytest.mark.parametrize("operation", [qml.PhaseShift, qml.RX, qml.RY, qml.RZ])
    @pytest.mark.parametrize("par", [[0.1], [0.2], [0.3]])
    @pytest.mark.parametrize("Wires", [0, numQubits - 1])
    def test_apply_operation_1gatequbit_1param_gate_qnode_param(self, operation, par, Wires):
        apply_operation_gates_qnode_param(operation, par, Wires)