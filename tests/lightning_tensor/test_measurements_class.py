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
Unit tests for measurements class.
"""
import numpy as np
import pennylane as qml
import pytest
from conftest import LightningDevice, device_name  # tested device

if device_name != "lightning.tensor":
    pytest.skip(
        "Skipping tests for the LightningTensorMeasurements class.", allow_module_level=True
    )

from pennylane_lightning.lightning_tensor._measurements import LightningTensorMeasurements
from pennylane_lightning.lightning_tensor._tensornet import LightningTensorNet

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)


# General LightningTensorNet fixture, for any number of wires.
@pytest.fixture(
    params=[np.complex64, np.complex128],
)
def lightning_tn(request):
    """Fixture for creating a LightningTensorNet object."""
    return LightningTensorNet(num_wires=5, max_bond_dim=128, c_dtype=request.param)


class TestMeasurementFunction:
    """Tests for the measurement method."""

    def test_initialization(self, lightning_tn):
        """Tests for the initialization of the LightningTensorMeasurements class."""
        tensornetwork = lightning_tn
        m = LightningTensorMeasurements(tensornetwork)

        assert m.dtype == tensornetwork.dtype

    def test_not_implemented_state_measurements(self, lightning_tn):
        """Test than a NotImplementedError is raised if the measurement is not a state measurement."""

        tensornetwork = lightning_tn
        m = LightningTensorMeasurements(tensornetwork)

        mp = qml.counts(wires=(0, 1))
        with pytest.raises(NotImplementedError):
            m.get_measurement_function(mp)

    def test_not_measure_tensor_network(self, lightning_tn):
        """Test than a NotImplementedError is raised if the measurement is not a state measurement."""

        tensornetwork = lightning_tn
        m = LightningTensorMeasurements(tensornetwork)

        tape = qml.tape.QuantumScript(
            [qml.RX(0.1, wires=0), qml.Hadamard(1), qml.PauliZ(1)],
            [qml.expval(qml.prod(qml.PauliZ(0), qml.PauliX(1)))],
            shots=1000,
        )

        with pytest.raises(NotImplementedError):
            m.measure_tensor_network(tape)

    @pytest.mark.parametrize("n_qubits", range(4, 12, 4))
    @pytest.mark.parametrize("n_targets", list(range(1, 4)) + list(range(4, 12, 4)))
    def test_probs_many_wires(self, n_qubits, n_targets, tol):
        """Test probs measuring many wires of a random quantum state."""
        if n_targets >= n_qubits:
            pytest.skip("Number of targets cannot exceed the number of wires.")

        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit", wires=n_qubits)

        init_state = np.random.rand(2**n_qubits) + 1.0j * np.random.rand(2**n_qubits)
        init_state /= np.linalg.norm(init_state)

        def circuit():
            qml.StatePrep(init_state, wires=range(n_qubits))
            return qml.probs(wires=range(0, n_targets))

        res = qml.QNode(circuit, dev)()
        ref = qml.QNode(circuit, dq)()

        assert np.allclose(res, ref, atol=tol, rtol=0)
