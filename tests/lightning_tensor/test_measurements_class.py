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

import numpy as np
import pennylane as qml
import pytest
from conftest import LightningDevice, device_name  # tested device
from scipy.sparse import csr_matrix, random_array

try:
    from pennylane_lightning.lightning_tensor_ops import MeasurementsC64, MeasurementsC128
except ImportError:
    pass

from pennylane_lightning.lightning_tensor._measurements import LightningMeasurements
from pennylane_lightning.lightning_tensor._state_tensor import LightningStateTensor

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)


# General LightningStateTensor fixture, for any number of wires.
@pytest.fixture(
    scope="function",
    params=[np.complex64, np.complex128],
)
def lightning_st(request):
    def _statetensor(num_wires):
        return LightningStateTensor(num_wires=num_wires, maxBondDim = 128, dtype=request.param)

    return _statetensor


def get_hermitian_matrix(n):
    H = np.random.rand(n, n) + 1.0j * np.random.rand(n, n)
    return H + np.conj(H).T


def get_sparse_hermitian_matrix(n):
    H = random_array((n, n), density=0.15)
    H = H + 1.0j * random_array((n, n), density=0.15)
    return csr_matrix(H + H.conj().T)


class CustomStateMeasurement(qml.measurements.StateMeasurement):
    """Custom state measurement class for testing."""
    def process_state(self, state, wire_order):
        return 1


def test_initialization(lightning_st):
    """Tests for the initialization of the LightningMeasurements class."""
    statetensor = lightning_st(num_wires=5)
    m = LightningMeasurements(statetensor)

    assert m.dtype == statetensor.dtype

class TestGetMeasurementFunction:
    """Tests for the get_measurement_function method."""

    def test_not_implemented_state_measurements(self, lightning_st):
        """Test than a NotImplementedError is raised if the measurement is not a state measurement."""

        statetensor = lightning_st(num_wires=5)
        m = LightningMeasurements(statetensor)

        mp = qml.counts(wires=(0, 1))
        with pytest.raises(NotImplementedError):
            m.get_measurement_function(mp)


class TestMeasurement:
    """Tests for the measurement method."""

    def test_not_measure_final_state(self, lightning_st):
        """Test than a NotImplementedError is raised if the measurement is not a state measurement."""

        statetensor = lightning_st(num_wires=5)
        m = LightningMeasurements(statetensor)

        tape = qml.tape.QuantumScript(
            [qml.RX(0.1, wires=0), qml.Hadamard(1), qml.PauliZ(1)],
            [qml.expval(qml.prod(qml.PauliZ(0), qml.PauliX(1)))],
            shots=1000
        )

        with pytest.raises(NotImplementedError):
            m.measure_final_state(tape)
