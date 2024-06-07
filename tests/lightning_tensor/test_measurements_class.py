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
else:
    from pennylane_lightning.lightning_tensor._measurements import LightningTensorMeasurements
    from pennylane_lightning.lightning_tensor._state_tensor import LightningStateTensor

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)


# General LightningStateTensor fixture, for any number of wires.
@pytest.fixture(
    params=[np.complex64, np.complex128],
)
def lightning_st(request):
    """Fixture for creating a LightningStateTensor object."""
    return LightningStateTensor(num_wires=5, maxBondDim=128, dtype=request.param)


class TestMeasurementFunction:
    """Tests for the measurement method."""

    def test_initialization(self, lightning_st):
        """Tests for the initialization of the LightningMeasurements class."""
        statetensor = lightning_st
        m = LightningMeasurements(statetensor)

        assert m.dtype == statetensor.dtype

    def test_not_implemented_state_measurements(self, lightning_st):
        """Test than a NotImplementedError is raised if the measurement is not a state measurement."""

        statetensor = lightning_st
        m = LightningMeasurements(statetensor)

        mp = qml.counts(wires=(0, 1))
        with pytest.raises(NotImplementedError):
            m.get_measurement_function(mp)

    def test_not_measure_final_state(self, lightning_st):
        """Test than a NotImplementedError is raised if the measurement is not a state measurement."""

        statetensor = lightning_st
        m = LightningMeasurements(statetensor)

        tape = qml.tape.QuantumScript(
            [qml.RX(0.1, wires=0), qml.Hadamard(1), qml.PauliZ(1)],
            [qml.expval(qml.prod(qml.PauliZ(0), qml.PauliX(1)))],
            shots=1000,
        )

        with pytest.raises(NotImplementedError):
            m.measure_final_state(tape)
