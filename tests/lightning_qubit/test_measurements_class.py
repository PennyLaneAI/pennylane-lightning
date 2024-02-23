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

import pytest
from conftest import LightningDevice  # tested device

import numpy as np

import pennylane as qml

try:
    from pennylane_lightning.lightning_qubit_ops import (
        MeasurementsC64,
        MeasurementsC128,
    )
except ImportError:
    pass

from pennylane_lightning.lightning_qubit._state_vector import LightningStateVector
from pennylane_lightning.lightning_qubit._measurements import LightningMeasurements

from pennylane_lightning.lightning_qubit import LightningQubit

if not LightningQubit._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

if LightningDevice != LightningQubit:
    pytest.skip("Exclusive tests for lightning.qubit. Skipping.", allow_module_level=True)

class CustomStateMeasurement(qml.measurements.StateMeasurement):
    def process_state(self, state, wire_order):
        return 1


def test_initialization_complex64():
    """Tests for the initialization of the LightningMeasurements class with np.complex64."""
    statevector = LightningStateVector(num_wires=5, dtype=np.complex64)
    m = LightningMeasurements(statevector)

    assert m._qubit_state is statevector
    assert m.state is statevector.state_vector
    assert m.dtype == np.complex64
    assert isinstance(m.measurement_lightning, MeasurementsC64)


def test_initialization_complex128():
    """Tests for the initialization of the LightningMeasurements class."""
    statevector = LightningStateVector(num_wires=5, dtype=np.complex128)
    m = LightningMeasurements(statevector)

    assert m._qubit_state is statevector
    assert m.state is statevector.state_vector
    assert m.dtype == np.complex128
    assert isinstance(m.measurement_lightning, MeasurementsC128)


class TestGetMeasurementFunction:
    """Tests for the get_measurement_function method."""

    def test_only_support_state_measurements(self):
        """Test than a NotImplementedError is raised if the measurement is not a state measurement."""

        statevector = LightningStateVector(num_wires=5, dtype=np.complex64)
        m = LightningMeasurements(statevector)

        mp = qml.counts(wires=(0, 1))
        with pytest.raises(NotImplementedError):
            m.get_measurement_function(mp)

    @pytest.mark.parametrize(
        "mp",
        (
            qml.probs(wires=0),
            qml.var(qml.Z(0)),
            qml.vn_entropy(wires=0),
            CustomStateMeasurement(),
            qml.expval(qml.Identity(0)),
            qml.expval(qml.Projector([1, 0], wires=(0, 1))),
        ),
    )
    def test_state_diagonalizing_gates_measurements(self, mp):
        """Test that any non-expval measurement is"""
        statevector = LightningStateVector(num_wires=5, dtype=np.complex64)
        m = LightningMeasurements(statevector)

        assert m.get_measurement_function(mp) == m.state_diagonalizing_gates

    @pytest.mark.parametrize(
        "obs",
        (
            qml.X(0),
            qml.Y(0),
            qml.Z(0),
            qml.sum(qml.X(0), qml.Y(0)),
            qml.prod(qml.X(0), qml.Y(1)),
            qml.s_prod(2.0, qml.X(0)),
            qml.Hamiltonian([1.0, 2.0], [qml.X(0), qml.Y(0)]),
            qml.Hermitian(np.eye(2), wires=0),
            qml.SparseHamiltonian(qml.X.compute_sparse_matrix(), wires=0),
        ),
    )
    def test_expval_selected(self, obs):
        """Test that expval is chosen for a variety of different expectation values."""
        statevector = LightningStateVector(num_wires=5, dtype=np.complex64)
        m = LightningMeasurements(statevector)
        mp = qml.expval(obs)
        assert m.get_measurement_function(mp) == m.expval


def expected_entropy_ising_xx(param):
    """
    Return the analytical entropy for the IsingXX.
    """
    eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
    eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
    eigs = [eig_1, eig_2]
    eigs = [eig for eig in eigs if eig > 0]

    expected_entropy = eigs * np.log(eigs)

    expected_entropy = -np.sum(expected_entropy)
    return expected_entropy


@pytest.mark.parametrize("method_name", ("state_diagonalizing_gates", "measurement"))
class TestStateDiagonalizingGates:
    """Tests for various measurements that go through state_diagonalizing_gates"""

    def test_vn_entropy(self, method_name):
        """Test that state_diagonalizing_gates can handle an arbitrary measurement process."""
        phi = 0.5
        statevector = LightningStateVector(num_wires=5, dtype=np.complex64)
        statevector.apply_operations([qml.IsingXX(phi, wires=(0, 1))])
        m = LightningMeasurements(statevector)
        measurement = qml.vn_entropy(wires=0)
        result = getattr(m, method_name)(measurement)
        assert qml.math.allclose(result, expected_entropy_ising_xx(phi))

    def test_custom_measurement(self, method_name):
        """Test that LightningMeasurements can handle a custom state based measurement."""
        statevector = LightningStateVector(num_wires=5, dtype=np.complex64)
        m = LightningMeasurements(statevector)
        measurement = CustomStateMeasurement()
        result = getattr(m, method_name)(measurement)
        assert result == 1

    def test_measurement_with_diagonalizing_gates(self, method_name):
        statevector = LightningStateVector(num_wires=5, dtype=np.complex64)
        m = LightningMeasurements(statevector)
        measurement = qml.probs(op=qml.X(0))
        result = getattr(m, method_name)(measurement)
        assert qml.math.allclose(result, [0.5, 0.5])

    def test_identity_expval(self, method_name):
        """Test that the expectation value of an identity is always one."""
        statevector = LightningStateVector(num_wires=5, dtype=np.complex64)
        statevector.apply_operations([qml.Rot(0.5, 4.2, 6.8, wires=4)])
        m = LightningMeasurements(statevector)
        result = getattr(m, method_name)(qml.expval(qml.I(4)))
        assert np.allclose(result, 1.0)

    def test_basis_state_projector_expval(self, method_name):
        """Test expectation value for a basis state projector."""
        phi = 0.8
        statevector = LightningStateVector(num_wires=1, dtype=np.complex64)
        statevector.apply_operations([qml.RX(phi, 0)])
        m = LightningMeasurements(statevector)
        result = getattr(m, method_name)(qml.expval(qml.Projector([0], wires=0)))
        assert qml.math.allclose(result, np.cos(phi / 2) ** 2)

    def test_state_vector_projector_expval(self, method_name):
        """Test expectation value for a state vector projector."""
        phi = -0.6
        statevector = LightningStateVector(num_wires=1, dtype=np.complex64)
        statevector.apply_operations([qml.RX(phi, 0)])
        m = LightningMeasurements(statevector)
        result = getattr(m, method_name)(qml.expval(qml.Projector([0, 1], wires=0)))
        assert qml.math.allclose(result, np.sin(phi / 2) ** 2)
