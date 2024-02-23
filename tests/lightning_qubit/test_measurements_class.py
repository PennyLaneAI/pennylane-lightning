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
import math

import pennylane as qml
from pennylane.tape import QuantumScript

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


# General LightningStateVector fixture, for any number of wires.
@pytest.fixture(
    scope="function",
    params=[np.complex64, np.complex128],
)
def lightning_sv(request):
    def _statevector(num_wires):
        return LightningStateVector(num_wires=num_wires, dtype=request.param)

    return _statevector


class CustomStateMeasurement(qml.measurements.StateMeasurement):
    def process_state(self, state, wire_order):
        return 1


def test_initialization(lightning_sv):
    """Tests for the initialization of the LightningMeasurements class."""
    statevector = lightning_sv(num_wires=5)
    m = LightningMeasurements(statevector)

    assert m.qubit_state is statevector
    assert m.state is statevector.state_vector
    assert m.dtype == statevector.dtype


class TestGetMeasurementFunction:
    """Tests for the get_measurement_function method."""

    def test_only_support_state_measurements(self, lightning_sv):
        """Test than a NotImplementedError is raised if the measurement is not a state measurement."""

        statevector = lightning_sv(num_wires=5)
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
    def test_state_diagonalizing_gates_measurements(self, lightning_sv, mp):
        """Test that any non-expval measurement calls the state_diagonalizing_gates method"""
        statevector = lightning_sv(num_wires=5)
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
    def test_expval_selected(self, lightning_sv, obs):
        """Test that expval is chosen for a variety of different expectation values."""
        statevector = lightning_sv(num_wires=5)
        m = LightningMeasurements(statevector)
        mp = qml.expval(obs)
        assert m.get_measurement_function(mp) == m.expval


@pytest.mark.parametrize("method_name", ("state_diagonalizing_gates", "measurement"))
class TestStateDiagonalizingGates:
    """Tests for various measurements that go through state_diagonalizing_gates"""

    def expected_entropy_Ising_XX(self, param):
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

    def test_vn_entropy(self, lightning_sv, method_name):
        """Test that state_diagonalizing_gates can handle an arbitrary measurement process."""
        phi = 0.5
        statevector = lightning_sv(num_wires=5)
        statevector.apply_operations([qml.IsingXX(phi, wires=(0, 1))])
        m = LightningMeasurements(statevector)
        measurement = qml.vn_entropy(wires=0)
        result = getattr(m, method_name)(measurement)
        assert qml.math.allclose(result, self.expected_entropy_Ising_XX(phi))

    def test_custom_measurement(self, lightning_sv, method_name):
        """Test that LightningMeasurements can handle a custom state based measurement."""
        statevector = lightning_sv(num_wires=5)
        m = LightningMeasurements(statevector)
        measurement = CustomStateMeasurement()
        result = getattr(m, method_name)(measurement)
        assert result == 1

    def test_measurement_with_diagonalizing_gates(self, lightning_sv, method_name):
        statevector = lightning_sv(num_wires=5)
        m = LightningMeasurements(statevector)
        measurement = qml.probs(op=qml.X(0))
        result = getattr(m, method_name)(measurement)
        assert qml.math.allclose(result, [0.5, 0.5])

    def test_identity_expval(self, lightning_sv, method_name):
        """Test that the expectation value of an identity is always one."""
        statevector = lightning_sv(num_wires=5)
        statevector.apply_operations([qml.Rot(0.5, 4.2, 6.8, wires=4)])
        m = LightningMeasurements(statevector)
        result = getattr(m, method_name)(qml.expval(qml.I(4)))
        assert np.allclose(result, 1.0)

    def test_basis_state_projector_expval(self, lightning_sv, method_name):
        """Test expectation value for a basis state projector."""
        phi = 0.8
        statevector = lightning_sv(num_wires=1)
        statevector.apply_operations([qml.RX(phi, 0)])
        m = LightningMeasurements(statevector)
        result = getattr(m, method_name)(qml.expval(qml.Projector([0], wires=0)))
        assert qml.math.allclose(result, np.cos(phi / 2) ** 2)

    def test_state_vector_projector_expval(self, lightning_sv, method_name):
        """Test expectation value for a state vector projector."""
        phi = -0.6
        statevector = lightning_sv(num_wires=1)
        statevector.apply_operations([qml.RX(phi, 0)])
        m = LightningMeasurements(statevector)
        result = getattr(m, method_name)(qml.expval(qml.Projector([0, 1], wires=0)))
        assert qml.math.allclose(result, np.sin(phi / 2) ** 2)


@pytest.mark.parametrize("method_name", ("expval", "measurement"))
class TestExpval:
    """Tests for the expval function"""

    wires = 2

    @pytest.mark.parametrize(
        "obs, expected",
        [
            [qml.PauliX(0), -0.041892271271228736],
            [qml.PauliX(1), 0.0],
            [qml.PauliY(0), -0.5516350865364075],
            [qml.PauliY(1), 0.0],
            [qml.PauliZ(0), 0.8330328980789793],
            [qml.PauliZ(1), 1.0],
        ],
    )
    def test_expval_qml_tape_wire0(self, obs, expected, tol, lightning_sv, method_name):
        """Test expval with a circuit on wires=[0]"""

        x, y, z = [0.5, 0.3, -0.7]
        statevector = lightning_sv(self.wires)
        statevector.apply_operations(
            [qml.RX(0.4, wires=[0]), qml.Rot(x, y, z, wires=[0]), qml.RY(-0.2, wires=[0])]
        )

        m = LightningMeasurements(statevector)
        result = getattr(m, method_name)(qml.expval(obs))

        assert np.allclose(result, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "obs, expected",
        [
            [qml.PauliX(0), 0.0],
            [qml.PauliX(1), -0.19866933079506122],
            [qml.PauliY(0), -0.3894183423086505],
            [qml.PauliY(1), 0.0],
            [qml.PauliZ(0), 0.9210609940028852],
            [qml.PauliZ(1), 0.9800665778412417],
        ],
    )
    def test_expval_wire01(self, obs, expected, tol, lightning_sv, method_name):
        """Test expval with a circuit on wires=[0,1]"""

        statevector = lightning_sv(self.wires)
        statevector.apply_operations([qml.RX(0.4, wires=[0]), qml.RY(-0.2, wires=[1])])

        m = LightningMeasurements(statevector)
        result = getattr(m, method_name)(qml.expval(obs))

        assert np.allclose(result, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "obs, coeffs, expected",
        [
            ([qml.PauliX(0) @ qml.PauliZ(1)], [1.0], 0.0),
            ([qml.PauliZ(0) @ qml.PauliZ(1)], [1.0], math.cos(0.4) * math.cos(-0.2)),
            (
                [
                    qml.PauliX(0) @ qml.PauliZ(1),
                    qml.Hermitian(
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 3.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 1.0],
                            [0.0, 0.0, 1.0, -2.0],
                        ],
                        wires=[0, 1],
                    ),
                ],
                [0.3, 1.0],
                0.9319728930156066,
            ),
        ],
    )
    def test_expval_hamiltonian(self, obs, coeffs, expected, tol, lightning_sv, method_name):
        """Test expval with Hamiltonian"""
        ham = qml.Hamiltonian(coeffs, obs)

        statevector = lightning_sv(self.wires)
        statevector.apply_operations([qml.RX(0.4, wires=[0]), qml.RY(-0.2, wires=[1])])

        m = LightningMeasurements(statevector)
        result = getattr(m, method_name)(qml.expval(ham))

        assert np.allclose(result, expected, atol=tol, rtol=0)


THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)


@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
class TestExpval:
    """Test expectation value calculations"""

    def test_Identity(self, theta, phi, tol, lightning_sv):
        """Tests applying identities."""

        wires = 3
        ops = [
            qml.Identity(0),
            qml.Identity((0, 1)),
            qml.Identity((1, 2)),
            qml.RX(theta, 0),
            qml.RX(phi, 1),
        ]
        measurements = [qml.expval(qml.PauliZ(0))]
        tape = qml.tape.QuantumScript(ops, measurements)

        statevector = lightning_sv(wires)
        statevector = statevector.get_final_state(tape)
        m = LightningMeasurements(statevector)
        result = m.measure_final_state(tape)
        expected = np.cos(theta)

        assert np.allclose(result, expected, tol)

    def test_identity_expectation(self, theta, phi, tol, lightning_sv):
        """Tests identity expectations."""

        wires = 2
        tape = qml.tape.QuantumScript(
            [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            [qml.expval(qml.Identity(wires=[0])), qml.expval(qml.Identity(wires=[1]))],
        )
        statevector = lightning_sv(wires)
        statevector = statevector.get_final_state(tape)
        m = LightningMeasurements(statevector)
        result = m.measure_final_state(tape)
        expected = 1.0

        assert np.allclose(result, expected, tol)

    def test_multi_wire_identity_expectation(self, theta, phi, tol, lightning_sv):
        """Tests multi-wire identity."""
        wires = 2
        tape = qml.tape.QuantumScript(
            [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            [qml.expval(qml.Identity(wires=[0, 1]))],
        )
        statevector = lightning_sv(wires)
        statevector = statevector.get_final_state(tape)
        m = LightningMeasurements(statevector)
        result = m.measure_final_state(tape)
        expected = 1.0

        assert np.allclose(result, expected, tol)

    @pytest.mark.parametrize(
        "Obs, Op, expected_fn",
        [
            (
                [qml.PauliX(wires=[0]), qml.PauliX(wires=[1])],
                qml.RY,
                lambda theta, phi: np.array([np.sin(theta) * np.sin(phi), np.sin(phi)]),
            ),
            (
                [qml.PauliY(wires=[0]), qml.PauliY(wires=[1])],
                qml.RX,
                lambda theta, phi: np.array([0, -np.cos(theta) * np.sin(phi)]),
            ),
            (
                [qml.PauliZ(wires=[0]), qml.PauliZ(wires=[1])],
                qml.RX,
                lambda theta, phi: np.array([np.cos(theta), np.cos(theta) * np.cos(phi)]),
            ),
            (
                [qml.Hadamard(wires=[0]), qml.Hadamard(wires=[1])],
                qml.RY,
                lambda theta, phi: np.array(
                    [
                        np.sin(theta) * np.sin(phi) + np.cos(theta),
                        np.cos(theta) * np.cos(phi) + np.sin(phi),
                    ]
                )
                / np.sqrt(2),
            ),
        ],
    )
    def test_single_wire_observables_expectation(
        self, Obs, Op, expected_fn, theta, phi, tol, lightning_sv
    ):
        """Test that expectation values for single wire observables are correct"""
        wires = 3
        tape = qml.tape.QuantumScript(
            [Op(theta, wires=[0]), Op(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            [qml.expval(Obs[0]), qml.expval(Obs[1])],
        )
        statevector = lightning_sv(wires)
        statevector = statevector.get_final_state(tape)
        m = LightningMeasurements(statevector)
        result = m.measure_final_state(tape)
        expected = expected_fn(theta, phi)

        assert np.allclose(result, expected, tol)


@pytest.mark.parametrize("phi", PHI)
class TestExpOperatorArithmetic:
    """Test integration with SProd, Prod, and Sum."""

    wires = 2

    def test_sprod(self, phi, lightning_sv, tol):
        """Test the `SProd` class."""
        tape = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0)],
            [qml.expval(qml.s_prod(0.5, qml.PauliZ(0)))],
        )
        statevector = lightning_sv(self.wires)
        statevector = statevector.get_final_state(tape)
        m = LightningMeasurements(statevector)
        result = m.measure_final_state(tape)
        expected = 0.5 * np.cos(phi)

        assert np.allclose(result, expected, tol)

    def test_prod(self, phi, lightning_sv, tol):
        """Test the `Prod` class."""
        tape = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0), qml.Hadamard(1), qml.PauliZ(1)],
            [qml.expval(qml.prod(qml.PauliZ(0), qml.PauliX(1)))],
        )
        statevector = lightning_sv(self.wires)
        statevector = statevector.get_final_state(tape)
        m = LightningMeasurements(statevector)
        result = m.measure_final_state(tape)
        expected = -np.cos(phi)

        assert np.allclose(result, expected, tol)

    @pytest.mark.parametrize("theta", THETA)
    def test_sum(self, phi, theta, lightning_sv, tol):
        """Test the `Sum` class."""
        tape = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0), qml.RY(theta, wires=1)],
            [qml.expval(qml.sum(qml.PauliZ(0), qml.PauliX(1)))],
        )
        statevector = lightning_sv(self.wires)
        statevector = statevector.get_final_state(tape)
        m = LightningMeasurements(statevector)
        result = m.measure_final_state(tape)
        expected = np.cos(phi) + np.sin(theta)

        assert np.allclose(result, expected, tol)
