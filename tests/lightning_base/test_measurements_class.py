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

import itertools
import math
from typing import Sequence

import numpy as np
import pennylane as qp
import pytest
from conftest import (  # tested device
    PHI,
    THETA,
    LightningDevice,
    LightningMeasurements,
    LightningStateVector,
    device_name,
    get_hermitian_matrix,
    get_random_normalized_state,
    get_sparse_hermitian_matrix,
    validate_counts,
    validate_others,
    validate_samples,
)
from pennylane.devices import DefaultQubit
from pennylane.measurements import VarianceMP

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


class CustomStateMeasurement(qp.measurements.StateMeasurement):
    def process_state(self, state, wire_order):
        return 1


# Observables not supported in lightning.tensor
def obs_not_supported_in_ltensor(obs):
    if device_name == "lightning.tensor":
        if isinstance(obs, qp.Projector) or isinstance(obs, qp.SparseHamiltonian):
            return True
        if isinstance(obs, qp.Hamiltonian):
            return any([obs_not_supported_in_ltensor(o) for o in obs])
        if isinstance(obs, qp.Hermitian) and len(obs.wires) > 1:
            return True
        if isinstance(obs, list) and all([isinstance(o, int) for o in obs]):  # out of order probs
            return obs != sorted(obs)
        return False
    else:
        return False


def get_final_state(statevector, tape):
    if device_name == "lightning.tensor":
        return statevector.set_tensor_network(tape)
    return statevector.get_final_state(tape)


def measure_final_state(m, tape):
    if device_name == "lightning.tensor":
        return m.measure_tensor_network(tape)
    return m.measure_final_state(tape)


def test_initialization(lightning_sv):
    """Tests for the initialization of the LightningMeasurements class."""
    statevector = lightning_sv(num_wires=5)
    m = LightningMeasurements(statevector)

    if device_name == "lightning.tensor":
        assert m.dtype == statevector.dtype
    else:
        assert m.qubit_state is statevector
        assert m.dtype == statevector.dtype


class TestGetMeasurementFunction:
    """Tests for the get_measurement_function method."""

    def test_only_support_state_measurements(self, lightning_sv):
        """Test than a NotImplementedError is raised if the measurement is not a state measurement."""

        statevector = lightning_sv(num_wires=5)
        m = LightningMeasurements(statevector)

        mp = qp.counts(wires=(0, 1))
        with pytest.raises(NotImplementedError):
            m.get_measurement_function(mp)

    @pytest.mark.parametrize(
        "mp",
        (
            qp.vn_entropy(wires=0),
            CustomStateMeasurement(),
            qp.expval(qp.Identity(0)),
            qp.expval(qp.Projector([1, 0], wires=(0, 1))),
            qp.var(qp.Identity(0)),
            qp.var(qp.Projector([1, 0], wires=(0, 1))),
        ),
    )
    def test_state_diagonalizing_gates_measurements(self, lightning_sv, mp):
        """Test that any non-expval measurement calls the state_diagonalizing_gates method"""
        if obs_not_supported_in_ltensor(mp.obs):
            pytest.skip("Observable not supported in lightning.tensor.")

        statevector = lightning_sv(num_wires=5)
        m = LightningMeasurements(statevector)

        assert m.get_measurement_function(mp) == m.state_diagonalizing_gates

    @pytest.mark.parametrize(
        "obs",
        (
            qp.PauliX(0),
            qp.PauliY(0),
            qp.PauliZ(0),
            qp.sum(qp.PauliX(0), qp.PauliY(0)),
            qp.prod(qp.PauliX(0), qp.PauliY(1)),
            qp.s_prod(2.0, qp.PauliX(0)),
            qp.Hamiltonian([1.0, 2.0], [qp.PauliX(0), qp.PauliY(0)]),
            qp.Hermitian(np.eye(2), wires=0),
            qp.SparseHamiltonian(qp.PauliX.compute_sparse_matrix(), wires=0),
        ),
    )
    def test_expval_selected(self, lightning_sv, obs):
        """Test that expval is chosen for a variety of different expectation values."""
        if obs_not_supported_in_ltensor(obs):
            pytest.skip("Observable not supported in lightning.tensor.")

        statevector = lightning_sv(num_wires=5)
        m = LightningMeasurements(statevector)
        mp = qp.expval(obs)
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
        statevector.apply_operations([qp.IsingXX(phi, wires=(0, 1))])
        m = LightningMeasurements(statevector)
        measurement = qp.vn_entropy(wires=0)
        result = getattr(m, method_name)(measurement)
        assert qp.math.allclose(result, self.expected_entropy_Ising_XX(phi))

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
        measurement = qp.probs(op=qp.PauliX(0))
        result = getattr(m, method_name)(measurement)
        assert qp.math.allclose(result, [0.5, 0.5])

    def test_identity_expval(self, lightning_sv, method_name):
        """Test that the expectation value of an identity is always one."""
        statevector = lightning_sv(num_wires=5)
        statevector.apply_operations([qp.Rot(0.5, 4.2, 6.8, wires=4)])
        m = LightningMeasurements(statevector)
        result = getattr(m, method_name)(qp.expval(qp.I(4)))
        assert np.allclose(result, 1.0)

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support a single-wire circuit.",
    )
    def test_basis_state_projector_expval(self, lightning_sv, method_name):
        """Test expectation value for a basis state projector."""
        phi = 0.8
        statevector = lightning_sv(num_wires=1)
        statevector.apply_operations([qp.RX(phi, 0)])
        m = LightningMeasurements(statevector)
        result = getattr(m, method_name)(qp.expval(qp.Projector([0], wires=0)))
        assert qp.math.allclose(result, np.cos(phi / 2) ** 2)

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support a single-wire circuit.",
    )
    def test_state_vector_projector_expval(self, lightning_sv, method_name):
        """Test expectation value for a state vector projector."""
        phi = -0.6
        statevector = lightning_sv(num_wires=1)
        statevector.apply_operations([qp.RX(phi, 0)])
        m = LightningMeasurements(statevector)
        result = getattr(m, method_name)(qp.expval(qp.Projector([0, 1], wires=0)))
        assert qp.math.allclose(result, np.sin(phi / 2) ** 2)


@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
class TestExpval:
    """Test expectation value calculations."""

    def test_identity(self, theta, phi, tol, lightning_sv):
        """Tests applying identities."""

        wires = 3
        ops = [
            qp.Identity(0),
            qp.Identity((0, 1)),
            qp.Identity((1, 2)),
            qp.RX(theta, 0),
            qp.RX(phi, 1),
        ]
        measurements = [qp.expval(qp.PauliZ(0))]
        tape = qp.tape.QuantumScript(ops, measurements)

        statevector = lightning_sv(wires)
        statevector = get_final_state(statevector, tape)
        m = LightningMeasurements(statevector)
        result = measure_final_state(m, tape)
        expected = np.cos(theta)

        assert np.allclose(result, expected, tol)

    def test_identity_expectation(self, theta, phi, tol, lightning_sv):
        """Tests identity expectations."""

        wires = 2
        tape = qp.tape.QuantumScript(
            [qp.RX(theta, wires=[0]), qp.RX(phi, wires=[1]), qp.CNOT(wires=[0, 1])],
            [qp.expval(qp.Identity(wires=[0])), qp.expval(qp.Identity(wires=[1]))],
        )
        statevector = lightning_sv(wires)
        statevector = get_final_state(statevector, tape)
        m = LightningMeasurements(statevector)
        result = measure_final_state(m, tape)
        expected = 1.0

        assert np.allclose(result, expected, tol)

    def test_multi_wire_identity_expectation(self, theta, phi, tol, lightning_sv):
        """Tests multi-wire identity."""
        wires = 2
        tape = qp.tape.QuantumScript(
            [qp.RX(theta, wires=[0]), qp.RX(phi, wires=[1]), qp.CNOT(wires=[0, 1])],
            [qp.expval(qp.Identity(wires=[0, 1]))],
        )
        statevector = lightning_sv(wires)
        statevector = get_final_state(statevector, tape)
        m = LightningMeasurements(statevector)
        result = measure_final_state(m, tape)
        expected = 1.0

        assert np.allclose(result, expected, tol)

    @pytest.mark.parametrize(
        "Obs, Op, expected_fn",
        [
            (
                [qp.PauliX(wires=[0]), qp.PauliX(wires=[1])],
                qp.RY,
                lambda theta, phi: np.array([np.sin(theta) * np.sin(phi), np.sin(phi)]),
            ),
            (
                [qp.PauliY(wires=[0]), qp.PauliY(wires=[1])],
                qp.RX,
                lambda theta, phi: np.array([0, -np.cos(theta) * np.sin(phi)]),
            ),
            (
                [qp.PauliZ(wires=[0]), qp.PauliZ(wires=[1])],
                qp.RX,
                lambda theta, phi: np.array([np.cos(theta), np.cos(theta) * np.cos(phi)]),
            ),
            (
                [qp.Hadamard(wires=[0]), qp.Hadamard(wires=[1])],
                qp.RY,
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
        tape = qp.tape.QuantumScript(
            [Op(theta, wires=[0]), Op(phi, wires=[1]), qp.CNOT(wires=[0, 1])],
            [qp.expval(Obs[0]), qp.expval(Obs[1])],
        )
        statevector = lightning_sv(wires)
        statevector = get_final_state(statevector, tape)
        m = LightningMeasurements(statevector)
        result = measure_final_state(m, tape)
        expected = expected_fn(theta, phi)

        assert np.allclose(result, expected, tol)


@pytest.mark.parametrize("method_name", ("expval", "measurement"))
class TestExpvalHamiltonian:
    """Tests expval for Hamiltonians"""

    wires = 2

    @pytest.mark.parametrize(
        "obs, coeffs, expected",
        [
            ([qp.PauliX(0) @ qp.PauliZ(1)], [1.0], 0.0),
            ([qp.PauliZ(0) @ qp.PauliZ(1)], [1.0], math.cos(0.4) * math.cos(-0.2)),
            (
                [
                    qp.PauliX(0) @ qp.PauliZ(1),
                    qp.Hermitian(
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

        if any(isinstance(o, qp.Hermitian) for o in obs) and device_name == "lightning.tensor":
            pytest.skip("Hermitian with 1+ wires target not supported in lightning.tensor.")

        ham = qp.Hamiltonian(coeffs, obs)

        statevector = lightning_sv(self.wires)
        statevector.apply_operations([qp.RX(0.4, wires=[0]), qp.RY(-0.2, wires=[1])])

        m = LightningMeasurements(statevector)
        result = getattr(m, method_name)(qp.expval(ham))

        assert np.allclose(result, expected, atol=tol, rtol=0)


@pytest.mark.skipif(
    device_name not in ("lightning.kokkos", "lightning.gpu"),
    reason="Specialized expectation value for Pauli Sentences only available for lightning.kokkos and lightning.gpu",
)
class TestExpvalPauliSentence:
    """Tests expval for Pauli Sentences"""

    wires = 2

    @pytest.mark.parametrize(
        "obs",
        [
            qp.Hadamard(0),
            qp.Hermitian(np.eye(2), wires=0),
        ],
    )
    def test_no_paulirep(self, obs, lightning_sv):
        """Test _expval_pauli_sentence method with obs with no pauli_rep"""

        statevector = lightning_sv(self.wires)
        statevector.apply_operations([qp.RX(0.4, wires=[0]), qp.RY(-0.2, wires=[1])])

        m = LightningMeasurements(statevector)
        with pytest.raises(AttributeError, match="object has no attribute"):
            m._expval_pauli_sentence(qp.expval(obs))

    @pytest.mark.parametrize(
        "obs, expected",
        (
            [qp.Identity(1), 1.0],
            [qp.PauliX(1), -0.19866933079506138],
            [qp.PauliY(0), -0.3894183423086505],
            [qp.PauliZ(0), 0.9210609940028852],
            [0.1 * qp.PauliZ(0) + 0.2 * qp.PauliY(0), 0.01422243094],
            [qp.sum(qp.PauliX(1), qp.PauliY(0)), -0.5880876731037115],
            [qp.prod(qp.PauliX(1), qp.PauliY(0)), 0.07736548146578165],
            [qp.s_prod(2.0, qp.PauliX(1)), -0.39733866159012277],
            [
                qp.sum(
                    qp.prod(qp.PauliX(1), qp.PauliY(0)),
                    qp.s_prod(-1.0, qp.prod(qp.PauliY(0), qp.PauliX(1))),
                ),
                0.0,
            ],
            [qp.Hamiltonian([1.0, 2.0], [qp.PauliX(1), qp.PauliY(0)]), -0.9775060154123624],
            [
                qp.sum(
                    qp.Hamiltonian([1.0, 2.0], [qp.PauliX(1), qp.PauliY(0)]),
                    qp.prod(qp.PauliX(1), qp.PauliY(0)),
                ),
                -0.9001405339,
            ],
        ),
    )
    def test_pauli_sentence(self, obs, expected, tol, lightning_sv):
        """Test _expval_pauli_sentence method with obs"""

        statevector = lightning_sv(self.wires)
        statevector.apply_operations([qp.RX(0.4, wires=[0]), qp.RY(-0.2, wires=[1])])

        m = LightningMeasurements(statevector)
        result = m._expval_pauli_sentence(qp.expval(obs))

        assert np.allclose(result, expected, atol=tol, rtol=0)


@pytest.mark.skipif(
    device_name == "lightning.tensor",
    reason="lightning.tensor does not support sparse observables.",
)
class TestSparseExpval:
    """Tests for the expval function with sparse observables."""

    wires = 2

    @pytest.mark.parametrize(
        "ham_terms, expected",
        [
            [qp.PauliX(0) @ qp.Identity(1), 0.00000000000000000],
            [qp.Identity(0) @ qp.PauliX(1), -0.19866933079506122],
            [qp.PauliY(0) @ qp.Identity(1), -0.38941834230865050],
            [qp.Identity(0) @ qp.PauliY(1), 0.00000000000000000],
            [qp.PauliZ(0) @ qp.Identity(1), 0.92106099400288520],
            [qp.Identity(0) @ qp.PauliZ(1), 0.98006657784124170],
        ],
    )
    @pytest.mark.parametrize(
        "obs",
        [
            qp.SparseHamiltonian,
        ],
    )
    def test_sparse_pauli_words(self, obs, ham_terms, expected, tol, lightning_sv):
        """Test expval of some simple sparse observables"""

        ops = [qp.RX(0.4, wires=[0]), qp.RY(-0.2, wires=[1])]
        sparse_matrix = qp.Hamiltonian([1], [ham_terms]).sparse_matrix()
        measurements = [qp.expval(obs(sparse_matrix, wires=[0, 1]))]
        tape = qp.tape.QuantumScript(ops, measurements)

        statevector = lightning_sv(self.wires)
        statevector = get_final_state(statevector, tape)
        m = LightningMeasurements(statevector)
        result = measure_final_state(m, tape)

        assert np.allclose(result, expected, tol)


@pytest.mark.skipif(
    device_name == "lightning.tensor",
    reason="lightning.tensor does not support sparse observables.",
)
class TestSparseMeasurements:
    """Tests all sparse measurements"""

    sparse_observables = [
        qp.SparseHamiltonian,
    ]

    @staticmethod
    def calculate_reference(tape, lightning_sv):
        # Using the dense version as a reference.
        new_meas = []
        for m in tape.measurements:
            new_meas.append(
                m.__class__(qp.Hermitian(m.obs.sparse_matrix().toarray(), wires=m.obs.wires))
            )

        tape = qp.tape.QuantumScript(tape.operations, new_meas)
        statevector = lightning_sv(tape.num_wires)
        statevector = get_final_state(statevector, tape)
        m = LightningMeasurements(statevector)
        return measure_final_state(m, tape)

    @pytest.mark.local_salt(42)
    @pytest.mark.parametrize("measurement", [qp.expval, qp.var])
    @pytest.mark.parametrize(
        "observable",
        sparse_observables,
    )
    def test_single_return_value(self, measurement, observable, lightning_sv, tol, seed):
        n_qubits = 4
        observable = observable(get_sparse_hermitian_matrix(2**n_qubits), wires=range(n_qubits))

        n_layers = 1
        rng = np.random.default_rng(seed)
        weights = rng.random((n_layers, n_qubits, 3))
        ops = [qp.Hadamard(i) for i in range(n_qubits)] + [
            qp.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        ]
        measurements = (
            [measurement(wires=observable)]
            if isinstance(observable, list)
            else [measurement(op=observable)]
        )
        tape = qp.tape.QuantumScript(ops, measurements)

        statevector = lightning_sv(n_qubits)
        statevector = get_final_state(statevector, tape)
        m = LightningMeasurements(statevector)

        result = measure_final_state(m, tape)

        expected = self.calculate_reference(tape, lightning_sv)

        assert np.allclose(
            result,
            expected,
            max(tol, 1.0e-4),
            1e-6 if statevector.dtype == np.complex64 else 1e-8,
        )

    @pytest.mark.local_salt(42)
    @pytest.mark.parametrize("measurement", [qp.expval, qp.var])
    @pytest.mark.parametrize(
        "obs0_",
        sparse_observables,
    )
    @pytest.mark.parametrize(
        "obs1_",
        sparse_observables,
    )
    def test_double_return_value(self, measurement, obs0_, obs1_, lightning_sv, tol, seed):
        n_qubits = 4
        obs0_ = obs0_(get_sparse_hermitian_matrix(2**4), wires=range(n_qubits))
        obs1_ = obs1_(get_sparse_hermitian_matrix(2**4), wires=range(n_qubits))

        n_layers = 1
        rng = np.random.default_rng(seed)
        weights = rng.random((n_layers, n_qubits, 3))
        ops = [qp.Hadamard(i) for i in range(n_qubits)] + [
            qp.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        ]
        measurements = [measurement(op=obs0_), measurement(op=obs1_)]
        tape = qp.tape.QuantumScript(ops, measurements)

        statevector = lightning_sv(n_qubits)
        statevector = get_final_state(statevector, tape)
        m = LightningMeasurements(statevector)

        result = measure_final_state(m, tape)

        expected = self.calculate_reference(tape, lightning_sv)
        if len(expected) == 1:
            expected = expected[0]

        assert isinstance(result, Sequence)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert np.allclose(r, e, atol=tol, rtol=tol)


class TestMeasurements:
    """Tests all measurements"""

    @staticmethod
    def calculate_reference(tape, lightning_sv):
        use_default = True
        new_meas = []
        for m in tape.measurements:
            # NotImplementedError in DefaultQubit
            # We therefore validate against `qp.Hermitian`
            if isinstance(m, VarianceMP) and isinstance(m.obs, (qp.SparseHamiltonian)):
                use_default = False
                new_meas.append(m.__class__(qp.Hermitian(qp.matrix(m.obs), wires=m.obs.wires)))
                continue
            new_meas.append(m)
        if use_default:
            dev = DefaultQubit()
            program, _ = dev.preprocess()
            tapes, transf_fn = program([tape])
            results = dev.execute(tapes)
            return transf_fn(results)

        tape = qp.tape.QuantumScript(tape.operations, new_meas)
        statevector = lightning_sv(tape.num_wires)
        statevector = get_final_state(statevector, tape)
        m = LightningMeasurements(statevector)
        return measure_final_state(m, tape)

    @pytest.mark.local_salt(42)
    @pytest.mark.parametrize("shots", [None, 500_000, [500_000, 500_000]])
    @pytest.mark.parametrize("measurement", [qp.expval, qp.probs, qp.var])
    @pytest.mark.parametrize(
        "observable",
        (
            [0],
            [1, 2],
            [1, 0],
            qp.PauliX(0),
            qp.PauliY(1),
            qp.PauliZ(2),
            qp.sum(qp.PauliX(0), qp.PauliY(0)),
            qp.prod(qp.PauliX(0), qp.PauliY(1)),
            qp.s_prod(2.0, qp.PauliX(0)),
            qp.Hermitian(get_hermitian_matrix(2**2), wires=[0, 1]),
            qp.Hermitian(get_hermitian_matrix(2**2), wires=[2, 3]),
            qp.Hamiltonian(
                [1.0, 2.0, 3.0],
                [qp.PauliX(0), qp.PauliY(1), qp.PauliZ(2) @ qp.PauliZ(3)],
            ),
        ),
    )
    def test_single_return_value(self, shots, measurement, observable, lightning_sv, tol, seed):
        if obs_not_supported_in_ltensor(observable):
            pytest.skip("Observable not supported in lightning.tensor.")

        if measurement is qp.probs and isinstance(
            observable,
            (
                qp.ops.Sum,
                qp.ops.SProd,
                qp.ops.Prod,
            ),
        ):
            pytest.skip(
                f"Observable of type {type(observable).__name__} is not supported for rotating probabilities."
            )

        if measurement is not qp.probs and isinstance(observable, list):
            pytest.skip(
                f"Measurement of type {type(measurement).__name__} does not have a keyword argument 'wires'."
            )
        if shots != None and measurement is qp.expval:
            # Increase the number of shots
            if isinstance(shots, int):
                shots = 1_000_000
            else:
                shots = [1_000_000, 1_000_000]

        n_qubits = 4
        n_layers = 1
        rng = np.random.default_rng(seed)
        weights = rng.random((n_layers, n_qubits, 3))
        ops = [qp.Hadamard(i) for i in range(n_qubits)]
        if device_name != "lightning.tensor":
            ops += [qp.StronglyEntanglingLayers(weights, wires=range(n_qubits))]
        measurements = (
            [measurement(wires=observable)]
            if isinstance(observable, list)
            else [measurement(op=observable)]
        )
        tape = qp.tape.QuantumScript(ops, measurements, shots=shots)

        statevector = lightning_sv(n_qubits, seed=seed)
        statevector = get_final_state(statevector, tape)
        m = LightningMeasurements(statevector)

        skip_list = (qp.ops.Sum,)
        do_skip = measurement is qp.var and isinstance(observable, skip_list)
        do_skip = do_skip and shots is not None
        if do_skip:
            with pytest.raises(TypeError):
                _ = measure_final_state(m, tape)
            return
        else:
            result = measure_final_state(m, tape)

        expected = self.calculate_reference(tape, lightning_sv)

        # a few tests may fail in single precision, and hence we increase the tolerance
        if shots is None:
            assert np.allclose(
                result,
                expected,
                max(tol, 1.0e-4),
                1e-6 if statevector.dtype == np.complex64 else 1e-8,
            )
        else:
            # TODO Set better atol and rtol
            dtol = max(tol, 2.0e-2)
            # allclose -> absolute(a - b) <= (atol + rtol * absolute(b))
            assert np.allclose(result, expected, rtol=dtol, atol=dtol)

    @pytest.mark.local_salt(42)
    @pytest.mark.parametrize("shots", [None, 400_000, (400_000, 400_000)])
    @pytest.mark.parametrize("measurement", [qp.expval, qp.probs, qp.var])
    @pytest.mark.parametrize(
        "obs0_",
        (
            qp.PauliX(0),
            qp.PauliY(1),
            qp.PauliZ(2),
            qp.sum(qp.PauliX(0), qp.PauliY(0)),
            qp.prod(qp.PauliX(0), qp.PauliY(1)),
            qp.s_prod(2.0, qp.PauliX(0)),
            qp.Hermitian(get_hermitian_matrix(2), wires=[0]),
            qp.Hermitian(get_hermitian_matrix(2**2), wires=[2, 3]),
            qp.Hamiltonian(
                [1.0, 2.0, 3.0],
                [qp.PauliX(0), qp.PauliY(1), qp.PauliZ(2) @ qp.PauliZ(3)],
            ),
        ),
    )
    @pytest.mark.parametrize(
        "obs1_",
        (
            qp.PauliX(0),
            qp.PauliY(1),
            qp.PauliZ(2),
            qp.sum(qp.PauliX(0), qp.PauliY(0)),
            qp.prod(qp.PauliX(0), qp.PauliY(1)),
            qp.s_prod(2.0, qp.PauliX(0)),
            qp.Hermitian(get_hermitian_matrix(2), wires=[0]),
            qp.Hermitian(get_hermitian_matrix(2**2), wires=[2, 3]),
            qp.Hamiltonian(
                [1.0, 2.0, 3.0],
                [qp.PauliX(0), qp.PauliY(1), qp.PauliZ(2) @ qp.PauliZ(3)],
            ),
        ),
    )
    def test_double_return_value(self, shots, measurement, obs0_, obs1_, lightning_sv, tol, seed):
        if obs_not_supported_in_ltensor(obs0_) or obs_not_supported_in_ltensor(obs1_):
            pytest.skip("Observable not supported in lightning.tensor.")

        skip_list = (
            qp.ops.Sum,
            qp.ops.SProd,
            qp.ops.Prod,
            qp.Hamiltonian,
        )
        if measurement is qp.probs and (
            isinstance(obs0_, skip_list) or isinstance(obs1_, skip_list)
        ):
            pytest.skip(
                f"Observable of type {type(obs0_).__name__} is not supported for rotating probabilities."
            )

        if shots != None and measurement is qp.expval:
            # Increase the number of shots
            if isinstance(shots, int):
                shots = 1_000_000
            else:
                shots = [1_000_000, 1_000_000]

        n_qubits = 4
        n_layers = 1
        rng = np.random.default_rng(seed)
        weights = rng.random((n_layers, n_qubits, 3))
        ops = [qp.Hadamard(i) for i in range(n_qubits)]
        if device_name != "lightning.tensor":
            ops += [qp.StronglyEntanglingLayers(weights, wires=range(n_qubits))]
        measurements = [measurement(op=obs0_), measurement(op=obs1_)]
        tape = qp.tape.QuantumScript(ops, measurements, shots=shots)

        statevector = lightning_sv(n_qubits, seed=seed)
        statevector = get_final_state(statevector, tape)
        m = LightningMeasurements(statevector)

        skip_list = (
            qp.ops.Sum,
            qp.Hamiltonian,
        )
        do_skip = measurement is qp.var and (
            isinstance(obs0_, skip_list) or isinstance(obs1_, skip_list)
        )
        do_skip = do_skip and shots is not None
        if do_skip:
            with pytest.raises(TypeError):
                _ = measure_final_state(m, tape)
            return
        else:
            result = measure_final_state(m, tape)

        expected = self.calculate_reference(tape, lightning_sv)
        if len(expected) == 1:
            expected = expected[0]

        assert isinstance(result, Sequence)
        assert len(result) == len(expected)

        # a few tests may fail in single precision, and hence we increase the tolerance
        dtol = tol if shots is None else max(tol, 2.0e-2)

        # var has larger error
        if measurement is qp.var:
            dtol = max(dtol, 1.0e-4)

        if device_name == "lightning.tensor" and statevector.dtype == np.complex64:
            dtol = max(dtol, 1.0e-4)

        # TODO Might need to update atol/rtol
        for r, e in zip(result, expected):
            if isinstance(shots, tuple) and isinstance(r[0], np.ndarray):
                r = np.concatenate(r)
                e = np.concatenate(e)
            # allclose -> absolute(r - e) <= (atol + rtol * absolute(e))
            assert np.allclose(r, e, atol=dtol, rtol=dtol)

    @pytest.mark.skipif(
        device_name in ("lightning.tensor"),
        reason=f"{device_name} does not support out of order probs.",
    )
    @pytest.mark.parametrize(
        "cases",
        [
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]],
        ],
    )
    def test_probs_tape_unordered_wires(self, cases, tol):
        """Test probs with a circuit on wires=[0] fails for out-of-order wires passed to probs."""

        x, y, z = [0.5, 0.3, -0.7]
        dev = qp.device(device_name, wires=cases[1])

        def circuit():
            qp.RX(0.4, wires=[0])
            qp.Rot(x, y, z, wires=[0])
            qp.RY(-0.2, wires=[0])
            return qp.probs(wires=cases[0])

        expected = qp.QNode(circuit, qp.device("default.qubit", wires=cases[1]))()
        results = qp.QNode(circuit, dev)()
        assert np.allclose(expected, results, tol)

    @pytest.mark.skipif(
        device_name in ("lightning.tensor"),
        reason=f"{device_name} does not support seeding device.",
    )
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_seeded_measurement_rngstate(self, dtype):
        """Test that seeded measurement uses identical rng state"""

        n_qubits = 4

        rng_1 = np.random.default_rng(123)
        rng_2 = np.random.default_rng(123)
        rng_3 = np.random.default_rng(321)

        statevector1 = LightningStateVector(n_qubits, dtype, rng=rng_1)
        statevector2 = LightningStateVector(n_qubits, dtype, rng=rng_2)
        statevector3 = LightningStateVector(n_qubits, dtype, rng=rng_3)
        LightningMeasurements(statevector1)
        LightningMeasurements(statevector2)
        LightningMeasurements(statevector3)

        assert statevector1._rng.bit_generator.state == statevector2._rng.bit_generator.state
        assert statevector1._rng.bit_generator.state != statevector3._rng.bit_generator.state

    @pytest.mark.local_salt(42)
    @pytest.mark.skipif(
        device_name in ("lightning.tensor"),
        reason=f"{device_name} does not support seeding device.",
    )
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    @pytest.mark.parametrize("shots", [10, [10, 10]])
    @pytest.mark.parametrize("measurement", [qp.expval, qp.probs, qp.var, qp.sample, qp.counts])
    @pytest.mark.parametrize(
        "observable",
        (
            [0],
            [0, 1],
            [0, 1, 2],
            [0, 1, 2, 3],
            qp.PauliX(0),
            qp.PauliY(1),
            qp.PauliZ(2),
            qp.sum(qp.PauliX(0), qp.PauliY(0)),
            qp.prod(qp.PauliX(0), qp.PauliY(1)),
            qp.s_prod(2.0, qp.PauliX(0)),
            qp.Hermitian(get_hermitian_matrix(2**2), wires=[0, 1]),
            qp.Hermitian(get_hermitian_matrix(2**2), wires=[2, 3]),
            qp.Hamiltonian(
                [1.0, 2.0, 3.0],
                [qp.PauliX(0), qp.PauliY(1), qp.PauliZ(2) @ qp.PauliZ(3)],
            ),
        ),
    )
    def test_seeded_shots_measurement(self, dtype, shots, measurement, observable, tol, seed):
        """Test that seeded measurements with shots return same results with same seed."""
        if measurement is qp.probs and isinstance(
            observable,
            (
                qp.ops.Sum,
                qp.ops.SProd,
                qp.ops.Prod,
            ),
        ):
            pytest.skip(
                f"Observable of type {type(observable).__name__} is not supported for rotating probabilities."
            )

        if measurement in (qp.expval, qp.var) and isinstance(observable, Sequence):
            pytest.skip("qp.expval, qp.var do not take wire arguments.")
        n_qubits = 4
        n_layers = 1
        rng = np.random.default_rng(seed)
        weights = rng.random((n_layers, n_qubits, 3))
        ops = [qp.Hadamard(i) for i in range(n_qubits)]
        ops += [qp.StronglyEntanglingLayers(weights, wires=range(n_qubits))]
        measurements = (
            [measurement(wires=observable)]
            if isinstance(observable, list)
            else [measurement(op=observable)]
        )
        tape = qp.tape.QuantumScript(ops, measurements, shots=shots)

        skip_list = (qp.ops.Sum,)
        do_skip = measurement is qp.var and isinstance(observable, skip_list)
        if not do_skip:
            rng_1 = np.random.default_rng(123)
            rng_2 = np.random.default_rng(123)
            statevector1 = LightningStateVector(n_qubits, dtype, rng=rng_1)
            statevector1 = get_final_state(statevector1, tape)
            statevector2 = LightningStateVector(n_qubits, dtype, rng=rng_2)
            statevector2 = get_final_state(statevector2, tape)
            m_1 = LightningMeasurements(statevector1)
            m_2 = LightningMeasurements(statevector2)
            result_1 = measure_final_state(m_1, tape)
            result_2 = measure_final_state(m_2, tape)

            if measurement is qp.sample:
                validate_samples(shots, result_1, result_2, rtol=0.0, atol=0.0)
            elif measurement is qp.counts:
                validate_counts(shots, result_1, result_2, rtol=0.0, atol=0.0)
            else:
                validate_others(shots, result_1, result_2, rtol=0.0, atol=0.0)


class TestControlledOps:
    """Tests for controlled operations"""

    @staticmethod
    def calculate_reference(tape):
        dev = DefaultQubit()
        program, _ = dev.preprocess()
        tapes, transf_fn = program([tape])
        results = dev.execute(tapes)
        return transf_fn(results)

    @pytest.mark.local_salt(42)
    @pytest.mark.parametrize(
        "operation",
        [
            qp.PauliX,
            qp.PauliY,
            qp.PauliZ,
            qp.Hadamard,
            qp.S,
            qp.T,
            qp.PhaseShift,
            qp.RX,
            qp.RY,
            qp.RZ,
            qp.Rot,
            qp.SWAP,
            qp.IsingXX,
            qp.IsingXY,
            qp.IsingYY,
            qp.IsingZZ,
            qp.SingleExcitation,
            qp.SingleExcitationMinus,
            qp.SingleExcitationPlus,
            qp.DoubleExcitation,
            qp.DoubleExcitationMinus,
            qp.DoubleExcitationPlus,
            qp.MultiRZ,
            qp.GlobalPhase,
            qp.PCPhase,
        ],
    )
    @pytest.mark.parametrize("control_value", [False, True])
    @pytest.mark.parametrize("n_qubits", list(range(2, 5)))
    def test_controlled_qubit_gates(
        self, operation, n_qubits, control_value, tol, lightning_sv, seed
    ):
        """Test that multi-controlled gates are correctly applied to a state"""
        threshold = 250 if device_name != "lightning.tensor" else 5
        num_wires = max(operation.num_wires, 1) if operation.num_wires else 1
        rng = np.random.default_rng(seed)

        if device_name not in ["lightning.qubit", "lightning.gpu"] and operation == qp.PCPhase:
            pytest.skip("PCPhase only supported on lightning.qubit and lightning.gpu.")

        for n_wires in range(num_wires + 1, num_wires + 4):
            wire_lists = list(itertools.permutations(range(0, n_qubits), n_wires))
            n_perms = len(wire_lists) * n_wires
            if n_perms > threshold:
                wire_lists = wire_lists[0 :: (n_perms // threshold)]
            for all_wires in wire_lists:
                target_wires = all_wires[0:num_wires]
                control_wires = all_wires[num_wires:]
                init_state = rng.random(2**n_qubits) + 1.0j * rng.random(2**n_qubits)
                init_state /= np.linalg.norm(init_state)

                if operation.num_params == 0:
                    operation_params = []
                else:
                    operation_params = tuple([0.1234] * operation.num_params) + (target_wires,)
                    if operation == qp.PCPhase:
                        # Hyperparameter for PCPhase is the dimension of the control space
                        operation_params = (0.1234, 2) + (target_wires,)

                ops = [
                    qp.StatePrep(init_state, wires=range(n_qubits)),
                ]

                if operation.num_params == 0:
                    ops += [
                        qp.ctrl(
                            operation(target_wires),
                            control_wires,
                            control_values=(
                                [control_value or bool(i % 2) for i, _ in enumerate(control_wires)]
                            ),
                        ),
                    ]
                else:
                    ops += [
                        qp.ctrl(
                            operation(*operation_params),
                            control_wires,
                            control_values=(
                                [control_value or bool(i % 2) for i, _ in enumerate(control_wires)]
                            ),
                        ),
                    ]

                measurements = [qp.state()]
                tape = qp.tape.QuantumScript(ops, measurements)

                statevector = lightning_sv(n_qubits)
                if device_name == "lightning.tensor" and statevector.method == "tn":
                    pytest.skip("StatePrep not supported in lightning.tensor with the tn method.")

                statevector = get_final_state(statevector, tape)
                m = LightningMeasurements(statevector)
                result = measure_final_state(m, tape)
                expected = self.calculate_reference(tape)
                if device_name == "lightning.tensor":
                    assert np.allclose(result, expected, 1e-4)
                else:
                    assert np.allclose(result, expected, tol * 10)

    def test_controlled_qubit_unitary_from_op(self, tol, lightning_sv):
        n_qubits = 10
        par = 0.1234

        tape = qp.tape.QuantumScript(
            [qp.ControlledQubitUnitary(qp.RX.compute_matrix(par), wires=range(6))],
            [qp.expval(qp.PauliX(0))],
        )

        statevector = lightning_sv(n_qubits)
        statevector = get_final_state(statevector, tape)
        m = LightningMeasurements(statevector)
        result = measure_final_state(m, tape)
        expected = self.calculate_reference(tape)

        assert np.allclose(result, expected, tol)

    @pytest.mark.parametrize("control_wires", range(4))
    @pytest.mark.parametrize("target_wires", range(4))
    def test_cnot_controlled_qubit_unitary(self, control_wires, target_wires, tol, lightning_sv):
        """Test that ControlledQubitUnitary is correctly applied to a state"""
        if control_wires == target_wires:
            return
        n_qubits = 4
        control_wires = [control_wires]
        target_wires = [target_wires]
        wires = control_wires + target_wires
        U = qp.matrix(qp.PauliX(target_wires))
        init_state = get_random_normalized_state(2**n_qubits)

        tape = qp.tape.QuantumScript(
            [
                qp.StatePrep(init_state, wires=range(n_qubits)),
                qp.ControlledQubitUnitary(U, wires=control_wires + target_wires),
            ],
            [qp.state()],
        )
        tape_cnot = qp.tape.QuantumScript(
            [qp.StatePrep(init_state, wires=range(n_qubits)), qp.CNOT(wires=wires)],
            [qp.state()],
        )

        statevector = lightning_sv(n_qubits)
        if device_name == "lightning.tensor" and statevector.method == "tn":
            pytest.skip("StatePrep not supported in lightning.tensor with the tn method.")

        statevector = get_final_state(statevector, tape)
        m = LightningMeasurements(statevector)
        result = measure_final_state(m, tape)
        expected = self.calculate_reference(tape_cnot)

        if device_name == "lightning.tensor":
            assert np.allclose(result, expected, 1e-4)
        else:
            assert np.allclose(result, expected, tol)

    @pytest.mark.parametrize("control_value", [False, True])
    @pytest.mark.parametrize("n_qubits", list(range(2, 8)))
    def test_controlled_globalphase(self, n_qubits, control_value, tol, lightning_sv):
        """Test that multi-controlled gates are correctly applied to a state"""
        threshold = 250 if device_name != "lightning.tensor" else 5
        operation = qp.GlobalPhase
        num_wires = max(operation.num_wires, 1) if operation.num_wires else 1

        for n_wires in range(num_wires + 1, num_wires + 4):
            wire_lists = list(itertools.permutations(range(0, n_qubits), n_wires))
            n_perms = len(wire_lists) * n_wires
            if n_perms > threshold:
                wire_lists = wire_lists[0 :: (n_perms // threshold)]
            for all_wires in wire_lists:
                target_wires = all_wires[0:num_wires]
                control_wires = all_wires[num_wires:]
                init_state = get_random_normalized_state(2**n_qubits)

                tape = qp.tape.QuantumScript(
                    [
                        qp.StatePrep(init_state, wires=range(n_qubits)),
                        qp.ctrl(
                            operation(0.1234, target_wires),
                            control_wires,
                            control_values=(
                                [control_value or bool(i % 2) for i, _ in enumerate(control_wires)]
                            ),
                        ),
                    ],
                    [qp.state()],
                )
                statevector = lightning_sv(n_qubits)
                if device_name == "lightning.tensor" and statevector.method == "tn":
                    pytest.skip("StatePrep not supported in lightning.tensor with the tn method.")

                statevector = get_final_state(statevector, tape)
                m = LightningMeasurements(statevector)
                result = measure_final_state(m, tape)
                expected = self.calculate_reference(tape)
                if device_name == "lightning.tensor" and statevector.dtype == np.complex64:
                    assert np.allclose(result, expected, 1e-4)
                else:
                    assert np.allclose(result, expected, tol)


@pytest.mark.parametrize("phi", PHI)
class TestExpOperatorArithmetic:
    """Test integration with SProd, Prod, and Sum."""

    wires = 2

    def test_sprod(self, phi, lightning_sv, tol):
        """Test the `SProd` class."""
        tape = qp.tape.QuantumScript(
            [qp.RX(phi, wires=0)],
            [qp.expval(qp.s_prod(0.5, qp.PauliZ(0)))],
        )
        statevector = lightning_sv(self.wires)
        statevector = get_final_state(statevector, tape)
        m = LightningMeasurements(statevector)
        result = measure_final_state(m, tape)
        expected = 0.5 * np.cos(phi)

        assert np.allclose(result, expected, tol)

    def test_prod(self, phi, lightning_sv, tol):
        """Test the `Prod` class."""
        tape = qp.tape.QuantumScript(
            [qp.RX(phi, wires=0), qp.Hadamard(1), qp.PauliZ(1)],
            [qp.expval(qp.prod(qp.PauliZ(0), qp.PauliX(1)))],
        )
        statevector = lightning_sv(self.wires)
        statevector = get_final_state(statevector, tape)
        m = LightningMeasurements(statevector)
        result = measure_final_state(m, tape)
        expected = -np.cos(phi)

        assert np.allclose(result, expected, tol)

    @pytest.mark.parametrize("theta", THETA)
    def test_sum(self, phi, theta, lightning_sv, tol):
        """Test the `Sum` class."""
        tape = qp.tape.QuantumScript(
            [qp.RX(phi, wires=0), qp.RY(theta, wires=1)],
            [qp.expval(qp.sum(qp.PauliZ(0), qp.PauliX(1)))],
        )
        statevector = lightning_sv(self.wires)
        statevector = get_final_state(statevector, tape)
        m = LightningMeasurements(statevector)
        result = measure_final_state(m, tape)
        expected = np.cos(phi) + np.sin(theta)

        assert np.allclose(result, expected, tol)


@pytest.mark.parametrize(
    "op,par,wires,expected",
    [
        (qp.StatePrep, [0, 1], [1], [1, -1]),
        (qp.StatePrep, [0, 1], [0], [-1, 1]),
        (qp.StatePrep, [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], [1], [1, 0]),
        (qp.StatePrep, [1j / 2.0, np.sqrt(3) / 2.0], [1], [1, -0.5]),
        (qp.StatePrep, [(2 - 1j) / 3.0, 2j / 3.0], [0], [1 / 9.0, 1]),
    ],
)
def test_state_vector_2_qubit_subset(tol, op, par, wires, expected, lightning_sv):
    """Tests qubit state vector preparation and measure on subsets of 2 qubits"""

    tape = qp.tape.QuantumScript(
        [op(par, wires=wires)], [qp.expval(qp.PauliZ(0)), qp.expval(qp.PauliZ(1))]
    )

    statevector = lightning_sv(2)
    if device_name == "lightning.tensor" and statevector.method == "tn":
        pytest.skip("StatePrep not supported in lightning.tensor with the tn method.")

    statevector = get_final_state(statevector, tape)

    m = LightningMeasurements(statevector)
    result = measure_final_state(m, tape)

    assert np.allclose(result, expected, tol)
