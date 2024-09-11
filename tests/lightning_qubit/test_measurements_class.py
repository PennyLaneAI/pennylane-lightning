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
import pennylane as qml
import pytest
from conftest import (  # tested device
    PHI,
    THETA,
    LightningDevice,
    LightningMeasurements,
    LightningStateVector,
    device_name,
)
from flaky import flaky
from pennylane.devices import DefaultQubit
from pennylane.measurements import VarianceMP
from scipy.sparse import csr_matrix, random_array

if not LightningDevice._new_API:
    pytest.skip(
        "Exclusive tests for new API devices. Skipping.",
        allow_module_level=True,
    )

if device_name == "lightning.tensor":
    pytest.skip("Skipping tests for the LightningTensor class.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


def get_hermitian_matrix(n):
    H = np.random.rand(n, n) + 1.0j * np.random.rand(n, n)
    return H + np.conj(H).T


def get_sparse_hermitian_matrix(n):
    H = random_array((n, n), density=0.15)
    H = H + 1.0j * random_array((n, n), density=0.15)
    return csr_matrix(H + H.conj().T)


class CustomStateMeasurement(qml.measurements.StateMeasurement):
    def process_state(self, state, wire_order):
        return 1


def test_initialization(lightning_sv):
    """Tests for the initialization of the LightningMeasurements class."""
    statevector = lightning_sv(num_wires=5)
    m = LightningMeasurements(statevector)

    assert m.qubit_state is statevector
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
            qml.vn_entropy(wires=0),
            CustomStateMeasurement(),
            qml.expval(qml.Identity(0)),
            qml.expval(qml.Projector([1, 0], wires=(0, 1))),
            qml.var(qml.Identity(0)),
            qml.var(qml.Projector([1, 0], wires=(0, 1))),
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
            qml.PauliX(0),
            qml.PauliY(0),
            qml.PauliZ(0),
            qml.sum(qml.PauliX(0), qml.PauliY(0)),
            qml.prod(qml.PauliX(0), qml.PauliY(1)),
            qml.s_prod(2.0, qml.PauliX(0)),
            qml.Hamiltonian([1.0, 2.0], [qml.PauliX(0), qml.PauliY(0)]),
            qml.Hermitian(np.eye(2), wires=0),
            qml.SparseHamiltonian(qml.PauliX.compute_sparse_matrix(), wires=0),
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
        measurement = qml.probs(op=qml.PauliX(0))
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


@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
class TestExpval:
    """Test expectation value calculations."""

    def test_identity(self, theta, phi, tol, lightning_sv):
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


@pytest.mark.parametrize("method_name", ("expval", "measurement"))
class TestExpvalHamiltonian:
    """Tests expval for Hamiltonians"""

    wires = 2

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


class TestSparseExpval:
    """Tests for the expval function"""

    wires = 2

    @pytest.mark.parametrize(
        "ham_terms, expected",
        [
            [qml.PauliX(0) @ qml.Identity(1), 0.00000000000000000],
            [qml.Identity(0) @ qml.PauliX(1), -0.19866933079506122],
            [qml.PauliY(0) @ qml.Identity(1), -0.38941834230865050],
            [qml.Identity(0) @ qml.PauliY(1), 0.00000000000000000],
            [qml.PauliZ(0) @ qml.Identity(1), 0.92106099400288520],
            [qml.Identity(0) @ qml.PauliZ(1), 0.98006657784124170],
        ],
    )
    def test_sparse_Pauli_words(self, ham_terms, expected, tol, lightning_sv):
        """Test expval of some simple sparse Hamiltonian"""

        ops = [qml.RX(0.4, wires=[0]), qml.RY(-0.2, wires=[1])]
        measurements = [
            qml.expval(
                qml.SparseHamiltonian(
                    qml.Hamiltonian([1], [ham_terms]).sparse_matrix(), wires=[0, 1]
                )
            )
        ]
        tape = qml.tape.QuantumScript(ops, measurements)

        statevector = lightning_sv(self.wires)
        statevector = statevector.get_final_state(tape)
        m = LightningMeasurements(statevector)
        result = m.measure_final_state(tape)

        assert np.allclose(result, expected, tol)


class TestMeasurements:
    """Tests all measurements"""

    @staticmethod
    def calculate_reference(tape, lightning_sv):
        use_default = True
        new_meas = []
        for m in tape.measurements:
            # NotImplementedError in DefaultQubit
            # We therefore validate against `qml.Hermitian`
            if isinstance(m, VarianceMP) and isinstance(
                m.obs, (qml.Hamiltonian, qml.SparseHamiltonian)
            ):
                use_default = False
                new_meas.append(m.__class__(qml.Hermitian(qml.matrix(m.obs), wires=m.obs.wires)))
                continue
            new_meas.append(m)
        if use_default:
            dev = DefaultQubit(max_workers=1)
            program, _ = dev.preprocess()
            tapes, transf_fn = program([tape])
            results = dev.execute(tapes)
            return transf_fn(results)

        tape = qml.tape.QuantumScript(tape.operations, new_meas)
        statevector = lightning_sv(tape.num_wires)
        statevector = statevector.get_final_state(tape)
        m = LightningMeasurements(statevector)
        return m.measure_final_state(tape)

    @flaky(max_runs=2)
    @pytest.mark.parametrize("shots", [None, 600_000, [790_000, 790_000]])
    @pytest.mark.parametrize("measurement", [qml.expval, qml.probs, qml.var])
    @pytest.mark.parametrize(
        "observable",
        (
            [0],
            [1, 2],
            [1, 0],
            qml.PauliX(0),
            qml.PauliY(1),
            qml.PauliZ(2),
            qml.sum(qml.PauliX(0), qml.PauliY(0)),
            qml.prod(qml.PauliX(0), qml.PauliY(1)),
            qml.s_prod(2.0, qml.PauliX(0)),
            qml.Hermitian(get_hermitian_matrix(2**2), wires=[0, 1]),
            qml.Hermitian(get_hermitian_matrix(2**2), wires=[2, 3]),
            qml.Hamiltonian(
                [1.0, 2.0, 3.0], [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2) @ qml.PauliZ(3)]
            ),
            qml.SparseHamiltonian(get_sparse_hermitian_matrix(2**4), wires=range(4)),
        ),
    )
    def test_single_return_value(self, shots, measurement, observable, lightning_sv, tol):
        if measurement is qml.probs and isinstance(
            observable,
            (
                qml.ops.Sum,
                qml.ops.SProd,
                qml.ops.Prod,
                qml.SparseHamiltonian,
            ),
        ):
            pytest.skip(
                f"Observable of type {type(observable).__name__} is not supported for rotating probabilities."
            )

        if measurement is not qml.probs and isinstance(observable, list):
            pytest.skip(
                f"Measurement of type {type(measurement).__name__} does not have a keyword argument 'wires'."
            )
        rtol = 1.0e-2  # 1% of expected value as tolerance
        if shots != None and measurement is qml.expval:
            # Increase the number of shots
            if isinstance(shots, int):
                shots *= 10
            else:
                shots = [i * 10 for i in shots]

            # Extra tolerance
            rtol = 5.0e-2  # 5% of expected value as tolerance

        n_qubits = 4
        n_layers = 1
        np.random.seed(0)
        weights = np.random.rand(n_layers, n_qubits, 3)
        ops = [qml.Hadamard(i) for i in range(n_qubits)]
        ops += [qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))]
        measurements = (
            [measurement(wires=observable)]
            if isinstance(observable, list)
            else [measurement(op=observable)]
        )
        tape = qml.tape.QuantumScript(ops, measurements, shots=shots)

        statevector = lightning_sv(n_qubits)
        statevector = statevector.get_final_state(tape)
        m = LightningMeasurements(statevector)

        skip_list = (
            qml.ops.Sum,
            qml.SparseHamiltonian,
        )
        do_skip = measurement is qml.var and isinstance(observable, skip_list)
        do_skip = do_skip or (
            measurement is qml.expval and isinstance(observable, qml.SparseHamiltonian)
        )
        do_skip = do_skip and shots is not None
        if do_skip:
            with pytest.raises(TypeError):
                _ = m.measure_final_state(tape)
            return
        else:
            result = m.measure_final_state(tape)

        expected = self.calculate_reference(tape, lightning_sv)

        # a few tests may fail in single precision, and hence we increase the tolerance
        if shots is None:
            assert np.allclose(result, expected, max(tol, 1.0e-4))
        else:
            atol = max(tol, 1.0e-2) if statevector.dtype == np.complex64 else max(tol, 1.0e-3)
            rtol = max(tol, rtol)  # % of expected value as tolerance

            # allclose -> absolute(a - b) <= (atol + rtol * absolute(b))
            assert np.allclose(result, expected, rtol=rtol, atol=atol)

    @flaky(max_runs=10)
    @pytest.mark.parametrize("shots", [None, 100_000, (90_000, 90_000)])
    @pytest.mark.parametrize("measurement", [qml.expval, qml.probs, qml.var])
    @pytest.mark.parametrize(
        "obs0_",
        (
            qml.PauliX(0),
            qml.PauliY(1),
            qml.PauliZ(2),
            qml.sum(qml.PauliX(0), qml.PauliY(0)),
            qml.prod(qml.PauliX(0), qml.PauliY(1)),
            qml.s_prod(2.0, qml.PauliX(0)),
            qml.Hermitian(get_hermitian_matrix(2), wires=[0]),
            qml.Hermitian(get_hermitian_matrix(2**2), wires=[2, 3]),
            qml.Hamiltonian(
                [1.0, 2.0, 3.0], [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2) @ qml.PauliZ(3)]
            ),
            qml.SparseHamiltonian(get_sparse_hermitian_matrix(2**4), wires=range(4)),
        ),
    )
    @pytest.mark.parametrize(
        "obs1_",
        (
            qml.PauliX(0),
            qml.PauliY(1),
            qml.PauliZ(2),
            qml.sum(qml.PauliX(0), qml.PauliY(0)),
            qml.prod(qml.PauliX(0), qml.PauliY(1)),
            qml.s_prod(2.0, qml.PauliX(0)),
            qml.Hermitian(get_hermitian_matrix(2), wires=[0]),
            qml.Hermitian(get_hermitian_matrix(2**2), wires=[2, 3]),
            qml.Hamiltonian(
                [1.0, 2.0, 3.0], [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2) @ qml.PauliZ(3)]
            ),
            qml.SparseHamiltonian(get_sparse_hermitian_matrix(2**4), wires=range(4)),
        ),
    )
    def test_double_return_value(self, shots, measurement, obs0_, obs1_, lightning_sv, tol):
        skip_list = (
            qml.ops.Sum,
            qml.ops.SProd,
            qml.ops.Prod,
            qml.Hamiltonian,
            qml.SparseHamiltonian,
        )
        if measurement is qml.probs and (
            isinstance(obs0_, skip_list) or isinstance(obs1_, skip_list)
        ):
            pytest.skip(
                f"Observable of type {type(obs0_).__name__} is not supported for rotating probabilities."
            )

        rtol = 1.0e-2  # 1% of expected value as tolerance
        if shots != None and measurement is qml.expval:
            # Increase the number of shots
            if isinstance(shots, int):
                shots *= 10
            else:
                shots = [i * 10 for i in shots]

            # Extra tolerance
            rtol = 5.0e-2  # 5% of expected value as tolerance

        n_qubits = 4
        n_layers = 1
        np.random.seed(0)
        weights = np.random.rand(n_layers, n_qubits, 3)
        ops = [qml.Hadamard(i) for i in range(n_qubits)]
        ops += [qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))]
        measurements = [measurement(op=obs0_), measurement(op=obs1_)]
        tape = qml.tape.QuantumScript(ops, measurements, shots=shots)

        statevector = lightning_sv(n_qubits)
        statevector = statevector.get_final_state(tape)
        m = LightningMeasurements(statevector)

        skip_list = (
            qml.ops.Sum,
            qml.Hamiltonian,
            qml.SparseHamiltonian,
        )
        do_skip = measurement is qml.var and (
            isinstance(obs0_, skip_list) or isinstance(obs1_, skip_list)
        )
        do_skip = do_skip or (
            measurement is qml.expval
            and (
                isinstance(obs0_, qml.SparseHamiltonian) or isinstance(obs1_, qml.SparseHamiltonian)
            )
        )
        do_skip = do_skip and shots is not None
        if do_skip:
            with pytest.raises(TypeError):
                _ = m.measure_final_state(tape)
            return
        else:
            result = m.measure_final_state(tape)

        expected = self.calculate_reference(tape, lightning_sv)
        if len(expected) == 1:
            expected = expected[0]

        assert isinstance(result, Sequence)
        assert len(result) == len(expected)
        # a few tests may fail in single precision, and hence we increase the tolerance
        atol = tol if shots is None else max(tol, 1.0e-2)
        rtol = max(tol, rtol)  # % of expected value as tolerance
        for r, e in zip(result, expected):
            if isinstance(shots, tuple) and isinstance(r[0], np.ndarray):
                r = np.concatenate(r)
                e = np.concatenate(e)
            # allclose -> absolute(r - e) <= (atol + rtol * absolute(e))
            assert np.allclose(r, e, atol=atol, rtol=rtol)

    @pytest.mark.skipif(
        device_name == "lightning.gpu",
        reason="lightning.gpu does not support out of order prob.",
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
        dev = qml.device(device_name, wires=cases[1])

        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.probs(wires=cases[0])

        expected = qml.QNode(circuit, qml.device("default.qubit", wires=cases[1]))()
        results = qml.QNode(circuit, dev)()
        assert np.allclose(expected, results, tol)


class TestControlledOps:
    """Tests for controlled operations"""

    @staticmethod
    def calculate_reference(tape):
        dev = DefaultQubit(max_workers=1)
        program, _ = dev.preprocess()
        tapes, transf_fn = program([tape])
        results = dev.execute(tapes)
        return transf_fn(results)

    @pytest.mark.parametrize(
        "operation",
        [
            qml.PauliX,
            qml.PauliY,
            qml.PauliZ,
            qml.Hadamard,
            qml.S,
            qml.T,
            qml.PhaseShift,
            qml.RX,
            qml.RY,
            qml.RZ,
            qml.Rot,
            qml.SWAP,
            qml.IsingXX,
            qml.IsingXY,
            qml.IsingYY,
            qml.IsingZZ,
            qml.SingleExcitation,
            qml.SingleExcitationMinus,
            qml.SingleExcitationPlus,
            qml.DoubleExcitation,
            qml.DoubleExcitationMinus,
            qml.DoubleExcitationPlus,
            qml.MultiRZ,
            qml.GlobalPhase,
        ],
    )
    @pytest.mark.parametrize("control_value", [False, True])
    @pytest.mark.parametrize("n_qubits", list(range(2, 5)))
    def test_controlled_qubit_gates(self, operation, n_qubits, control_value, tol, lightning_sv):
        """Test that multi-controlled gates are correctly applied to a state"""
        threshold = 250
        num_wires = max(operation.num_wires, 1)
        np.random.seed(0)

        for n_wires in range(num_wires + 1, num_wires + 4):
            wire_lists = list(itertools.permutations(range(0, n_qubits), n_wires))
            n_perms = len(wire_lists) * n_wires
            if n_perms > threshold:
                wire_lists = wire_lists[0 :: (n_perms // threshold)]
            for all_wires in wire_lists:
                target_wires = all_wires[0:num_wires]
                control_wires = all_wires[num_wires:]
                init_state = np.random.rand(2**n_qubits) + 1.0j * np.random.rand(2**n_qubits)
                init_state /= np.linalg.norm(init_state)

                ops = [
                    qml.StatePrep(init_state, wires=range(n_qubits)),
                ]

                if operation.num_params == 0:
                    ops += [
                        qml.ctrl(
                            operation(target_wires),
                            control_wires,
                            control_values=[
                                control_value or bool(i % 2) for i, _ in enumerate(control_wires)
                            ],
                        ),
                    ]
                else:
                    ops += [
                        qml.ctrl(
                            operation(*tuple([0.1234] * operation.num_params), target_wires),
                            control_wires,
                            control_values=[
                                control_value or bool(i % 2) for i, _ in enumerate(control_wires)
                            ],
                        ),
                    ]

                measurements = [qml.state()]
                tape = qml.tape.QuantumScript(ops, measurements)

                statevector = lightning_sv(n_qubits)
                statevector = statevector.get_final_state(tape)
                m = LightningMeasurements(statevector)
                result = m.measure_final_state(tape)
                expected = self.calculate_reference(tape)

                assert np.allclose(result, expected, tol * 10)

    def test_controlled_qubit_unitary_from_op(self, tol, lightning_sv):
        n_qubits = 10
        par = 0.1234

        tape = qml.tape.QuantumScript(
            [
                qml.ControlledQubitUnitary(
                    qml.QubitUnitary(qml.RX.compute_matrix(par), wires=5), control_wires=range(5)
                )
            ],
            [qml.expval(qml.PauliX(0))],
        )

        statevector = lightning_sv(n_qubits)
        statevector = statevector.get_final_state(tape)
        m = LightningMeasurements(statevector)
        result = m.measure_final_state(tape)
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
        U = qml.matrix(qml.PauliX(target_wires))
        init_state = np.random.rand(2**n_qubits) + 1.0j * np.random.rand(2**n_qubits)
        init_state /= np.linalg.norm(init_state)

        tape = qml.tape.QuantumScript(
            [
                qml.StatePrep(init_state, wires=range(n_qubits)),
                qml.ControlledQubitUnitary(U, control_wires=control_wires, wires=target_wires),
            ],
            [qml.state()],
        )
        tape_cnot = qml.tape.QuantumScript(
            [qml.StatePrep(init_state, wires=range(n_qubits)), qml.CNOT(wires=wires)], [qml.state()]
        )

        statevector = lightning_sv(n_qubits)
        statevector = statevector.get_final_state(tape)
        m = LightningMeasurements(statevector)
        result = m.measure_final_state(tape)
        expected = self.calculate_reference(tape_cnot)

        assert np.allclose(result, expected, tol)

    @pytest.mark.parametrize("control_value", [False, True])
    @pytest.mark.parametrize("n_qubits", list(range(2, 8)))
    def test_controlled_globalphase(self, n_qubits, control_value, tol, lightning_sv):
        """Test that multi-controlled gates are correctly applied to a state"""
        threshold = 250
        operation = qml.GlobalPhase
        num_wires = max(operation.num_wires, 1)
        for n_wires in range(num_wires + 1, num_wires + 4):
            wire_lists = list(itertools.permutations(range(0, n_qubits), n_wires))
            n_perms = len(wire_lists) * n_wires
            if n_perms > threshold:
                wire_lists = wire_lists[0 :: (n_perms // threshold)]
            for all_wires in wire_lists:
                target_wires = all_wires[0:num_wires]
                control_wires = all_wires[num_wires:]
                init_state = np.random.rand(2**n_qubits) + 1.0j * np.random.rand(2**n_qubits)
                init_state /= np.linalg.norm(init_state)

                tape = qml.tape.QuantumScript(
                    [
                        qml.StatePrep(init_state, wires=range(n_qubits)),
                        qml.ctrl(
                            operation(0.1234, target_wires),
                            control_wires,
                            control_values=[
                                control_value or bool(i % 2) for i, _ in enumerate(control_wires)
                            ],
                        ),
                    ],
                    [qml.state()],
                )
                statevector = lightning_sv(n_qubits)
                statevector = statevector.get_final_state(tape)
                m = LightningMeasurements(statevector)
                result = m.measure_final_state(tape)
                expected = self.calculate_reference(tape)

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


@pytest.mark.parametrize(
    "op,par,wires,expected",
    [
        (qml.QubitStateVector, [0, 1], [1], [1, -1]),
        (qml.QubitStateVector, [0, 1], [0], [-1, 1]),
        (qml.QubitStateVector, [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], [1], [1, 0]),
        (qml.QubitStateVector, [1j / 2.0, np.sqrt(3) / 2.0], [1], [1, -0.5]),
        (qml.QubitStateVector, [(2 - 1j) / 3.0, 2j / 3.0], [0], [1 / 9.0, 1]),
    ],
)
def test_state_vector_2_qubit_subset(tol, op, par, wires, expected, lightning_sv):
    """Tests qubit state vector preparation and measure on subsets of 2 qubits"""

    tape = qml.tape.QuantumScript(
        [op(par, wires=wires)], [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
    )

    statevector = lightning_sv(2)
    statevector = statevector.get_final_state(tape)

    m = LightningMeasurements(statevector)
    result = m.measure_final_state(tape)

    assert np.allclose(result, expected, tol)
