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
"""
Tests for process and execute (expval calculation).
"""
import pytest

import numpy as np
import pennylane as qml
from pennylane_lightning.lightning_qubit import LightningQubit2 as LightningQubit
from pennylane.devices import DefaultQubit

from conftest import LightningDevice  # tested device

if LightningDevice != LightningQubit:
    pytest.skip("Exclusive tests for lightning.qubit. Skipping.", allow_module_level=True)

if not LightningQubit._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


@pytest.mark.parametrize("c_dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
class TestExpval:
    """Test expectation value calculations"""

    @staticmethod
    def calculate_reference(tape):
        dev = DefaultQubit(max_workers=1)
        program, _ = dev.preprocess()
        tapes, transf_fn = program([tape])
        results = dev.execute(tapes)
        return transf_fn(results)

    @staticmethod
    def process_and_execute(dev, tape):
        program, _ = dev.preprocess()
        tapes, transf_fn = program([tape])
        results = dev.execute(tapes)
        return transf_fn(results)

    def test_Identity(self, theta, phi, c_dtype, tol):
        """Tests applying identities."""

        ops = [
            qml.Identity(0),
            qml.Identity((0, 1)),
            qml.Identity((1, 2)),
            qml.RX(theta, 0),
            qml.RX(phi, 1),
        ]
        measurements = [qml.expval(qml.PauliZ(0))]
        tape = qml.tape.QuantumScript(ops, measurements)

        dev = LightningQubit(c_dtype=c_dtype, wires=3)
        result = dev.execute(tape)
        expected = np.cos(theta)
        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(result, expected, tol)

    def test_identity_expectation(self, theta, phi, c_dtype, tol):
        """Tests identity expectations."""

        dev = LightningQubit(wires=2, c_dtype=c_dtype)
        tape = qml.tape.QuantumScript(
            [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            [qml.expval(qml.Identity(wires=[0])), qml.expval(qml.Identity(wires=[1]))],
        )
        result = dev.execute(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(1.0, result, tol)

    def test_multi_wire_identity_expectation(self, theta, phi, c_dtype, tol):
        """Tests multi-wire identity."""

        dev = LightningQubit(wires=2, c_dtype=c_dtype)
        tape = qml.tape.QuantumScript(
            [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            [qml.expval(qml.Identity(wires=[0, 1]))],
        )
        result = dev.execute(tape)
        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(1.0, result, tol)

    @pytest.mark.parametrize(
        "wires",
        [
            ([0, 1]),
            (["a", 1]),
            (["b", "a"]),
        ],
    )
    def test_PauliZ_expectation(self, theta, phi, c_dtype, tol, wires):
        """Tests PauliZ."""

        dev = LightningQubit(wires=wires, c_dtype=c_dtype)
        tape = qml.tape.QuantumScript(
            [qml.RX(theta, wires=wires[0]), qml.RX(phi, wires=wires[1]), qml.CNOT(wires=wires)],
            [qml.expval(qml.PauliZ(wires=wires[0])), qml.expval(qml.PauliZ(wires=wires[1]))],
        )

        calculated_val = self.process_and_execute(dev, tape)
        reference_val = self.calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_PauliX_expectation(self, theta, phi, dev, tol):
        """Tests PauliX."""

        tape = qml.tape.QuantumScript(
            [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            [qml.expval(qml.PauliX(wires=[0])), qml.expval(qml.PauliX(wires=[1]))],
        )

        calculated_val = self.process_and_execute(dev, tape)
        reference_val = self.calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_PauliY_expectation(self, theta, phi, dev, tol):
        """Tests PauliY."""

        tape = qml.tape.QuantumScript(
            [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            [qml.expval(qml.PauliY(wires=[0])), qml.expval(qml.PauliY(wires=[1]))],
        )

        calculated_val = self.process_and_execute(dev, tape)
        reference_val = self.calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_hadamard_expectation(self, theta, phi, dev, tol):
        """Tests Hadamard."""

        tape = qml.tape.QuantumScript(
            [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            [qml.expval(qml.Hadamard(wires=[0])), qml.expval(qml.Hadamard(wires=[1]))],
        )

        calculated_val = self.process_and_execute(dev, tape)
        reference_val = self.calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_hermitian_expectation(self, theta, phi, dev, tol):
        """Tests an Hermitian operator."""

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=0)
            qml.RX(phi, wires=1)
            qml.RX(theta + phi, wires=2)

            for idx in range(3):
                qml.expval(qml.Hermitian([[1, 0], [0, -1]], wires=[idx]))

        calculated_val = self.process_and_execute(dev, tape)
        reference_val = self.calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_hamiltonian_expectation(self, theta, phi, dev, tol):
        """Tests a Hamiltonian."""

        ham = qml.Hamiltonian(
            [1.0, 0.3, 0.3, 0.4],
            [
                qml.PauliX(0) @ qml.PauliX(1),
                qml.PauliZ(0),
                qml.PauliZ(1),
                qml.PauliX(0) @ qml.PauliY(1),
            ],
        )

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=0)
            qml.RX(phi, wires=1)
            qml.RX(theta + phi, wires=2)

            qml.expval(ham)

        calculated_val = self.process_and_execute(dev, tape)
        reference_val = self.calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_sparse_hamiltonian_expectation(self, theta, phi, dev, tol):
        """Tests a Hamiltonian."""

        ham = qml.SparseHamiltonian(
            qml.Hamiltonian(
                [1.0, 0.3, 0.3, 0.4],
                [
                    qml.PauliX(0) @ qml.PauliX(1),
                    qml.PauliZ(0),
                    qml.PauliZ(1),
                    qml.PauliX(0) @ qml.PauliY(1),
                ],
            ).sparse_matrix(),
            wires=[0, 1],
        )

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=0)
            qml.RX(phi, wires=1)

            qml.expval(ham)

        calculated_val = self.process_and_execute(dev, tape)
        reference_val = self.calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)


@pytest.mark.parametrize("phi", PHI)
class TestOperatorArithmetic:
    """Test integration with SProd, Prod, and Sum."""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return LightningQubit(c_dtype=request.param)

    @staticmethod
    def calculate_reference(tape):
        dev = DefaultQubit(max_workers=1)
        program, _ = dev.preprocess()
        tapes, transf_fn = program([tape])
        results = dev.execute(tapes)
        return transf_fn(results)

    @staticmethod
    def process_and_execute(dev, tape):
        program, _ = dev.preprocess()
        tapes, transf_fn = program([tape])
        results = dev.execute(tapes)
        return transf_fn(results)

    def test_s_prod(self, phi, dev, tol):
        """Tests the `SProd` class."""

        tape = qml.tape.QuantumScript(
            [qml.RX(phi, wires=[0])],
            [qml.expval(qml.s_prod(0.5, qml.PauliZ(0)))],
        )

        calculated_val = self.process_and_execute(dev, tape)
        reference_val = self.calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_prod(self, phi, dev, tol):
        """Tests the `Prod` class."""

        tape = qml.tape.QuantumScript(
            [qml.RX(phi, wires=[0]), qml.Hadamard(wires=[1]), qml.PauliZ(wires=[1])],
            [qml.expval(qml.prod(qml.PauliZ(0), qml.PauliX(1)))],
        )

        calculated_val = self.process_and_execute(dev, tape)
        reference_val = self.calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_sum(self, phi, dev, tol):
        """Tests the `Sum` class."""
        tape = qml.tape.QuantumScript(
            [qml.RX(phi, wires=[0]), qml.RX(-1.1 * phi, wires=[0])],
            [qml.expval(qml.sum(qml.PauliZ(0), qml.PauliX(1)))],
        )

        calculated_val = self.process_and_execute(dev, tape)
        reference_val = self.calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_integration(self, phi, dev, tol):
        """Test a Combination of `Sum`, `SProd`, and `Prod`."""

        obs = qml.sum(qml.s_prod(2.3, qml.PauliZ(0)), -0.5 * qml.prod(qml.PauliY(0), qml.PauliZ(1)))

        tape = qml.tape.QuantumScript(
            [qml.RX(phi, wires=[0]), qml.RX(-1.1 * phi, wires=[0])],
            [qml.expval(obs)],
        )

        calculated_val = self.process_and_execute(dev, tape)
        reference_val = self.calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)


@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorExpval:
    """Test tensor expectation values"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return LightningQubit(c_dtype=request.param)

    @staticmethod
    def calculate_reference(tape):
        dev = DefaultQubit(max_workers=1)
        program, _ = dev.preprocess()
        tapes, transf_fn = program([tape])
        results = dev.execute(tapes)
        return transf_fn(results)

    @staticmethod
    def process_and_execute(dev, tape):
        program, _ = dev.preprocess()
        tapes, transf_fn = program([tape])
        results = dev.execute(tapes)
        return transf_fn(results)

    def test_PauliX_PauliY(self, theta, phi, varphi, dev, tol):
        """Tests a tensor product involving PauliX and PauliY."""

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.expval(qml.PauliX(0) @ qml.PauliY(2))

        calculated_val = self.process_and_execute(dev, tape)
        reference_val = self.calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_PauliZ_identity(self, theta, phi, varphi, dev, tol):
        """Tests a tensor product involving PauliZ and Identity."""

        with qml.tape.QuantumTape() as tape:
            qml.Identity(wires=[0])
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.expval(qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2))

        calculated_val = self.process_and_execute(dev, tape)
        reference_val = self.calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_PauliZ_hadamard_PauliY(self, theta, phi, varphi, dev, tol):
        """Tests a tensor product involving PauliY, PauliZ and Hadamard."""

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.expval(qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2))

        calculated_val = self.process_and_execute(dev, tape)
        reference_val = self.calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)
