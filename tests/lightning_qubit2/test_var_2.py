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
Tests for process and execute (variance calculation).
"""
# pylint: disable=too-many-arguments, redefined-outer-name
import pytest
from conftest import LightningDevice, PHI, THETA, VARPHI

import numpy as np
import pennylane as qml
from pennylane.tape import QuantumScript

from pennylane_lightning.lightning_qubit import LightningQubit, LightningQubit2

if LightningDevice != LightningQubit:
    pytest.skip("Exclusive tests for lightning.qubit. Skipping.", allow_module_level=True)

if not LightningQubit2._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.fixture(params=[np.complex64, np.complex128])
def dev(request):
    return LightningQubit2(wires=3, c_dtype=request.param)


def calculate_reference(tape):
    dev = qml.device("default.qubit", max_workers=1)
    program, _ = dev.preprocess()
    tapes, transf_fn = program([tape])
    results = dev.execute(tapes)
    return transf_fn(results)


def process_and_execute(dev, tape):
    program, _ = dev.preprocess()
    tapes, transf_fn = program([tape])
    results = dev.execute(tapes)
    return transf_fn(results)


@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
class TestVar:
    """Tests for the variance"""

    def test_Identity(self, theta, phi, dev):
        """Tests applying identities."""

        ops = [
            qml.Identity(0),
            qml.Identity((0, 1)),
            qml.Identity((1, 2)),
            qml.RX(theta, 0),
            qml.RX(phi, 1),
        ]
        measurements = [qml.var(qml.PauliZ(0))]
        tape = qml.tape.QuantumScript(ops, measurements)

        result = dev.execute(tape)
        expected = 1 - np.cos(theta) ** 2
        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(result, expected, atol=tol, rtol=0)

    def test_identity_variance(self, theta, phi, dev):
        """Tests identity variances."""

        tape = qml.tape.QuantumScript(
            [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            [qml.var(qml.Identity(wires=[0])), qml.var(qml.Identity(wires=[1]))],
        )
        result = dev.execute(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(0.0, result, atol=tol, rtol=0)

    def test_multi_wire_identity_variance(self, theta, phi, dev):
        """Tests multi-wire identity."""

        tape = qml.tape.QuantumScript(
            [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            [qml.var(qml.Identity(wires=[0, 1]))],
        )
        result = dev.execute(tape)
        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(0.0, result, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "wires",
        [
            ([0, 1]),
            (["a", 1]),
            (["b", "a"]),
        ],
    )
    def test_custom_wires(self, theta, phi, wires):
        """Tests custom wires."""
        device = LightningQubit2(wires=wires)

        tape = qml.tape.QuantumScript(
            [qml.RX(theta, wires=wires[0]), qml.RX(phi, wires=wires[1]), qml.CNOT(wires=wires)],
            [qml.var(qml.PauliZ(wires=wires[0])), qml.var(qml.PauliZ(wires=wires[1]))],
        )

        calculated_val = process_and_execute(device, tape)
        reference_val = np.array(
            [1 - np.cos(theta) ** 2, 1 - np.cos(theta) ** 2 * np.cos(phi) ** 2]
        )

        tol = 1e-5 if device.c_dtype == np.complex64 else 1e-7
        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "Obs, Op, expected_fn",
        [
            (
                [qml.PauliX(wires=[0]), qml.PauliX(wires=[1])],
                qml.RY,
                lambda theta, phi: np.array(
                    [1 - np.sin(theta) ** 2 * np.sin(phi) ** 2, 1 - np.sin(phi) ** 2]
                ),
            ),
            (
                [qml.PauliY(wires=[0]), qml.PauliY(wires=[1])],
                qml.RX,
                lambda theta, phi: np.array([1, 1 - np.cos(theta) ** 2 * np.sin(phi) ** 2]),
            ),
            (
                [qml.PauliZ(wires=[0]), qml.PauliZ(wires=[1])],
                qml.RX,
                lambda theta, phi: np.array(
                    [1 - np.cos(theta) ** 2, 1 - np.cos(theta) ** 2 * np.cos(phi) ** 2]
                ),
            ),
            (
                [qml.Hadamard(wires=[0]), qml.Hadamard(wires=[1])],
                qml.RY,
                lambda theta, phi: np.array(
                    [
                        1 - (np.sin(theta) * np.sin(phi) + np.cos(theta)) ** 2 / 2,
                        1 - (np.cos(theta) * np.cos(phi) + np.sin(phi)) ** 2 / 2,
                    ]
                ),
            ),
        ],
    )
    def test_single_wire_observables_variance(self, Obs, Op, expected_fn, theta, phi, dev):
        """Test that variance values for single wire observables are correct"""

        tape = qml.tape.QuantumScript(
            [Op(theta, wires=[0]), Op(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            [qml.var(Obs[0]), qml.var(Obs[1])],
        )
        result = process_and_execute(dev, tape)
        expected = expected_fn(theta, phi)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(result, expected, atol=tol, rtol=0)

    def test_hermitian_variance(self, theta, phi, dev):
        """Tests an Hermitian operator."""

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=0)
            qml.RX(phi, wires=1)
            qml.RX(theta + phi, wires=2)

            for idx in range(3):
                qml.var(qml.Hermitian([[1, 0], [0, -1]], wires=[idx]))

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

    def test_hamiltonian_variance(self, theta, phi, dev):
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

        with qml.tape.QuantumTape() as tape1:
            qml.RX(theta, wires=0)
            qml.RX(phi, wires=1)
            qml.RX(theta + phi, wires=2)

            qml.var(ham)

        tape2 = QuantumScript(tape1.operations, [qml.var(qml.dot(ham.coeffs, ham.ops))])

        calculated_val = process_and_execute(dev, tape1)
        reference_val = calculate_reference(tape2)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

    def test_sparse_hamiltonian_variance(self, theta, phi, dev):
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

        with qml.tape.QuantumTape() as tape1:
            qml.RX(theta, wires=0)
            qml.RX(phi, wires=1)

            qml.var(ham)

        tape2 = QuantumScript(
            tape1.operations, [qml.var(qml.Hermitian(ham.matrix(), wires=[0, 1]))]
        )

        calculated_val = process_and_execute(dev, tape1)
        reference_val = calculate_reference(tape2)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)


@pytest.mark.parametrize("phi", PHI)
class TestOperatorArithmetic:
    """Test integration with SProd, Prod, and Sum."""

    def test_s_prod(self, phi, dev, tol):
        """Tests the `SProd` class."""

        tape = qml.tape.QuantumScript(
            [qml.RX(phi, wires=[0])],
            [qml.var(qml.s_prod(0.5, qml.PauliZ(0)))],
        )

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_prod(self, phi, dev, tol):
        """Tests the `Prod` class."""

        tape = qml.tape.QuantumScript(
            [qml.RX(phi, wires=[0]), qml.Hadamard(wires=[1]), qml.PauliZ(wires=[1])],
            [qml.var(qml.prod(qml.PauliZ(0), qml.PauliX(1)))],
        )

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_sum(self, phi, dev, tol):
        """Tests the `Sum` class."""
        tape = qml.tape.QuantumScript(
            [qml.RX(phi, wires=[0]), qml.RX(-1.1 * phi, wires=[0])],
            [qml.var(qml.sum(qml.PauliZ(0), qml.PauliX(1)))],
        )

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_integration(self, phi, dev, tol):
        """Test a Combination of `Sum`, `SProd`, and `Prod`."""

        obs = qml.sum(qml.s_prod(2.3, qml.PauliZ(0)), -0.5 * qml.prod(qml.PauliY(0), qml.PauliZ(1)))

        tape = qml.tape.QuantumScript(
            [qml.RX(phi, wires=[0]), qml.RX(-1.1 * phi, wires=[0])],
            [qml.var(obs)],
        )

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)


@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorVar:
    """Test tensor variances"""

    def test_PauliX_PauliY(self, theta, phi, varphi, dev, tol):
        """Tests a tensor product involving PauliX and PauliY."""

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.var(qml.PauliX(0) @ qml.PauliY(2))

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

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
            qml.var(qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2))

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

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
            qml.var(qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2))

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)
