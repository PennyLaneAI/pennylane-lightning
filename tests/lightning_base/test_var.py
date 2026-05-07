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

import numpy as np
import pennylane as qp

# pylint: disable=too-many-arguments, redefined-outer-name
import pytest
from conftest import PHI, THETA, VARPHI, LightningDevice, device_name
from pennylane.tape import QuantumScript

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.fixture(params=[np.complex64, np.complex128])
def dev(request):
    return LightningDevice(wires=3, c_dtype=request.param)


def calculate_reference(tape):
    dev = qp.device("default.qubit", max_workers=1)
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
            qp.Identity(0),
            qp.Identity((0, 1)),
            qp.RX(theta, 0),
            qp.Identity((1, 2)),
            qp.RX(phi, 1),
        ]
        measurements = [qp.var(qp.PauliZ(0))]
        tape = qp.tape.QuantumScript(ops, measurements)

        result = dev.execute(tape)
        expected = 1 - np.cos(theta) ** 2
        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(result, expected, atol=tol, rtol=0)

    def test_identity_variance(self, theta, phi, dev):
        """Tests identity variances."""

        tape = qp.tape.QuantumScript(
            [qp.RX(theta, wires=[0]), qp.RX(phi, wires=[1]), qp.CNOT(wires=[0, 1])],
            [qp.var(qp.Identity(wires=[0])), qp.var(qp.Identity(wires=[1]))],
        )
        result = dev.execute(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(0.0, result, atol=tol, rtol=0)

    def test_multi_wire_identity_variance(self, theta, phi, dev):
        """Tests multi-wire identity."""

        tape = qp.tape.QuantumScript(
            [qp.RX(theta, wires=[0]), qp.RX(phi, wires=[1]), qp.CNOT(wires=[0, 1])],
            [qp.var(qp.Identity(wires=[0, 1]))],
        )
        result = dev.execute(tape)
        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(0.0, result, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "wires",
        [([0, 1]), (["a", 1]), (["b", "a"]), ([-1, 2.5])],
    )
    def test_custom_wires(self, theta, phi, wires):
        """Tests custom wires."""
        device = LightningDevice(wires=wires)

        tape = qp.tape.QuantumScript(
            [qp.RX(theta, wires=wires[0]), qp.RX(phi, wires=wires[1]), qp.CNOT(wires=wires)],
            [qp.var(qp.PauliZ(wires=wires[0])), qp.var(qp.PauliZ(wires=wires[1]))],
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
                [qp.PauliX(wires=[0]), qp.PauliX(wires=[1])],
                qp.RY,
                lambda theta, phi: np.array(
                    [1 - np.sin(theta) ** 2 * np.sin(phi) ** 2, 1 - np.sin(phi) ** 2]
                ),
            ),
            (
                [qp.PauliY(wires=[0]), qp.PauliY(wires=[1])],
                qp.RX,
                lambda theta, phi: np.array([1, 1 - np.cos(theta) ** 2 * np.sin(phi) ** 2]),
            ),
            (
                [qp.PauliZ(wires=[0]), qp.PauliZ(wires=[1])],
                qp.RX,
                lambda theta, phi: np.array(
                    [1 - np.cos(theta) ** 2, 1 - np.cos(theta) ** 2 * np.cos(phi) ** 2]
                ),
            ),
            (
                [qp.Hadamard(wires=[0]), qp.Hadamard(wires=[1])],
                qp.RY,
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

        tape = qp.tape.QuantumScript(
            [Op(theta, wires=[0]), Op(phi, wires=[1]), qp.CNOT(wires=[0, 1])],
            [qp.var(Obs[0]), qp.var(Obs[1])],
        )
        result = process_and_execute(dev, tape)
        expected = expected_fn(theta, phi)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(result, expected, atol=tol, rtol=0)

    def test_hermitian_variance(self, theta, phi, dev):
        """Tests an Hermitian operator."""

        with qp.tape.QuantumTape() as tape:
            qp.RX(theta, wires=0)
            qp.RX(phi, wires=1)
            qp.RX(theta + phi, wires=2)

            for idx in range(3):
                qp.var(qp.Hermitian([[1, 0], [0, -1]], wires=[idx]))

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

    def test_hamiltonian_variance(self, theta, phi, dev):
        """Tests a Hamiltonian."""

        ham = qp.Hamiltonian(
            [1.0, 0.3, 0.3, 0.4],
            [
                qp.PauliX(0) @ qp.PauliX(1),
                qp.PauliZ(0),
                qp.PauliZ(1),
                qp.PauliX(0) @ qp.PauliY(1),
            ],
        )

        with qp.tape.QuantumTape() as tape1:
            qp.RX(theta, wires=0)
            qp.RX(phi, wires=1)
            qp.RX(theta + phi, wires=2)

            qp.var(ham)

        tape2 = QuantumScript(tape1.operations, [qp.var(qp.dot(ham.coeffs, ham.ops))])

        calculated_val = process_and_execute(dev, tape1)
        reference_val = calculate_reference(tape2)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

    @pytest.mark.skipif(
        device_name == "lightning.tensor", reason="SparseH not supported on lightning.tensor."
    )
    def test_sparse_hamiltonian_variance(self, theta, phi, dev):
        """Tests a Hamiltonian."""

        ham = qp.SparseHamiltonian(
            qp.Hamiltonian(
                [1.0, 0.3, 0.3, 0.4],
                [
                    qp.PauliX(0) @ qp.PauliX(1),
                    qp.PauliZ(0),
                    qp.PauliZ(1),
                    qp.PauliX(0) @ qp.PauliY(1),
                ],
            ).sparse_matrix(),
            wires=[0, 1],
        )

        with qp.tape.QuantumTape() as tape1:
            qp.RX(theta, wires=0)
            qp.RX(phi, wires=1)

            qp.var(ham)

        tape2 = QuantumScript(tape1.operations, [qp.var(qp.Hermitian(ham.matrix(), wires=[0, 1]))])

        calculated_val = process_and_execute(dev, tape1)
        reference_val = calculate_reference(tape2)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)


@pytest.mark.parametrize("phi", PHI)
class TestOperatorArithmetic:
    """Test integration with SProd, Prod, and Sum."""

    @pytest.mark.parametrize(
        "obs",
        [
            qp.s_prod(0.5, qp.PauliZ(0)),
            qp.prod(qp.PauliZ(0), qp.PauliX(1)),
            qp.sum(qp.PauliZ(0), qp.PauliX(1)),
        ],
    )
    def test_op_math(self, phi, dev, obs, tol):
        """Tests the `SProd`, `Prod`, and `Sum` classes."""

        tape = qp.tape.QuantumScript(
            [
                qp.RX(phi, wires=[0]),
                qp.Hadamard(wires=[1]),
                qp.PauliZ(wires=[1]),
                qp.RX(-1.1 * phi, wires=[1]),
            ],
            [qp.var(obs)],
        )

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_integration(self, phi, dev, tol):
        """Test a Combination of `Sum`, `SProd`, and `Prod`."""

        obs = qp.sum(qp.s_prod(2.3, qp.PauliZ(0)), -0.5 * qp.prod(qp.PauliY(0), qp.PauliZ(1)))

        tape = qp.tape.QuantumScript(
            [qp.RX(phi, wires=[0]), qp.RX(-1.1 * phi, wires=[0])],
            [qp.var(obs)],
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

        with qp.tape.QuantumTape() as tape:
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            qp.var(qp.PauliX(0) @ qp.PauliY(2))

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_PauliZ_identity(self, theta, phi, varphi, dev, tol):
        """Tests a tensor product involving PauliZ and Identity."""

        with qp.tape.QuantumTape() as tape:
            qp.Identity(wires=[0])
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            qp.var(qp.PauliZ(0) @ qp.Identity(1) @ qp.PauliZ(2))

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_PauliZ_hadamard_PauliY(self, theta, phi, varphi, dev, tol):
        """Tests a tensor product involving PauliY, PauliZ and Hadamard."""

        with qp.tape.QuantumTape() as tape:
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            qp.var(qp.PauliZ(0) @ qp.Hadamard(1) @ qp.PauliY(2))

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)
