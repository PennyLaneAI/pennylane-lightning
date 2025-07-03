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
import itertools

# pylint: disable=too-many-arguments, redefined-outer-name
import numpy as np
import pennylane as qml
import pytest
from conftest import PHI, THETA, VARPHI, LightningDevice, device_name
from pennylane.devices import DefaultQubit

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.fixture(params=[np.complex64, np.complex128])
def dev(request):
    return LightningDevice(wires=3, c_dtype=request.param)


def calculate_reference(tape):
    dev = DefaultQubit()
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
class TestExpval:
    """Test expectation value calculations"""

    def test_Identity(self, theta, phi, dev, tol):
        """Tests applying identities."""

        ops = [
            qml.Identity(0),
            qml.Identity((0, 1)),
            qml.RX(theta, 0),
            qml.Identity((1, 2)),
            qml.RX(phi, 1),
        ]
        measurements = [qml.expval(qml.PauliZ(0))]
        tape = qml.tape.QuantumScript(ops, measurements)

        result = dev.execute(tape)
        expected = np.cos(theta)
        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(result, expected, tol)

    def test_identity_expectation(self, theta, phi, dev, tol):
        """Tests identity expectations."""

        tape = qml.tape.QuantumScript(
            [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            [qml.expval(qml.Identity(wires=[0])), qml.expval(qml.Identity(wires=[1]))],
        )
        result = dev.execute(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(1.0, result, tol)

    def test_multi_wire_identity_expectation(self, theta, phi, dev, tol):
        """Tests multi-wire identity."""

        tape = qml.tape.QuantumScript(
            [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            [qml.expval(qml.Identity(wires=[0, 1]))],
        )
        result = dev.execute(tape)
        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(1.0, result, tol)

    @pytest.mark.parametrize(
        "wires",
        [([0, 1]), (["a", 1]), (["b", "a"]), ([-1, 2.5])],
    )
    def test_custom_wires(self, theta, phi, tol, wires):
        """Tests custom wires."""
        dev = LightningDevice(wires=wires)

        tape = qml.tape.QuantumScript(
            [qml.RX(theta, wires=wires[0]), qml.RX(phi, wires=wires[1]), qml.CNOT(wires=wires)],
            [qml.expval(qml.PauliZ(wires=wires[0])), qml.expval(qml.PauliZ(wires=wires[1]))],
        )

        calculated_val = process_and_execute(dev, tape)
        reference_val = np.array([np.cos(theta), np.cos(theta) * np.cos(phi)])

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

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
    def test_single_wire_observables_expectation(self, Obs, Op, expected_fn, theta, phi, tol, dev):
        """Test that expectation values for single wire observables are correct"""

        tape = qml.tape.QuantumScript(
            [Op(theta, wires=[0]), Op(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            [qml.expval(Obs[0]), qml.expval(Obs[1])],
        )
        result = process_and_execute(dev, tape)
        expected = expected_fn(theta, phi)

        assert np.allclose(result, expected, tol)

    def test_hermitian_expectation(self, theta, phi, tol, dev):
        """Tests an Hermitian operator."""

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=0)
            qml.RX(phi, wires=1)
            qml.RX(theta + phi, wires=2)

            for idx in range(3):
                qml.expval(qml.Hermitian([[1, 0], [0, -1]], wires=[idx]))

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

        assert np.allclose(calculated_val, reference_val, tol)

    @pytest.mark.local_salt(42)
    @pytest.mark.parametrize(
        "n_op_wires, wires",
        [
            [1, [3]],
            [2, [1, 3]],
            [3, [2, 7, 4]],
            [4, [0, 3, 7, 6]],
            [5, [0, 4, 3, 1, 6]],
            [6, [6, 1, 2, 5, 4, 3]],
            [7, [6, 3, 2, 7, 1, 4, 0]],
            [8, [7, 5, 6, 4, 3, 0, 1, 2]],
        ],
    )
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_hermitian_arbitrary_size_expectation(self, dtype, theta, phi, n_op_wires, wires, tol, seed):
        """Tests the expectation value of an arbitrary size Hermitian matrix."""
        if device_name == "lightning.tensor" and n_op_wires > 1:
            pytest.skip(
                "lightning.tensor does not support Hermitian matrices with n_op_wires not equal to 1"
            )
        n_qubits = 8
        dev = LightningDevice(wires=n_qubits, c_dtype=dtype)
        
        np.random.seed(seed)
        mat = np.random.rand(2**n_op_wires, 2**n_op_wires) + 1j * np.random.rand(
            2**n_op_wires, 2**n_op_wires
        )
        mat = mat + mat.conj().T

        tape = qml.tape.QuantumScript(
            [
                qml.RX(theta, wires=0),
                qml.RX(phi, wires=1),
                qml.RX(theta + phi, wires=2),
                qml.RX(theta, wires=3),
                qml.RX(phi, wires=4),
                qml.RX(theta + phi, wires=5),
                qml.RX(phi, wires=6),
                qml.RX(theta + phi, wires=7),
            ],
            [qml.expval(qml.Hermitian(mat, wires=wires))],
        )
        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)
        
        assert np.allclose(calculated_val, reference_val, tol if dtype == np.complex128 else 1e-5)

    def test_hamiltonian_expectation(self, theta, phi, tol, dev):
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

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

        assert np.allclose(calculated_val, reference_val, tol)

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support sparse Hamiltonians",
    )
    def test_sparse_hamiltonian_expectation(self, theta, phi, tol, dev):
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

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

        assert np.allclose(calculated_val, reference_val, tol)


@pytest.mark.parametrize("phi", PHI)
class TestOperatorArithmetic:
    """Test integration with SProd, Prod, and Sum."""

    @pytest.mark.parametrize(
        "obs",
        [
            qml.s_prod(0.5, qml.PauliZ(0)),
            qml.prod(qml.PauliZ(0), qml.PauliX(1)),
            qml.sum(qml.PauliZ(0), qml.PauliX(1)),
        ],
    )
    def test_op_math(self, phi, dev, obs, tol):
        """Tests the `SProd`, `Prod`, and `Sum` classes."""

        tape = qml.tape.QuantumScript(
            [
                qml.RX(phi, wires=[0]),
                qml.Hadamard(wires=[1]),
                qml.PauliZ(wires=[1]),
                qml.RX(-1.1 * phi, wires=[1]),
            ],
            [qml.expval(obs)],
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
            [qml.expval(obs)],
        )

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)


@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorExpval:
    """Test tensor expectation values"""

    def test_PauliX_PauliY(self, theta, phi, varphi, dev, tol):
        """Tests a tensor product involving PauliX and PauliY."""

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.expval(qml.PauliX(0) @ qml.PauliY(2))

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
            qml.expval(qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2))

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
            qml.expval(qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2))

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_permute_order(self, theta, phi, varphi, dev, tol):
        """Tests a tensor product with permuted term order."""

        subroutine = [
            qml.RX(theta, wires=[0]),
            qml.RX(phi, wires=[1]),
            qml.RX(varphi, wires=[2]),
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=[1, 2]),
        ]

        ob1 = qml.PauliZ(0) @ qml.X(1) @ qml.PauliY(2)
        ob2 = qml.PauliY(2) @ qml.X(1) @ qml.PauliZ(0)

        tape1 = qml.tape.QuantumScript(subroutine, [qml.expval(ob1)])
        tape2 = qml.tape.QuantumScript(subroutine, [qml.expval(ob2)])

        calculated_val_1 = process_and_execute(dev, tape1)
        calculated_val_2 = process_and_execute(dev, tape2)
        reference_val = calculate_reference(tape1)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val_1, calculated_val_2, tol)
        assert np.allclose(calculated_val_1, reference_val, tol)

    def test_hamiltonian(self, theta, phi, varphi, dev, tol):
        """Tests the expectation value of a Hamiltonian with multiple pauli words and coefficients."""

        subroutine = [
            qml.RX(theta, wires=[0]),
            qml.RX(phi, wires=[1]),
            qml.RX(varphi, wires=[2]),
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=[1, 2]),
        ]

        ob1 = qml.PauliZ(0) @ qml.X(1) @ qml.PauliY(2)
        ob2 = qml.PauliY(0) @ qml.Z(1) @ qml.PauliX(2)
        ob3 = qml.Identity(0) @ qml.X(1)
        ob4 = qml.PauliY(2) @ qml.Z(1)
        ob5 = qml.PauliZ(2)

        tape = qml.tape.QuantumScript(
            subroutine, [qml.expval(0.3 * ob1 + 0.4 * ob2 + 0.5 * ob3 + 0.6 * ob4 + 0.7 * ob5)]
        )

        calculated_val = process_and_execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)
