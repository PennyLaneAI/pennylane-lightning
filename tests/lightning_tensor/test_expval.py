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
Tests for the expectation value calculations on the LightningTensor device.
"""

import pennylane as qml
import pytest
from conftest import PHI, THETA, VARPHI, LightningDevice, device_name
from pennylane import DeviceError
from pennylane import numpy as np
from pennylane.devices import DefaultQubit

if device_name != "lightning.tensor":
    pytest.skip("Exclusive tests for Lightning Tensor device. Skipping.", allow_module_level=True)
else:
    from pennylane_lightning.lightning_tensor import LightningTensor
    from pennylane_lightning.lightning_tensor._measurements import LightningTensorMeasurements
    from pennylane_lightning.lightning_tensor._state_tensor import LightningStateTensor

if not LightningDevice._new_API:  # pylint: disable=protected-access
    pytest.skip("Exclusive tests for new API. Skipping.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


def calculate_reference(tape):
    """Calculates the reference value for the given tape."""
    dev = DefaultQubit(max_workers=1)
    program, _ = dev.preprocess()
    tapes, transf_fn = program([tape])
    results = dev.execute(tapes)
    return transf_fn(results)


def execute(dev, tape):
    """Executes the tape on the device and returns the result."""
    results = dev.execute(tape)
    return results


@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
class TestExpval:
    """Test expectation value calculations"""

    def test_Identity(self, theta, phi, qubit_device, tol):
        """Tests applying identities."""
        dev = qubit_device(wires=3)

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

    def test_identity_expectation(self, theta, phi, qubit_device, tol):
        """Tests identity expectations."""
        dev = qubit_device(wires=3)

        tape = qml.tape.QuantumScript(
            [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            [qml.expval(qml.Identity(wires=[0])), qml.expval(qml.Identity(wires=[1]))],
        )
        result = dev.execute(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(1.0, result, tol)

    def test_multi_wire_identity_expectation(self, theta, phi, qubit_device, tol):
        """Tests multi-wire identity."""
        dev = qubit_device(wires=3)

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
        dev = LightningTensor(wires=wires, c_dtype=np.complex128)

        tape = qml.tape.QuantumScript(
            [
                qml.RX(theta, wires=wires[0]),
                qml.RX(phi, wires=wires[1]),
                qml.CNOT(wires=wires),
            ],
            [
                qml.expval(qml.PauliZ(wires=wires[0])),
                qml.expval(qml.PauliZ(wires=wires[1])),
            ],
        )

        calculated_val = execute(dev, tape)
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
    def test_single_wire_observables_expectation(
        self, Obs, Op, expected_fn, theta, phi, tol, qubit_device
    ):  # pylint: disable=too-many-arguments
        """Test that expectation values for single wire observables are correct"""
        dev = qubit_device(wires=3)

        tape = qml.tape.QuantumScript(
            [Op(theta, wires=[0]), Op(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            [qml.expval(Obs[0]), qml.expval(Obs[1])],
        )
        result = execute(dev, tape)
        expected = expected_fn(theta, phi)

        assert np.allclose(result, expected, tol)

    def test_hermitian_expectation(self, theta, phi, tol, qubit_device):
        """Tests an Hermitian operator."""
        dev = qubit_device(wires=3)

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=0)
            qml.RX(phi, wires=1)
            qml.RX(theta + phi, wires=2)

            for idx in range(3):
                qml.expval(qml.Hermitian([[1, 0], [0, -1]], wires=[idx]))

        calculated_val = execute(dev, tape)
        reference_val = calculate_reference(tape)

        assert np.allclose(calculated_val, reference_val, tol)

    def test_hermitian_expectation_qnode(self, theta, phi, tol, qubit_device):
        """Tests an Hermitian operator."""
        dev = qubit_device(wires=3)
        dev_def = qml.device("default.qubit", wires=3)
        obs = qml.Hermitian([[1, 0], [0, -1]], wires=[0])

        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(theta + phi, wires=[2])
            return qml.expval(obs)

        circ = qml.QNode(circuit, dev)
        circ_def = qml.QNode(circuit, dev_def)
        assert np.allclose(circ(), circ_def(), tol)

    def test_hermitian_expectation_with_basisstate(self, theta, phi, qubit_device):
        """Tests an Hermitian operator."""
        dev = qubit_device(wires=8)
        dev_def = qml.device("default.qubit", wires=8)
        obs = qml.Hermitian([[1, 0], [0, -1]], wires=[0])

        @qml.qnode(dev)
        def circuit_dev():
            qml.BasisState(np.array([1, 0, 1, 0, 1, 0, 1, 0]), wires=range(8))
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(theta + phi, wires=[2])
            return qml.expval(obs)

        @qml.qnode(dev_def)
        def circuit_def():
            qml.BasisState(np.array([1, 0, 1, 0, 1, 0, 1, 0]), wires=range(8))
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(theta + phi, wires=[2])
            return qml.expval(obs)

        assert np.allclose(circuit_dev(), circuit_def(), rtol=5e-2)

    def test_hamiltonian_expectation(self, theta, phi, tol, qubit_device):
        """Tests a Hamiltonian."""
        dev = qubit_device(wires=3)

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

        calculated_val = execute(dev, tape)
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
    def test_op_math(self, phi, qubit_device, obs, tol):
        """Tests the `SProd`, `Prod`, and `Sum` classes."""
        dev = qubit_device(wires=3)

        tape = qml.tape.QuantumScript(
            [
                qml.RX(phi, wires=[0]),
                qml.Hadamard(wires=[1]),
                qml.PauliZ(wires=[1]),
                qml.RX(-1.1 * phi, wires=[1]),
            ],
            [qml.expval(obs)],
        )

        calculated_val = execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_integration(self, phi, qubit_device, tol):
        """Test a Combination of `Sum`, `SProd`, and `Prod`."""
        dev = qubit_device(wires=3)

        obs = qml.sum(
            qml.s_prod(2.3, qml.PauliZ(0)),
            -0.5 * qml.prod(qml.PauliY(0), qml.PauliZ(1)),
        )

        tape = qml.tape.QuantumScript(
            [qml.RX(phi, wires=[0]), qml.RX(-1.1 * phi, wires=[0])],
            [qml.expval(obs)],
        )

        calculated_val = execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)


@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorExpval:
    """Test tensor expectation values"""

    def test_PauliX_PauliY(
        self, theta, phi, varphi, qubit_device, tol
    ):  # pylint: disable=too-many-arguments
        """Tests a tensor product involving PauliX and PauliY."""
        dev = qubit_device(wires=3)

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.expval(qml.PauliX(0) @ qml.PauliY(2))

        calculated_val = execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_PauliZ_identity(
        self, theta, phi, varphi, qubit_device, tol
    ):  # pylint: disable=too-many-arguments
        """Tests a tensor product involving PauliZ and Identity."""
        dev = qubit_device(wires=3)

        with qml.tape.QuantumTape() as tape:
            qml.Identity(wires=[0])
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.expval(qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2))

        calculated_val = execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_PauliZ_hadamard_PauliY(
        self, theta, phi, varphi, qubit_device, tol
    ):  # pylint: disable=too-many-arguments
        """Tests a tensor product involving PauliY, PauliZ and Hadamard."""
        dev = qubit_device(wires=3)

        with qml.tape.QuantumTape() as tape:
            qml.QubitUnitary(np.eye(4), wires=[0, 1])
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.expval(qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2))

        calculated_val = execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)


# Define the parameter values
THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)


@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
def test_multi_qubit_gates(theta, phi, qubit_device, tol):  # pylint: disable=too-many-arguments
    """Tests a simple circuit with multi-qubit gates."""

    ops = [
        qml.PauliX(wires=[0]),
        qml.RX(theta, wires=[0]),
        qml.RX(phi, wires=[1]),
        qml.CNOT(wires=[3, 4]),
        qml.CZ(wires=[3, 5]),
        qml.Hadamard(wires=[4]),
        qml.CNOT(wires=[2, 4]),
    ]

    meas = [
        qml.expval(qml.PauliY(2)),
        qml.expval(qml.Hamiltonian([1, 5, 6], [qml.Z(6), qml.X(0), qml.Hadamard(4)])),
        qml.expval(
            qml.Hamiltonian(
                [4, 5, 7],
                [
                    qml.Z(6) @ qml.Y(4),
                    qml.X(7),
                    qml.Hadamard(4),
                ],
            )
        ),
    ]

    tape = qml.tape.QuantumScript(ops=ops, measurements=meas)

    reference_val = calculate_reference(tape)
    dev = qubit_device(wires=tape.wires)
    calculated_val = dev.execute(tape)

    assert np.allclose(calculated_val, reference_val, tol)


random_unitary = np.array(
    [
        [
            -0.48401572 - 0.11012304j,
            -0.44806504 + 0.46775911j,
            -0.36968281 + 0.19235993j,
            -0.37561358 + 0.13887962j,
        ],
        [
            -0.12838047 + 0.13992187j,
            0.14531831 + 0.45319438j,
            0.28902175 - 0.71158765j,
            -0.24333677 - 0.29721109j,
        ],
        [
            0.26400811 - 0.72519269j,
            0.13965687 + 0.35092711j,
            0.09141515 - 0.14367072j,
            0.14894673 + 0.45886629j,
        ],
        [
            -0.04067799 + 0.34681783j,
            -0.45852968 - 0.03214391j,
            -0.10528164 - 0.4431247j,
            0.50251451 + 0.45476965j,
        ],
    ]
)


def circuit_ansatz(params, wires):
    """Circuit ansatz containing all the parametrized gates"""
    qml.Identity(wires=wires[0])
    qml.QubitUnitary(random_unitary, wires=[wires[1], wires[3]])
    qml.PauliX(wires=wires[1])
    qml.PauliY(wires=wires[2])
    qml.PauliZ(wires=wires[3])
    qml.Hadamard(wires=wires[4])
    qml.S(wires=wires[5])
    qml.T(wires=wires[0])
    qml.SX(wires=wires[1])
    qml.CNOT(wires=[wires[6], wires[7]])
    qml.SWAP(wires=[wires[2], wires[3]])
    qml.ISWAP(wires=[wires[4], wires[5]])
    qml.PSWAP(params[0], wires=[wires[6], wires[7]])
    qml.SISWAP(wires=[wires[4], wires[5]])
    qml.SQISW(wires=[wires[1], wires[0]])
    qml.CY(wires=[wires[0], wires[2]])
    qml.CZ(wires=[wires[1], wires[3]])
    qml.PhaseShift(params[1], wires=wires[2])
    qml.ControlledPhaseShift(params[2], wires=[wires[0], wires[5]])
    qml.RX(params[3], wires=wires[0])
    qml.RY(params[4], wires=wires[1])
    qml.RZ(params[5], wires=wires[3])
    qml.Rot(params[6], params[7], params[8], wires=wires[0])
    qml.CRX(params[9], wires=[wires[1], wires[0]])
    qml.CRY(params[10], wires=[wires[3], wires[2]])
    qml.CRZ(params[11], wires=[wires[2], wires[1]])
    qml.IsingXX(params[12], wires=[wires[1], wires[0]])
    qml.IsingYY(params[13], wires=[wires[3], wires[2]])
    qml.IsingZZ(params[14], wires=[wires[2], wires[1]])
    qml.SingleExcitation(params[15], wires=[wires[2], wires[0]])
    qml.SingleExcitationPlus(params[16], wires=[wires[3], wires[1]])
    qml.SingleExcitationMinus(params[17], wires=[wires[4], wires[2]])
    qml.QFT(wires=[wires[0]])
    qml.CSWAP(wires=[wires[3], wires[4], wires[5]])
    qml.Toffoli(wires=[wires[0], wires[1], wires[2]])
    qml.DoubleExcitation(params[18], wires=[wires[0], wires[1], wires[2], wires[3]])
    qml.QubitCarry(wires=[wires[0], wires[1], wires[6], wires[7]])
    qml.QubitSum(wires=[wires[2], wires[3], wires[7]])
    qml.OrbitalRotation(params[19], wires=[wires[0], wires[1], wires[5], wires[6]])


@pytest.mark.parametrize(
    "returns",
    [
        (qml.PauliX(0),),
        (qml.PauliY(0),),
        (qml.PauliZ(0),),
        (qml.PauliX(1),),
        (qml.PauliY(1),),
        (qml.PauliZ(1),),
        (qml.PauliX(2),),
        (qml.PauliY(2),),
        (qml.PauliZ(2),),
        (qml.PauliX(3),),
        (qml.PauliY(3),),
        (qml.PauliZ(3),),
        (qml.PauliX(0), qml.PauliY(1)),
        (
            qml.PauliZ(0),
            qml.PauliX(1),
            qml.PauliY(2),
        ),
        (
            qml.PauliY(0),
            qml.PauliZ(1),
            qml.PauliY(3),
        ),
        (qml.PauliZ(0) @ qml.PauliY(3),),
        (qml.Hadamard(2),),
        (qml.Hadamard(3) @ qml.PauliZ(2),),
        (qml.PauliX(0) @ qml.PauliY(3),),
        (qml.PauliY(0) @ qml.PauliY(2) @ qml.PauliY(3),),
        (qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2),),
        (0.5 * qml.PauliZ(0) @ qml.PauliZ(2),),
        (qml.ops.LinearCombination([1.0, 2.0], [qml.X(0) @ qml.Z(1), qml.Y(3) @ qml.Z(2)])),
        (qml.ops.prod(qml.X(0), qml.Y(1))),
    ],
)
def test_integration(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations"""
    num_wires = 8
    dev_default = qml.device("default.qubit", wires=range(num_wires))
    dev_ltensor = LightningTensor(wires=range(num_wires), maxBondDim=16, c_dtype=np.complex128)

    def circuit(params):
        qml.BasisState(np.array([1, 0, 1, 0, 1, 0, 1, 0]), wires=range(num_wires))
        circuit_ansatz(params, wires=range(num_wires))
        return qml.math.hstack([qml.expval(r) for r in returns])

    n_params = 20
    np.random.seed(1337)
    params_init = np.random.rand(n_params)

    params = np.array(params_init, requires_grad=True)

    qnode_ltensor = qml.QNode(circuit, dev_ltensor)
    qnode_default = qml.QNode(circuit, dev_default)

    j_ltensor = qml.jacobian(qnode_ltensor)(params)
    j_default = qml.jacobian(qnode_default)(params)

    assert np.allclose(j_ltensor, j_default, rtol=1e-1)


@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
def test_state_prep_not_support(qubit_device, theta, phi):
    """Test that state preparation is not supported on the device."""
    dev = qubit_device(wires=3)
    obs = qml.Hermitian([[1, 0], [0, -1]], wires=[0])

    @qml.qnode(dev)
    def circuit_dev():
        qml.StatePrep(np.array([0, 0, 0, 1, 1, 0, 1, 0]), wires=range(3))
        qml.RX(theta, wires=[0])
        qml.RX(phi, wires=[1])
        qml.RX(theta + phi, wires=[2])
        return qml.expval(obs)

    with pytest.raises(ValueError):
        circuit_dev()


class TestSparseHExpval:
    """Test sparseH expectation values"""

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0) @ qml.Identity(1), 0.00000000000000000, 1.000000000000000000],
            [qml.Identity(0) @ qml.PauliX(1), -0.19866933079506122, 0.960530638694763184],
            [qml.PauliY(0) @ qml.Identity(1), -0.38941834230865050, 0.848353326320648193],
            [qml.Identity(0) @ qml.PauliY(1), 0.00000000000000000, 1.000000119209289551],
            [qml.PauliZ(0) @ qml.Identity(1), 0.92106099400288520, 0.151646673679351807],
            [qml.Identity(0) @ qml.PauliZ(1), 0.98006657784124170, 0.039469480514526367],
        ],
    )
    def test_sparse_Pauli_words(self, cases, qubit_device):
        """Test expval of some simple sparse Hamiltonian"""
        dev = qubit_device(wires=4)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit_expval():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.expval(
                qml.SparseHamiltonian(
                    qml.Hamiltonian([1], [cases[0]]).sparse_matrix(), wires=[0, 1]
                )
            )

        with pytest.raises(DeviceError):
            circuit_expval()

    def test_expval_sparseH(self):
        """Test that expval is chosen for a variety of different expectation values."""
        obs = [
            qml.expval(qml.SparseHamiltonian(qml.PauliX.compute_sparse_matrix(), wires=0)),
        ]

        state_tensor = LightningStateTensor(4, 10)
        tape = qml.tape.QuantumScript(measurements=obs)
        m = LightningTensorMeasurements(state_tensor)

        with pytest.raises(NotImplementedError):
            m.expval(tape.measurements[0])


class TestStronglyEntanglingLayers:
    """Test the StronglyEntanglingLayers algorithm."""

    @pytest.mark.parametrize("n_qubits", range(2, 10, 2))
    def test_stronglyentanglinglayers(self, n_qubits):
        dev = qml.device("lightning.tensor", wires=n_qubits)
        dq = qml.device("default.qubit", wires=n_qubits)

        def circuit(weights):
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n_qubits)
        init_weights = np.random.random(size=shape)
        weights = np.array(init_weights, requires_grad=True)

        circ = qml.QNode(circuit, dev, diff_method="parameter-shift")
        circ_def = qml.QNode(circuit, dq, diff_method="parameter-shift")

        jac_ps = qml.jacobian(circ)(weights)
        jac_def = qml.jacobian(circ_def)(weights)

        assert np.allclose(jac_ps, jac_ps, atol=1e-7)
