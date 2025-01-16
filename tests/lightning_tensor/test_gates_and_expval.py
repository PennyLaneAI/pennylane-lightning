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
from conftest import PHI, THETA, LightningDevice, device_name
from pennylane import DeviceError
from pennylane import numpy as np

if device_name != "lightning.tensor":
    pytest.skip("Exclusive tests for Lightning Tensor device. Skipping.", allow_module_level=True)
else:
    from pennylane_lightning.lightning_tensor import LightningTensor
    from pennylane_lightning.lightning_tensor._measurements import LightningTensorMeasurements
    from pennylane_lightning.lightning_tensor._tensornet import LightningTensorNet

if not LightningDevice._new_API:  # pylint: disable=protected-access
    pytest.skip("Exclusive tests for new API. Skipping.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


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
    qml.ControlledQubitUnitary(
        qml.matrix(qml.PauliX([wires[1]])), control_wires=[wires[0]], wires=wires[1]
    )
    qml.DiagonalQubitUnitary(np.array([1, 1]), wires=wires[2])
    qml.MultiControlledX(wires=[wires[0], wires[1], wires[3]], control_values=[0, 1])
    qml.PauliX(wires=wires[1])
    qml.PauliY(wires=wires[2])
    qml.PauliZ(wires=wires[3])
    qml.Hadamard(wires=wires[4])
    qml.adjoint(qml.S(wires=wires[4]))
    qml.S(wires=wires[5])
    qml.adjoint(qml.T(wires=wires[1]))
    qml.T(wires=wires[0])
    qml.adjoint(qml.SX(wires=wires[0]))
    qml.SX(wires=wires[1])
    qml.CNOT(wires=[wires[6], wires[7]])
    qml.SWAP(wires=[wires[2], wires[3]])
    qml.adjoint(qml.ISWAP(wires=[wires[0], wires[1]]))
    qml.ISWAP(wires=[wires[4], wires[5]])
    qml.ISWAP(wires=[wires[4], wires[6]])
    qml.PSWAP(params[0], wires=[wires[6], wires[7]])
    qml.PSWAP(params[1], wires=[wires[0], wires[7]])
    qml.adjoint(qml.SISWAP(wires=[wires[0], wires[1]]))
    qml.adjoint(qml.SISWAP(wires=[wires[0], wires[4]]))
    qml.SISWAP(wires=[wires[4], wires[5]])
    qml.SISWAP(wires=[wires[2], wires[5]])
    qml.SQISW(wires=[wires[1], wires[0]])
    qml.SQISW(wires=[wires[5], wires[0]])
    qml.CSWAP(wires=[wires[2], wires[4], wires[5]])
    qml.Toffoli(wires=[wires[0], wires[1], wires[2]])
    qml.Toffoli(wires=[wires[0], wires[1], wires[5]])
    qml.CY(wires=[wires[0], wires[2]])
    qml.CZ(wires=[wires[1], wires[3]])
    qml.PhaseShift(params[2], wires=wires[2])
    qml.ControlledPhaseShift(params[3], wires=[wires[0], wires[5]])
    qml.RX(params[4], wires=wires[0])
    qml.RY(params[5], wires=wires[1])
    qml.RZ(params[6], wires=wires[3])
    qml.Rot(params[7], params[8], params[9], wires=wires[0])
    qml.CRX(params[10], wires=[wires[1], wires[0]])
    qml.CRY(params[11], wires=[wires[3], wires[2]])
    qml.CRZ(params[12], wires=[wires[2], wires[1]])
    qml.CRX(params[13], wires=[wires[1], wires[5]])
    qml.CRY(params[14], wires=[wires[3], wires[6]])
    qml.CRZ(params[15], wires=[wires[2], wires[0]])
    qml.IsingXX(params[16], wires=[wires[1], wires[0]])
    qml.IsingYY(params[17], wires=[wires[3], wires[2]])
    qml.IsingXY(params[18], wires=[wires[2], wires[1]])
    qml.IsingZZ(params[19], wires=[wires[2], wires[1]])
    qml.IsingXX(params[20], wires=[wires[1], wires[5]])
    qml.IsingYY(params[21], wires=[wires[3], wires[0]])
    qml.IsingXY(params[22], wires=[wires[2], wires[4]])
    qml.IsingZZ(params[23], wires=[wires[2], wires[0]])
    qml.SingleExcitation(params[24], wires=[wires[2], wires[0]])
    qml.SingleExcitationPlus(params[25], wires=[wires[3], wires[1]])
    qml.SingleExcitationMinus(params[26], wires=[wires[4], wires[2]])
    qml.DoubleExcitation(params[27], wires=[wires[0], wires[1], wires[2], wires[3]])
    qml.DoubleExcitationPlus(params[28], wires=[wires[1], wires[2], wires[3], wires[4]])
    qml.DoubleExcitationMinus(params[29], wires=[wires[2], wires[3], wires[4], wires[5]])
    qml.DoubleExcitation(params[30], wires=[wires[0], wires[2], wires[4], wires[6]])
    qml.DoubleExcitationPlus(params[31], wires=[wires[0], wires[2], wires[4], wires[6]])
    qml.DoubleExcitationMinus(params[32], wires=[wires[0], wires[2], wires[4], wires[6]])
    qml.QubitCarry(wires=[wires[0], wires[1], wires[6], wires[7]])
    qml.QubitSum(wires=[wires[2], wires[3], wires[7]])
    qml.OrbitalRotation(params[33], wires=[wires[0], wires[1], wires[5], wires[6]])
    qml.QFT(wires=[wires[0]])
    qml.ECR(wires=[wires[1], wires[3]])


# The expected values were generated using default.qubit
@pytest.mark.parametrize("method", [{"method": "mps", "max_bond_dim": 128}, {"method": "tn"}])
@pytest.mark.parametrize(
    "returns,expected_value",
    [
        ((qml.PauliX(0),), -0.094606003),
        ((qml.PauliY(0),), -0.138130983),
        ((qml.PauliZ(0),), 0.052683073),
        ((qml.PauliX(1),), -0.027114956),
        ((qml.PauliY(1),), 0.035227835),
        ((qml.PauliZ(1),), 0.130383680),
        ((qml.PauliX(2),), -0.112239026),
        ((qml.PauliY(2),), -0.043408985),
        ((qml.PauliZ(2),), -0.186733557),
        ((qml.PauliX(3),), 0.081030290),
        ((qml.PauliY(3),), 0.136389367),
        ((qml.PauliZ(3),), -0.024382650),
        ((qml.PauliX(0), qml.PauliY(1)), [-0.094606, 0.03522784]),
        (
            (
                qml.PauliZ(0),
                qml.PauliX(1),
                qml.PauliY(2),
            ),
            [0.05268307, -0.02711496, -0.04340899],
        ),
        (
            (
                qml.PauliY(0),
                qml.PauliZ(1),
                qml.PauliY(3),
            ),
            [-0.13813098, 0.13038368, 0.13638937],
        ),
        ((qml.PauliZ(0) @ qml.PauliY(3),), 0.174335019),
        ((qml.Hadamard(2),), -0.211405541),
        ((qml.Hadamard(3) @ qml.PauliZ(2),), -0.024206963),
        ((qml.PauliX(0) @ qml.PauliY(3),), 0.088232689),
        ((qml.PauliY(0) @ qml.PauliY(2) @ qml.PauliY(3),), 0.193644667),
        ((qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2),), -0.034583947),
        ((0.5 * qml.PauliZ(0) @ qml.PauliZ(2),), 0.002016079),
        (
            (qml.ops.LinearCombination([1.0, 2.0], [qml.X(0) @ qml.Z(1), qml.Y(3) @ qml.Z(2)])),
            [0.08618213, 0.09506244],
        ),
        ((qml.ops.prod(qml.X(0), qml.Y(1))), [-0.094606, 0.03522784]),
    ],
)
def test_integration_for_all_supported_gates(returns, expected_value, method):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations"""

    num_wires = 8
    dev_ltensor = LightningTensor(wires=range(num_wires), c_dtype=np.complex128, **method)

    def circuit(params):
        qml.BasisState(np.array([1, 0, 1, 0, 1, 0, 1, 0]), wires=range(num_wires))
        circuit_ansatz(params, wires=range(num_wires))
        return qml.math.hstack([qml.expval(r) for r in returns])

    n_params = 34
    np.random.seed(1337)
    params_init = np.random.rand(n_params)

    params = np.array(params_init, requires_grad=True)
    qnode_ltensor = qml.QNode(circuit, dev_ltensor)
    j_ltensor = qnode_ltensor(params)

    assert np.allclose(j_ltensor, expected_value, rtol=1e-6)


@pytest.mark.parametrize("method", [{"method": "mps", "max_bond_dim": 128}, {"method": "tn"}])
class TestSparseHExpval:
    """Test sparseH expectation values"""

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0) @ qml.Identity(1), 0.000000000, 1.000000000],
            [qml.Identity(0) @ qml.PauliX(1), -0.198669330, 0.960530638],
            [qml.PauliY(0) @ qml.Identity(1), -0.389418342, 0.848353326],
            [qml.Identity(0) @ qml.PauliY(1), 0.000000000, 1.000000119],
            [qml.PauliZ(0) @ qml.Identity(1), 0.921060994, 0.151646673],
            [qml.Identity(0) @ qml.PauliZ(1), 0.980066577, 0.039469480],
        ],
    )
    def test_sparse_Pauli_words(self, cases, qubit_device, method):
        """Test expval of some simple sparse Hamiltonian"""
        dev = qml.device(device_name, wires=4, **method)

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

    def test_expval_sparseH_not_supported(self, method):
        """Test that expval of SparseH is not supported."""
        with qml.queuing.AnnotatedQueue() as q:
            qml.expval(qml.SparseHamiltonian(qml.PauliX.compute_sparse_matrix(), wires=0))

        tensornet = LightningTensorNet(4, **method)

        m = LightningTensorMeasurements(tensornet)

        with pytest.raises(NotImplementedError, match="Sparse Hamiltonians are not supported."):
            m.expval(q.queue[0])

    def test_var_sparseH_not_supported(self, method):
        """Test that var of SparseH is not supported."""
        with qml.queuing.AnnotatedQueue() as q:
            qml.var(qml.SparseHamiltonian(qml.PauliX.compute_sparse_matrix(), wires=0))

        tensornet = LightningTensorNet(4, **method)

        m = LightningTensorMeasurements(tensornet)

        with pytest.raises(
            NotImplementedError,
            match="The var measurement does not support sparse Hamiltonian observables.",
        ):
            m.var(q.queue[0])

    def test_expval_hermitian_not_supported(self, method):
        """Test that expval of Hermitian with 1+ wires is not supported."""
        with qml.queuing.AnnotatedQueue() as q:
            qml.expval(qml.Hermitian(np.eye(4), wires=[0, 1]))

        tensornet = LightningTensorNet(4, **method)

        m = LightningTensorMeasurements(tensornet)

        with pytest.raises(
            ValueError, match="The number of Hermitian observables target wires should be 1."
        ):
            m.expval(q.queue[0])

    def test_var_hermitian_not_supported(self, method):
        """Test that var of Hermitian with 1+ wires is not supported."""
        with qml.queuing.AnnotatedQueue() as q:
            qml.var(qml.Hermitian(np.eye(4), wires=[0, 1]))

        tensornet = LightningTensorNet(4, **method)

        m = LightningTensorMeasurements(tensornet)

        with pytest.raises(
            ValueError, match="The number of Hermitian observables target wires should be 1."
        ):
            m.var(q.queue[0])


@pytest.mark.parametrize("method", [{"method": "mps", "max_bond_dim": 128}, {"method": "tn"}])
class TestQChem:
    """Integration tests for qchem module by parameter-shift and finite-diff differentiation methods."""

    # The expected values were generated using default.qubit
    @pytest.mark.parametrize(
        "diff_approach, expected_value",
        [
            ("parameter-shift", -0.17987143),
            ("finite-diff", -0.17987139),
        ],
    )
    def test_integration_H2_Hamiltonian(self, diff_approach, expected_value, method):
        symbols = ["H", "H"]

        geometry = np.array(
            [[-0.676411907, 0.000000000, 0.000000000], [0.676411907, 0.000000000, 0.000000000]],
            requires_grad=False,
        )

        mol = qml.qchem.Molecule(symbols, geometry, basis_name="STO-3G")

        H, qubits = qml.qchem.molecular_hamiltonian(
            symbols,
            geometry,
            basis="STO-3G",
        )

        singles, doubles = qml.qchem.excitations(mol.n_electrons, len(H.wires))

        num_params = len(singles + doubles)
        params = np.zeros(num_params, requires_grad=True)

        hf_state = qml.qchem.hf_state(mol.n_electrons, qubits)

        # Choose different batching supports here
        dev = qml.device(device_name, wires=qubits, **method)

        @qml.qnode(dev, diff_method=diff_approach)
        def circuit(params, excitations):
            qml.BasisState(hf_state, wires=range(qubits))
            for i, excitation in enumerate(excitations):
                if len(excitation) == 4:
                    qml.DoubleExcitation(params[i], wires=excitation)
                else:
                    qml.SingleExcitation(params[i], wires=excitation)
            return qml.expval(H)

        jac_func = qml.jacobian(circuit)

        params = qml.numpy.array([0.0] * len(doubles), requires_grad=True)
        jacs = jac_func(params, excitations=doubles)

        assert np.allclose(jacs, expected_value)
