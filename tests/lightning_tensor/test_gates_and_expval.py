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

import pennylane as qp
import pytest
from conftest import PHI, THETA, LightningDevice, device_name
from pennylane import numpy as np
from pennylane.exceptions import DeviceError

if device_name != "lightning.tensor":
    pytest.skip(
        "Exclusive tests for Lightning Tensor device. Skipping.",
        allow_module_level=True,
    )
else:
    from pennylane_lightning.lightning_tensor import LightningTensor
    from pennylane_lightning.lightning_tensor._measurements import LightningTensorMeasurements
    from pennylane_lightning.lightning_tensor._tensornet import LightningTensorNet

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
    qp.Identity(wires=wires[0])
    qp.QubitUnitary(random_unitary, wires=[wires[1], wires[3]])
    qp.ControlledQubitUnitary(qp.matrix(qp.PauliX([wires[1]])), wires=[wires[0], wires[1]])
    qp.DiagonalQubitUnitary(np.array([1, 1]), wires=wires[2])
    qp.MultiControlledX(wires=[wires[0], wires[1], wires[3]], control_values=[0, 1])
    qp.MultiControlledX(wires=[wires[0], wires[1], wires[3]], control_values=[1, 1])
    qp.MultiControlledX(wires=[wires[0], wires[1], wires[3]], control_values=[0, 0])
    qp.PauliX(wires=wires[1])
    qp.PauliY(wires=wires[2])
    qp.PauliZ(wires=wires[3])
    qp.Hadamard(wires=wires[4])
    qp.adjoint(qp.S(wires=wires[4]))
    qp.S(wires=wires[5])
    qp.adjoint(qp.T(wires=wires[1]))
    qp.T(wires=wires[0])
    qp.adjoint(qp.SX(wires=wires[0]))
    qp.SX(wires=wires[1])
    qp.CNOT(wires=[wires[6], wires[7]])
    qp.SWAP(wires=[wires[2], wires[3]])
    qp.adjoint(qp.ISWAP(wires=[wires[0], wires[1]]))
    qp.ISWAP(wires=[wires[4], wires[5]])
    qp.ISWAP(wires=[wires[4], wires[6]])
    qp.PSWAP(params[0], wires=[wires[6], wires[7]])
    qp.PSWAP(params[1], wires=[wires[0], wires[7]])
    qp.adjoint(qp.SISWAP(wires=[wires[0], wires[1]]))
    qp.adjoint(qp.SISWAP(wires=[wires[0], wires[4]]))
    qp.SISWAP(wires=[wires[4], wires[5]])
    qp.SISWAP(wires=[wires[2], wires[5]])
    qp.SQISW(wires=[wires[1], wires[0]])
    qp.SQISW(wires=[wires[5], wires[0]])
    qp.CSWAP(wires=[wires[2], wires[4], wires[5]])
    qp.Toffoli(wires=[wires[0], wires[1], wires[2]])
    qp.Toffoli(wires=[wires[0], wires[1], wires[5]])
    qp.CY(wires=[wires[0], wires[2]])
    qp.CZ(wires=[wires[1], wires[3]])
    qp.PhaseShift(params[2], wires=wires[2])
    qp.ControlledPhaseShift(params[3], wires=[wires[0], wires[5]])
    qp.RX(params[4], wires=wires[0])
    qp.RY(params[5], wires=wires[1])
    qp.RZ(params[6], wires=wires[3])
    qp.Rot(params[7], params[8], params[9], wires=wires[0])
    qp.CRX(params[10], wires=[wires[1], wires[0]])
    qp.CRY(params[11], wires=[wires[3], wires[2]])
    qp.CRZ(params[12], wires=[wires[2], wires[1]])
    qp.CRX(params[13], wires=[wires[1], wires[5]])
    qp.CRY(params[14], wires=[wires[3], wires[6]])
    qp.CRZ(params[15], wires=[wires[2], wires[0]])
    qp.IsingXX(params[16], wires=[wires[1], wires[0]])
    qp.IsingYY(params[17], wires=[wires[3], wires[2]])
    qp.IsingXY(params[18], wires=[wires[2], wires[1]])
    qp.IsingZZ(params[19], wires=[wires[2], wires[1]])
    qp.IsingXX(params[20], wires=[wires[1], wires[5]])
    qp.IsingYY(params[21], wires=[wires[3], wires[0]])
    qp.IsingXY(params[22], wires=[wires[2], wires[4]])
    qp.IsingZZ(params[23], wires=[wires[2], wires[0]])
    qp.SingleExcitation(params[24], wires=[wires[2], wires[0]])
    qp.SingleExcitationPlus(params[25], wires=[wires[3], wires[1]])
    qp.SingleExcitationMinus(params[26], wires=[wires[4], wires[2]])
    qp.DoubleExcitation(params[27], wires=[wires[0], wires[1], wires[2], wires[3]])
    qp.DoubleExcitationPlus(params[28], wires=[wires[1], wires[2], wires[3], wires[4]])
    qp.DoubleExcitationMinus(params[29], wires=[wires[2], wires[3], wires[4], wires[5]])
    qp.DoubleExcitation(params[30], wires=[wires[0], wires[2], wires[4], wires[6]])
    qp.DoubleExcitationPlus(params[31], wires=[wires[0], wires[2], wires[4], wires[6]])
    qp.DoubleExcitationMinus(params[32], wires=[wires[0], wires[2], wires[4], wires[6]])
    qp.QubitCarry(wires=[wires[0], wires[1], wires[6], wires[7]])
    qp.QubitSum(wires=[wires[2], wires[3], wires[7]])
    qp.OrbitalRotation(params[33], wires=[wires[0], wires[1], wires[5], wires[6]])
    qp.QFT(wires=[wires[0]])
    qp.ECR(wires=[wires[1], wires[3]])


# The expected values were generated using default.qubit
@pytest.mark.local_salt(42)
@pytest.mark.parametrize("method", [{"method": "mps", "max_bond_dim": 128}, {"method": "tn"}])
@pytest.mark.parametrize(
    "returns",
    [
        (qp.PauliX(0),),
        (qp.PauliY(0),),
        (qp.PauliZ(0),),
        (qp.PauliX(1),),
        (qp.PauliY(1),),
        (qp.PauliZ(1),),
        (qp.PauliX(2),),
        (qp.PauliY(2),),
        (qp.PauliZ(2),),
        (qp.PauliX(3),),
        (qp.PauliY(3),),
        (qp.PauliZ(3),),
        (qp.PauliX(0), qp.PauliY(1)),
        (
            qp.PauliZ(0),
            qp.PauliX(1),
            qp.PauliY(2),
        ),
        (
            qp.PauliY(0),
            qp.PauliZ(1),
            qp.PauliY(3),
        ),
        (qp.PauliZ(0) @ qp.PauliY(3),),
        (qp.Hadamard(2),),
        (qp.Hadamard(3) @ qp.PauliZ(2),),
        (qp.PauliX(0) @ qp.PauliY(3),),
        (qp.PauliY(0) @ qp.PauliY(2) @ qp.PauliY(3),),
        (qp.PauliZ(0) @ qp.PauliZ(1) @ qp.PauliZ(2),),
        (0.5 * qp.PauliZ(0) @ qp.PauliZ(2),),
        (qp.ops.LinearCombination([1.0, 2.0], [qp.X(0) @ qp.Z(1), qp.Y(3) @ qp.Z(2)])),
        (qp.ops.prod(qp.X(0), qp.Y(1))),
    ],
)
def test_integration_for_all_supported_gates(returns, method, seed):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations"""

    num_wires = 8
    dev_ltensor = LightningTensor(wires=range(num_wires), c_dtype=np.complex128, **method)
    dev_default = qp.device("default.qubit", wires=range(num_wires))

    def circuit(params):
        qp.BasisState(np.array([1, 0, 1, 0, 1, 0, 1, 0]), wires=range(num_wires))
        circuit_ansatz(params, wires=range(num_wires))
        return qp.math.hstack([qp.expval(r) for r in returns])

    n_params = 34
    rng = np.random.default_rng(seed)
    params_init = rng.random(n_params)

    params = np.array(params_init, requires_grad=True)
    qnode_ltensor = qp.QNode(circuit, dev_ltensor)
    j_ltensor = qnode_ltensor(params)

    ref = qp.QNode(circuit, dev_default)(params)

    assert np.allclose(j_ltensor, ref, rtol=2e-6)


@pytest.mark.parametrize("method", [{"method": "mps", "max_bond_dim": 128}, {"method": "tn"}])
class TestSparseHExpval:
    """Test sparseH expectation values"""

    @pytest.mark.parametrize(
        "cases",
        [
            [qp.PauliX(0) @ qp.Identity(1), 0.000000000, 1.000000000],
            [qp.Identity(0) @ qp.PauliX(1), -0.198669330, 0.960530638],
            [qp.PauliY(0) @ qp.Identity(1), -0.389418342, 0.848353326],
            [qp.Identity(0) @ qp.PauliY(1), 0.000000000, 1.000000119],
            [qp.PauliZ(0) @ qp.Identity(1), 0.921060994, 0.151646673],
            [qp.Identity(0) @ qp.PauliZ(1), 0.980066577, 0.039469480],
        ],
    )
    def test_sparse_pauli_words(self, cases, qubit_device, method):
        """Test expval of some simple sparse Hamiltonian"""
        dev = qp.device(device_name, wires=4, **method)

        @qp.qnode(dev, diff_method="parameter-shift")
        def circuit_expval():
            qp.RX(0.4, wires=[0])
            qp.RY(-0.2, wires=[1])
            return qp.expval(
                qp.SparseHamiltonian(qp.Hamiltonian([1], [cases[0]]).sparse_matrix(), wires=[0, 1])
            )

        with pytest.raises(DeviceError):
            circuit_expval()

    def test_expval_sparseH_not_supported(self, method):
        """Test that expval of SparseH is not supported."""
        with qp.queuing.AnnotatedQueue() as q:
            qp.expval(qp.SparseHamiltonian(qp.PauliX.compute_sparse_matrix(), wires=0))

        tensornet = LightningTensorNet(4, **method)

        m = LightningTensorMeasurements(tensornet)

        with pytest.raises(NotImplementedError, match="Sparse Observables are not supported."):
            m.expval(q.queue[0])

    def test_var_sparseH_not_supported(self, method):
        """Test that var of SparseH is not supported."""
        with qp.queuing.AnnotatedQueue() as q:
            qp.var(qp.SparseHamiltonian(qp.PauliX.compute_sparse_matrix(), wires=0))

        tensornet = LightningTensorNet(4, **method)

        m = LightningTensorMeasurements(tensornet)

        with pytest.raises(
            NotImplementedError,
            match="The var measurement does not support sparse observables.",
        ):
            m.var(q.queue[0])

    def test_expval_hermitian_not_supported(self, method):
        """Test that expval of Hermitian with 1+ wires is not supported."""
        with qp.queuing.AnnotatedQueue() as q:
            qp.expval(qp.Hermitian(np.eye(4), wires=[0, 1]))

        tensornet = LightningTensorNet(4, **method)

        m = LightningTensorMeasurements(tensornet)

        with pytest.raises(
            ValueError,
            match="The number of Hermitian observables target wires should be 1.",
        ):
            m.expval(q.queue[0])

    def test_var_hermitian_not_supported(self, method):
        """Test that var of Hermitian with 1+ wires is not supported."""
        with qp.queuing.AnnotatedQueue() as q:
            qp.var(qp.Hermitian(np.eye(4), wires=[0, 1]))

        tensornet = LightningTensorNet(4, **method)

        m = LightningTensorMeasurements(tensornet)

        with pytest.raises(
            ValueError,
            match="The number of Hermitian observables target wires should be 1.",
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
            [
                [-0.676411907, 0.000000000, 0.000000000],
                [0.676411907, 0.000000000, 0.000000000],
            ],
            requires_grad=False,
        )

        mol = qp.qchem.Molecule(symbols, geometry, basis_name="STO-3G")

        H, qubits = qp.qchem.molecular_hamiltonian(
            symbols,
            geometry,
            basis="STO-3G",
        )

        singles, doubles = qp.qchem.excitations(mol.n_electrons, len(H.wires))

        num_params = len(singles + doubles)
        params = np.zeros(num_params, requires_grad=True)

        hf_state = qp.qchem.hf_state(mol.n_electrons, qubits)

        # Choose different batching supports here
        dev = qp.device(device_name, wires=qubits, **method)

        @qp.qnode(dev, diff_method=diff_approach)
        def circuit(params, excitations):
            qp.BasisState(hf_state, wires=range(qubits))
            for i, excitation in enumerate(excitations):
                if len(excitation) == 4:
                    qp.DoubleExcitation(params[i], wires=excitation)
                else:
                    qp.SingleExcitation(params[i], wires=excitation)
            return qp.expval(H)

        jac_func = qp.jacobian(circuit)

        params = qp.numpy.array([0.0] * len(doubles), requires_grad=True)
        jacs = jac_func(params, excitations=doubles)

        assert np.allclose(jacs, expected_value)
