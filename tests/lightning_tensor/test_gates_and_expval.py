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
    qml.MultiControlledX(wires=[wires[0], wires[1], wires[3]], control_values=[wires[0], wires[1]])
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
    qml.PSWAP(params[0], wires=[wires[6], wires[7]])
    qml.adjoint(qml.SISWAP(wires=[wires[0], wires[1]]))
    qml.SISWAP(wires=[wires[4], wires[5]])
    qml.SQISW(wires=[wires[1], wires[0]])
    # qml.CSWAP(wires=[wires[2], wires[4], wires[5]])
    qml.Toffoli(wires=[wires[0], wires[1], wires[2]])
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
    qml.IsingXY(params[14], wires=[wires[2], wires[1]])
    qml.IsingZZ(params[15], wires=[wires[2], wires[1]])
    qml.SingleExcitation(params[16], wires=[wires[2], wires[0]])
    qml.SingleExcitationPlus(params[17], wires=[wires[3], wires[1]])
    qml.SingleExcitationMinus(params[18], wires=[wires[4], wires[2]])
    qml.DoubleExcitation(params[19], wires=[wires[0], wires[1], wires[2], wires[3]])
    qml.QubitCarry(wires=[wires[0], wires[1], wires[6], wires[7]])
    qml.QubitSum(wires=[wires[2], wires[3], wires[7]])
    qml.OrbitalRotation(params[20], wires=[wires[0], wires[1], wires[5], wires[6]])
    qml.QFT(wires=[wires[0]])
    qml.ECR(wires=[wires[1], wires[3]])


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
def test_integration_for_all_supported_gates(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations"""
    num_wires = 8
    dev_default = qml.device("default.qubit", wires=range(num_wires))
    dev_ltensor = LightningTensor(wires=range(num_wires), max_bond_dim=16, c_dtype=np.complex128)

    def circuit(params):
        qml.BasisState(np.array([1, 0, 1, 0, 1, 0, 1, 0]), wires=range(num_wires))
        circuit_ansatz(params, wires=range(num_wires))
        return qml.math.hstack([qml.expval(r) for r in returns])

    n_params = 22
    np.random.seed(1337)
    params_init = np.random.rand(n_params)

    params = np.array(params_init, requires_grad=True)

    qnode_ltensor = qml.QNode(circuit, dev_ltensor)
    qnode_default = qml.QNode(circuit, dev_default)

    j_ltensor = qnode_ltensor(params)
    j_default = qnode_default(params)

    assert np.allclose(j_ltensor, j_default, rtol=1e-6)


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

    def test_expval_sparseH_not_supported(self):
        """Test that expval of SparseH is not supported."""
        with qml.queuing.AnnotatedQueue() as q:
            qml.expval(qml.SparseHamiltonian(qml.PauliX.compute_sparse_matrix(), wires=0))

        tensornet = LightningTensorNet(4, 10)
        m = LightningTensorMeasurements(tensornet)

        with pytest.raises(NotImplementedError, match="Sparse Hamiltonians are not supported."):
            m.expval(q.queue[0])

    def test_var_sparseH_not_supported(self):
        """Test that var of SparseH is not supported."""
        with qml.queuing.AnnotatedQueue() as q:
            qml.var(qml.SparseHamiltonian(qml.PauliX.compute_sparse_matrix(), wires=0))

        tensornet = LightningTensorNet(4, 10)
        m = LightningTensorMeasurements(tensornet)

        with pytest.raises(
            NotImplementedError,
            match="The var measurement does not support sparse Hamiltonian observables.",
        ):
            m.var(q.queue[0])

    def test_expval_hermitian_not_supported(self):
        """Test that expval of Hermitian with 1+ wires is not supported."""
        with qml.queuing.AnnotatedQueue() as q:
            qml.expval(qml.Hermitian(np.eye(4), wires=[0, 1]))

        tensornet = LightningTensorNet(4, 10)
        m = LightningTensorMeasurements(tensornet)

        with pytest.raises(
            ValueError, match="The number of Hermitian observables target wires should be 1."
        ):
            m.expval(q.queue[0])

    def test_var_hermitian_not_supported(self):
        """Test that var of Hermitian with 1+ wires is not supported."""
        with qml.queuing.AnnotatedQueue() as q:
            qml.var(qml.Hermitian(np.eye(4), wires=[0, 1]))

        tensornet = LightningTensorNet(4, 10)
        m = LightningTensorMeasurements(tensornet)

        with pytest.raises(
            ValueError, match="The number of Hermitian observables target wires should be 1."
        ):
            m.var(q.queue[0])


class QChem:
    """Integration tests for qchem module by parameter-shift and finite-diff differentiation methods."""

    @pytest.mark.parametrize("diff_approach", ["parameter-shift", "finite-diff"])
    def test_integration_H2_Hamiltonian(self, diff_approach):
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

        excitations = singles + doubles
        num_params = len(singles + doubles)
        params = np.zeros(num_params, requires_grad=True)

        hf_state = qml.qchem.hf_state(mol.n_electrons, qubits)

        # Choose different batching supports here
        dev = qml.device(device_name, wires=qubits)
        dev_comp = qml.device("default.qubit", wires=qubits)

        @qml.qnode(dev, diff_method=diff_approach)
        def circuit(params, excitations):
            qml.BasisState(hf_state, wires=range(qubits))
            for i, excitation in enumerate(excitations):
                if len(excitation) == 4:
                    qml.DoubleExcitation(params[i], wires=excitation)
                else:
                    qml.SingleExcitation(params[i], wires=excitation)
            return qml.expval(H)

        @qml.qnode(dev_comp, diff_method=diff_approach)
        def circuit_compare(params, excitations):
            qml.BasisState(hf_state, wires=range(qubits))

            for i, excitation in enumerate(excitations):
                if len(excitation) == 4:
                    qml.DoubleExcitation(params[i], wires=excitation)
                else:
                    qml.SingleExcitation(params[i], wires=excitation)
            return qml.expval(H)

        jac_func = qml.jacobian(circuit)
        jac_func_comp = qml.jacobian(circuit_compare)

        params = qml.numpy.array([0.0] * len(doubles), requires_grad=True)
        jacs = jac_func(params, excitations=doubles)
        jacs_comp = jac_func_comp(params, excitations=doubles)

        assert np.allclose(jacs, jacs_comp)
