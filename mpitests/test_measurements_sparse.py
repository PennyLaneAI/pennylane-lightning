# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Unit tests for sparse measurements on :mod:`pennylane_lightning` MPI-enabled devices.
"""

# pylint: disable=protected-access,too-few-public-methods,unused-import,missing-function-docstring,too-many-arguments,too-many-positional-arguments

import numpy as np
import pennylane as qp
import pytest
from conftest import LightningDevice as ld
from conftest import device_name
from mpi4py import MPI
from pennylane import qchem

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

if device_name != "lightning.gpu":
    pytest.skip("Kokkos MPI does not yet support Sparse.", allow_module_level=True)


class TestSparseExpval:
    """Tests for the expval function"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return qp.device(device_name, mpi=True, wires=2, c_dtype=request.param)

    @pytest.mark.parametrize(
        "cases",
        [
            [
                qp.PauliX(0) @ qp.Identity(1),
                0.00000000000000000,
                1.000000000000000000,
            ],
            [
                qp.Identity(0) @ qp.PauliX(1),
                -0.19866933079506122,
                0.960530638694763184,
            ],
            [
                qp.PauliY(0) @ qp.Identity(1),
                -0.38941834230865050,
                0.848353326320648193,
            ],
            [
                qp.Identity(0) @ qp.PauliY(1),
                0.00000000000000000,
                1.000000119209289551,
            ],
            [
                qp.PauliZ(0) @ qp.Identity(1),
                0.92106099400288520,
                0.151646673679351807,
            ],
            [
                qp.Identity(0) @ qp.PauliZ(1),
                0.98006657784124170,
                0.039469480514526367,
            ],
        ],
    )
    def test_sparse_pauli_words(self, cases, tol, dev):
        """Test expval of some simple sparse Hamiltonian"""

        # Compute the sparse matrix of the input operator
        # This is done outside of the QNode to avoid queuing the `Hamiltonian`
        matrix = qp.Hamiltonian([1], [cases[0]]).sparse_matrix()

        @qp.qnode(dev, diff_method="parameter-shift")
        def circuit_expval():
            qp.RX(0.4, wires=[0])
            qp.RY(-0.2, wires=[1])
            return qp.expval(qp.SparseHamiltonian(matrix, wires=[0, 1]))

        assert np.allclose(circuit_expval(), cases[1], atol=tol, rtol=0)

        @qp.qnode(dev, diff_method="parameter-shift")
        def circuit_var():
            qp.RX(0.4, wires=[0])
            qp.RY(-0.2, wires=[1])
            return qp.var(qp.SparseHamiltonian(matrix, wires=[0, 1]))

        assert np.allclose(circuit_var(), cases[2], atol=tol, rtol=0)


class TestSparseExpvalQChem:
    """Tests for the expval function with qchem workflow"""

    symbols = ["Li", "H"]
    geometry = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.969280527])

    H, qubits = qchem.molecular_hamiltonian(
        symbols,
        geometry,
    )

    active_electrons = 1

    hf_state = qchem.hf_state(active_electrons, qubits)

    singles, doubles = qchem.excitations(active_electrons, qubits)
    excitations = singles + doubles

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    @pytest.mark.parametrize(
        "qubits, wires, H, hf_state, excitations",
        [
            [qubits, range(qubits), H, hf_state, excitations],
            [
                qubits,
                [2, 10, 5, 6, 9, 3, 4, 1, 0, 8, 11, 7],
                H,
                hf_state,
                excitations,
            ],
        ],
    )
    def test_sparse_pauli_words(self, qubits, wires, H, hf_state, excitations, tol, dtype):
        """Test expval of some simple sparse Hamiltonian"""

        H_sparse = H.sparse_matrix(wires)

        dev = qp.device(device_name, mpi=True, wires=wires, c_dtype=dtype)

        @qp.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qp.BasisState(hf_state, wires=range(qubits))

            for excitation in excitations:
                if len(excitation) == 4:
                    qp.DoubleExcitation(1, wires=excitation)
                elif len(excitation) == 2:
                    qp.SingleExcitation(1, wires=excitation)

            return qp.expval(qp.SparseHamiltonian(H_sparse, wires=wires))

        dev_default = qp.device("default.qubit", wires=qubits)

        @qp.qnode(dev_default, diff_method="parameter-shift")
        def circuit_default():
            qp.BasisState(hf_state, wires=range(qubits))

            for excitation in excitations:
                if len(excitation) == 4:
                    qp.DoubleExcitation(1, wires=excitation)
                elif len(excitation) == 2:
                    qp.SingleExcitation(1, wires=excitation)

            return qp.expval(qp.SparseHamiltonian(H_sparse, wires=wires))

        assert np.allclose(circuit(), circuit_default(), atol=tol, rtol=0)
