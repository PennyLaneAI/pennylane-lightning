# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Integration tests for the ``execute`` method of Lightning devices.
"""
import functools

import pennylane as qml
import pytest
from conftest import LightningDevice, device_name
from pennylane import numpy as np

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.mark.usefixtures("use_legacy_and_new_opmath")
@pytest.mark.parametrize("diff_method", ("param_shift", "finite_diff"))
class TestQChem:
    """Test tapes returning the expectation values of a Hamiltonian, with a qchem workflow."""

    def test_VQE_gradients(self, diff_method, tol):
        """Test if the VQE procedure returns the expected gradients."""

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

        with qml.tape.QuantumTape() as tape:
            qml.BasisState(hf_state, wires=range(qubits))

            for i, excitation in enumerate(excitations):
                if len(excitation) == 4:
                    qml.DoubleExcitation(params[i], wires=excitation)
                elif len(excitation) == 2:
                    qml.SingleExcitation(params[i], wires=excitation)

            qml.expval(H)

        num_params = len(excitations)
        tape.trainable_params = np.linspace(1, num_params, num_params, dtype=int).tolist()

        gradient_tapes, fn_grad = getattr(qml.gradients, diff_method)(tape)

        dev_l = qml.device(device_name, wires=qubits)
        dev_d = qml.device("default.qubit", wires=qubits)

        def dev_l_execute(t):
            dev = qml.device(device_name, wires=qubits)
            return dev.execute(t)

        grad_dev_l = fn_grad([dev_l_execute(t) for t in gradient_tapes])
        grad_qml_l = fn_grad(qml.execute(gradient_tapes, dev_l))

        grad_qml_d = fn_grad(qml.execute(gradient_tapes, dev_d))

        assert np.allclose(grad_dev_l, grad_qml_l, tol)
        assert np.allclose(grad_dev_l, grad_qml_d, tol)
