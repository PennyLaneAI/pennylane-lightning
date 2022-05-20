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
Unit tests for Sparse Measures in lightning.qubit.
"""
import numpy as np
import pennylane as qml
from pennylane.measurements import (
    Expectation,
)

import pytest

try:
    from pennylane_lightning.lightning_qubit_ops import (
        MeasuresC64,
        MeasuresC128,
    )
except (ImportError, ModuleNotFoundError):
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


class TestSparseExpval:
    """Tests for the expval function"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return qml.device("lightning.qubit", wires=2, c_dtype=request.param)

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0) @ qml.Identity(1),  0.00000000000000000],
            [qml.Identity(0) @ qml.PauliX(1), -0.19866933079506122],
            [qml.PauliY(0) @ qml.Identity(1), -0.38941834230865050],
            [qml.Identity(0) @ qml.PauliY(1),  0.00000000000000000],
            [qml.PauliZ(0) @ qml.Identity(1),  0.92106099400288520],
            [qml.Identity(0) @ qml.PauliZ(1),  0.98006657784124170],
        ],
    )
    def test_sparse_2_gates(self, cases, tol, dev):
        """Test expval of some simple sparse Hamiltonian"""

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.expval(
                qml.SparseHamiltonian(
                    qml.utils.sparse_hamiltonian(qml.Hamiltonian([1], [cases[0]])), wires=[0, 1]
                )
            )

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)