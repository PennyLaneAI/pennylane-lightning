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
Unit tests for the :mod:`pennylane_lightning_gpu.LightningGPU` device (MPI).
"""
# pylint: disable=protected-access,cell-var-from-loop
from mpi4py import MPI
import numpy as np
import pennylane as qml
from pennylane import numpy as np
from pennylane import QNode, qnode
import pytest

import itertools

fixture_params = itertools.product(
    [np.complex64, np.complex128],
    [True, False],
)

class TestAdjointJacobian:
    """Tests for the adjoint_jacobian method"""

    @pytest.fixture(params=fixture_params)
    def dev_mpi(self, request):
        params = request.param
        return qml.device("lightning.gpu", wires=8, mpi=True, c_dtype=params[0], batch_obs=params[1])

    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_not_expval_adj_mpi(self, isBatch_obs, dev_mpi):        
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""
        
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.1, wires=0)
            qml.var(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match="Adjoint differentiation method does"):
            dev_mpi.adjoint_jacobian(tape)
'''
    @pytest.fixture(params=[np.complex64, np.complex128])
    @pytest.mark.parametrize("par", [1, -2, 1.623, -0.051, 0])  # integers, floats, zero
    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_ry_gradient(self, par, tol, isBatch_obs, request):
        """Test that the gradient of the RY gate matches the exact analytic formula."""
        num_wires = 3
        dev_gpumpi = qml.device(
            "lightning.gpu",
            wires=num_wires,
            mpi=True,
            c_dtype=request.params,
            batch_obs=isBatch_obs,
        )

        with qml.tape.QuantumTape() as tape:
            qml.RY(par, wires=[0])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {0}

        # gradients
        exact = np.cos(par)
        gtapes, fn = qml.gradients.param_shift(tape)
        grad_PS = fn(qml.execute(gtapes, dev_gpumpi, gradient_fn=None))
        grad_A = dev_gpumpi.adjoint_jacobian(tape)

        # different methods must agree
        assert np.allclose(grad_PS, exact, atol=tol, rtol=0)
        assert np.allclose(grad_A, exact, atol=tol, rtol=0)
'''