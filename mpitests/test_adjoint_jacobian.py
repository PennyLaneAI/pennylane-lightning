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

    def test_not_expval_adj_mpi(self, dev_mpi):        
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""
        
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.1, wires=0)
            qml.var(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match="Adjoint differentiation method does"):
            dev_mpi.adjoint_jacobian(tape)
    
    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_finite_shots_warns_adj_mpi(self, isBatch_obs):
        """Tests warning raised when finite shots specified"""

        dev_mpi = qml.device("lightning.gpu", wires=1, shots=1, batch_obs=isBatch_obs)

        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        with pytest.warns(
            UserWarning,
            match="Requested adjoint differentiation to be computed with finite shots.",
        ):
            dev_mpi.adjoint_jacobian(tape)
    
    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_unsupported_op(self, isBatch_obs):
        """Test if a QuantumFunctionError is raised for an unsupported operation, i.e.,
        multi-parameter operations that are not qml.Rot"""
        num_wires = 8
        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128, batch_obs=isBatch_obs
        )

        with qml.tape.QuantumTape() as tape:
            qml.CRot(0.1, 0.2, 0.3, wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The CRot operation is not supported using the",
        ):
            dev_gpumpi.adjoint_jacobian(tape)
    
    @pytest.mark.parametrize("isBatch_obs", [False, True])
    def test_proj_unsupported(self, isBatch_obs):
        """Test if a QuantumFunctionError is raised for a Projector observable"""
        num_wires = 8
        dev_gpumpi = qml.device(
            "lightning.gpu", wires=num_wires, mpi=True, c_dtype=np.complex128, batch_obs=isBatch_obs
        )

        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0, 1], wires=[0, 1]))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="differentiation method does not support the Projector",
        ):
            dev_gpumpi.adjoint_jacobian(tape)

        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0], wires=[0]) @ qml.PauliZ(0))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="differentiation method does not support the Projector",
        ):
            dev_gpumpi.adjoint_jacobian(tape)
    
    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ])
    @pytest.mark.parametrize("stateprep", [qml.QubitStateVector, qml.StatePrep])
    def test_pauli_rotation_gradient(self, stateprep, G, theta, tol, dev_mpi):
        """Tests that the automatic gradients of Pauli rotations are correct."""

        num_wires = 8
        dev_cpu = qml.device("lightning.qubit", wires=num_wires)

        with qml.tape.QuantumTape() as tape:
            stateprep(np.array([1.0, -1.0]) / np.sqrt(2), wires=0)
            G(theta, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1}

        calculated_val = dev_mpi.adjoint_jacobian(tape)
        expected_val = dev_cpu.adjoint_jacobian(tape)

        assert np.allclose(calculated_val, expected_val, atol=tol, rtol=0)
    
    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("stateprep", [qml.QubitStateVector, qml.StatePrep])
    def test_Rot_gradient(self, stateprep, theta, tol, dev_mpi):
        """Tests that the device gradient of an arbitrary Euler-angle-parameterized gate is
        correct."""

        num_wires = 8
        dev_cpu = qml.device("lightning.qubit", wires=num_wires)

        params = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.tape.QuantumTape() as tape:
            stateprep(np.array([1.0, -1.0]) / np.sqrt(2), wires=0)
            qml.Rot(*params, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        calculated_val = dev_mpi.adjoint_jacobian(tape)
        expected_val = dev_cpu.adjoint_jacobian(tape)

        assert np.allclose(calculated_val, expected_val, atol=tol, rtol=0)
    
    @pytest.mark.parametrize("par", [1, -2, 1.623, -0.051, 0])  # integers, floats, zero
    def test_ry_gradient(self, par, tol, dev_mpi):
        """Test that the gradient of the RY gate matches the exact analytic formula."""
        with qml.tape.QuantumTape() as tape:
            qml.RY(par, wires=[0])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {0}

        # gradients
        exact = np.cos(par)
        gtapes, fn = qml.gradients.param_shift(tape)
        grad_PS = fn(qml.execute(gtapes, dev_mpi, gradient_fn=None))
        grad_A = dev_mpi.adjoint_jacobian(tape)

        # different methods must agree
        assert np.allclose(grad_PS, exact, atol=tol, rtol=0)
        assert np.allclose(grad_A, exact, atol=tol, rtol=0)