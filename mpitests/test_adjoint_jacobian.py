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
# pylint: disable=protected-access,cell-var-from-loop,c-extension-no-member
import itertools
import math

import pennylane as qml
import pytest
from conftest import LightningDevice as ld
from conftest import device_name
from mpi4py import MPI
from pennylane import QNode
from pennylane import numpy as np
from pennylane import qnode
from scipy.stats import unitary_group

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

I, X, Y, Z = (
    np.eye(2),
    qml.PauliX.compute_matrix(),
    qml.PauliY.compute_matrix(),
    qml.PauliZ.compute_matrix(),
)

# Tuple passed to distributed device ctor
# np.complex for data type and True or False
# for enabling batched_obs.
fixture_params = itertools.product(
    [np.complex64, np.complex128],
    [True, False],
)


@pytest.fixture(name="dev", params=fixture_params)
def fixture_dev(request):
    """Returns a PennyLane device."""
    return qml.device(
        device_name,
        wires=8,
        mpi=True,
        c_dtype=request.param[0],
        batch_obs=request.param[1],
    )


def Rx(theta):
    r"""One-qubit rotation about the x axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}`
    """
    return math.cos(theta / 2) * I + 1j * math.sin(-theta / 2) * X


def Ry(theta):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    return math.cos(theta / 2) * I + 1j * math.sin(-theta / 2) * Y


def Rz(theta):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    return math.cos(theta / 2) * I + 1j * math.sin(-theta / 2) * Z


class TestAdjointJacobian:  # pylint: disable=too-many-public-methods
    """Tests for the adjoint_jacobian method"""

    def test_not_expval(self, dev):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.1, wires=0)
            qml.var(qml.PauliZ(0))

        with pytest.raises(
            qml.QuantumFunctionError, match="Adjoint differentiation method does not"
        ):
            dev.adjoint_jacobian(tape)

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.1, wires=0)
            qml.state()

        if device_name == "lightning.gpu":
            message = "Adjoint differentiation does not support State measurements."
        else:
            message = "Adjoint differentiation method does not support measurement StateMP."
        with pytest.raises(
            qml.QuantumFunctionError,
            match=message,
        ):
            dev.adjoint_jacobian(tape)

    def test_finite_shots_warns(self):
        """Tests warning raised when finite shots specified"""

        dev = qml.device(device_name, wires=8, mpi=True, shots=1)

        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        with pytest.warns(
            UserWarning,
            match="Requested adjoint differentiation to be computed with finite shots.",
        ):
            dev.adjoint_jacobian(tape)

    def test_empty_measurements(self, dev):
        """Tests if an empty array is returned when the measurements of the tape is empty."""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])

        jac = dev.adjoint_jacobian(tape)
        assert len(jac) == 0

    def test_unsupported_op(self, dev):
        """Test if a QuantumFunctionError is raised for an unsupported operation, i.e.,
        multi-parameter operations that are not qml.Rot"""

        with qml.tape.QuantumTape() as tape:
            qml.CRot(0.1, 0.2, 0.3, wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The CRot operation is not supported using the",
        ):
            dev.adjoint_jacobian(tape)

    def test_proj_unsupported(self, dev):
        """Test if a QuantumFunctionError is raised for a Projector observable"""
        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0, 1], wires=[0, 1]))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="differentiation method does not support the Projector",
        ):
            dev.adjoint_jacobian(tape)

        with qml.tape.QuantumTape() as tape:
            qml.CRX(0.1, wires=[0, 1])
            qml.expval(qml.Projector([0], wires=[0]) @ qml.PauliZ(0))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="differentiation method does not support the Projector",
        ):
            dev.adjoint_jacobian(tape)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ])
    @pytest.mark.parametrize("stateprep", [qml.QubitStateVector, qml.StatePrep])
    def test_pauli_rotation_gradient(self, stateprep, G, theta, dev):
        """Tests that the automatic gradients of Pauli rotations are correct."""
        random_state = np.array(
            [0.43593284 - 0.02945156j, 0.40812291 + 0.80158023j], requires_grad=False
        )

        tape = qml.tape.QuantumScript(
            [stateprep(random_state, 0), G(theta, 0)], [qml.expval(qml.PauliZ(0))]
        )

        tape.trainable_params = {1}

        calculated_val = dev.adjoint_jacobian(tape)

        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        # compare to finite differences
        tapes, fn = qml.gradients.param_shift(tape)
        numeric_val = fn(qml.execute(tapes, dev, None))
        assert np.allclose(calculated_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("stateprep", [qml.QubitStateVector, qml.StatePrep])
    def test_Rot_gradient(self, stateprep, theta, dev):
        """Tests that the device gradient of an arbitrary Euler-angle-parameterized gate is
        correct."""
        params = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.tape.QuantumTape() as tape:
            stateprep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            qml.Rot(*params, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        calculated_val = dev.adjoint_jacobian(tape)

        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        # compare to finite differences
        tapes, fn = qml.gradients.param_shift(tape)
        numeric_val = fn(qml.execute(tapes, dev, None))
        assert np.allclose(calculated_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("par", [1, -2, 1.623, -0.051, 0])  # integers, floats, zero
    def test_ry_gradient(self, par, tol, dev):
        """Test that the gradient of the RY gate matches the exact analytic formula."""
        with qml.tape.QuantumTape() as tape:
            qml.RY(par, wires=[0])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {0}

        # gradients
        exact = np.cos(par)
        grad_A = dev.adjoint_jacobian(tape)

        # different methods must agree
        assert np.allclose(grad_A, exact, atol=tol, rtol=0)

    def test_rx_gradient(self, tol, dev):
        """Test that the gradient of the RX gate matches the known formula."""
        a = 0.7418

        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.expval(qml.PauliZ(0))

        # circuit jacobians
        dev_jacobian = dev.adjoint_jacobian(tape)
        expected_jacobian = -np.sin(a)
        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    def test_multiple_rx_gradient_pauliz(self, tol, dev):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result."""
        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        with qml.tape.QuantumTape() as tape:
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[2], wires=2)

            for idx in range(3):
                qml.expval(qml.PauliZ(idx))

        # circuit jacobians
        dev_jacobian = dev.adjoint_jacobian(tape)
        expected_jacobian = -np.diag(np.sin(params))
        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    def test_multiple_rx_gradient_hermitian(self, tol, dev):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result
        with Hermitian observable
        """
        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        with qml.tape.QuantumTape() as tape:
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[2], wires=2)

            for idx in range(3):
                qml.expval(qml.Hermitian([[1, 0], [0, -1]], wires=[idx]))

        tape.trainable_params = {0, 1, 2}
        # circuit jacobians
        dev_jacobian = dev.adjoint_jacobian(tape)
        expected_jacobian = -np.diag(np.sin(params))

        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    qubit_ops = [getattr(qml, name) for name in qml.ops._qubit__ops__]  # pylint: disable=no-member
    ops = {qml.RX, qml.RY, qml.RZ, qml.PhaseShift, qml.CRX, qml.CRY, qml.CRZ, qml.Rot}

    def test_multiple_rx_gradient_expval_hermitian(self, tol, dev):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result
        with Hermitian observable
        """
        params = np.array([np.pi / 3, np.pi / 4, np.pi / 5])

        with qml.tape.QuantumTape() as tape:
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[2], wires=2)

            qml.expval(
                qml.Hermitian(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                    wires=[0, 2],
                )
            )

        tape.trainable_params = {0, 1, 2}
        dev_jacobian = dev.adjoint_jacobian(tape)
        expected_jacobian = np.array(
            [
                -np.sin(params[0]) * np.cos(params[2]),
                0,
                -np.cos(params[0]) * np.sin(params[2]),
            ]
        )

        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    qubit_ops = [getattr(qml, name) for name in qml.ops._qubit__ops__]  # pylint: disable=no-member
    ops = {qml.RX, qml.RY, qml.RZ, qml.PhaseShift, qml.CRX, qml.CRY, qml.CRZ, qml.Rot}

    def test_multiple_rx_gradient_expval_hamiltonian(self, tol, dev):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result
        with Hermitian observable
        """
        params = np.array([np.pi / 3, np.pi / 4, np.pi / 5])

        ham = qml.Hamiltonian(
            [1.0, 0.3, 0.3, 0.4],
            [
                qml.PauliX(0) @ qml.PauliX(1),
                qml.PauliZ(0),
                qml.PauliZ(1),
                qml.Hermitian(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                    wires=[0, 2],
                ),
            ],
        )

        with qml.tape.QuantumTape() as tape:
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[2], wires=2)

            qml.expval(ham)

        tape.trainable_params = {0, 1, 2}
        dev_jacobian = dev.adjoint_jacobian(tape)
        expected_jacobian = (
            0.3 * np.array([-np.sin(params[0]), 0, 0])
            + 0.3 * np.array([0, -np.sin(params[1]), 0])
            + 0.4
            * np.array(
                [
                    -np.sin(params[0]) * np.cos(params[2]),
                    0,
                    -np.cos(params[0]) * np.sin(params[2]),
                ]
            )
        )

        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    qubit_ops = [getattr(qml, name) for name in qml.ops._qubit__ops__]  # pylint: disable=no-member
    ops = {qml.RX, qml.RY, qml.RZ, qml.PhaseShift, qml.CRX, qml.CRY, qml.CRZ, qml.Rot}

    @pytest.mark.parametrize("obs", [qml.PauliX, qml.PauliY])
    @pytest.mark.parametrize(
        "op",
        [
            qml.RX(0.4, wires=0),
            qml.RY(0.6, wires=0),
            qml.RZ(0.8, wires=0),
            qml.CRX(1.0, wires=[0, 1]),
            qml.CRY(2.0, wires=[0, 1]),
            qml.CRZ(3.0, wires=[0, 1]),
            qml.Rot(0.2, -0.1, 0.2, wires=0),
        ],
    )
    def test_gradients_pauliz(self, op, obs, dev):
        """Tests that the gradients of circuits match between the finite difference and device
        methods."""
        # op.num_wires and op.num_params must be initialized a priori
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.RX(0.543, wires=0)
            qml.CNOT(wires=[0, 1])

            op  # pylint: disable=pointless-statement

            qml.Rot(1.3, -2.3, 0.5, wires=[0])
            qml.RZ(-0.5, wires=0)
            qml.adjoint(qml.RY(0.5, wires=1), lazy=False)
            qml.CNOT(wires=[0, 1])

            qml.expval(obs(wires=0))
            qml.expval(qml.PauliZ(wires=1))

        tape.trainable_params = set(range(1, 1 + op.num_params))

        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        # pylint: disable=unnecessary-direct-lambda-call
        grad_F = (lambda t, fn: fn(qml.execute(t, dev, None)))(*qml.gradients.param_shift(tape))
        grad_D = dev.adjoint_jacobian(tape)

        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "op",
        [
            qml.RX(0.4, wires=0),
            qml.RY(0.6, wires=0),
            qml.RZ(0.8, wires=0),
            qml.CRX(1.0, wires=[0, 1]),
            qml.CRY(2.0, wires=[0, 1]),
            qml.CRZ(3.0, wires=[0, 1]),
            qml.Rot(0.2, -0.1, 0.2, wires=0),
        ],
    )
    def test_gradients_hermitian(self, op, dev):
        """Tests that the gradients of circuits match between the finite difference and device
        methods."""
        # op.num_wires and op.num_params must be initialized a priori
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.RX(0.543, wires=0)
            qml.CNOT(wires=[0, 1])

            op.queue()

            qml.Rot(1.3, -2.3, 0.5, wires=[0])
            qml.RZ(-0.5, wires=0)
            qml.adjoint(qml.RY(0.5, wires=1), lazy=False)
            qml.CNOT(wires=[0, 1])

            qml.expval(
                qml.Hermitian(
                    [[0, 0, 1, 1], [0, 1, 2, 1], [1, 2, 1, 0], [1, 1, 0, 0]],
                    wires=[0, 1],
                )
            )

        tape.trainable_params = set(range(1, 1 + op.num_params))

        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        # pylint: disable=unnecessary-direct-lambda-call
        grad_F = (lambda t, fn: fn(qml.execute(t, dev, None)))(*qml.gradients.param_shift(tape))
        grad_D = dev.adjoint_jacobian(tape)

        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    def test_gradient_gate_with_multiple_parameters_pauliz(self, dev):
        """Tests that gates with multiple free parameters yield correct gradients."""
        x, y, z = [0.5, 0.3, -0.7]

        tape = qml.tape.QuantumScript(
            [
                qml.RX(0.4, wires=[0]),
                qml.Rot(x, y, z, wires=[0]),
                qml.RY(-0.2, wires=[0]),
            ],
            [qml.expval(qml.PauliZ(0))],
        )

        tape.trainable_params = {1, 2, 3}

        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        grad_D = dev.adjoint_jacobian(tape)
        tapes, fn = qml.gradients.param_shift(tape)
        grad_F = fn(qml.execute(tapes, dev, None))

        # gradient has the correct shape and every element is nonzero
        assert len(grad_D) == 3
        assert all(isinstance(v, np.ndarray) for v in grad_D)
        assert np.count_nonzero(grad_D) == 3
        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    def test_gradient_gate_with_multiple_parameters_hermitian(self, dev):
        """Tests that gates with multiple free parameters yield correct gradients."""
        x, y, z = [0.5, 0.3, -0.7]

        tape = qml.tape.QuantumScript(
            [
                qml.RX(0.4, wires=[0]),
                qml.Rot(x, y, z, wires=[0]),
                qml.RY(-0.2, wires=[0]),
            ],
            [qml.expval(qml.Hermitian([[0, 1], [1, 1]], wires=0))],
        )

        tape.trainable_params = {1, 2, 3}

        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        grad_D = dev.adjoint_jacobian(tape)
        tapes, fn = qml.gradients.param_shift(tape)
        grad_F = fn(qml.execute(tapes, dev, None))

        # gradient has the correct shape and every element is nonzero
        assert len(grad_D) == 3
        assert all(isinstance(v, np.ndarray) for v in grad_D)
        assert np.count_nonzero(grad_D) == 3
        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    def test_gradient_gate_with_multiple_parameters_hamiltonian(self, dev):
        """Tests that gates with multiple free parameters yield correct gradients."""
        x, y, z = [0.5, 0.3, -0.7]

        ham = qml.Hamiltonian(
            [1.0, 0.3, 0.3],
            [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0), qml.PauliZ(1)],
        )

        tape = qml.tape.QuantumScript(
            [
                qml.RX(0.4, wires=[0]),
                qml.Rot(x, y, z, wires=[0]),
                qml.RY(-0.2, wires=[0]),
            ],
            [qml.expval(ham)],
        )

        tape.trainable_params = {1, 2, 3}

        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        grad_D = dev.adjoint_jacobian(tape)
        tapes, fn = qml.gradients.param_shift(tape)
        grad_F = fn(qml.execute(tapes, dev, None))

        # gradient has the correct shape and every element is nonzero
        assert len(grad_D) == 3
        assert all(isinstance(v, np.ndarray) for v in grad_D)
        assert np.count_nonzero(grad_D) == 3
        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    def test_use_device_state(self, tol, dev):
        """Tests that when using the device state, the correct answer is still returned."""

        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        dM1 = dev.adjoint_jacobian(tape)

        qml.execute([tape], dev, None)
        dM2 = dev.adjoint_jacobian(tape, use_device_state=True)

        assert np.allclose(dM1, dM2, atol=tol, rtol=0)

    def test_provide_starting_state(self, tol, dev):
        """Tests provides correct answer when provided starting state."""
        comm = MPI.COMM_WORLD

        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        dM1 = dev.adjoint_jacobian(tape)

        if device_name == "lightning.gpu":
            local_state_vector = dev.state
            complex_type = np.complex128 if dev.R_DTYPE == np.float64 else np.complex64
            state_vector = np.zeros(1 << 8).astype(complex_type)
            comm.Allgather(local_state_vector, state_vector)
            qml.execute([tape], dev, None)
            dM2 = dev.adjoint_jacobian(tape, starting_state=state_vector)
            assert np.allclose(dM1, dM2, atol=tol, rtol=0)

    def test_provide_wrong_starting_state(self, dev):
        """Tests raise an exception when provided starting state mismatches."""
        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The number of qubits of starting_state must be the same as",
        ):
            dev.adjoint_jacobian(tape, starting_state=np.ones(7))

    @pytest.mark.skipif(
        device_name == "lightning.gpu",
        reason="Adjoint differentiation does not support State measurements.",
    )
    def test_state_return_type(self, dev):
        """Tests raise an exception when the return type is State"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.state()

        tape.trainable_params = {0}

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Adjoint differentiation method does not support measurement StateMP.",
        ):
            dev.adjoint_jacobian(tape)


class TestAdjointJacobianQNode:
    """Test QNode integration with the adjoint_jacobian method"""

    @pytest.fixture(params=fixture_params)
    def dev(self, request):
        """Returns a PennyLane device."""
        return qml.device(
            device_name,
            wires=8,
            mpi=True,
            c_dtype=request.param[0],
            batch_obs=request.param[1],
        )

    def test_finite_shots_error(self):
        """Tests that an error is raised when computing the adjoint diff on a device with finite shots"""

        dev = qml.device(device_name, wires=8, mpi=True, shots=1)

        with pytest.raises(
            qml.QuantumFunctionError, match="does not support adjoint with requested circuit."
        ):

            @qml.qnode(dev, diff_method="adjoint")
            def circ(x):
                qml.RX(x, wires=0)
                return qml.expval(qml.PauliZ(0))

            qml.grad(circ)(0.1)

    def test_qnode(self, mocker, dev):
        """Test that specifying diff_method allows the adjoint method to be selected"""
        args = np.array([0.54, 0.1, 0.5], requires_grad=True)

        def circuit(x, y, z):
            qml.Hadamard(wires=0)
            qml.RX(0.543, wires=0)
            qml.CNOT(wires=[0, 1])

            qml.Rot(x, y, z, wires=0)

            qml.Rot(1.3, -2.3, 0.5, wires=[0])
            qml.RZ(-0.5, wires=0)
            qml.RY(0.5, wires=1)
            qml.CNOT(wires=[0, 1])

            return qml.expval(qml.PauliX(0) @ qml.PauliZ(1))

        qnode1 = QNode(circuit, dev, diff_method="adjoint")
        spy = mocker.spy(dev.target_device, "adjoint_jacobian")

        grad_fn = qml.grad(qnode1)
        grad_A = grad_fn(*args)

        spy.assert_called()

        h = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7
        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        qnode2 = QNode(circuit, dev, diff_method="finite-diff", h=h)
        grad_fn = qml.grad(qnode2)
        grad_F = grad_fn(*args)

        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

    thetas = np.linspace(-2 * np.pi, 2 * np.pi, 8)

    @pytest.mark.parametrize("reused_p", thetas**3 / 19)
    @pytest.mark.parametrize("other_p", thetas**2 / 1)
    def test_fanout_multiple_params(
        self, reused_p, other_p, tol, mocker, dev
    ):  # pylint: disable=too-many-arguments
        """Tests that the correct gradient is computed for qnodes which
        use the same parameter in multiple gates."""

        def expZ(state):
            return np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2

        extra_param = np.array(0.31, requires_grad=False)

        @qnode(dev, diff_method="adjoint")
        def cost(p1, p2):
            qml.RX(extra_param, wires=[0])
            qml.RY(p1, wires=[0])
            qml.RZ(p2, wires=[0])
            qml.RX(p1, wires=[0])
            return qml.expval(qml.PauliZ(0))

        zero_state = np.array([1.0, 0.0])
        cost(reused_p, other_p)

        spy = mocker.spy(dev.target_device, "adjoint_jacobian")

        # analytic gradient
        grad_fn = qml.grad(cost)
        grad_D = grad_fn(reused_p, other_p)

        spy.assert_called_once()

        # manual gradient
        grad_true0 = (
            expZ(
                Rx(reused_p) @ Rz(other_p) @ Ry(reused_p + np.pi / 2) @ Rx(extra_param) @ zero_state
            )
            - expZ(
                Rx(reused_p) @ Rz(other_p) @ Ry(reused_p - np.pi / 2) @ Rx(extra_param) @ zero_state
            )
        ) / 2
        grad_true1 = (
            expZ(
                Rx(reused_p + np.pi / 2) @ Rz(other_p) @ Ry(reused_p) @ Rx(extra_param) @ zero_state
            )
            - expZ(
                Rx(reused_p - np.pi / 2) @ Rz(other_p) @ Ry(reused_p) @ Rx(extra_param) @ zero_state
            )
        ) / 2
        expected = grad_true0 + grad_true1  # product rule

        assert np.allclose(grad_D[0], expected, atol=tol, rtol=0)

    def test_gradient_repeated_gate_parameters(self, mocker, dev):
        """Tests that repeated use of a free parameter in a multi-parameter gate yields correct
        gradients."""
        params = np.array([0.8, 1.3], requires_grad=True)

        def circuit(params):
            qml.RX(np.array(np.pi / 4, requires_grad=False), wires=[0])
            qml.Rot(params[1], params[0], 2 * params[0], wires=[0])
            return qml.expval(qml.PauliX(0))

        spy_analytic = mocker.spy(dev.target_device, "adjoint_jacobian")

        h = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7
        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        cost = QNode(circuit, dev, diff_method="finite-diff", h=h)

        grad_fn = qml.grad(cost)
        grad_F = grad_fn(params)

        spy_analytic.assert_not_called()

        cost = QNode(circuit, dev, diff_method="adjoint")
        grad_fn = qml.grad(cost)
        grad_D = grad_fn(params)

        spy_analytic.assert_called_once()

        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    def test_interface_tf(self, dev):
        """Test if gradients agree between the adjoint and finite-diff methods when using the
        TensorFlow interface"""

        tf = pytest.importorskip("tensorflow")

        def f(params1, params2):
            qml.RX(0.4, wires=[0])
            qml.RZ(params1 * tf.sqrt(params2), wires=[0])
            qml.RY(tf.cos(params2), wires=[0])
            return qml.expval(qml.PauliZ(0))

        if dev.R_DTYPE == np.float32:
            tf_r_dtype = tf.float32
        else:
            tf_r_dtype = tf.float64

        params1 = tf.Variable(0.3, dtype=tf_r_dtype)
        params2 = tf.Variable(0.4, dtype=tf_r_dtype)

        h = 2e-3 if dev.R_DTYPE == np.float32 else 1e-7
        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        qnode1 = QNode(f, dev, interface="tf", diff_method="adjoint")
        qnode2 = QNode(f, dev, interface="tf", diff_method="finite-diff", h=h)

        with tf.GradientTape() as tape:
            res1 = qnode1(params1, params2)

        g1 = tape.gradient(res1, [params1, params2])

        with tf.GradientTape() as tape:
            res2 = qnode2(params1, params2)

        g2 = tape.gradient(res2, [params1, params2])

        assert np.allclose(g1, g2, atol=tol)

    def test_interface_torch(self, dev):
        """Test if gradients agree between the adjoint and finite-diff methods when using the
        Torch interface"""

        torch = pytest.importorskip("torch")

        def f(params1, params2):
            qml.RX(0.4, wires=[0])
            qml.RZ(params1 * torch.sqrt(params2), wires=[0])
            qml.RY(torch.cos(params2), wires=[0])
            return qml.expval(qml.PauliZ(0))

        params1 = torch.tensor(0.3, requires_grad=True)
        params2 = torch.tensor(0.4, requires_grad=True)

        h = 2e-3 if dev.R_DTYPE == np.float32 else 1e-7

        qnode1 = QNode(f, dev, interface="torch", diff_method="adjoint")
        qnode2 = QNode(f, dev, interface="torch", diff_method="finite-diff", h=h)

        res1 = qnode1(params1, params2)
        res1.backward()

        grad_adjoint = params1.grad, params2.grad

        res2 = qnode2(params1, params2)
        res2.backward()

        grad_fd = params1.grad, params2.grad

        assert np.allclose(grad_adjoint, grad_fd)

    def test_interface_jax(self, dev):
        """Test if the gradients agree between adjoint and finite-difference methods in the
        jax interface"""

        jax = pytest.importorskip("jax")
        if dev.R_DTYPE == np.float64:
            from jax import config  # pylint: disable=import-outside-toplevel

            config.update("jax_enable_x64", True)

        def f(params1, params2):
            qml.RX(0.4, wires=[0])
            qml.RZ(params1 * jax.numpy.sqrt(params2), wires=[0])
            qml.RY(jax.numpy.cos(params2), wires=[0])
            return qml.expval(qml.PauliZ(0))

        params1 = jax.numpy.array(0.3, dev.R_DTYPE)
        params2 = jax.numpy.array(0.4, dev.R_DTYPE)

        h = 2e-3 if dev.R_DTYPE == np.float32 else 1e-7
        tol = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7

        qnode_adjoint = QNode(f, dev, interface="jax", diff_method="adjoint")
        qnode_fd = QNode(f, dev, interface="jax", diff_method="finite-diff", h=h)

        grad_adjoint = jax.grad(qnode_adjoint)(params1, params2)
        grad_fd = jax.grad(qnode_fd)(params1, params2)

        assert np.allclose(grad_adjoint, grad_fd, atol=tol)


def circuit_ansatz(params, wires):
    """Circuit ansatz containing all the parametrized gates"""
    qml.QubitStateVector(unitary_group.rvs(2**8, random_state=0)[0], wires=wires)
    qml.RX(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.adjoint(qml.RX(params[2], wires=wires[2]))
    qml.RZ(params[0], wires=wires[3])
    qml.CRX(params[3], wires=[wires[3], wires[0]])
    qml.PhaseShift(params[4], wires=wires[2])
    qml.CRY(params[5], wires=[wires[2], wires[1]])
    qml.adjoint(qml.CRZ(params[5], wires=[wires[0], wires[3]]))
    qml.adjoint(qml.PhaseShift(params[6], wires=wires[0]))
    qml.Rot(params[6], params[7], params[8], wires=wires[0])
    qml.adjoint(qml.Rot(params[8], params[8], params[9], wires=wires[1]))
    qml.MultiRZ(params[11], wires=[wires[0], wires[1]])
    qml.PauliRot(params[12], "XXYZ", wires=[wires[0], wires[1], wires[2], wires[3]])
    qml.CPhase(params[12], wires=[wires[3], wires[2]])
    qml.IsingXX(params[13], wires=[wires[1], wires[0]])
    qml.IsingXY(params[14], wires=[wires[3], wires[2]])
    qml.IsingYY(params[14], wires=[wires[3], wires[2]])
    qml.IsingZZ(params[14], wires=[wires[2], wires[1]])
    qml.U1(params[15], wires=wires[0])
    qml.U2(params[16], params[17], wires=wires[0])
    qml.U3(params[18], params[19], params[20], wires=wires[1])
    qml.adjoint(qml.CRot(params[21], params[22], params[23], wires=[wires[1], wires[2]]))
    qml.SingleExcitation(params[24], wires=[wires[2], wires[0]])
    qml.DoubleExcitation(params[25], wires=[wires[2], wires[0], wires[1], wires[3]])
    qml.SingleExcitationPlus(params[26], wires=[wires[0], wires[2]])
    qml.SingleExcitationMinus(params[27], wires=[wires[0], wires[2]])
    qml.DoubleExcitationPlus(params[27], wires=[wires[2], wires[0], wires[1], wires[3]])
    qml.DoubleExcitationMinus(params[27], wires=[wires[2], wires[0], wires[1], wires[3]])
    qml.RX(params[28], wires=wires[0])
    qml.RX(params[29], wires=wires[1])


@pytest.mark.parametrize(
    "returns",
    [
        qml.PauliZ(0),
        qml.PauliX(2),
        qml.PauliZ(0) @ qml.PauliY(3),
        qml.Hadamard(2),
        qml.Hadamard(3) @ qml.PauliZ(2),
        qml.PauliX(0) @ qml.PauliY(3),
        qml.PauliY(0) @ qml.PauliY(2) @ qml.PauliY(3),
        qml.Hermitian(
            np.kron(qml.PauliY.compute_matrix(), qml.PauliZ.compute_matrix()),
            wires=[3, 2],
        ),
        qml.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=0),
        qml.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=0) @ qml.PauliZ(2),
    ],
)
def test_integration(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations"""
    dev_def = qml.device("default.qubit", wires=range(8))
    dev_lightning = qml.device(device_name, wires=range(8), mpi=True)

    def circuit(params):
        circuit_ansatz(params, wires=range(8))
        return qml.expval(returns), qml.expval(qml.PauliY(1))

    n_params = 30
    params = np.linspace(0, 10, n_params)

    qnode_def = qml.QNode(circuit, dev_def)
    qnode_lightning = qml.QNode(circuit, dev_lightning, diff_method="adjoint")

    def casted_to_array_def(params):
        return np.array(qnode_def(params))

    def casted_to_array_lightning(params):
        return np.array(qnode_lightning(params))

    j_def = qml.jacobian(casted_to_array_def)(params)
    j_lightning = qml.jacobian(casted_to_array_lightning)(params)

    assert np.allclose(j_def, j_lightning)


custom_wires = ["alice", 3.14, -1, 0, "bob", 1, "unit", "test"]


@pytest.mark.parametrize(
    "returns",
    [
        qml.PauliZ(custom_wires[0]),
        qml.PauliX(custom_wires[2]),
        qml.PauliZ(custom_wires[0]) @ qml.PauliY(custom_wires[3]),
        qml.Hadamard(custom_wires[2]),
        qml.Hadamard(custom_wires[3]) @ qml.PauliZ(custom_wires[2]),
        # qml.Projector([0, 1], wires=[custom_wires[0], custom_wires[2]]) @ qml.Hadamard(custom_wires[3])
        # qml.Projector([0, 0], wires=[custom_wires[2], custom_wires[0]])
        qml.PauliX(custom_wires[0]) @ qml.PauliY(custom_wires[3]),
        qml.PauliY(custom_wires[0]) @ qml.PauliY(custom_wires[2]) @ qml.PauliY(custom_wires[3]),
        qml.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=custom_wires[0]),
        qml.Hermitian(
            np.kron(qml.PauliY.compute_matrix(), qml.PauliZ.compute_matrix()),
            wires=[custom_wires[3], custom_wires[2]],
        ),
        qml.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=custom_wires[0])
        @ qml.PauliZ(custom_wires[2]),
    ],
)
def test_integration_custom_wires(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""
    dev_def = qml.device("default.qubit", wires=custom_wires)
    dev_lightning = qml.device(device_name, wires=custom_wires, mpi=True, batch_obs=False)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qml.expval(returns), qml.expval(qml.PauliY(custom_wires[1]))

    n_params = 30
    params = np.linspace(0, 10, n_params)

    qnode_def = qml.QNode(circuit, dev_def)
    qnode_lightning = qml.QNode(circuit, dev_lightning, diff_method="adjoint")

    def casted_to_array_def(params):
        return np.array(qnode_def(params))

    def casted_to_array_lightning(params):
        return np.array(qnode_lightning(params))

    j_def = qml.jacobian(casted_to_array_def)(params)
    j_lightning = qml.jacobian(casted_to_array_lightning)(params)

    assert np.allclose(j_def, j_lightning)


@pytest.mark.parametrize(
    "returns",
    [
        (qml.PauliZ(custom_wires[0]),),
        (qml.PauliZ(custom_wires[0]), qml.PauliZ(custom_wires[1])),
        (
            qml.PauliZ(custom_wires[0]),
            qml.PauliZ(custom_wires[1]),
            qml.PauliZ(custom_wires[3]),
        ),
        (
            qml.PauliZ(custom_wires[0]),
            qml.PauliZ(custom_wires[1]),
            qml.PauliZ(custom_wires[3]),
            qml.PauliZ(custom_wires[2]),
        ),
        (
            qml.PauliZ(custom_wires[0]) @ qml.PauliY(custom_wires[3]),
            qml.PauliZ(custom_wires[1]) @ qml.PauliY(custom_wires[2]),
        ),
        (
            qml.PauliZ(custom_wires[0]) @ qml.PauliY(custom_wires[3]),
            qml.PauliZ(custom_wires[1]),
        ),
    ],
)
def test_integration_custom_wires_batching(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    dev_def = qml.device("default.qubit", wires=custom_wires)
    dev_gpu = qml.device("lightning.gpu", wires=custom_wires, mpi=True, batch_obs=True)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return [qml.expval(r) for r in returns] + [qml.expval(qml.PauliY(custom_wires[1]))]

    n_params = 30
    np.random.seed(1337)
    params = np.random.rand(n_params)

    qnode_gpu = qml.QNode(circuit, dev_gpu, diff_method="adjoint")
    qnode_def = qml.QNode(circuit, dev_def)

    def convert_to_array_gpu(params):
        return np.hstack(qnode_gpu(params))

    def convert_to_array_def(params):
        return np.hstack(qnode_def(params))

    j_gpu = qml.jacobian(convert_to_array_gpu)(params)
    j_def = qml.jacobian(convert_to_array_def)(params)

    assert np.allclose(j_gpu, j_def, atol=1e-7)


@pytest.mark.parametrize(
    "returns",
    [
        (0.5 * qml.PauliZ(custom_wires[0]),),
        (0.5 * qml.PauliZ(custom_wires[0]), qml.PauliZ(custom_wires[1])),
        (
            qml.PauliZ(custom_wires[0]),
            0.5 * qml.PauliZ(custom_wires[1]),
            qml.PauliZ(custom_wires[3]),
        ),
        (
            qml.PauliZ(custom_wires[0]),
            qml.PauliZ(custom_wires[1]),
            qml.PauliZ(custom_wires[3]),
            0.5 * qml.PauliZ(custom_wires[2]),
        ),
        (
            qml.PauliZ(custom_wires[0]) @ qml.PauliY(custom_wires[3]),
            0.5 * qml.PauliZ(custom_wires[1]) @ qml.PauliY(custom_wires[2]),
        ),
        (
            qml.PauliZ(custom_wires[0]) @ qml.PauliY(custom_wires[3]),
            0.5 * qml.PauliZ(custom_wires[1]),
        ),
        (
            0.0 * qml.PauliZ(custom_wires[0]) @ qml.PauliZ(custom_wires[1]),
            1.0 * qml.Identity(10),
            1.2 * qml.PauliZ(custom_wires[2]) @ qml.PauliZ(custom_wires[3]),
        ),
    ],
)
def test_batching_H(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    dev_cpu = qml.device("default.qubit", wires=custom_wires + [10, 72])
    dev_gpu = qml.device(device_name, wires=custom_wires + [10, 72], batch_obs=True)
    dev_gpu_default = qml.device(device_name, wires=custom_wires + [10, 72], batch_obs=False)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qml.math.hstack([qml.expval(r) for r in returns])

    n_params = 30
    np.random.seed(1337)
    params = np.random.rand(n_params)

    qnode_cpu = qml.QNode(circuit, dev_cpu, diff_method="parameter-shift")
    qnode_gpu = qml.QNode(circuit, dev_gpu, diff_method="adjoint")
    qnode_gpu_default = qml.QNode(circuit, dev_gpu_default, diff_method="adjoint")

    j_cpu = qml.jacobian(qnode_cpu)(params)
    j_gpu = qml.jacobian(qnode_gpu)(params)
    j_gpu_default = qml.jacobian(qnode_gpu_default)(params)

    assert np.allclose(j_cpu, j_gpu)
    assert np.allclose(j_gpu, j_gpu_default)


@pytest.fixture(scope="session")
def create_xyz_file(tmp_path_factory):
    """Creates a coordinate file for an H2 molecule in the XYZ format."""
    directory = tmp_path_factory.mktemp("tmp")
    file = directory / "h2.xyz"
    file.write_text("""2\nH2, Unoptimized\nH  1.0 0.0 0.0\nH -1.0 0.0 0.0""")
    yield file


@pytest.mark.parametrize(
    "batches",
    [False, True, 1, 2, 3, 4],
)
def test_integration_H2_Hamiltonian(
    create_xyz_file, batches
):  # pylint: disable=redefined-outer-name
    """Tests getting the total energy and its derivatives for an H2 Hamiltonian."""
    comm = MPI.COMM_WORLD
    _ = pytest.importorskip("openfermionpyscf")

    n_electrons = 2
    np.random.seed(1337)

    if comm.Get_rank() == 0:
        str_path = create_xyz_file
        symbols, coordinates = qml.qchem.read_structure(str(str_path), outpath=str(str_path.parent))
        H, qubits = qml.qchem.molecular_hamiltonian(
            symbols,
            coordinates,
            method="pyscf",
            basis="6-31G",
            active_electrons=n_electrons,
            name="h2",
            outpath=str(str_path.parent),
            load_data=True,
        )
    else:
        symbols = None
        coordinates = None
        H = None
        qubits = None

    symbols = comm.bcast(symbols, root=0)
    coordinates = comm.bcast(coordinates, root=0)
    H = comm.bcast(H, root=0)
    qubits = comm.bcast(qubits, root=0)

    hf_state = qml.qchem.hf_state(n_electrons, qubits)
    _, doubles = qml.qchem.excitations(n_electrons, qubits)

    # Choose different batching supports here
    dev = qml.device(device_name, wires=qubits, mpi=True, batch_obs=batches)
    dev_comp = qml.device("default.qubit", wires=qubits)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(params, excitations):
        qml.BasisState(hf_state, wires=H.wires)
        for i, excitation in enumerate(excitations):
            if len(excitation) == 4:
                qml.DoubleExcitation(params[i], wires=excitation)
            else:
                qml.SingleExcitation(params[i], wires=excitation)
        return qml.expval(H)

    @qml.qnode(dev_comp, diff_method="parameter-shift")
    def circuit_compare(params, excitations):
        qml.BasisState(hf_state, wires=H.wires)

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

    comm.Barrier()

    assert np.allclose(jacs, jacs_comp)


@pytest.mark.parametrize(
    "returns",
    [
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [0.1], [qml.PauliX(wires=custom_wires[0]) @ qml.PauliY(wires=custom_wires[1])]
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [2.0], [qml.PauliX(wires=custom_wires[2]) @ qml.PauliZ(wires=custom_wires[0])]
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [2.0], [qml.PauliX(wires=custom_wires[1]) @ qml.PauliZ(wires=custom_wires[2])]
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [1.1], [qml.PauliX(wires=custom_wires[0]) @ qml.PauliZ(wires=custom_wires[2])]
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
    ],
)
def test_adjoint_SparseHamiltonian_custom_wires(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    comm = MPI.COMM_WORLD
    dev_gpu = qml.device("lightning.gpu", wires=custom_wires, mpi=True)
    dev_cpu = qml.device("default.qubit", wires=custom_wires)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qml.expval(returns)

    if comm.Get_rank() == 0:
        n_params = 30
        np.random.seed(1337)
        params = np.random.rand(n_params)
    else:
        params = None

    params = comm.bcast(params, root=0)

    qnode_gpu = qml.QNode(circuit, dev_gpu, diff_method="adjoint")
    qnode_cpu = qml.QNode(circuit, dev_cpu, diff_method="parameter-shift")

    j_gpu = qml.jacobian(qnode_gpu)(params)
    j_cpu = qml.jacobian(qnode_cpu)(params)

    comm.Barrier()

    assert np.allclose(j_cpu, j_gpu)


@pytest.mark.parametrize(
    "returns",
    [
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [0.1],
                [qml.PauliZ(1) @ qml.PauliX(0) @ qml.Identity(2) @ qml.PauliX(4) @ qml.Identity(5)],
            ).sparse_matrix(range(len(custom_wires))),
            wires=range(len(custom_wires)),
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [0.1],
                [qml.PauliX(1) @ qml.PauliZ(0)],
            ).sparse_matrix(range(len(custom_wires))),
            wires=range(len(custom_wires)),
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [0.1],
                [qml.PauliX(0)],
            ).sparse_matrix(range(len(custom_wires))),
            wires=range(len(custom_wires)),
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [0.1],
                [qml.PauliX(5)],
            ).sparse_matrix(range(len(custom_wires))),
            wires=range(len(custom_wires)),
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian(
                [0.1],
                [qml.PauliX(0) @ qml.PauliZ(1)],
            ).sparse_matrix(range(len(custom_wires))),
            wires=range(len(custom_wires)),
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian([2.0], [qml.PauliX(1) @ qml.PauliZ(2)]).sparse_matrix(
                range(len(custom_wires))
            ),
            wires=range(len(custom_wires)),
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian([2.0], [qml.PauliX(2) @ qml.PauliZ(4)]).sparse_matrix(
                range(len(custom_wires))
            ),
            wires=range(len(custom_wires)),
        ),
        qml.SparseHamiltonian(
            qml.Hamiltonian([1.1], [qml.PauliX(2) @ qml.PauliZ(0)]).sparse_matrix(
                range(len(custom_wires))
            ),
            wires=range(len(custom_wires)),
        ),
    ],
)
def test_adjoint_SparseHamiltonian(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    comm = MPI.COMM_WORLD
    dev_gpu = qml.device("lightning.gpu", wires=len(custom_wires), mpi=True)
    dev_cpu = qml.device("default.qubit", wires=len(custom_wires))

    def circuit(params):
        circuit_ansatz(params, wires=range(len(custom_wires)))
        return qml.expval(returns)

    if comm.Get_rank() == 0:
        n_params = 30
        np.random.seed(1337)
        params = np.random.rand(n_params)
    else:
        params = None

    params = comm.bcast(params, root=0)

    qnode_gpu = qml.QNode(circuit, dev_gpu, diff_method="adjoint")
    qnode_cpu = qml.QNode(circuit, dev_cpu, diff_method="parameter-shift")

    j_gpu = qml.jacobian(qnode_gpu)(params)
    j_cpu = qml.jacobian(qnode_cpu)(params)

    comm.Barrier()

    assert np.allclose(j_cpu, j_gpu)


@pytest.mark.parametrize("n_targets", range(1, 5))
def test_qubit_unitary(dev, n_targets):
    """Tests that ``qml.QubitUnitary`` can be included in circuits differentiated with the adjoint method."""
    n_wires = len(dev.wires)
    dev_def = qml.device("default.qubit", wires=n_wires)
    h = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7
    c_dtype = np.complex64 if dev.R_DTYPE == np.float32 else np.complex128

    np.random.seed(1337)
    par = 2 * np.pi * np.random.rand(n_wires)
    U = np.random.rand(2**n_targets, 2**n_targets) + 1j * np.random.rand(
        2**n_targets, 2**n_targets
    )
    U, _ = np.linalg.qr(U)
    init_state = np.random.rand(2**n_wires) + 1j * np.random.rand(2**n_wires)
    init_state /= np.sqrt(np.dot(np.conj(init_state), init_state))

    comm = MPI.COMM_WORLD
    par = comm.bcast(par, root=0)
    U = comm.bcast(U, root=0)
    init_state = comm.bcast(init_state, root=0)

    init_state = np.array(init_state, requires_grad=False, dtype=c_dtype)
    U = np.array(U, requires_grad=False, dtype=c_dtype)
    obs = qml.prod(*(qml.PauliZ(i) for i in range(n_wires)))

    def circuit(x):
        qml.StatePrep(init_state, wires=range(n_wires))
        for i in range(n_wires // 2):
            qml.RY(x[i], wires=i)
        qml.QubitUnitary(U, wires=range(n_targets))
        for i in range(n_wires // 2, n_wires):
            qml.RY(x[i], wires=i)
        return qml.expval(obs)

    circ = qml.QNode(circuit, dev, diff_method="adjoint")
    circ_ps = qml.QNode(circuit, dev, diff_method="parameter-shift")
    circ_def = qml.QNode(circuit, dev_def, diff_method="adjoint")
    jac = qml.jacobian(circ)(par)
    jac_ps = qml.jacobian(circ_ps)(par)
    jac_def = qml.jacobian(circ_def)(par)

    comm.Barrier()

    assert len(jac) == n_wires
    assert not np.allclose(jac, 0.0)
    assert np.allclose(jac, jac_ps, atol=h, rtol=0)
    assert np.allclose(jac, jac_def, atol=h, rtol=0)


@pytest.mark.parametrize("n_targets", [1, 2])
def test_diff_qubit_unitary(dev, n_targets):
    """Tests that ``qml.QubitUnitary`` can be differentiated with the adjoint method."""
    n_wires = len(dev.wires)
    dev_def = qml.device("default.qubit", wires=n_wires)
    h = 1e-3 if dev.R_DTYPE == np.float32 else 1e-7
    c_dtype = np.complex64 if dev.R_DTYPE == np.float32 else np.complex128

    np.random.seed(1337)
    par = 2 * np.pi * np.random.rand(n_wires)
    U = np.random.rand(2**n_targets, 2**n_targets) + 1j * np.random.rand(
        2**n_targets, 2**n_targets
    )
    U, _ = np.linalg.qr(U)
    init_state = np.random.rand(2**n_wires) + 1j * np.random.rand(2**n_wires)
    init_state /= np.sqrt(np.dot(np.conj(init_state), init_state))

    comm = MPI.COMM_WORLD
    par = comm.bcast(par, root=0)
    U = comm.bcast(U, root=0)
    init_state = comm.bcast(init_state, root=0)

    init_state = np.array(init_state, requires_grad=False, dtype=c_dtype)
    U = np.array(U, requires_grad=False, dtype=c_dtype)
    obs = qml.prod(*(qml.PauliZ(i) for i in range(n_wires)))

    def circuit(x, u_mat):
        qml.StatePrep(init_state, wires=range(n_wires))
        for i in range(n_wires // 2):
            qml.RY(x[i], wires=i)
        qml.QubitUnitary(u_mat, wires=range(n_targets))
        for i in range(n_wires // 2, n_wires):
            qml.RY(x[i], wires=i)
        return qml.expval(obs)

    circ = qml.QNode(circuit, dev, diff_method="adjoint")
    circ_def = qml.QNode(circuit, dev_def, diff_method="adjoint")
    circ_fd = qml.QNode(circuit, dev, diff_method="finite-diff", h=h)
    circ_ps = qml.QNode(circuit, dev, diff_method="parameter-shift")
    jacs = qml.jacobian(circ)(par, U)
    jacs_def = qml.jacobian(circ_def)(par, U)
    jacs_fd = qml.jacobian(circ_fd)(par, U)
    jacs_ps = qml.jacobian(circ_ps)(par, U)

    comm.Barrier()

    for jac, jac_def, jac_fd, jac_ps in zip(jacs, jacs_def, jacs_fd, jacs_ps):
        assert not np.allclose(jac, 0.0)
        assert np.allclose(jac, jac_fd, atol=h, rtol=0)
        assert np.allclose(jac, jac_ps, atol=h, rtol=0)
        assert np.allclose(jac, jac_def, atol=h, rtol=0)
