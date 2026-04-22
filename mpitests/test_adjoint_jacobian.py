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
Unit tests for adjoint Jacobian on :mod:`pennylane_lightning` MPI-enabled devices.
"""

# pylint: disable=protected-access,cell-var-from-loop,c-extension-no-member
import itertools
import math
from functools import partial

import pennylane as qp
import pytest
from conftest import LightningDevice as ld
from conftest import device_name
from mpi4py import MPI
from pennylane import QNode
from pennylane import numpy as np
from pennylane import qnode
from pennylane.devices import ExecutionConfig
from pennylane.exceptions import QuantumFunctionError
from pennylane.tape import QuantumScript
from scipy.stats import unitary_group

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


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
    return qp.device(
        device_name,
        wires=8,
        mpi=True,
        c_dtype=request.param[0],
        batch_obs=request.param[1],
    )


def create_random_init_state(numWires, c_dtype, seed=None):
    """Returns a random normalized state of c_dtype with 2**numWires elements."""
    rng = np.random.default_rng(seed)
    r_dtype = np.float64 if c_dtype == np.complex128 else np.float32

    num_elements = 2**numWires
    init_state = rng.random(num_elements).astype(r_dtype) + 1j * rng.random(num_elements).astype(
        r_dtype
    )
    return init_state / np.linalg.norm(init_state)


class TestAdjointJacobian:  # pylint: disable=too-many-public-methods
    """Tests for the adjoint_jacobian method"""

    @pytest.mark.parametrize("batch_obs", [True, False])
    def test_not_expval(self, dev, batch_obs):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""

        qs = QuantumScript([qp.RX(1.23, 0)], [qp.var(qp.PauliZ(0))], trainable_params=[0])
        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})

        with pytest.raises(QuantumFunctionError, match="Adjoint differentiation method does not"):
            dev.compute_derivatives(qs, config)

        qs = QuantumScript([qp.RX(1.23, 0)], [qp.state()], trainable_params=[0])

        with pytest.raises(
            QuantumFunctionError,
            match="Adjoint differentiation method does not support measurement StateMP.",
        ):
            dev.compute_derivatives(qs, config)

    @pytest.mark.parametrize("batch_obs", [True, False])
    def test_finite_shots_warns(self, dev, batch_obs):
        """Tests warning raised when finite shots specified"""

        qs = QuantumScript([qp.RX(1.23, 0)], [qp.expval(qp.Z(0))], shots=10, trainable_params=[0])
        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})

        with pytest.raises(
            QuantumFunctionError,
            match="Requested adjoint differentiation to be computed with finite shots.",
        ):
            dev.compute_derivatives(qs, config)

    def test_empty_measurements(self, dev):
        """Tests if an empty array is returned when the measurements of the tape is empty."""

        def circuit():
            qp.RX(0.4, wires=[0])
            return qp.expval(qp.PauliZ(0))

        result = QNode(circuit, dev, diff_method="adjoint")

        jac = qp.grad(result)()

        assert len(jac) == 0

    @pytest.mark.parametrize("batch_obs", [True, False])
    def test_unsupported_op(self, batch_obs, dev):
        """Test if a QuantumFunctionError is raised for an unsupported operation, i.e.,
        multi-parameter operations that are not qp.Rot"""

        qs = QuantumScript(
            [qp.CRot(0.1, 0.2, 0.3, wires=[0, 1])],
            [qp.expval(qp.PauliZ(0))],
            trainable_params=[0],
        )
        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})

        with pytest.raises(
            RuntimeError,
            match="The operation is not supported using the adjoint differentiation method",
        ):
            dev.compute_derivatives(qs, config)

    @pytest.mark.skip("WIP: Need a deep review if LGPU accept Projector")
    @pytest.mark.parametrize("batch_obs", [True, False])
    def test_proj_unsupported(self, batch_obs, dev):
        """Test if a QuantumFunctionError is raised for a Projector observable"""

        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})

        qs = QuantumScript(
            [qp.CRX(0.1, wires=[0, 1])],
            [qp.expval(qp.Projector([0, 1], wires=[0, 1]))],
            trainable_params=[0],
        )

        with pytest.raises(
            QuantumFunctionError,
            match="differentiation method does not support the Projector",
        ):
            dev.compute_derivatives(qs, config)

        qs = QuantumScript(
            [qp.CRX(0.1, wires=[0, 1])],
            [qp.expval(qp.Projector([0], wires=[0]) @ qp.PauliZ(0))],
            trainable_params=[0],
        )

        with pytest.raises(
            QuantumFunctionError,
            match="differentiation method does not support the Projector",
        ):
            dev.compute_derivatives(qs, config)

    @staticmethod
    def tol_for_allclose(c_dtype):
        """Compute the tolerance for allclose"""
        return 1e-3 if c_dtype == np.complex64 else 1e-7

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qp.RX, qp.RY, qp.RZ])
    @pytest.mark.parametrize("batch_obs", [True, False])
    def test_pauli_rotation_gradient(
        self, G, theta, batch_obs, dev
    ):  # pylint: disable=too-many-arguments
        """Tests that the automatic gradients of Pauli rotations are correct."""
        random_state = np.array(
            [0.43593284 - 0.02945156j, 0.40812291 + 0.80158023j], requires_grad=False
        )

        qs = QuantumScript(
            [qp.StatePrep(random_state, 0), G(theta, 0)],
            [qp.expval(qp.PauliZ(0))],
            trainable_params=[1],
        )
        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})

        calculated_val = dev.compute_derivatives(qs, config)

        tol = self.tol_for_allclose(dev.c_dtype)

        # compare to finite differences
        tapes, fn = qp.gradients.param_shift(qs)
        numeric_val = fn(qp.execute(tapes, dev, None))
        assert np.allclose(calculated_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("batch_obs", [True, False])
    def test_Rot_gradient(self, theta, batch_obs, dev):
        """Tests that the device gradient of an arbitrary Euler-angle-parameterized gate is
        correct."""
        params = np.array([theta, theta**3, np.sqrt(2) * theta])

        qs = QuantumScript(
            [
                qp.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0),
                qp.Rot(*params, wires=[0]),
            ],
            [qp.expval(qp.PauliZ(0))],
            trainable_params=[1, 2, 3],
        )

        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})

        calculated_val = dev.compute_derivatives(qs, config)

        tol = self.tol_for_allclose(dev.c_dtype)

        # compare to finite differences
        tapes, fn = qp.gradients.param_shift(qs)
        numeric_val = fn(qp.execute(tapes, dev, None))
        assert np.allclose(calculated_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("param", [1, -2, 1.623, -0.051, 0])  # integers, floats, zero
    @pytest.mark.parametrize(
        "rotation, meas, expected_func",
        [
            (qp.RY, qp.PauliX, lambda x: np.cos(x)),  # pylint: disable=unnecessary-lambda
            (qp.RX, qp.PauliZ, lambda x: -np.sin(x)),  # pylint: disable=unnecessary-lambda
        ],
    )
    @pytest.mark.parametrize("batch_obs", [True, False])
    def test_r_gradient(
        self, tol, param, rotation, meas, expected_func, batch_obs, dev
    ):  # pylint: disable=too-many-arguments
        """Test for the gradient of the rotation gate matches the known formula."""

        qs = QuantumScript(
            [rotation(param, wires=0)],
            [qp.expval(meas(0))],
            trainable_params=[0],
        )

        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})

        # circuit jacobians
        dev_jacobian = dev.compute_derivatives(qs, config)
        expected_jacobian = expected_func(param)
        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    @staticmethod
    def process_and_execute_multiple_rx(dev, params, meas, batch_obs):
        """Compute the circuit with multiple RX gates"""
        qs = QuantumScript(
            [qp.RX(params[0], wires=0), qp.RX(params[1], wires=1), qp.RX(params[2], wires=2)],
            meas,
            trainable_params=[0, 1, 2],
        )
        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})

        # circuit jacobians
        dev_jacobian = dev.compute_derivatives(qs, config)

        return dev_jacobian

    @pytest.mark.parametrize("batch_obs", [True, False])
    def test_multiple_rx_gradient_pauliz(self, tol, batch_obs, dev):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result."""
        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        meas = [qp.expval(qp.PauliZ(idx)) for idx in range(3)]

        # circuit jacobians
        dev_jacobian = self.process_and_execute_multiple_rx(dev, params, meas, batch_obs)
        expected_jacobian = -np.diag(np.sin(params))
        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    @pytest.mark.parametrize("batch_obs", [True, False])
    def test_multiple_rx_gradient_hermitian(self, tol, batch_obs, dev):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result
        with Hermitian observable
        """

        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        meas = [qp.expval(qp.Hermitian([[1, 0], [0, -1]], wires=[idx])) for idx in range(3)]

        # circuit jacobians
        dev_jacobian = self.process_and_execute_multiple_rx(dev, params, meas, batch_obs)
        expected_jacobian = -np.diag(np.sin(params))

        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    @pytest.mark.parametrize("batch_obs", [True, False])
    def test_multiple_rx_gradient_expval_hermitian(self, tol, batch_obs, dev):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result
        with Hermitian observable
        """
        params = np.array([np.pi / 3, np.pi / 4, np.pi / 5])

        meas = [
            qp.expval(
                qp.Hermitian(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                    wires=[0, 2],
                )
            )
        ]

        dev_jacobian = self.process_and_execute_multiple_rx(dev, params, meas, batch_obs)
        expected_jacobian = np.array(
            [
                -np.sin(params[0]) * np.cos(params[2]),
                0,
                -np.cos(params[0]) * np.sin(params[2]),
            ]
        )

        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    @pytest.mark.parametrize("batch_obs", [True, False])
    def test_multiple_rx_gradient_expval_hamiltonian(self, tol, batch_obs, dev):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result
        with Hermitian observable
        """
        params = np.array([np.pi / 3, np.pi / 4, np.pi / 5])

        meas = [
            qp.expval(
                qp.Hamiltonian(
                    [1.0, 0.3, 0.3, 0.4],
                    [
                        qp.PauliX(0) @ qp.PauliX(1),
                        qp.PauliZ(0),
                        qp.PauliZ(1),
                        qp.Hermitian(
                            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                            wires=[0, 2],
                        ),
                    ],
                )
            )
        ]

        dev_jacobian = self.process_and_execute_multiple_rx(dev, params, meas, batch_obs)
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

    @pytest.mark.parametrize(
        "meas",
        [
            [qp.expval(qp.PauliX(wires=0)), qp.expval(qp.PauliZ(wires=1))],
            [qp.expval(qp.PauliY(wires=0)), qp.expval(qp.PauliZ(wires=1))],
            [
                qp.expval(
                    qp.Hermitian(
                        [[0, 0, 1, 1], [0, 1, 2, 1], [1, 2, 1, 0], [1, 1, 0, 0]],
                        wires=[0, 1],
                    )
                )
            ],
        ],
    )
    @pytest.mark.parametrize(
        "op",
        [
            qp.RX(0.4, wires=0),
            qp.RY(0.6, wires=0),
            qp.RZ(0.8, wires=0),
            qp.CRX(1.0, wires=[0, 1]),
            qp.CRY(2.0, wires=[0, 1]),
            qp.CRZ(3.0, wires=[0, 1]),
            qp.Rot(0.2, -0.1, 0.2, wires=0),
        ],
    )
    @pytest.mark.parametrize("batch_obs", [True, False])
    def test_gradients_pauliz_hermitian(self, op, meas, batch_obs, dev):
        """Tests that the gradients of circuits match between the finite difference and device
        methods."""
        # op.num_wires and op.num_params must be initialized a priori
        qs = QuantumScript(
            [
                qp.Hadamard(wires=0),
                qp.RX(0.543, wires=0),
                qp.CNOT(wires=[0, 1]),
                op,
                qp.Rot(1.3, -2.3, 0.5, wires=[0]),
                qp.RZ(-0.5, wires=0),
                qp.adjoint(qp.RY(0.5, wires=1), lazy=False),
                qp.CNOT(wires=[0, 1]),
            ],
            meas,
            trainable_params=list(range(1, 1 + op.num_params)),
        )
        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})

        tol = self.tol_for_allclose(dev.c_dtype)

        tapes, fn = qp.gradients.param_shift(qs)
        grad_F = fn(qp.execute(tapes, dev, None))

        # circuit jacobians
        grad_D = dev.compute_derivatives(qs, config)
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "meas",
        [
            [qp.expval(qp.PauliZ(0))],
            [qp.expval(qp.Hermitian([[0, 1], [1, 1]], wires=0))],
            [
                qp.expval(
                    qp.Hamiltonian(
                        [1.0, 0.3, 0.3],
                        [qp.PauliX(0) @ qp.PauliX(1), qp.PauliZ(0), qp.PauliZ(1)],
                    )
                )
            ],
        ],
    )
    @pytest.mark.parametrize("batch_obs", [True, False])
    def test_gradient_gate_with_multiple_parameters(self, meas, batch_obs, dev):
        """Tests that gates with multiple free parameters yield correct gradients."""
        x, y, z = [0.5, 0.3, -0.7]

        qs = QuantumScript(
            [
                qp.RX(0.4, wires=[0]),
                qp.Rot(x, y, z, wires=[0]),
                qp.RY(-0.2, wires=[0]),
            ],
            meas,
            trainable_params=[1, 2, 3],
        )
        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})

        tol = self.tol_for_allclose(dev.c_dtype)

        # circuit jacobians
        grad_D = dev.compute_derivatives(qs, config)
        tapes, fn = qp.gradients.param_shift(qs)
        grad_F = fn(qp.execute(tapes, dev, None))

        # gradient has the correct shape and every element is nonzero
        assert len(grad_D) == 3
        assert all(isinstance(v, np.ndarray) for v in grad_D)
        assert np.count_nonzero(grad_D) == 3
        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)


class TestAdjointJacobianQNode:
    """Test QNode integration with the adjoint_jacobian method"""

    # def analytic_rotation(self):
    I = np.eye(2)
    X = qp.PauliX.compute_matrix()
    Y = qp.PauliY.compute_matrix()
    Z = qp.PauliZ.compute_matrix()

    def Rx(self, theta):
        r"""One-qubit rotation about the x axis.

        Args:
            theta (float): rotation angle
        Returns:
            array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}`
        """
        return math.cos(theta / 2) * self.I + 1j * math.sin(-theta / 2) * self.X

    def Ry(self, theta):
        r"""One-qubit rotation about the y axis.

        Args:
            theta (float): rotation angle
        Returns:
            array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
        """
        return math.cos(theta / 2) * self.I + 1j * math.sin(-theta / 2) * self.Y

    def Rz(self, theta):
        r"""One-qubit rotation about the z axis.

        Args:
            theta (float): rotation angle
        Returns:
            array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_z \theta/2}`
        """
        return math.cos(theta / 2) * self.I + 1j * math.sin(-theta / 2) * self.Z

    def test_finite_shots_error(self):
        """Tests that an error is raised when computing the adjoint diff on a device with finite shots"""

        dev = qp.device(device_name, wires=8, mpi=True)

        with pytest.raises(
            QuantumFunctionError,
            match="does not support adjoint with requested circuit.",
        ):

            @partial(qp.set_shots, shots=1)
            @qp.qnode(dev, diff_method="adjoint")
            def circ(x):
                qp.RX(x, wires=0)
                return qp.expval(qp.PauliZ(0))

            qp.grad(circ)(0.1)

    @staticmethod
    def tol_for_allclose(c_dtype):
        """Compute the tolerance for allclose"""
        return 1e-3 if c_dtype == np.complex64 else 1e-7

    def test_qnode(self, mocker, dev):
        """Test that specifying diff_method allows the adjoint method to be selected"""
        args = np.array([0.54, 0.1, 0.5], requires_grad=True)

        def circuit(x, y, z):
            qp.Hadamard(wires=0)
            qp.RX(0.543, wires=0)
            qp.CNOT(wires=[0, 1])

            qp.Rot(x, y, z, wires=0)

            qp.Rot(1.3, -2.3, 0.5, wires=[0])
            qp.RZ(-0.5, wires=0)
            qp.RY(0.5, wires=1)
            qp.CNOT(wires=[0, 1])

            return qp.expval(qp.PauliX(0) @ qp.PauliZ(1))

        qnode1 = QNode(circuit, dev, diff_method="adjoint")
        spy = mocker.spy(dev, "LightningAdjointJacobian")

        grad_fn = qp.grad(qnode1)
        grad_A = grad_fn(*args)

        spy.assert_called()

        h = self.tol_for_allclose(dev.c_dtype)
        tol = self.tol_for_allclose(dev.c_dtype)

        qnode2 = QNode(circuit, dev, diff_method="finite-diff", gradient_kwargs={"h": h})
        grad_fn = qp.grad(qnode2)
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
            qp.RX(extra_param, wires=[0])
            qp.RY(p1, wires=[0])
            qp.RZ(p2, wires=[0])
            qp.RX(p1, wires=[0])
            return qp.expval(qp.PauliZ(0))

        zero_state = np.array([1.0, 0.0])
        cost(reused_p, other_p)

        spy = mocker.spy(dev, "LightningAdjointJacobian")

        # analytic gradient
        grad_fn = qp.grad(cost)
        grad_D = grad_fn(reused_p, other_p)

        spy.assert_called_once()

        # manual gradient
        grad_true0 = (
            expZ(
                self.Rx(reused_p)
                @ self.Rz(other_p)
                @ self.Ry(reused_p + np.pi / 2)
                @ self.Rx(extra_param)
                @ zero_state
            )
            - expZ(
                self.Rx(reused_p)
                @ self.Rz(other_p)
                @ self.Ry(reused_p - np.pi / 2)
                @ self.Rx(extra_param)
                @ zero_state
            )
        ) / 2
        grad_true1 = (
            expZ(
                self.Rx(reused_p + np.pi / 2)
                @ self.Rz(other_p)
                @ self.Ry(reused_p)
                @ self.Rx(extra_param)
                @ zero_state
            )
            - expZ(
                self.Rx(reused_p - np.pi / 2)
                @ self.Rz(other_p)
                @ self.Ry(reused_p)
                @ self.Rx(extra_param)
                @ zero_state
            )
        ) / 2
        expected = grad_true0 + grad_true1  # product rule

        assert np.allclose(grad_D[0], expected, atol=tol, rtol=0)

    def test_gradient_repeated_gate_parameters(self, mocker, dev):
        """Tests that repeated use of a free parameter in a multi-parameter gate yields correct
        gradients."""
        params = np.array([0.8, 1.3], requires_grad=True)

        def circuit(params):
            qp.RX(np.array(np.pi / 4, requires_grad=False), wires=[0])
            qp.Rot(params[1], params[0], 2 * params[0], wires=[0])
            return qp.expval(qp.PauliX(0))

        spy_analytic = mocker.spy(dev, "LightningAdjointJacobian")

        h = self.tol_for_allclose(dev.c_dtype)
        tol = self.tol_for_allclose(dev.c_dtype)

        cost = QNode(circuit, dev, diff_method="finite-diff", gradient_kwargs={"h": h})

        grad_fn = qp.grad(cost)
        grad_F = grad_fn(params)

        spy_analytic.assert_not_called()

        cost = QNode(circuit, dev, diff_method="adjoint")
        grad_fn = qp.grad(cost)
        grad_D = grad_fn(params)

        spy_analytic.assert_called_once()

        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    def test_interface_torch(self, dev):
        """Test if gradients agree between the adjoint and finite-diff methods when using the
        Torch interface"""

        torch = pytest.importorskip("torch")

        def f(params1, params2):
            qp.RX(0.4, wires=[0])
            qp.RZ(params1 * torch.sqrt(params2), wires=[0])
            qp.RY(torch.cos(params2), wires=[0])
            return qp.expval(qp.PauliZ(0))

        params1 = torch.tensor(0.3, requires_grad=True)
        params2 = torch.tensor(0.4, requires_grad=True)

        h = self.tol_for_allclose(dev.c_dtype)

        qnode1 = QNode(f, dev, interface="torch", diff_method="adjoint")
        qnode2 = QNode(
            f, dev, interface="torch", diff_method="finite-diff", gradient_kwargs={"h": h}
        )

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
        if dev.c_dtype == np.complex128:
            from jax import config  # pylint: disable=import-outside-toplevel

            config.update("jax_enable_x64", True)

        def f(params1, params2):
            qp.RX(0.4, wires=[0])
            qp.RZ(params1 * jax.numpy.sqrt(params2), wires=[0])
            qp.RY(jax.numpy.cos(params2), wires=[0])
            return qp.expval(qp.PauliZ(0))

        r_dtype = np.float32 if dev.c_dtype == np.complex64 else np.float64

        params1 = jax.numpy.array(0.3, r_dtype)
        params2 = jax.numpy.array(0.4, r_dtype)

        h = self.tol_for_allclose(dev.c_dtype)
        tol = self.tol_for_allclose(dev.c_dtype)

        qnode_adjoint = QNode(f, dev, interface="jax", diff_method="adjoint")
        qnode_fd = QNode(
            f, dev, interface="jax", diff_method="finite-diff", gradient_kwargs={"h": h}
        )

        grad_adjoint = jax.grad(qnode_adjoint)(params1, params2)
        grad_fd = jax.grad(qnode_fd)(params1, params2)

        assert np.allclose(grad_adjoint, grad_fd, atol=tol)


def circuit_ansatz(params, wires):
    """Circuit ansatz containing all the parametrized gates"""
    qp.StatePrep(unitary_group.rvs(2**8, random_state=0)[0], wires=wires)
    qp.RX(params[0], wires=wires[0])
    qp.RY(params[1], wires=wires[1])
    qp.adjoint(qp.RX(params[2], wires=wires[2]))
    qp.RZ(params[0], wires=wires[3])
    qp.CRX(params[3], wires=[wires[3], wires[0]])
    qp.PhaseShift(params[4], wires=wires[2])
    qp.CRY(params[5], wires=[wires[2], wires[1]])
    qp.adjoint(qp.CRZ(params[5], wires=[wires[0], wires[3]]))
    qp.adjoint(qp.PhaseShift(params[6], wires=wires[0]))
    qp.Rot(params[6], params[7], params[8], wires=wires[0])
    qp.adjoint(qp.Rot(params[8], params[8], params[9], wires=wires[1]))
    qp.MultiRZ(params[11], wires=[wires[0], wires[1]])
    qp.PauliRot(params[12], "XXYZ", wires=[wires[0], wires[1], wires[2], wires[3]])
    qp.CPhase(params[12], wires=[wires[3], wires[2]])
    qp.IsingXX(params[13], wires=[wires[1], wires[0]])
    qp.IsingXY(params[14], wires=[wires[3], wires[2]])
    qp.IsingYY(params[14], wires=[wires[3], wires[2]])
    qp.IsingZZ(params[14], wires=[wires[2], wires[1]])
    qp.U1(params[15], wires=wires[0])
    qp.U2(params[16], params[17], wires=wires[0])
    qp.U3(params[18], params[19], params[20], wires=wires[1])
    qp.adjoint(qp.CRot(params[21], params[22], params[23], wires=[wires[1], wires[2]]))
    qp.SingleExcitation(params[24], wires=[wires[2], wires[0]])
    qp.DoubleExcitation(params[25], wires=[wires[2], wires[0], wires[1], wires[3]])
    qp.SingleExcitationPlus(params[26], wires=[wires[0], wires[2]])
    qp.SingleExcitationMinus(params[27], wires=[wires[0], wires[2]])
    qp.DoubleExcitationPlus(params[27], wires=[wires[2], wires[0], wires[1], wires[3]])
    qp.DoubleExcitationMinus(params[27], wires=[wires[2], wires[0], wires[1], wires[3]])
    qp.RX(params[28], wires=wires[0])
    qp.RX(params[29], wires=wires[1])
    qp.PSWAP(params[29], wires=[wires[0], wires[1]])


@pytest.mark.parametrize(
    "returns",
    [
        qp.PauliZ(0),
        qp.PauliX(2),
        qp.PauliZ(0) @ qp.PauliY(3),
        qp.Hadamard(2),
        qp.Hadamard(3) @ qp.PauliZ(2),
        qp.PauliX(0) @ qp.PauliY(3),
        qp.PauliY(0) @ qp.PauliY(2) @ qp.PauliY(3),
        qp.Hermitian(
            np.kron(qp.PauliY.compute_matrix(), qp.PauliZ.compute_matrix()),
            wires=[3, 2],
        ),
        qp.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=0),
        qp.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=0) @ qp.PauliZ(2),
    ],
)
def test_integration(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations"""
    dev_def = qp.device("default.qubit", wires=range(8))
    dev_lightning = qp.device(device_name, wires=range(8), mpi=True)

    def circuit(params):
        circuit_ansatz(params, wires=range(8))
        return qp.expval(returns), qp.expval(qp.PauliY(1))

    n_params = 30
    params = np.linspace(0, 10, n_params)

    qnode_def = qp.QNode(circuit, dev_def)
    qnode_lightning = qp.QNode(circuit, dev_lightning, diff_method="adjoint")

    def casted_to_array_def(params):
        return np.array(qnode_def(params))

    def casted_to_array_lightning(params):
        return np.array(qnode_lightning(params))

    j_def = qp.jacobian(casted_to_array_def)(params)
    j_lightning = qp.jacobian(casted_to_array_lightning)(params)

    assert np.allclose(j_def, j_lightning)


custom_wires = ["alice", 3.14, -1, 0, "bob", 1, "unit", "test"]


@pytest.mark.parametrize(
    "returns",
    [
        qp.PauliZ(custom_wires[0]),
        qp.PauliX(custom_wires[2]),
        qp.PauliZ(custom_wires[0]) @ qp.PauliY(custom_wires[3]),
        qp.Hadamard(custom_wires[2]),
        qp.Hadamard(custom_wires[3]) @ qp.PauliZ(custom_wires[2]),
        # qp.Projector([0, 1], wires=[custom_wires[0], custom_wires[2]]) @ qp.Hadamard(custom_wires[3])
        # qp.Projector([0, 0], wires=[custom_wires[2], custom_wires[0]])
        qp.PauliX(custom_wires[0]) @ qp.PauliY(custom_wires[3]),
        qp.PauliY(custom_wires[0]) @ qp.PauliY(custom_wires[2]) @ qp.PauliY(custom_wires[3]),
        qp.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=custom_wires[0]),
        qp.Hermitian(
            np.kron(qp.PauliY.compute_matrix(), qp.PauliZ.compute_matrix()),
            wires=[custom_wires[3], custom_wires[2]],
        ),
        qp.Hermitian(np.array([[0, 1], [1, 0]], requires_grad=False), wires=custom_wires[0])
        @ qp.PauliZ(custom_wires[2]),
    ],
)
def test_integration_custom_wires(returns):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""
    dev_def = qp.device("default.qubit", wires=custom_wires)
    dev_lightning = qp.device(device_name, wires=custom_wires, mpi=True, batch_obs=False)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qp.expval(returns), qp.expval(qp.PauliY(custom_wires[1]))

    n_params = 30
    params = np.linspace(0, 10, n_params)

    qnode_def = qp.QNode(circuit, dev_def)
    qnode_lightning = qp.QNode(circuit, dev_lightning, diff_method="adjoint")

    def casted_to_array_def(params):
        return np.array(qnode_def(params))

    def casted_to_array_lightning(params):
        return np.array(qnode_lightning(params))

    j_def = qp.jacobian(casted_to_array_def)(params)
    j_lightning = qp.jacobian(casted_to_array_lightning)(params)

    assert np.allclose(j_def, j_lightning)


@pytest.mark.local_salt(42)
@pytest.mark.parametrize(
    "returns",
    [
        (qp.PauliZ(custom_wires[0]),),
        (qp.PauliZ(custom_wires[0]), qp.PauliZ(custom_wires[1])),
        (
            qp.PauliZ(custom_wires[0]),
            qp.PauliZ(custom_wires[1]),
            qp.PauliZ(custom_wires[3]),
        ),
        (
            qp.PauliZ(custom_wires[0]),
            qp.PauliZ(custom_wires[1]),
            qp.PauliZ(custom_wires[3]),
            qp.PauliZ(custom_wires[2]),
        ),
        (
            qp.PauliZ(custom_wires[0]) @ qp.PauliY(custom_wires[3]),
            qp.PauliZ(custom_wires[1]) @ qp.PauliY(custom_wires[2]),
        ),
        (
            qp.PauliZ(custom_wires[0]) @ qp.PauliY(custom_wires[3]),
            qp.PauliZ(custom_wires[1]),
        ),
    ],
)
def test_integration_custom_wires_batching(returns, seed):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    dev_def = qp.device("default.qubit", wires=custom_wires)
    dev_mpi = qp.device(device_name, wires=custom_wires, mpi=True, batch_obs=True)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return [qp.expval(r) for r in returns] + [qp.expval(qp.PauliY(custom_wires[1]))]

    n_params = 30
    rng = np.random.default_rng(seed)
    params = rng.random(n_params)

    qnode_mpi = qp.QNode(circuit, dev_mpi, diff_method="adjoint")
    qnode_def = qp.QNode(circuit, dev_def)

    def convert_to_array_gpu(params):
        return np.hstack(qnode_mpi(params))

    def convert_to_array_def(params):
        return np.hstack(qnode_def(params))

    j_gpu = qp.jacobian(convert_to_array_gpu)(params)
    j_def = qp.jacobian(convert_to_array_def)(params)

    assert np.allclose(j_gpu, j_def, atol=1e-7)


@pytest.mark.local_salt(42)
@pytest.mark.parametrize(
    "returns",
    [
        (0.5 * qp.PauliZ(custom_wires[0]),),
        (0.5 * qp.PauliZ(custom_wires[0]), qp.PauliZ(custom_wires[1])),
        (
            qp.PauliZ(custom_wires[0]),
            0.5 * qp.PauliZ(custom_wires[1]),
            qp.PauliZ(custom_wires[3]),
        ),
        (
            qp.PauliZ(custom_wires[0]),
            qp.PauliZ(custom_wires[1]),
            qp.PauliZ(custom_wires[3]),
            0.5 * qp.PauliZ(custom_wires[2]),
        ),
        (
            qp.PauliZ(custom_wires[0]) @ qp.PauliY(custom_wires[3]),
            0.5 * qp.PauliZ(custom_wires[1]) @ qp.PauliY(custom_wires[2]),
        ),
        (
            qp.PauliZ(custom_wires[0]) @ qp.PauliY(custom_wires[3]),
            0.5 * qp.PauliZ(custom_wires[1]),
        ),
        (
            0.0 * qp.PauliZ(custom_wires[0]) @ qp.PauliZ(custom_wires[1]),
            1.0 * qp.Identity(10),
            1.2 * qp.PauliZ(custom_wires[2]) @ qp.PauliZ(custom_wires[3]),
        ),
    ],
)
def test_batching_H(returns, seed):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    dev_cpu = qp.device("default.qubit", wires=custom_wires + [10, 72])
    dev_mpi = qp.device(device_name, wires=custom_wires + [10, 72], batch_obs=True)
    dev_mpi_default = qp.device(device_name, wires=custom_wires + [10, 72], batch_obs=False)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qp.math.hstack([qp.expval(r) for r in returns])

    n_params = 30
    rng = np.random.default_rng(seed)
    params = rng.random(n_params)

    qnode_cpu = qp.QNode(circuit, dev_cpu, diff_method="parameter-shift")
    qnode_mpi = qp.QNode(circuit, dev_mpi, diff_method="adjoint")
    qnode_mpi_default = qp.QNode(circuit, dev_mpi_default, diff_method="adjoint")

    j_cpu = qp.jacobian(qnode_cpu)(params)
    j_gpu = qp.jacobian(qnode_mpi)(params)
    j_gpu_default = qp.jacobian(qnode_mpi_default)(params)

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
        symbols, coordinates = qp.qchem.read_structure(str(str_path), outpath=str(str_path.parent))
        H, qubits = qp.qchem.molecular_hamiltonian(
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

    hf_state = qp.qchem.hf_state(n_electrons, qubits)
    _, doubles = qp.qchem.excitations(n_electrons, qubits)

    # Choose different batching supports here
    dev = qp.device(device_name, wires=qubits, mpi=True, batch_obs=batches)
    dev_comp = qp.device("default.qubit", wires=qubits)

    @qp.qnode(dev, diff_method="adjoint")
    def circuit(params, excitations):
        qp.BasisState(hf_state, wires=H.wires)
        for i, excitation in enumerate(excitations):
            if len(excitation) == 4:
                qp.DoubleExcitation(params[i], wires=excitation)
            else:
                qp.SingleExcitation(params[i], wires=excitation)
        return qp.expval(H)

    @qp.qnode(dev_comp, diff_method="parameter-shift")
    def circuit_compare(params, excitations):
        qp.BasisState(hf_state, wires=H.wires)

        for i, excitation in enumerate(excitations):
            if len(excitation) == 4:
                qp.DoubleExcitation(params[i], wires=excitation)
            else:
                qp.SingleExcitation(params[i], wires=excitation)
        return qp.expval(H)

    jac_func = qp.jacobian(circuit)
    jac_func_comp = qp.jacobian(circuit_compare)

    params = qp.numpy.array([0.0] * len(doubles), requires_grad=True)

    jacs = jac_func(params, excitations=doubles)
    jacs_comp = jac_func_comp(params, excitations=doubles)

    comm.Barrier()

    assert np.allclose(jacs, jacs_comp)


@pytest.mark.local_salt(42)
@pytest.mark.skipif(
    device_name == "lightning.kokkos", reason="Kokkos MPI does not support SparseHamiltonian"
)
@pytest.mark.parametrize(
    "returns",
    [
        qp.SparseHamiltonian(
            qp.Hamiltonian(
                [0.1], [qp.PauliX(wires=custom_wires[0]) @ qp.PauliY(wires=custom_wires[1])]
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
        qp.SparseHamiltonian(
            qp.Hamiltonian(
                [2.0], [qp.PauliX(wires=custom_wires[2]) @ qp.PauliZ(wires=custom_wires[0])]
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
        qp.SparseHamiltonian(
            qp.Hamiltonian(
                [2.0], [qp.PauliX(wires=custom_wires[1]) @ qp.PauliZ(wires=custom_wires[2])]
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
        qp.SparseHamiltonian(
            qp.Hamiltonian(
                [1.1], [qp.PauliX(wires=custom_wires[0]) @ qp.PauliZ(wires=custom_wires[2])]
            ).sparse_matrix(custom_wires),
            wires=custom_wires,
        ),
    ],
)
def test_adjoint_SparseHamiltonian_custom_wires(returns, seed):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    comm = MPI.COMM_WORLD
    dev_mpi = qp.device(device_name, wires=custom_wires, mpi=True)
    dev_cpu = qp.device("default.qubit", wires=custom_wires)

    def circuit(params):
        circuit_ansatz(params, wires=custom_wires)
        return qp.expval(returns)

    if comm.Get_rank() == 0:
        n_params = 30
        rng = np.random.default_rng(seed)
        params = rng.random(n_params)
    else:
        params = None

    params = comm.bcast(params, root=0)

    qnode_mpi = qp.QNode(circuit, dev_mpi, diff_method="adjoint")
    qnode_cpu = qp.QNode(circuit, dev_cpu, diff_method="parameter-shift")

    j_gpu = qp.jacobian(qnode_mpi)(params)
    j_cpu = qp.jacobian(qnode_cpu)(params)

    comm.Barrier()

    assert np.allclose(j_cpu, j_gpu)


@pytest.mark.skipif(
    device_name == "lightning.kokkos", reason="Kokkos MPI does not support SparseHamiltonian"
)
@pytest.mark.local_salt(42)
@pytest.mark.parametrize(
    "returns",
    [
        qp.SparseHamiltonian(
            qp.Hamiltonian(
                [0.1],
                [qp.PauliZ(1) @ qp.PauliX(0) @ qp.Identity(2) @ qp.PauliX(4) @ qp.Identity(5)],
            ).sparse_matrix(range(len(custom_wires))),
            wires=range(len(custom_wires)),
        ),
        qp.SparseHamiltonian(
            qp.Hamiltonian(
                [0.1],
                [qp.PauliX(1) @ qp.PauliZ(0)],
            ).sparse_matrix(range(len(custom_wires))),
            wires=range(len(custom_wires)),
        ),
        qp.SparseHamiltonian(
            qp.Hamiltonian(
                [0.1],
                [qp.PauliX(0)],
            ).sparse_matrix(range(len(custom_wires))),
            wires=range(len(custom_wires)),
        ),
        qp.SparseHamiltonian(
            qp.Hamiltonian(
                [0.1],
                [qp.PauliX(5)],
            ).sparse_matrix(range(len(custom_wires))),
            wires=range(len(custom_wires)),
        ),
        qp.SparseHamiltonian(
            qp.Hamiltonian(
                [0.1],
                [qp.PauliX(0) @ qp.PauliZ(1)],
            ).sparse_matrix(range(len(custom_wires))),
            wires=range(len(custom_wires)),
        ),
        qp.SparseHamiltonian(
            qp.Hamiltonian([2.0], [qp.PauliX(1) @ qp.PauliZ(2)]).sparse_matrix(
                range(len(custom_wires))
            ),
            wires=range(len(custom_wires)),
        ),
        qp.SparseHamiltonian(
            qp.Hamiltonian([2.0], [qp.PauliX(2) @ qp.PauliZ(4)]).sparse_matrix(
                range(len(custom_wires))
            ),
            wires=range(len(custom_wires)),
        ),
        qp.SparseHamiltonian(
            qp.Hamiltonian([1.1], [qp.PauliX(2) @ qp.PauliZ(0)]).sparse_matrix(
                range(len(custom_wires))
            ),
            wires=range(len(custom_wires)),
        ),
    ],
)
def test_adjoint_SparseHamiltonian(returns, seed):
    """Integration tests that compare to default.qubit for a large circuit containing parametrized
    operations and when using custom wire labels"""

    comm = MPI.COMM_WORLD
    dev_mpi = qp.device(device_name, wires=len(custom_wires), mpi=True)
    dev_cpu = qp.device("default.qubit", wires=len(custom_wires))

    def circuit(params):
        circuit_ansatz(params, wires=range(len(custom_wires)))
        return qp.expval(returns)

    if comm.Get_rank() == 0:
        n_params = 30
        rng = np.random.default_rng(seed)
        params = rng.random(n_params)
    else:
        params = None

    params = comm.bcast(params, root=0)

    qnode_mpi = qp.QNode(circuit, dev_mpi, diff_method="adjoint")
    qnode_cpu = qp.QNode(circuit, dev_cpu, diff_method="parameter-shift")

    j_gpu = qp.jacobian(qnode_mpi)(params)
    j_cpu = qp.jacobian(qnode_cpu)(params)

    comm.Barrier()

    assert np.allclose(j_cpu, j_gpu)


@pytest.mark.local_salt(42)
@pytest.mark.parametrize("n_targets", range(1, 5))
def test_qubit_unitary(dev, n_targets, seed):
    """Tests that ``qp.QubitUnitary`` can be included in circuits differentiated with the adjoint method."""
    n_wires = len(dev.wires)
    dev_def = qp.device("default.qubit", wires=n_wires)
    h = 1e-3 if dev.c_dtype == np.complex64 else 1e-7
    c_dtype = dev.c_dtype

    rng = np.random.default_rng(seed)
    par = 2 * np.pi * rng.random(n_wires)
    U = rng.random((2**n_targets, 2**n_targets)) + 1j * rng.random((2**n_targets, 2**n_targets))
    U, _ = np.linalg.qr(U)
    init_state = create_random_init_state(n_wires, dev.c_dtype, seed)

    comm = MPI.COMM_WORLD
    par = comm.bcast(par, root=0)
    U = comm.bcast(U, root=0)
    init_state = comm.bcast(init_state, root=0)

    init_state = np.array(init_state, requires_grad=False, dtype=c_dtype)
    U = np.array(U, requires_grad=False, dtype=c_dtype)
    obs = qp.prod(*(qp.PauliZ(i) for i in range(n_wires)))

    def circuit(x):
        qp.StatePrep(init_state, wires=range(n_wires))
        for i in range(n_wires // 2):
            qp.RY(x[i], wires=i)
        qp.QubitUnitary(U, wires=range(n_targets))
        for i in range(n_wires // 2, n_wires):
            qp.RY(x[i], wires=i)
        return qp.expval(obs)

    circ = qp.QNode(circuit, dev, diff_method="adjoint")
    circ_ps = qp.QNode(circuit, dev, diff_method="parameter-shift")
    circ_def = qp.QNode(circuit, dev_def, diff_method="adjoint")
    jac = qp.jacobian(circ)(par)
    jac_ps = qp.jacobian(circ_ps)(par)
    jac_def = qp.jacobian(circ_def)(par)

    comm.Barrier()

    assert len(jac) == n_wires
    assert not np.allclose(jac, 0.0)
    assert np.allclose(jac, jac_ps, atol=h, rtol=0)
    assert np.allclose(jac, jac_def, atol=h, rtol=0)


@pytest.mark.local_salt(42)
@pytest.mark.parametrize("n_targets", [1, 2])
def test_diff_qubit_unitary(dev, n_targets, seed):
    """Tests that ``qp.QubitUnitary`` can be differentiated with the adjoint method."""
    n_wires = len(dev.wires)
    dev_def = qp.device("default.qubit", wires=n_wires)
    h = 1e-3 if dev.c_dtype == np.complex64 else 1e-7
    c_dtype = dev.c_dtype

    rng = np.random.default_rng(seed)
    par = 2 * np.pi * rng.random(n_wires)
    U = rng.random((2**n_targets, 2**n_targets)) + 1j * rng.random((2**n_targets, 2**n_targets))
    U, _ = np.linalg.qr(U)
    init_state = create_random_init_state(n_wires, dev.c_dtype, seed)

    comm = MPI.COMM_WORLD
    par = comm.bcast(par, root=0)
    U = comm.bcast(U, root=0)
    init_state = comm.bcast(init_state, root=0)

    init_state = np.array(init_state, requires_grad=False, dtype=c_dtype)
    U = np.array(U, requires_grad=False, dtype=c_dtype)
    obs = qp.prod(*(qp.PauliZ(i) for i in range(n_wires)))

    def circuit(x, u_mat):
        qp.StatePrep(init_state, wires=range(n_wires))
        for i in range(n_wires // 2):
            qp.RY(x[i], wires=i)
        qp.QubitUnitary(u_mat, wires=range(n_targets))
        for i in range(n_wires // 2, n_wires):
            qp.RY(x[i], wires=i)
        return qp.expval(obs)

    circ = qp.QNode(circuit, dev, diff_method="adjoint")
    circ_def = qp.QNode(circuit, dev_def, diff_method="adjoint")
    circ_fd = qp.QNode(circuit, dev, diff_method="finite-diff", gradient_kwargs={"h": h})
    circ_ps = qp.QNode(circuit, dev, diff_method="parameter-shift")
    jacs = qp.jacobian(circ)(par, U)
    jacs_def = qp.jacobian(circ_def)(par, U)
    jacs_fd = qp.jacobian(circ_fd)(par, U)
    jacs_ps = qp.jacobian(circ_ps)(par, U)

    comm.Barrier()

    for jac, jac_def, jac_fd, jac_ps in zip(jacs, jacs_def, jacs_fd, jacs_ps):
        assert not np.allclose(jac, 0.0)
        assert np.allclose(jac, jac_fd, atol=h, rtol=0)
        assert np.allclose(jac, jac_ps, atol=h, rtol=0)
        assert np.allclose(jac, jac_def, atol=h, rtol=0)
