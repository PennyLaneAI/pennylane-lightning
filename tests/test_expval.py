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
Unit tests for the expval method of Lightning devices.
"""
import itertools

import numpy as np
import pennylane as qml
import pytest
from conftest import PHI, THETA, VARPHI
from conftest import LightningDevice as ld
from conftest import device_name

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
class TestExpval:
    """Test expectation values"""

    def test_identity_expectation(self, theta, phi, qubit_device, tol):
        """Test that identity expectation value (i.e. the trace) is 1"""
        dev = qubit_device(wires=3)

        O1 = qml.Identity(wires=[0])
        O2 = qml.Identity(wires=[1])
        ops = [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1])]
        tape = qml.tape.QuantumScript(ops, [qml.expval(O1), qml.expval(O2)])
        res = dev.execute(tape)
        assert np.allclose(res, np.array([1, 1]), tol)

    def test_pauliz_expectation(self, theta, phi, qubit_device, tol):
        """Test that PauliZ expectation value is correct"""
        dev = qubit_device(wires=3)

        O1 = qml.PauliZ(wires=[0])
        O2 = qml.PauliZ(wires=[1])
        ops = [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1])]
        tape = qml.tape.QuantumScript(ops, [qml.expval(O1), qml.expval(O2)])
        res = dev.execute(tape)
        assert np.allclose(res, np.array([np.cos(theta), np.cos(theta) * np.cos(phi)]), tol)

    def test_paulix_expectation(self, theta, phi, qubit_device, tol):
        """Test that PauliX expectation value is correct"""
        dev = qubit_device(wires=3)

        O1 = qml.PauliX(wires=[0])
        O2 = qml.PauliX(wires=[1])
        ops = [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1])]
        tape = qml.tape.QuantumScript(ops, [qml.expval(O1), qml.expval(O2)])
        res = dev.execute(tape)

        assert np.allclose(
            res,
            np.array([np.sin(theta) * np.sin(phi), np.sin(phi)], dtype=dev.dtype),
            tol * 10,
        )

    def test_pauliy_expectation(self, theta, phi, qubit_device, tol):
        """Test that PauliY expectation value is correct"""
        dev = qubit_device(wires=3)

        O1 = qml.PauliY(wires=[0])
        O2 = qml.PauliY(wires=[1])
        ops = [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1])]
        tape = qml.tape.QuantumScript(ops, [qml.expval(O1), qml.expval(O2)])
        res = dev.execute(tape)

        assert np.allclose(res, np.array([0, -np.cos(theta) * np.sin(phi)]), tol)

    def test_hadamard_expectation(self, theta, phi, qubit_device, tol):
        """Test that Hadamard expectation value is correct"""
        dev = qubit_device(wires=3)

        O1 = qml.Hadamard(wires=[0])
        O2 = qml.Hadamard(wires=[1])
        ops = [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1])]
        tape = qml.tape.QuantumScript(ops, [qml.expval(O1), qml.expval(O2)])
        res = dev.execute(tape)

        expected = np.array(
            [np.sin(theta) * np.sin(phi) + np.cos(theta), np.cos(theta) * np.cos(phi) + np.sin(phi)]
        ) / np.sqrt(2)
        assert np.allclose(res, expected, tol)

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support qml.Projector()",
    )
    def test_projector_expectation(self, theta, phi, qubit_device, tol):
        """Test that Projector variance value is correct"""
        n_qubits = 2
        dev_def = qml.device("default.qubit", wires=n_qubits)
        dev = qubit_device(wires=n_qubits)

        init_state = np.random.rand(2**n_qubits) + 1j * np.random.rand(2**n_qubits)
        init_state /= np.linalg.norm(init_state)
        obs = qml.Projector(np.array([0, 1, 0, 0]) / np.sqrt(2), wires=[0, 1])

        def circuit():
            qml.StatePrep(init_state, wires=range(n_qubits))
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(obs)

        circ = qml.QNode(circuit, dev)
        circ_def = qml.QNode(circuit, dev_def)
        assert np.allclose(circ(), circ_def(), tol)

    @pytest.mark.parametrize("n_wires", range(1, 7))
    def test_hermitian_expectation(self, n_wires, theta, phi, qubit_device, tol):
        """Test that Hermitian expectation value is correct"""
        n_qubits = 7
        dev_def = qml.device("default.qubit", wires=n_qubits)
        dev = qubit_device(wires=n_qubits)

        m = 2**n_wires
        U = np.random.rand(m, m) + 1j * np.random.rand(m, m)
        U = U + np.conj(U.T)
        wires = list(range((n_qubits - n_wires), (n_qubits - n_wires) + n_wires))
        perms = list(itertools.permutations(wires))
        init_state = np.random.rand(2**n_qubits) + 1j * np.random.rand(2**n_qubits)
        init_state /= np.linalg.norm(init_state)
        if n_wires > 4:
            perms = perms[0::30]
        for perm in perms:
            obs = qml.Hermitian(U, wires=perm)

            def circuit():
                qml.StatePrep(init_state, wires=range(n_qubits))
                qml.RX(theta, wires=[0])
                qml.RY(phi, wires=[1])
                qml.RX(theta, wires=[2])
                qml.RY(phi, wires=[3])
                qml.RX(theta, wires=[4])
                qml.RY(phi, wires=[5])
                qml.RX(theta, wires=[6])
                qml.CNOT(wires=[0, 1])
                return qml.expval(obs)

            circ = qml.QNode(circuit, dev)
            circ_def = qml.QNode(circuit, dev_def)
            if device_name == "lightning.tensor":
                if n_wires > 1:
                    with pytest.raises(
                        ValueError,
                        match="The number of Hermitian observables target wires should be 1.",
                    ):
                        assert np.allclose(circ(), circ_def(), tol)
                else:
                    np.allclose(circ(), circ_def(), rtol=1e-6)
            else:
                assert np.allclose(circ(), circ_def(), tol)


@pytest.mark.parametrize(
    "diff_method",
    [
        "parameter-shift",
        pytest.param(
            "adjoint",
            marks=pytest.mark.skipif(
                device_name == "lightning.tensor",
                reason="lightning.tensor does not support the adjoint method",
            ),
        ),
    ],
)
class TestExpOperatorArithmetic:
    """Test integration of lightning with SProd, Prod, and Sum."""

    def test_sprod(self, diff_method, qubit_device):
        """Test the `SProd` class with lightning qubit."""

        dev = qubit_device(wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.s_prod(0.5, qml.PauliZ(0)))

        x = qml.numpy.array(0.123, requires_grad=True)
        res = circuit(x)
        assert qml.math.allclose(res, 0.5 * np.cos(x))

        g = qml.grad(circuit)(x)
        expected_grad = -0.5 * np.sin(x)
        assert qml.math.allclose(g, expected_grad)

    def test_prod(self, diff_method, qubit_device):
        """Test the `Prod` class with lightning qubit."""

        dev = qubit_device(wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(1)
            qml.PauliZ(1)
            return qml.expval(qml.prod(qml.PauliZ(0), qml.PauliX(1)))

        x = qml.numpy.array(0.123, requires_grad=True)
        res = circuit(x)
        assert qml.math.allclose(res, -np.cos(x))

        g = qml.grad(circuit)(x)
        expected_grad = np.sin(x)
        assert qml.math.allclose(g, expected_grad)

    def test_sum(self, diff_method, qubit_device):
        """Test the `Sum` class with lightning qubit."""

        dev = qubit_device(wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return qml.expval(qml.sum(qml.PauliZ(0), qml.PauliX(1)))

        x = qml.numpy.array(-3.21, requires_grad=True)
        y = qml.numpy.array(2.34, requires_grad=True)
        res = circuit(x, y)
        assert qml.math.allclose(res, np.cos(x) + np.sin(y))

        g = qml.grad(circuit)(x, y)
        expected = (-np.sin(x), np.cos(y))
        assert qml.math.allclose(g, expected)

    def test_integration(self, diff_method, qubit_device):
        """Test a Combination of `Sum`, `SProd`, and `Prod`."""

        obs = qml.sum(qml.s_prod(2.3, qml.PauliZ(0)), -0.5 * qml.prod(qml.PauliY(0), qml.PauliZ(1)))

        dev = qubit_device(wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return qml.expval(obs)

        x = qml.numpy.array(0.654, requires_grad=True)
        y = qml.numpy.array(-0.634, requires_grad=True)

        res = circuit(x, y)
        expected = 2.3 * np.cos(x) + 0.5 * np.sin(x) * np.cos(y)
        assert qml.math.allclose(res, expected)

        g = qml.grad(circuit)(x, y)
        expected = (-2.3 * np.sin(x) + 0.5 * np.cos(y) * np.cos(x), -0.5 * np.sin(x) * np.sin(y))
        assert qml.math.allclose(g, expected)


@pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorExpval:
    """Test tensor expectation values"""

    def test_paulix_pauliy(self, theta, phi, varphi, qubit_device, tol):
        """Test that a tensor product involving PauliX and PauliY works
        correctly"""
        dev = qubit_device(wires=3)
        obs = qml.PauliX(0) @ qml.PauliY(2)
        ops = [
            qml.RX(theta, wires=[0]),
            qml.RX(phi, wires=[1]),
            qml.RX(varphi, wires=[2]),
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=[1, 2]),
        ]
        tape = qml.tape.QuantumScript(ops, [qml.expval(op=obs)])
        res = dev.execute(tape)

        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)

        assert np.allclose(res, expected, atol=tol)

    def test_pauliz_identity(self, theta, phi, varphi, qubit_device, tol):
        """Test that a tensor product involving PauliZ and Identity works
        correctly"""
        dev = qubit_device(wires=3)
        obs = qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2)
        ops = [
            qml.RX(theta, wires=[0]),
            qml.RX(phi, wires=[1]),
            qml.RX(varphi, wires=[2]),
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=[1, 2]),
        ]
        tape = qml.tape.QuantumScript(ops, [qml.expval(op=obs)])
        res = dev.execute(tape)

        expected = np.cos(varphi) * np.cos(phi)

        assert np.allclose(res, expected, tol)

    def test_pauliz_hadamard_pauliy(self, theta, phi, varphi, qubit_device, tol):
        """Test that a tensor product involving PauliZ and PauliY and Hadamard
        works correctly"""
        dev = qubit_device(wires=3)
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)
        ops = [
            qml.RX(theta, wires=[0]),
            qml.RX(phi, wires=[1]),
            qml.RX(varphi, wires=[2]),
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=[1, 2]),
        ]
        tape = qml.tape.QuantumScript(ops, [qml.expval(op=obs)])
        res = dev.execute(tape)
        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)

        assert np.allclose(res, expected, tol)
