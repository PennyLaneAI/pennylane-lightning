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
Unit tests for Lightning devices.
"""
import math
from functools import partial

import numpy as np
import pennylane as qml
import pytest
from conftest import PHI, THETA
from conftest import LightningDevice as ld
from conftest import device_name, get_random_matrix, get_random_normalized_state
from pennylane.exceptions import DeviceError
from pennylane.operation import Operation
from pennylane.wires import Wires

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


class TestExpval:
    """Tests that expectation values are properly calculated or that the proper errors are raised."""

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support single wire devices",
    )
    @pytest.mark.parametrize(
        "operation,input,expected_output",
        [
            (qml.PauliX, [1 / math.sqrt(2), 1 / math.sqrt(2)], 1),
            (qml.PauliX, [1 / math.sqrt(2), -1 / math.sqrt(2)], -1),
            (qml.PauliX, [1, 0], 0),
            (qml.PauliY, [1 / math.sqrt(2), 1j / math.sqrt(2)], 1),
            (qml.PauliY, [1 / math.sqrt(2), -1j / math.sqrt(2)], -1),
            (qml.PauliY, [1, 0], 0),
            (qml.PauliZ, [1, 0], 1),
            (qml.PauliZ, [0, 1], -1),
            (qml.PauliZ, [1 / math.sqrt(2), 1 / math.sqrt(2)], 0),
            (qml.Hadamard, [1, 0], 1 / math.sqrt(2)),
            (qml.Hadamard, [0, 1], -1 / math.sqrt(2)),
            (qml.Hadamard, [1 / math.sqrt(2), 1 / math.sqrt(2)], 1 / math.sqrt(2)),
            (qml.Identity, [1, 0], 1),
            (qml.Identity, [0, 1], 1),
            (qml.Identity, [1 / math.sqrt(2), -1 / math.sqrt(2)], 1),
        ],
    )
    def test_expval_single_wire_no_parameters(
        self, qubit_device, tol, operation, input, expected_output
    ):
        """Tests that expectation values are properly calculated for single-wire observables without parameters."""
        dev = qubit_device(wires=1)
        obs = operation(wires=[0])
        ops = [qml.StatePrep(np.array(input), wires=[0])]
        tape = qml.tape.QuantumScript(ops, [qml.expval(op=obs)])
        res = dev.execute(tape)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support single wire devices",
    )
    def test_expval_estimate(self):
        """Test that the expectation value is not analytically calculated"""
        dev = qml.device(device_name, wires=1)

        @partial(qml.set_shots, shots=3)
        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliX(0))

        expval = circuit()

        # With 3 samples we are guaranteed to see a difference between
        # an estimated variance an an analytically calculated one
        assert expval != 0.0


class TestVar:
    """Tests that variances are properly calculated."""

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support single wire devices",
    )
    @pytest.mark.parametrize(
        "operation,input,expected_output",
        [
            (qml.PauliX, [1 / math.sqrt(2), 1 / math.sqrt(2)], 0),
            (qml.PauliX, [1 / math.sqrt(2), -1 / math.sqrt(2)], 0),
            (qml.PauliX, [1, 0], 1),
            (qml.PauliY, [1 / math.sqrt(2), 1j / math.sqrt(2)], 0),
            (qml.PauliY, [1 / math.sqrt(2), -1j / math.sqrt(2)], 0),
            (qml.PauliY, [1, 0], 1),
            (qml.PauliZ, [1, 0], 0),
            (qml.PauliZ, [0, 1], 0),
            (qml.PauliZ, [1 / math.sqrt(2), 1 / math.sqrt(2)], 1),
            (qml.Hadamard, [1, 0], 1 / 2),
            (qml.Hadamard, [0, 1], 1 / 2),
            (qml.Hadamard, [1 / math.sqrt(2), 1 / math.sqrt(2)], 1 / 2),
            (qml.Identity, [1, 0], 0),
            (qml.Identity, [0, 1], 0),
            (qml.Identity, [1 / math.sqrt(2), -1 / math.sqrt(2)], 0),
        ],
    )
    def test_var_single_wire_no_parameters(
        self, qubit_device, tol, operation, input, expected_output
    ):
        """Tests that variances are properly calculated for single-wire observables without parameters."""
        dev = qubit_device(wires=1)
        obs = operation(wires=[0])
        ops = [qml.StatePrep(np.array(input), wires=[0])]
        tape = qml.tape.QuantumScript(ops, [qml.var(op=obs)])
        res = dev.execute(tape)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support single-wire devices",
    )
    def test_var_estimate(self):
        """Test that the variance is not analytically calculated"""

        dev = qml.device(device_name, wires=1)

        @partial(qml.set_shots, shots=3)
        @qml.qnode(dev)
        def circuit():
            return qml.var(qml.PauliX(0))

        var = circuit()

        # With 3 samples we are guaranteed to see a difference between
        # an estimated variance and an analytically calculated one
        assert var != 1.0


class TestSample:
    """Tests that samples are properly calculated."""

    def test_sample_dimensions(self, qubit_device):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """

        # Explicitly resetting is necessary as the internal
        # state is set to None in __init__ and only properly
        # initialized during reset
        dev = qubit_device(wires=2)
        ops = [qml.RX(1.5708, wires=[0]), qml.RX(1.5708, wires=[1])]

        shots = 10
        obs = qml.PauliZ(wires=[0])
        tape = qml.tape.QuantumScript(ops, [qml.sample(op=obs)], shots=shots)
        s1 = dev.execute(tape)

        assert np.array_equal(s1.shape, (shots,))

        shots = 12
        obs = qml.PauliZ(wires=[1])
        tape = qml.tape.QuantumScript(ops, [qml.sample(op=obs)], shots=shots)
        s2 = dev.execute(tape)
        assert np.array_equal(s2.shape, (shots,))

        shots = 17
        obs = qml.PauliX(0) @ qml.PauliZ(1)
        tape = qml.tape.QuantumScript(ops, [qml.sample(op=obs)], shots=shots)
        s3 = dev.execute(tape)

        assert np.array_equal(s3.shape, (shots,))

    def test_sample_values(self, qubit_device, tol):
        """Tests if the samples returned by sample have
        the correct values
        """

        # Explicitly resetting is necessary as the internal
        # state is set to None in __init__ and only properly
        # initialized during reset
        dev = qubit_device(wires=2)

        ops = [qml.RX(1.5708, wires=[0])]

        shots = qml.measurements.Shots(1000)
        obs = qml.PauliZ(0)
        tape = qml.tape.QuantumScript(ops, [qml.sample(op=obs)], shots=shots)
        s1 = dev.execute(tape)

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1**2, 1, atol=tol, rtol=0)


class TestLightningDeviceIntegration:
    """Integration tests for lightning device. This test ensures it integrates
    properly with the PennyLane interface, in particular QNode."""

    def test_load_default_qubit_device(self):
        """Test that the default plugin loads correctly"""

        dev = qml.device(device_name, wires=2)
        assert not dev.shots
        assert len(dev.wires) == 2

    @pytest.mark.parametrize(
        "wires, expected_state",
        [
            ([3, 0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0]),
            ([1, 0], [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0]),
        ],
    )
    def test_dynamic_allocate_two_qubits(self, qubit_device, tol, wires, expected_state):
        """Test the dynamic allocation of qubits in Lightning devices"""

        def circuit():
            qml.Identity(wires=wires[0])
            qml.Hadamard(wires=wires[1])
            return qml.state()

        dev = qubit_device(None)
        results = qml.qnode(dev)(circuit)()
        assert np.allclose(results, expected_state, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "wires, expected_state",
        [
            ([3, 1, 0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0, 0, 0, 0, 0]),
            ([2, 1, 0], [1 / np.sqrt(2), 0, 0, 0, 1 / np.sqrt(2), 0, 0, 0]),
        ],
    )
    def test_dynamic_allocate_three_qubits(self, qubit_device, tol, wires, expected_state):
        """Test the dynamic allocation of qubits in Lightning devices"""

        def circuit():
            qml.Identity(wires=wires[0])
            qml.Identity(wires=wires[1])
            qml.Hadamard(wires=wires[2])
            return qml.state()

        dev = qubit_device(None)
        results = qml.qnode(dev)(circuit)()
        assert np.allclose(results, expected_state, atol=tol, rtol=0)

    @pytest.mark.skipif(
        device_name == "lightning.tensor", reason="lightning.tensor requires num_wires > 1"
    )
    def test_qubit_circuit(self, qubit_device, tol):
        """Test that the default qubit plugin provides correct result for a simple circuit"""

        p = 0.543
        dev = qubit_device(wires=1)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -np.sin(p)

        assert np.isclose(circuit(p), expected, atol=tol, rtol=0)

    @pytest.mark.skipif(
        device_name == "lightning.tensor", reason="lightning.tensor requires num_wires > 1"
    )
    def test_qubit_identity(self, qubit_device, tol):
        """Test that the default qubit plugin provides correct result for the Identity expectation"""

        p = 0.543
        dev = qubit_device(wires=1)

        @qml.qnode(dev)
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.Identity(0))

        assert np.isclose(circuit(p), 1, atol=tol, rtol=0)

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support single wire devices",
    )
    def test_nonzero_shots(self, tol_stochastic):
        """Test that the default qubit plugin provides correct result for high shot number"""

        shots = 10**4
        dev = qml.device(device_name, wires=1)

        p = 0.543

        @partial(qml.set_shots, shots=shots)
        @qml.qnode(dev)
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        runs = []
        for _ in range(100):
            runs.append(circuit(p))

        assert np.isclose(np.mean(runs), -np.sin(p), atol=tol_stochastic, rtol=0)

    # This test is ran against the state |0> with one Z expval
    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support single-wire devices",
    )
    @pytest.mark.parametrize(
        "name,expected_output",
        [
            ("PauliX", -1),
            ("PauliY", -1),
            ("PauliZ", 1),
            ("Hadamard", 0),
        ],
    )
    def test_supported_gate_single_wire_no_parameters(
        self, qubit_device, tol, name, expected_output
    ):
        """Tests supported gates that act on a single wire that are not parameterized"""
        dev = qubit_device(wires=1)
        op = getattr(qml.ops, name)

        if hasattr(dev, "supports_operation"):
            assert dev.supports_operation(name)

        @qml.qnode(dev)
        def circuit():
            op(wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "name,expected_output",
        [
            ("CNOT", [-1 / 2, 1]),
            ("SWAP", [-1 / 2, -1 / 2]),
            ("CZ", [-1 / 2, -1 / 2]),
        ],
    )
    def test_supported_gate_two_wires_no_parameters(self, qubit_device, tol, name, expected_output):
        """Tests supported gates that act on two wires that are not parameterized"""
        dev = qubit_device(wires=2)
        op = getattr(qml.ops, name)

        if hasattr(dev, "supports_operation"):
            assert dev.supports_operation(name)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(np.array([1 / 2, 0, 0, math.sqrt(3) / 2]), wires=[0, 1])
            op(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "name,expected_output",
        [
            ("CSWAP", [-1, -1, 1]),
        ],
    )
    def test_supported_gate_three_wires_no_parameters(
        self, qubit_device, tol, name, expected_output
    ):
        """Tests supported gates that act on three wires that are not parameterized"""
        dev = qubit_device(wires=3)
        op = getattr(qml.ops, name)

        if hasattr(dev, "supports_operation"):
            assert dev.supports_operation(name)

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2])
            op(wires=[0, 1, 2])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran with two Z expvals
    @pytest.mark.parametrize(
        "name,par,expected_output",
        [
            ("BasisState", [0, 0], [1, 1]),
            ("BasisState", [1, 0], [-1, 1]),
            ("BasisState", [0, 1], [1, -1]),
            ("StatePrep", [1, 0, 0, 0], [1, 1]),
            ("StatePrep", [0, 0, 1, 0], [-1, 1]),
            ("StatePrep", [0, 1, 0, 0], [1, -1]),
        ],
    )
    def test_supported_state_preparation(self, qubit_device, tol, name, par, expected_output):
        """Tests supported state preparations"""
        dev = qubit_device(wires=2)
        op = getattr(qml.ops, name)

        if hasattr(dev, "supports_operation"):
            assert dev.supports_operation(name)

        @qml.qnode(dev)
        def circuit():
            op(np.array(par), wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran with two Z expvals
    @pytest.mark.parametrize(
        "name,par,wires,expected_output",
        [
            ("BasisState", [1, 1], [0, 1], [-1, -1]),
            pytest.param(
                "BasisState",
                [1],
                [0],
                [-1, 1],
                marks=pytest.mark.skipif(
                    device_name == "lightning.tensor",
                    reason="lightning.tensor requires a vector of length num_wires for qml.BasisState()",
                ),
            ),
            pytest.param(
                "BasisState",
                [1],
                [1],
                [1, -1],
                marks=pytest.mark.skipif(
                    device_name == "lightning.tensor",
                    reason="lightning.tensor requires a vector of length num_wires for qml.BasisState()",
                ),
            ),
        ],
    )
    def test_basis_state_2_qubit_subset(self, qubit_device, tol, name, par, wires, expected_output):
        """Tests qubit basis state preparation on subsets of qubits"""
        dev = qubit_device(wires=2)
        op = getattr(qml.ops, name)

        @qml.qnode(dev)
        def circuit():
            op(np.array(par), wires=wires)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is run with two expvals
    @pytest.mark.parametrize(
        "name,par,wires,expected_output",
        [
            ("StatePrep", [0, 1], [1], [1, -1]),
            ("StatePrep", [0, 1], [0], [-1, 1]),
            ("StatePrep", [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], [1], [1, 0]),
            ("StatePrep", [1j / 2.0, np.sqrt(3) / 2.0], [1], [1, -0.5]),
            ("StatePrep", [(2 - 1j) / 3.0, 2j / 3.0], [0], [1 / 9.0, 1]),
        ],
    )
    def test_state_vector_2_qubit_subset(
        self, qubit_device, tol, name, par, wires, expected_output
    ):
        """Tests qubit state vector preparation on subsets of 2 qubits"""
        dev = qubit_device(wires=2)
        op = getattr(qml.ops, name)

        par = np.array(par)

        @qml.qnode(dev)
        def circuit():
            op(par, wires=wires)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is run with three expvals
    @pytest.mark.parametrize(
        "name,par,wires,expected_output",
        [
            (
                "StatePrep",
                [1j / np.sqrt(10), (1 - 2j) / np.sqrt(10), 0, 0, 0, 2 / np.sqrt(10), 0, 0],
                [0, 1, 2],
                [1 / 5.0, 1.0, -4 / 5.0],
            ),
            ("StatePrep", [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], [0, 2], [0.0, 1.0, 0.0]),
            ("StatePrep", [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], [0, 1], [0.0, 0.0, 1.0]),
            ("StatePrep", [0, 1, 0, 0, 0, 0, 0, 0], [2, 1, 0], [-1.0, 1.0, 1.0]),
            ("StatePrep", [0, 1j, 0, 0, 0, 0, 0, 0], [0, 2, 1], [1.0, -1.0, 1.0]),
            ("StatePrep", [0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)], [1, 0], [-1.0, 0.0, 1.0]),
            ("StatePrep", [0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)], [0, 1], [0.0, -1.0, 1.0]),
        ],
    )
    def test_state_vector_3_qubit_subset(
        self, qubit_device, tol, name, par, wires, expected_output
    ):
        """Tests qubit state vector preparation on subsets of 3 qubits"""
        dev = qubit_device(wires=3)
        op = getattr(qml.ops, name)

        par = np.array(par)

        @qml.qnode(dev)
        def circuit():
            op(par, wires=wires)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran on the state |0> with one Z expvals
    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support single wire devices",
    )
    @pytest.mark.parametrize(
        "name,par,expected_output",
        [
            ("PhaseShift", [math.pi / 2], 1),
            ("PhaseShift", [-math.pi / 4], 1),
            ("RX", [math.pi / 2], 0),
            ("RX", [-math.pi / 4], 1 / math.sqrt(2)),
            ("RY", [math.pi / 2], 0),
            ("RY", [-math.pi / 4], 1 / math.sqrt(2)),
            ("RZ", [math.pi / 2], 1),
            ("RZ", [-math.pi / 4], 1),
            ("Rot", [math.pi / 2, 0, 0], 1),
            ("Rot", [0, math.pi / 2, 0], 0),
            ("Rot", [0, 0, math.pi / 2], 1),
            ("Rot", [math.pi / 2, -math.pi / 4, -math.pi / 4], 1 / math.sqrt(2)),
            ("Rot", [-math.pi / 4, math.pi / 2, math.pi / 4], 0),
            ("Rot", [-math.pi / 4, math.pi / 4, math.pi / 2], 1 / math.sqrt(2)),
        ],
    )
    def test_supported_gate_single_wire_with_parameters(
        self, qubit_device, tol, name, par, expected_output
    ):
        """Tests supported gates that act on a single wire that are parameterized"""
        dev = qubit_device(wires=1)
        op = getattr(qml.ops, name)

        if hasattr(dev, "supports_operation"):
            assert dev.supports_operation(name)

        @qml.qnode(dev)
        def circuit():
            op(*par, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran against the state 1/2|00>+sqrt(3)/2|11> with two Z expvals
    @pytest.mark.parametrize(
        "name,par,expected_output",
        [
            ("CRX", [0], [-1 / 2, -1 / 2]),
            ("CRX", [-math.pi], [-1 / 2, 1]),
            ("CRX", [math.pi / 2], [-1 / 2, 1 / 4]),
            ("CRY", [0], [-1 / 2, -1 / 2]),
            ("CRY", [-math.pi], [-1 / 2, 1]),
            ("CRY", [math.pi / 2], [-1 / 2, 1 / 4]),
            ("CRZ", [0], [-1 / 2, -1 / 2]),
            ("CRZ", [-math.pi], [-1 / 2, -1 / 2]),
            ("CRZ", [math.pi / 2], [-1 / 2, -1 / 2]),
            ("CRot", [math.pi / 2, 0, 0], [-1 / 2, -1 / 2]),
            ("CRot", [0, math.pi / 2, 0], [-1 / 2, 1 / 4]),
            ("CRot", [0, 0, math.pi / 2], [-1 / 2, -1 / 2]),
            ("CRot", [math.pi / 2, 0, -math.pi], [-1 / 2, -1 / 2]),
            ("CRot", [0, math.pi / 2, -math.pi], [-1 / 2, 1 / 4]),
            ("CRot", [-math.pi, 0, math.pi / 2], [-1 / 2, -1 / 2]),
            ("ControlledPhaseShift", [0], [-1 / 2, -1 / 2]),
            ("ControlledPhaseShift", [-math.pi], [-1 / 2, -1 / 2]),
            ("ControlledPhaseShift", [math.pi / 2], [-1 / 2, -1 / 2]),
            ("ControlledPhaseShift", [math.pi], [-1 / 2, -1 / 2]),
        ],
    )
    def test_supported_gate_two_wires_with_parameters(
        self, qubit_device, tol, name, par, expected_output
    ):
        """Tests supported gates that act on two wires that are parameterized"""
        dev = qubit_device(wires=2)
        op = getattr(qml.ops, name)

        if hasattr(dev, "supports_operation"):
            assert dev.supports_operation(name)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(np.array([1 / 2, 0, 0, math.sqrt(3) / 2]), wires=[0, 1])
            op(*par, wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support a single wire device",
    )
    @pytest.mark.parametrize(
        "name,state,expected_output",
        [
            ("PauliX", [1 / math.sqrt(2), 1 / math.sqrt(2)], 1),
            ("PauliX", [1 / math.sqrt(2), -1 / math.sqrt(2)], -1),
            ("PauliX", [1, 0], 0),
            ("PauliY", [1 / math.sqrt(2), 1j / math.sqrt(2)], 1),
            ("PauliY", [1 / math.sqrt(2), -1j / math.sqrt(2)], -1),
            ("PauliY", [1, 0], 0),
            ("PauliZ", [1, 0], 1),
            ("PauliZ", [0, 1], -1),
            ("PauliZ", [1 / math.sqrt(2), 1 / math.sqrt(2)], 0),
            ("Hadamard", [1, 0], 1 / math.sqrt(2)),
            ("Hadamard", [0, 1], -1 / math.sqrt(2)),
            ("Hadamard", [1 / math.sqrt(2), 1 / math.sqrt(2)], 1 / math.sqrt(2)),
        ],
    )
    def test_supported_observable_single_wire_no_parameters(
        self, qubit_device, tol, name, state, expected_output
    ):
        """Tests supported observables on single wires without parameters."""
        dev = qubit_device(wires=1)
        obs = getattr(qml.ops, name)

        if hasattr(dev, "supports_observable"):
            assert dev.supports_observable(name)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(np.array(state), wires=[0])
            return qml.expval(obs(wires=[0]))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support single wire devices",
    )
    @pytest.mark.parametrize(
        "name,state,expected_output,par",
        [
            ("Identity", [1, 0], 1, []),
            ("Identity", [0, 1], 1, []),
            ("Identity", [1 / math.sqrt(2), -1 / math.sqrt(2)], 1, []),
        ],
    )
    def test_supported_observable_single_wire_with_parameters(
        self, qubit_device, tol, name, state, expected_output, par
    ):
        """Tests supported observables on single wires with parameters."""
        dev = qubit_device(wires=1)
        obs = getattr(qml.ops, name)

        if hasattr(dev, "supports_observable"):
            assert dev.supports_observable(name)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(np.array(state), wires=[0])
            return qml.expval(obs(*par, wires=[0]))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    def test_multi_samples_return_correlated_results(self, qubit_device):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """
        dev = qubit_device(wires=2)

        @partial(qml.set_shots, shots=1000)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.sample(wires=[0, 1])

        outcomes = circuit()
        outcomes = outcomes.T

        assert np.array_equal(outcomes[0], outcomes[1])

    @pytest.mark.parametrize("num_wires", [3, 4, 5, 6, 7, 8])
    def test_multi_samples_return_correlated_results_more_wires_than_size_of_observable(
        self, num_wires
    ):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """

        dev = qml.device(device_name, wires=num_wires)

        @partial(qml.set_shots, shots=1000)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.sample(wires=[0, 1])

        outcomes = circuit()
        outcomes = outcomes.T

        assert np.array_equal(outcomes[0], outcomes[1])

    def test_snapshot_is_ignored_without_shot(self):
        """Tests if the Snapshot operator is ignored correctly"""
        dev = qml.device(device_name, wires=4)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.Snapshot()
            qml.adjoint(qml.Snapshot())
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        outcomes = circuit()

        assert np.allclose(outcomes, [0.0])

    def test_snapshot_is_ignored_with_shots(self):
        """Tests if the Snapshot operator is ignored correctly"""
        dev = qml.device(device_name, wires=4)

        @partial(qml.set_shots, shots=1000)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.Snapshot()
            qml.adjoint(qml.Snapshot())
            qml.CNOT(wires=[0, 1])
            return qml.sample(wires=[0, 1])

        outcomes = circuit()
        outcomes = outcomes.T

        assert np.array_equal(outcomes[0], outcomes[1])

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support _tensornet.state access",
    )
    def test_apply_qpe(self, qubit_device, tol):
        """Test the application of qml.QuantumPhaseEstimation"""
        dev = qubit_device(wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.QuantumPhaseEstimation(qml.matrix(qml.Hadamard, wire_order=[0])(wires=0), [0], [1])
            return qml.probs(wires=[0, 1])

        probs = circuit()

        # pylint: disable=protected-access
        res_sv = dev._statevector.state
        res_probs = probs

        expected_sv = np.array(
            [
                0.85355339 + 0.000000e00j,
                -0.14644661 - 6.123234e-17j,
                0.35355339 + 0.000000e00j,
                0.35355339 + 0.000000e00j,
            ]
        )
        expected_prob = np.array([0.72855339, 0.02144661, 0.125, 0.125])

        assert np.allclose(res_sv, expected_sv, atol=tol, rtol=0)
        assert np.allclose(res_probs, expected_prob, atol=tol, rtol=0)

    # Check the BlockEncode PennyLane page for details:
    # https://docs.pennylane.ai/en/stable/code/api/pennylane.BlockEncode.html
    @pytest.mark.parametrize(
        "op, op_wires",
        [
            [qml.BlockEncode, [0, 2]],
            [qml.ctrl(qml.BlockEncode, control=(1)), [0, 2]],
            [qml.ctrl(qml.BlockEncode, control=(2)), [0, 1]],
        ],
    )
    @pytest.mark.parametrize(
        "A",
        [
            np.array([[1, 1], [1, -1]]) / np.sqrt(2),
            np.array([[1, 1], [1, -1]]),
        ],
    )
    def test_apply_BlockEncode(self, op, op_wires, A, qubit_device, tol):
        """Test apply BlockEncode and C(BlockEncode)"""

        num_wires = 3
        dev = qubit_device(wires=num_wires)

        def circuit1(A):
            qml.Hadamard(0)
            qml.Hadamard(1)
            op(A, wires=op_wires)
            return qml.state()

        results = qml.qnode(dev)(circuit1)(A)

        dev_default = qml.device("default.qubit", wires=num_wires)
        expected = qml.qnode(dev_default)(circuit1)(A)

        assert np.allclose(results, expected, atol=tol, rtol=0)


class TestApplyLightningMethod:
    """Unit tests for the apply_lightning method."""

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor does not support direct access to the state",
    )
    @pytest.mark.parametrize(
        "ops0",
        [
            qml.PauliZ(0),
            qml.PauliY(0),
            qml.S(0),
            qml.RX(0.1234, 0),
            qml.Rot(0.1, 0.2, 0.3, 0),
            qml.T(0) @ qml.RY(0.1234, 0),
        ],
    )
    @pytest.mark.parametrize(
        "ops1",
        [
            qml.PauliZ(2),
            qml.PauliY(2),
            qml.S(2),
            qml.RX(0.1234, 2),
            qml.Rot(0.1, 0.2, 0.3, 2),
            qml.T(2) @ qml.RY(0.1234, 2),
        ],
    )
    def test_multiple_adjoint_operations(self, ops0, ops1, tol):
        """Test that multiple adjoint operations are handled correctly."""
        n_qubits = 4

        dev = qml.device(device_name, wires=n_qubits)
        dq = qml.device("default.qubit", wires=n_qubits)
        init_state = get_random_normalized_state(2**n_qubits)

        def circuit():
            qml.StatePrep(init_state, wires=range(n_qubits))
            qml.adjoint(ops0)
            qml.PhaseShift(0.1234, wires=0)
            qml.adjoint(ops1)
            return qml.state()

        results = qml.QNode(circuit, dev)()
        expected = qml.QNode(circuit, dq)()
        assert np.allclose(results, expected)


@pytest.mark.parametrize(
    "op",
    [
        qml.BasisState([0, 0], wires=[0, 1]),
        qml.StatePrep([0, 1, 0, 0], wires=[0, 1]),
    ],
)
@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
def test_circuit_with_stateprep(op, theta, phi, tol):
    """Test mid-circuit StatePrep"""
    n_qubits = 5
    n_wires = 2
    dev_def = qml.device("default.qubit", wires=n_qubits)
    dev = qml.device(device_name, wires=n_qubits)
    m = 2**n_wires
    U = get_random_matrix(m)
    U, _ = np.linalg.qr(U)
    init_state = get_random_normalized_state(2**n_qubits)

    def circuit():
        qml.StatePrep(init_state, wires=range(n_qubits))
        qml.RY(theta, wires=[0])
        qml.RY(phi, wires=[1])
        qml.CNOT(wires=[0, 1])
        op
        qml.QubitUnitary(U, wires=range(2, 2 + 2 * n_wires, 2))
        return qml.state()

    circ = qml.QNode(circuit, dev)
    circ_def = qml.QNode(circuit, dev_def)
    assert np.allclose(circ(), circ_def(), tol)
