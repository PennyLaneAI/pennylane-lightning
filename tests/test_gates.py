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
Unit tests for the correct application of gates with a Lightning device.
"""
import copy
import itertools

import numpy as np
import pennylane as qml
import pytest
from conftest import PHI, THETA
from conftest import LightningDevice as ld
from conftest import device_name

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.fixture
def op(op_name):
    ops_list = {
        "RX": [qml.RX, [], {"phi": 0.123, "wires": [0]}],
        "RY": [qml.RY, [], {"phi": 1.434, "wires": [0]}],
        "RZ": [qml.RZ, [], {"phi": 2.774, "wires": [0]}],
        "S": [qml.S, [], {"wires": [0]}],
        "SX": [qml.SX, [], {"wires": [0]}],
        "T": [qml.T, [], {"wires": [0]}],
        "CNOT": [qml.CNOT, [], {"wires": [0, 1]}],
        "CZ": [qml.CZ, [], {"wires": [0, 1]}],
        "CY": [qml.CY, [], {"wires": [0, 1]}],
        "SWAP": [qml.SWAP, [], {"wires": [0, 1]}],
        "ISWAP": [qml.ISWAP, [], {"wires": [0, 1]}],
        "SISWAP": [qml.SISWAP, [], {"wires": [0, 1]}],
        "SQISW": [qml.SQISW, [], {"wires": [0, 1]}],
        "CSWAP": [qml.CSWAP, [], {"wires": [0, 1, 2]}],
        "PauliRot": [qml.PauliRot, [0.123], {"pauli_word": "Y", "wires": [0]}],
        "IsingXX": [qml.IsingXX, [], {"phi": 0.123, "wires": [0, 1]}],
        "IsingXY": [qml.IsingXY, [], {"phi": 0.123, "wires": [0, 1]}],
        "IsingYY": [qml.IsingYY, [], {"phi": 0.123, "wires": [0, 1]}],
        "IsingZZ": [qml.IsingZZ, [], {"phi": 0.123, "wires": [0, 1]}],
        "Identity": [qml.Identity, [], {"wires": [0]}],
        "Rot": [qml.Rot, [], {"phi": 0.123, "theta": 0.456, "omega": 0.789, "wires": [0]}],
        "Toffoli": [qml.Toffoli, [], {"wires": [0, 1, 2]}],
        "PhaseShift": [qml.PhaseShift, [], {"phi": 2.133, "wires": [0]}],
        "ControlledPhaseShift": [qml.ControlledPhaseShift, [], {"phi": 1.777, "wires": [0, 1]}],
        "CPhase": [qml.CPhase, [], {"phi": 1.777, "wires": [0, 1]}],
        "MultiRZ": [qml.MultiRZ, [], {"theta": 0.112, "wires": [0, 1, 2]}],
        "GlobalPhase": [qml.GlobalPhase, [], {"phi": 0.112, "wires": [0, 1, 2]}],
        "CRX": [qml.CRX, [], {"phi": 0.123, "wires": [0, 1]}],
        "CRY": [qml.CRY, [], {"phi": 0.123, "wires": [0, 1]}],
        "CRZ": [qml.CRZ, [], {"phi": 0.123, "wires": [0, 1]}],
        "Hadamard": [qml.Hadamard, [], {"wires": [0]}],
        "PauliX": [qml.PauliX, [], {"wires": [0]}],
        "PauliY": [qml.PauliY, [], {"wires": [0]}],
        "PauliZ": [qml.PauliZ, [], {"wires": [0]}],
        "CRot": [qml.CRot, [], {"phi": 0.123, "theta": 0.456, "omega": 0.789, "wires": [0, 1]}],
        "DiagonalQubitUnitary": [qml.DiagonalQubitUnitary, [np.array([1.0, 1.0j])], {"wires": [0]}],
        "MultiControlledX": [
            qml.MultiControlledX,
            [],
            {"wires": [0, 1, 2], "control_values": "01"},
        ],
        "SingleExcitation": [qml.SingleExcitation, [0.123], {"wires": [0, 1]}],
        "SingleExcitationPlus": [qml.SingleExcitationPlus, [0.123], {"wires": [0, 1]}],
        "SingleExcitationMinus": [qml.SingleExcitationMinus, [0.123], {"wires": [0, 1]}],
        "DoubleExcitation": [qml.DoubleExcitation, [0.123], {"wires": [0, 1, 2, 3]}],
        "DoubleExcitationPlus": [qml.DoubleExcitationPlus, [0.123], {"wires": [0, 1, 2, 3]}],
        "DoubleExcitationMinus": [qml.DoubleExcitationMinus, [0.123], {"wires": [0, 1, 2, 3]}],
        "QFT": [qml.QFT, [], {"wires": [0]}],
        "QubitSum": [qml.QubitSum, [], {"wires": [0, 1, 2]}],
        "QubitCarry": [qml.QubitCarry, [], {"wires": [0, 1, 2, 3]}],
        "QubitUnitary": [qml.QubitUnitary, [], {"U": np.eye(16) * 1j, "wires": [0, 1, 2, 3]}],
        "BlockEncode": [qml.BlockEncode, [[[0.2, 0, 0.2], [-0.2, 0.2, 0]]], {"wires": [0, 1, 2]}],
    }
    return ops_list.get(op_name)


@pytest.mark.parametrize("op_name", ld.operations)
def test_gate_unitary_correct(op, op_name):
    """Test if lightning device correctly applies gates by reconstructing the unitary matrix and
    comparing to the expected version"""

    if op_name in ("BasisState", "QubitStateVector", "StatePrep"):
        pytest.skip("Skipping operation because it is a state preparation")
    if op == None:
        pytest.skip("Skipping operation.")

    wires = len(op[2]["wires"])

    if wires == 1 and device_name == "lightning.tensor":
        pytest.skip("Skipping single wire device on lightning.tensor.")

    if op_name == "QubitUnitary" and device_name == "lightning.tensor":
        pytest.skip(
            "Skipping QubitUnitary on lightning.tensor. as `lightning.tensor` cannot be cleaned up like other state-vector devices because the data is attached to the graph. It is recommended to use one device per circuit for `lightning.tensor`."
        )

    dev = qml.device(device_name, wires=wires)

    @qml.qnode(dev)
    def output(input):
        qml.BasisState(input, wires=range(wires))
        op[0](*op[1], **op[2])
        return qml.state()

    unitary = np.zeros((2**wires, 2**wires), dtype=np.complex128)

    for i, input in enumerate(itertools.product([0, 1], repeat=wires)):
        out = output(np.array(input))
        unitary[:, i] = out

    unitary_expected = qml.matrix(op[0](*op[1], **op[2]))

    assert np.allclose(unitary, unitary_expected)

    op1 = copy.deepcopy(op[1])
    if len(op1) > 0:
        op1 = (-np.array(op1)).tolist()
    op2 = copy.deepcopy(op[2])
    if "phi" in op2.keys():
        op2["phi"] *= np.sqrt(2)
    if "theta" in op2.keys():
        op2["theta"] *= np.sqrt(3)
    if "U" in op2.keys():
        op2["U"] *= np.sqrt(3)

    @qml.qnode(dev)
    def output(input):
        qml.BasisState(input, wires=range(wires))
        op[0](*op[1], **op[2])
        op[0](*op1, **op2)
        return qml.state()

    unitary = np.zeros((2**wires, 2**wires), dtype=np.complex128)

    for i, input in enumerate(itertools.product([0, 1], repeat=wires)):
        out = output(np.array(input))
        unitary[:, i] = out

    unitary_expected = qml.matrix(op[0](*op1, **op2)) @ qml.matrix(op[0](*op[1], **op[2]))
    assert np.allclose(unitary, unitary_expected)


@pytest.mark.parametrize("op_name", ld.operations)
def test_gate_unitary_correct_lt(op, op_name):
    """Test if lightning device correctly applies gates by reconstructing the unitary matrix and
    comparing to the expected version"""

    if op_name in ("BasisState", "QubitStateVector", "StatePrep"):
        pytest.skip("Skipping operation because it is a state preparation")
    if op == None:
        pytest.skip("Skipping operation.")

    wires = len(op[2]["wires"])

    if wires == 1 and device_name == "lightning.tensor":
        pytest.skip("Skipping single wire device on lightning.tensor.")

    unitary = np.zeros((2**wires, 2**wires), dtype=np.complex128)

    for i, input in enumerate(itertools.product([0, 1], repeat=wires)):
        dev = qml.device(device_name, wires=wires)

        @qml.qnode(dev)
        def output(input):
            qml.BasisState(input, wires=range(wires))
            op[0](*op[1], **op[2])
            return qml.state()

        out = output(np.array(input))
        unitary[:, i] = out

    unitary_expected = qml.matrix(op[0](*op[1], **op[2]))

    assert np.allclose(unitary, unitary_expected)


@pytest.mark.parametrize("op_name", ld.operations)
def test_inverse_unitary_correct(op, op_name):
    """Test if lightning device correctly applies inverse gates by reconstructing the unitary matrix
    and comparing to the expected version"""

    if op_name in ("BasisState", "QubitStateVector", "StatePrep"):
        pytest.skip("Skipping operation because it is a state preparation")
    if op == None:
        pytest.skip("Skipping operation.")

    wires = len(op[2]["wires"])

    if wires == 1 and device_name == "lightning.tensor":
        pytest.skip("Skipping single wire device on lightning.tensor.")

    dev = qml.device(device_name, wires=wires)

    @qml.qnode(dev)
    def output(input):
        qml.BasisState(input, wires=range(wires))
        qml.adjoint(op[0](*op[1], **op[2]))
        return qml.state()

    unitary = np.zeros((2**wires, 2**wires), dtype=np.complex128)

    for i, input in enumerate(itertools.product([0, 1], repeat=wires)):
        out = output(np.array(input))
        unitary[:, i] = out

    unitary_expected = qml.matrix(qml.adjoint(op[0](*op[1], **op[2])))

    assert np.allclose(unitary, unitary_expected)


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


def test_arbitrary_unitary_correct():
    """Test if lightning device correctly applies an arbitrary unitary by reconstructing its
    matrix"""
    wires = 2
    dev = qml.device(device_name, wires=wires)

    @qml.qnode(dev)
    def output(input):
        qml.BasisState(input, wires=range(wires))
        qml.QubitUnitary(random_unitary, wires=range(2))
        return qml.state()

    unitary = np.zeros((2**wires, 2**wires), dtype=np.complex128)

    for i, input in enumerate(itertools.product([0, 1], repeat=wires)):
        out = output(np.array(input))
        unitary[:, i] = out

    assert np.allclose(unitary, random_unitary)


def test_arbitrary_inv_unitary_correct():
    """Test if lightning device correctly applies the inverse of an arbitrary unitary by
    reconstructing its matrix"""
    wires = 2
    dev = qml.device(device_name, wires=wires)

    @qml.qnode(dev)
    def output(input):
        qml.BasisState(input, wires=range(wires))
        qml.adjoint(qml.QubitUnitary(random_unitary, wires=range(2)))
        return qml.state()

    unitary = np.zeros((2**wires, 2**wires), dtype=np.complex128)

    for i, input in enumerate(itertools.product([0, 1], repeat=wires)):
        out = output(np.array(input))
        unitary[:, i] = out

    random_unitary_inv = random_unitary.conj().T
    assert np.allclose(unitary, random_unitary_inv)


@pytest.mark.skipif(ld._new_API, reason="Old API required")
@pytest.mark.parametrize(
    "obs,has_rotation",
    [
        (qml.Hamiltonian([1], [qml.PauliY(0)]), False),
        (qml.sum(qml.PauliZ(0), qml.PauliX(1)), False),
        (qml.PauliX(0), True),
        (qml.sum(qml.PauliZ(0), qml.Hermitian(qml.PauliX(1).matrix(), 1)), True),
    ],
)
def test_get_diagonalizing_gates(obs, has_rotation):
    """Tests that _get_diagonalizing_gates filters measurements as expected."""
    dev = qml.device(device_name, wires=2)
    qs = qml.tape.QuantumScript(measurements=[qml.expval(obs)])
    actual = dev._get_diagonalizing_gates(qs)
    if has_rotation:
        expected = obs.diagonalizing_gates()
        assert len(actual) == len(expected)
        for rot_actual, rot_expected in zip(actual, expected):
            assert qml.equal(rot_actual, rot_expected)
    else:
        assert len(actual) == 0


@pytest.mark.parametrize("theta,phi", list(zip(THETA, PHI)))
def test_qubit_RY(theta, phi, tol):
    """Test that Hadamard expectation value is correct"""
    n_qubits = 4
    dev_def = qml.device("default.qubit", wires=n_qubits)
    dev = qml.device(device_name, wires=n_qubits)
    init_state = np.random.rand(2**n_qubits) + 1j * np.random.rand(2**n_qubits)
    init_state /= np.linalg.norm(init_state)

    def circuit():
        qml.StatePrep(init_state, wires=range(n_qubits))
        qml.RY(theta, wires=[0])
        qml.RY(phi, wires=[1])
        qml.RY(theta, wires=[2])
        qml.RY(phi, wires=[3])
        return qml.state()

    circ = qml.QNode(circuit, dev)
    circ_def = qml.QNode(circuit, dev_def)
    assert np.allclose(circ(), circ_def(), tol)


@pytest.mark.parametrize("theta,phi", list(zip(THETA, PHI)))
@pytest.mark.parametrize("n_wires", range(1, 7))
def test_qubit_unitary(n_wires, theta, phi, tol):
    """Test that Hadamard expectation value is correct"""
    n_qubits = 10
    dev_def = qml.device("default.qubit", wires=n_qubits)
    dev = qml.device(device_name, wires=n_qubits)
    m = 2**n_wires
    U = np.random.rand(m, m) + 1j * np.random.rand(m, m)
    U, _ = np.linalg.qr(U)
    init_state = np.random.rand(2**n_qubits) + 1j * np.random.rand(2**n_qubits)
    init_state /= np.linalg.norm(init_state)
    wires = list(range((n_qubits - n_wires), (n_qubits - n_wires) + n_wires))
    perms = list(itertools.permutations(wires))
    if n_wires > 4:
        perms = perms[0::30]
    for perm in perms:

        def circuit():
            qml.StatePrep(init_state, wires=range(n_qubits))
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.RY(theta, wires=[2])
            qml.RY(phi, wires=[3])
            qml.RY(theta, wires=[4])
            qml.RY(phi, wires=[5])
            qml.RY(theta, wires=[6])
            qml.RY(phi, wires=[7])
            qml.RY(phi, wires=[8])
            qml.RY(phi, wires=[9])
            qml.CNOT(wires=[0, 1])
            qml.QubitUnitary(U, wires=perm)
            return qml.state()

        circ = qml.QNode(circuit, dev)
        circ_def = qml.QNode(circuit, dev_def)
        assert np.allclose(circ(), circ_def(), tol)


@pytest.mark.skipif(
    device_name not in ("lightning.qubit", "lightning.kokkos"),
    reason="PennyLane-like StatePrep only implemented in lightning.qubit and lightning.kokkos.",
)
@pytest.mark.parametrize("n_targets", list(range(2, 8)))
def test_state_prep(n_targets, tol):
    """Test that StatePrep is correctly applied to a state."""
    n_wires = 7
    dq = qml.device("default.qubit", wires=n_wires)
    dev = qml.device(device_name, wires=n_wires)
    init_state = np.random.rand(2**n_targets) + 1.0j * np.random.rand(2**n_targets)
    init_state /= np.linalg.norm(init_state)
    for i in range(10):
        if i == 0:
            wires = np.arange(n_targets, dtype=int)
        else:
            wires = np.random.permutation(n_wires)[0:n_targets]
        tape = qml.tape.QuantumTape(
            [qml.StatePrep(init_state, wires=wires)] + [qml.X(i) for i in range(n_wires)],
            [qml.state()],
        )
        ref = dq.execute([tape])[0]
        res = dev.execute([tape])[0]
        assert np.allclose(res.ravel(), ref.ravel(), tol)


@pytest.mark.skipif(
    device_name != "lightning.qubit",
    reason="N-controlled operations only implemented in lightning.qubit.",
)
@pytest.mark.parametrize("control_value", [False, True])
@pytest.mark.parametrize("n_qubits", list(range(2, 8)))
def test_controlled_qubit_unitary(n_qubits, control_value, tol):
    """Test that ControlledQubitUnitary is correctly applied to a state"""
    dev_def = qml.device("default.qubit", wires=n_qubits)
    dev = qml.device(device_name, wires=n_qubits)
    threshold = 500
    for n_wires in range(1, 5):
        wire_lists = list(itertools.permutations(range(0, n_qubits), n_wires))
        n_perms = len(wire_lists) * (n_wires) ** 2
        if n_perms > threshold:
            wire_lists = wire_lists[0 :: (n_perms // threshold)]
        for all_wires in wire_lists:
            for i in range(1, len(all_wires)):
                control_wires = all_wires[0:i]
                target_wires = all_wires[i:]
                m = 2 ** len(target_wires)
                U = np.random.rand(m, m) + 1.0j * np.random.rand(m, m)
                U, _ = np.linalg.qr(U)
                init_state = np.random.rand(2**n_qubits) + 1.0j * np.random.rand(2**n_qubits)
                init_state /= np.linalg.norm(init_state)

                def circuit():
                    qml.StatePrep(init_state, wires=range(n_qubits))
                    qml.ControlledQubitUnitary(
                        U,
                        control_wires=control_wires,
                        wires=target_wires,
                        control_values=[
                            control_value or bool(i % 2) for i, _ in enumerate(control_wires)
                        ],
                    )
                    return qml.state()

                circ = qml.QNode(circuit, dev)
                circ_def = qml.QNode(circuit, dev_def)
                assert np.allclose(circ(), circ_def(), tol)


@pytest.mark.skipif(
    device_name not in ("lightning.qubit", "lightning.tensor"),
    reason="N-controlled operations only implemented in lightning.qubit and lightning.tensor.",
)
@pytest.mark.parametrize(
    "operation",
    [
        qml.PauliX,
        qml.PauliY,
        qml.PauliZ,
        qml.Hadamard,
        qml.S,
        qml.T,
        qml.PhaseShift,
        qml.RX,
        qml.RY,
        qml.RZ,
        qml.Rot,
        qml.SWAP,
        qml.IsingXX,
        qml.IsingXY,
        qml.IsingYY,
        qml.IsingZZ,
        qml.SingleExcitation,
        qml.SingleExcitationMinus,
        qml.SingleExcitationPlus,
        qml.DoubleExcitation,
        qml.DoubleExcitationMinus,
        qml.DoubleExcitationPlus,
        qml.MultiRZ,
        qml.GlobalPhase,
    ],
)
@pytest.mark.parametrize("control_value", [False, True])
@pytest.mark.parametrize("n_qubits", list(range(2, 8)))
def test_controlled_qubit_gates(operation, n_qubits, control_value, tol):
    """Test that multi-controlled gates are correctly applied to a state"""
    dev_def = qml.device("default.qubit", wires=n_qubits)
    dev = qml.device(device_name, wires=n_qubits)
    threshold = 5 if device_name == "lightning.tensor" else 250
    num_wires = max(operation.num_wires, 1)

    for n_wires in range(num_wires + 1, num_wires + 4):
        wire_lists = list(itertools.permutations(range(0, n_qubits), n_wires))
        n_perms = len(wire_lists) * n_wires
        if n_perms > threshold:
            wire_lists = wire_lists[0 :: (n_perms // threshold)]
        for all_wires in wire_lists:
            target_wires = all_wires[0:num_wires]
            control_wires = all_wires[num_wires:]
            init_state = np.random.rand(2**n_qubits) + 1.0j * np.random.rand(2**n_qubits)
            init_state /= np.linalg.norm(init_state)

            def circuit():
                qml.StatePrep(init_state, wires=range(n_qubits))
                if operation.num_params == 0:
                    qml.ctrl(
                        operation(target_wires),
                        control_wires,
                        control_values=(
                            [control_value or bool(i % 2) for i, _ in enumerate(control_wires)]
                            if device_name != "lightning.tensor"
                            else [control_value for _ in control_wires]
                        ),
                    )
                else:
                    qml.ctrl(
                        operation(*tuple([0.1234] * operation.num_params), target_wires),
                        control_wires,
                        control_values=(
                            [control_value or bool(i % 2) for i, _ in enumerate(control_wires)]
                            if device_name != "lightning.tensor"
                            else [control_value for _ in control_wires]
                        ),
                    )
                return qml.state()

            circ = qml.QNode(circuit, dev)
            circ_def = qml.QNode(circuit, dev_def)
            assert np.allclose(circ(), circ_def(), tol)


@pytest.mark.skipif(
    device_name != "lightning.qubit",
    reason="N-controlled operations only implemented in lightning.qubit.",
)
def test_controlled_qubit_unitary_from_op(tol):
    n_qubits = 10
    dev_def = qml.device("default.qubit", wires=n_qubits)
    dev = qml.device(device_name, wires=n_qubits)

    def circuit(x):
        qml.ControlledQubitUnitary(
            qml.QubitUnitary(qml.RX.compute_matrix(x), wires=5), control_wires=range(5)
        )
        return qml.expval(qml.PauliX(0))

    circ = qml.QNode(circuit, dev)
    circ_def = qml.QNode(circuit, dev_def)
    par = 0.1234
    assert np.allclose(circ(par), circ_def(par), tol)


@pytest.mark.skipif(
    device_name not in ("lightning.qubit", "lightning.kokkos"),
    reason="PauliRot operations only implemented in lightning.qubit and lightning.kokkos.",
)
@pytest.mark.parametrize("n_wires", [1, 2, 3, 4, 5, 10, 15])
@pytest.mark.parametrize("n_targets", [1, 2, 3, 4, 5, 10, 15])
def test_paulirot(n_wires, n_targets, tol):
    """Test that PauliRot is correctly applied to a state."""
    pws = dict((k, v) for k, v in enumerate(("X", "Y", "Z")))

    if n_wires < n_targets:
        pytest.skip("The number of targets cannot exceed the number of wires.")
    dev = qml.device(device_name, wires=n_wires)

    init_state = np.random.rand(2**n_wires) + 1.0j * np.random.rand(2**n_wires)
    init_state /= np.linalg.norm(init_state)
    theta = 0.3

    for i in range(10):
        word = (
            "Z" * n_targets
            if i == 0
            else "".join(pws[w] for w in np.random.randint(0, 3, n_targets))
        )
        wires = np.random.permutation(n_wires)[0:n_targets]
        stateprep = qml.StatePrep(init_state, wires=range(n_wires))
        op = qml.PauliRot(theta, word, wires=wires)

        tape0 = qml.tape.QuantumScript(
            [stateprep, op],
            [qml.state()],
        )

        tape1 = qml.tape.QuantumScript(
            [stateprep] + op.decomposition(),
            [qml.state()],
        )
        assert np.allclose(dev.execute(tape1), dev.execute(tape0), tol)


@pytest.mark.skipif(
    device_name not in ("lightning.qubit", "lightning.tensor"),
    reason="N-controlled operations only implemented in lightning.qubit.",
)
@pytest.mark.parametrize("control_wires", range(4))
@pytest.mark.parametrize("target_wires", range(4))
def test_cnot_controlled_qubit_unitary(control_wires, target_wires, tol):
    """Test that ControlledQubitUnitary is correctly applied to a state"""
    if control_wires == target_wires:
        return
    n_qubits = 4
    control_wires = [control_wires]
    target_wires = [target_wires]
    dev = qml.device(device_name, wires=n_qubits)
    wires = control_wires + target_wires
    U = qml.matrix(qml.PauliX(target_wires))
    init_state = np.random.rand(2**n_qubits) + 1.0j * np.random.rand(2**n_qubits)
    init_state /= np.linalg.norm(init_state)

    def circuit():
        qml.StatePrep(init_state, wires=range(n_qubits))
        qml.ControlledQubitUnitary(U, control_wires=control_wires, wires=target_wires)
        return qml.state()

    def cnot_circuit():
        qml.StatePrep(init_state, wires=range(n_qubits))
        qml.CNOT(wires=wires)
        return qml.state()

    circ = qml.QNode(circuit, dev)
    circ_def = qml.QNode(cnot_circuit, dev)
    assert np.allclose(circ(), circ_def(), atol=1e-4)


@pytest.mark.parametrize("control_value", [False, True])
@pytest.mark.parametrize("n_qubits", list(range(2, 8)))
def test_controlled_globalphase(n_qubits, control_value, tol):
    """Test that multi-controlled gates are correctly applied to a state"""
    dev_def = qml.device("default.qubit", wires=n_qubits)
    dev = qml.device(device_name, wires=n_qubits)
    threshold = 250
    operation = qml.GlobalPhase
    num_wires = max(operation.num_wires, 1)
    for n_wires in range(num_wires + 1, num_wires + 4):
        wire_lists = list(itertools.permutations(range(0, n_qubits), n_wires))
        n_perms = len(wire_lists) * n_wires
        if n_perms > threshold:
            wire_lists = wire_lists[0 :: (n_perms // threshold)]
        for all_wires in wire_lists:
            target_wires = all_wires[0:num_wires]
            control_wires = all_wires[num_wires:]
            init_state = np.random.rand(2**n_qubits) + 1.0j * np.random.rand(2**n_qubits)
            init_state /= np.linalg.norm(init_state)

            def circuit():
                qml.StatePrep(init_state, wires=range(n_qubits))
                qml.ctrl(
                    operation(0.1234, target_wires),
                    control_wires,
                    control_values=(
                        [control_value or bool(i % 2) for i, _ in enumerate(control_wires)]
                        if device_name != "lightning.tensor"
                        else [control_value for _ in control_wires]
                    ),
                )
                return qml.state()

            circ = qml.QNode(circuit, dev)
            circ_def = qml.QNode(circuit, dev_def)
            assert np.allclose(circ(), circ_def(), tol)
