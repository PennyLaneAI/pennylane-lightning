# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Unit tests for the LightningCliffish device.

lightning.cliffish runs only through Catalyst, so most tests compile circuits with
``qp.qjit(capture=True)`` and compare them with default.qubit.
"""

from importlib.util import find_spec
from pathlib import Path
from typing import Callable

import numpy as np
import pennylane as qp
import pytest
from conftest import LightningDevice, device_name
from pennylane.devices.capabilities import DeviceCapabilities
from pennylane.exceptions import DeviceError

if device_name != "lightning.cliffish":
    pytest.skip("Skipping tests for the LightningCliffish class.", allow_module_level=True)

requires_catalyst = pytest.mark.skipif(find_spec("catalyst") is None, reason="needs Catalyst")


try:
    import catalyst.passes as catalyst_passes
except ImportError:
    catalyst_passes = None

# Catalyst releases with delegated feed-forward also have the gadget check.
requires_delegation = pytest.mark.skipif(
    not hasattr(catalyst_passes, "verify_pbc_gadgets"),
    reason="needs Catalyst with delegated feed-forward",
)

# Clifford gates for random circuits, with their number of wires.
CLIFFORD_GATES: list[tuple[Callable, int]] = [
    (qp.Hadamard, 1),
    (qp.S, 1),
    (lambda wires: qp.adjoint(qp.S(wires)), 1),
    (qp.PauliX, 1),
    (qp.PauliY, 1),
    (qp.PauliZ, 1),
    (qp.CNOT, 2),
    (qp.CY, 2),
    (qp.CZ, 2),
    (qp.SWAP, 2),
    (qp.ISWAP, 2),
]


def random_circuit(num_wires: int, depth: int, seed: int) -> list[tuple[int, list[int]]]:
    """A random Clifford circuit, as (gate index, wires) pairs."""
    rng = np.random.default_rng(seed)
    circuit = []
    while len(circuit) < depth:
        index = int(rng.integers(len(CLIFFORD_GATES)))
        num_gate_wires = CLIFFORD_GATES[index][1]
        if num_gate_wires > num_wires:
            continue
        wires = [int(w) for w in rng.choice(num_wires, size=num_gate_wires, replace=False)]
        circuit.append((index, wires))
    return circuit


def apply_circuit(circuit: list[tuple[int, list[int]]]) -> None:
    """Queue the gates of a random circuit."""
    for index, wires in circuit:
        gate = CLIFFORD_GATES[index][0]
        gate(wires=wires if len(wires) > 1 else wires[0])


# Non-Clifford gates for random circuits, with their number of wires and parameters.
NON_CLIFFORD_GATES: list[tuple[Callable, int, int]] = [
    (lambda wires: qp.T(wires), 1, 0),
    (lambda wires: qp.adjoint(qp.T(wires)), 1, 0),
    (lambda angle, wires: qp.RX(angle, wires), 1, 1),
    (lambda angle, wires: qp.RY(angle, wires), 1, 1),
    (lambda angle, wires: qp.RZ(angle, wires), 1, 1),
    (lambda angle, wires: qp.IsingZZ(angle, wires), 2, 1),
    (lambda angle, wires: qp.PauliRot(angle, "XY", wires), 2, 1),
]


def random_clifford_t_circuit(
    num_wires: int, depth: int, seed: int, non_clifford_fraction: float = 0.3
) -> list[tuple[Callable, list[int], list[float]]]:
    """A random circuit of Clifford gates, T gates and rotations at random angles, as
    (gate, wires, parameters) triples."""
    rng = np.random.default_rng(seed)
    circuit = []
    while len(circuit) < depth:
        if rng.random() < non_clifford_fraction:
            gate, num_gate_wires, num_params = NON_CLIFFORD_GATES[
                int(rng.integers(len(NON_CLIFFORD_GATES)))
            ]
        else:
            gate, num_gate_wires = CLIFFORD_GATES[int(rng.integers(len(CLIFFORD_GATES)))]
            num_params = 0
        if num_gate_wires > num_wires:
            continue
        wires = [int(w) for w in rng.choice(num_wires, size=num_gate_wires, replace=False)]
        params = [float(a) for a in rng.uniform(-np.pi, np.pi, size=num_params)]
        circuit.append((gate, wires, params))
    return circuit


def apply_clifford_t_circuit(circuit: list[tuple[Callable, list[int], list[float]]]) -> None:
    """Queue the gates of a random Clifford+T circuit."""
    for gate, wires, params in circuit:
        gate(*params, wires=wires if len(wires) > 1 else wires[0])


# The gates that gate-level input uses. Under capture, Catalyst passes templates
# such as qp.QFT and qp.Toffoli to this device undecomposed, so tests decompose
# them first.
GATE_LEVEL_SET = {
    "Hadamard",
    "S",
    "Adjoint(S)",
    "T",
    "Adjoint(T)",
    "CNOT",
    "CZ",
    "SWAP",
    "PauliX",
    "PauliY",
    "PauliZ",
    "RX",
    "RY",
    "RZ",
    "PhaseShift",
    "GlobalPhase",
}


def gate_level(ops_fn: Callable) -> list[qp.operation.Operator]:
    """The ops `ops_fn` queues, decomposed to GATE_LEVEL_SET, without global phases."""
    pending = list(qp.tape.make_qscript(ops_fn)().operations)
    ops = []
    while pending:
        op = pending.pop(0)
        if op.name not in GATE_LEVEL_SET:
            pending[:0] = op.decomposition()
        elif op.name != "GlobalPhase":
            ops.append(op)
    return ops


def apply_ops(ops: list[qp.operation.Operator]) -> None:
    """Queue copies of `ops`."""
    for op in ops:
        if isinstance(op, qp.ops.op_math.Adjoint):
            qp.adjoint(type(op.base)(*op.base.parameters, wires=op.base.wires))
        else:
            type(op)(*op.parameters, wires=op.wires)


def random_pauli_word(num_wires: int, seed: int) -> list[tuple[str, int]]:
    """A random Pauli word on all wires, as (Pauli, wire) pairs; never the identity."""
    rng = np.random.default_rng(seed)
    word = [(str(rng.choice(list("IXYZ"))), wire) for wire in range(num_wires)]
    if all(pauli == "I" for pauli, _ in word):
        word[0] = ("Z", 0)
    return word


def pauli_observable(word: list[tuple[str, int]]) -> qp.operation.Operator:
    """The observable of a Pauli word."""
    paulis = {"X": qp.X, "Y": qp.Y, "Z": qp.Z}
    factors = [paulis[pauli](wire) for pauli, wire in word if pauli != "I"]
    return factors[0] if len(factors) == 1 else qp.prod(*factors)


def test_device_name():
    """Test the device name."""
    assert LightningDevice(wires=2).name == "lightning.cliffish"


def test_execute_is_refused():
    """Test that the device refuses execution outside Catalyst."""
    with pytest.raises(DeviceError, match=r"runs only under qp.qjit\(capture=True\)"):
        LightningDevice(wires=2).execute([])


def test_c_interface():
    """Test that the C interface names the Catalyst device and an existing library."""
    device, library = LightningDevice.get_c_interface()
    assert device == "LightningCliffishSimulator"
    assert Path(library).is_file()


def test_capabilities():
    """Test that the TOML file declares Clifford gates, T, rotations, Pauli
    observables and sampling, and nothing differentiable."""
    capabilities = DeviceCapabilities.from_toml_file(LightningDevice.config_filepath)
    assert capabilities.qjit_compatible
    for gate in ("Hadamard", "S", "CNOT", "CZ", "SWAP", "T", "RZ", "PauliRot", "PauliMeasure"):
        assert gate in capabilities.operations
        assert not capabilities.operations[gate].differentiable
        assert not capabilities.operations[gate].controllable
    assert "Toffoli" not in capabilities.operations
    assert "Hermitian" not in capabilities.observables
    assert "StateMP" not in capabilities.measurement_processes
    assert capabilities.supported_mcm_methods == ["device", "one-shot"]


@requires_catalyst
def test_ghz_counts_on_1000_qubits():
    """Test the minimal working example: a 1000-qubit GHZ state, counted on its
    first and last wires."""
    num_wires = 1000

    @qp.qjit(capture=True)
    @qp.set_shots(1000)
    @qp.qnode(qp.device("lightning.cliffish", wires=num_wires))
    def ghz():
        qp.Hadamard(0)

        @qp.for_loop(0, num_wires - 1, 1)
        def chain(i):
            qp.CNOT([i, i + 1])

        chain()
        return qp.counts(wires=[0, num_wires - 1])

    outcomes, counts = ghz()
    assert list(outcomes) == [0, 1, 2, 3]
    assert counts[1] == counts[2] == 0
    assert counts[0] + counts[3] == 1000
    assert counts[0] > 0 and counts[3] > 0


@requires_catalyst
@pytest.mark.parametrize("num_wires", [2, 3, 5])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exact_results_match_default_qubit(num_wires: int, seed: int):
    """Test exact expectation values, variances and probabilities on random Clifford
    circuits against default.qubit."""
    circuit = random_circuit(num_wires, depth=30, seed=seed)
    word = random_pauli_word(num_wires, seed=100 + seed)
    hamiltonian_words = [random_pauli_word(num_wires, seed=200 + seed + k) for k in range(3)]

    def qfunc():
        apply_circuit(circuit)
        hamiltonian = qp.sum(
            *(c * pauli_observable(w) for c, w in zip([0.5, -1.3, 0.7], hamiltonian_words))
        )
        return (
            qp.expval(pauli_observable(word)),
            qp.var(pauli_observable(word)),
            qp.expval(hamiltonian),
            qp.var(hamiltonian),
            qp.probs(wires=range(num_wires)),
        )

    expected = qp.qnode(qp.device("default.qubit", wires=num_wires))(qfunc)()
    actual = qp.qjit(qp.qnode(LightningDevice(wires=num_wires))(qfunc), capture=True)()
    for actual_value, expected_value in zip(actual, expected):
        assert np.allclose(actual_value, expected_value, atol=1e-10)


@requires_catalyst
@pytest.mark.parametrize("seed", [0, 1])
def test_samples_follow_default_qubit_probabilities(seed: int):
    """Test that samples follow the probabilities default.qubit computes."""
    num_wires = 3
    shots = 20000
    circuit = random_circuit(num_wires, depth=30, seed=seed)

    def probs():
        apply_circuit(circuit)
        return qp.probs(wires=range(num_wires))

    def sample():
        apply_circuit(circuit)
        return qp.sample(wires=range(num_wires))

    expected = qp.qnode(qp.device("default.qubit", wires=num_wires))(probs)()
    samples = qp.qjit(
        qp.set_shots(qp.qnode(LightningDevice(wires=num_wires))(sample), shots), capture=True
    )()

    assert samples.shape == (shots, num_wires)
    indices = samples.astype(int) @ (2 ** np.arange(num_wires - 1, -1, -1))
    frequencies = np.bincount(indices, minlength=2**num_wires) / shots
    sigma = np.sqrt(expected * (1 - expected) / shots)
    assert np.all(frequencies[expected == 0] == 0)
    assert np.all(np.abs(frequencies - expected) <= 5 * sigma + 1e-12)


@requires_catalyst
def test_expval_with_shots_is_estimated_from_samples():
    """Test that with shots, expectation values are estimated from samples."""
    shots = 10000

    @qp.qjit(capture=True)
    @qp.set_shots(shots)
    @qp.qnode(LightningDevice(wires=2))
    def bell():
        qp.Hadamard(0)
        qp.CNOT([0, 1])
        return qp.expval(qp.Z(0) @ qp.Z(1)), qp.expval(qp.Z(0)), qp.var(qp.Z(0))

    zz, z0, var_z0 = bell()
    assert zz == 1.0
    assert abs(z0) < 5 / np.sqrt(shots)
    assert 1 - 25 / shots < var_z0 <= 1


@requires_catalyst
def test_parametric_gates_at_clifford_angles_match_default_qubit():
    """Test rotations at Clifford angles against default.qubit."""

    def qfunc():
        qp.RX(np.pi, wires=0)
        qp.RY(np.pi / 2, wires=1)
        qp.RZ(-np.pi / 2, wires=1)
        qp.PhaseShift(np.pi / 2, wires=1)
        qp.IsingXX(np.pi / 2, wires=[0, 2])
        qp.IsingYY(np.pi, wires=[1, 2])
        qp.IsingZZ(np.pi / 2, wires=[0, 1])
        qp.MultiRZ(np.pi / 2, wires=[0, 1, 2])
        qp.PauliRot(np.pi / 2, "XYZ", wires=[0, 1, 2])
        qp.adjoint(qp.PauliRot(np.pi / 2, "ZX", wires=[1, 2]))
        return [qp.expval(pauli_observable(random_pauli_word(3, seed))) for seed in range(10)]

    expected = qp.qnode(qp.device("default.qubit", wires=3))(qfunc)()
    actual = qp.qjit(qp.qnode(LightningDevice(wires=3))(qfunc), capture=True)()
    assert np.allclose(actual, expected, atol=1e-10)


@requires_catalyst
def test_one_shot_mid_circuit_measurements():
    """Test mid-circuit measurements and feed-forward with mcm_method="one-shot"."""
    shots = 4000

    @qp.qjit(capture=True)
    @qp.set_shots(shots)
    @qp.qnode(LightningDevice(wires=2), mcm_method="one-shot")
    def circuit():
        qp.Hadamard(0)
        m = qp.measure(0)
        qp.cond(m, qp.PauliX)(1)
        return qp.expval(qp.Z(0) @ qp.Z(1)), qp.expval(qp.Z(1))

    zz, z1 = circuit()
    assert zz == 1.0
    assert abs(z1) < 5 / np.sqrt(shots)


@requires_catalyst
def test_single_branch_statistics_by_default():
    """Test that without mcm_method every shot comes from one measured branch."""

    @qp.qnode(LightningDevice(wires=2))
    def circuit():
        qp.Hadamard(0)
        m = qp.measure(0)
        qp.cond(m, qp.PauliX)(1)
        return qp.expval(qp.Z(1))

    outcomes = set()
    for seed in range(8):
        value = qp.qjit(qp.set_shots(circuit, 100), capture=True, seed=seed)()
        assert value in (-1.0, 1.0)
        outcomes.add(float(value))
    assert outcomes == {-1.0, 1.0}


@requires_catalyst
def test_pauli_measure_and_pauli_rot():
    """Test Pauli product measurements and rotations under qjit."""

    @qp.qjit(capture=True)
    @qp.qnode(LightningDevice(wires=4))
    def circuit():
        qp.Hadamard(0)
        qp.CNOT([0, 1])
        # The Bell state has XX = +1 and YY = -1.
        xx = qp.pauli_measure("XX", wires=[0, 1])
        qp.cond(xx, qp.PauliX)(2)
        yy = qp.pauli_measure("YY", wires=[0, 1])
        qp.cond(yy, qp.PauliX)(3)
        return qp.expval(qp.Z(2)), qp.expval(qp.Z(3))

    z2, z3 = circuit()
    assert z2 == 1.0
    assert z3 == -1.0


@requires_catalyst
def test_seeded_runs_are_reproducible():
    """Test that a seeded qjit run gives the same samples every time."""

    def sample():
        for wire in range(4):
            qp.Hadamard(wire)
        return qp.sample(wires=range(4))

    circuit = qp.set_shots(qp.qnode(LightningDevice(wires=4))(sample), 200)
    first = qp.qjit(circuit, capture=True, seed=42)
    second = qp.qjit(circuit, capture=True, seed=42)
    other = qp.qjit(circuit, capture=True, seed=43)
    assert np.array_equal(first(), second())
    assert not np.array_equal(first(), other())


def test_device_kwargs():
    """Test that the device passes its options to the runtime and checks them."""
    device = LightningDevice(
        wires=2,
        max_dense_qubits=10,
        recompression="t1",
        dense_checks=False,
        gadget_collapse="source",
    )
    assert device.device_kwargs == {
        "max_dense_qubits": 10,
        "recompression": "t1",
        "dense_checks": False,
        "gadget_collapse": "source",
    }
    with pytest.raises(ValueError, match="gadget_collapse must be one of"):
        LightningDevice(wires=2, gadget_collapse="maybe")
    with pytest.raises(ValueError, match="recompression must be one of"):
        LightningDevice(wires=2, recompression="eager")
    with pytest.raises(ValueError, match="max_dense_qubits must be in"):
        LightningDevice(wires=2, max_dense_qubits=41)


@requires_catalyst
@pytest.mark.parametrize("num_wires", [2, 3, 5])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_clifford_t_circuits_match_default_qubit(num_wires: int, seed: int):
    """Test exact expectation values, variances and probabilities on random circuits of
    Clifford gates, T gates and rotations at random angles against default.qubit."""
    circuit = random_clifford_t_circuit(num_wires, depth=40, seed=seed)
    word = random_pauli_word(num_wires, seed=300 + seed)
    hamiltonian_words = [random_pauli_word(num_wires, seed=400 + seed + k) for k in range(3)]

    def qfunc():
        apply_clifford_t_circuit(circuit)
        hamiltonian = qp.sum(
            *(c * pauli_observable(w) for c, w in zip([0.5, -1.3, 0.7], hamiltonian_words))
        )
        return (
            qp.expval(pauli_observable(word)),
            qp.var(pauli_observable(word)),
            qp.expval(hamiltonian),
            qp.var(hamiltonian),
            qp.probs(wires=range(num_wires)),
        )

    expected = qp.qnode(qp.device("default.qubit", wires=num_wires))(qfunc)()
    actual = qp.qjit(qp.qnode(LightningDevice(wires=num_wires))(qfunc), capture=True)()
    for actual_value, expected_value in zip(actual, expected):
        assert np.allclose(actual_value, expected_value, atol=1e-10)


@requires_catalyst
@pytest.mark.parametrize("recompression", ["none", "t1", "t2", "auto"])
@pytest.mark.parametrize("dense_checks", [False, True])
def test_recompression_options_agree(recompression: str, dense_checks: bool):
    """Test that every re-compression policy, with and without dense checks, gives
    the same probabilities."""
    num_wires = 4
    circuit = random_clifford_t_circuit(num_wires, depth=40, seed=7)

    def qfunc():
        apply_clifford_t_circuit(circuit)
        return qp.probs(wires=range(num_wires))

    expected = qp.qnode(qp.device("default.qubit", wires=num_wires))(qfunc)()
    device = LightningDevice(
        wires=num_wires, recompression=recompression, dense_checks=dense_checks
    )
    actual = qp.qjit(qp.qnode(device)(qfunc), capture=True)()
    assert np.allclose(actual, expected, atol=1e-10)


def measure_qft_outputs(num_wires: int):
    """Outputs that compare QFT results."""
    return (
        qp.probs(wires=range(num_wires)),
        qp.expval(qp.X(0) @ qp.Y(2)),
        qp.expval(qp.X(num_wires - 1)),
    )


@requires_catalyst
@pytest.mark.parametrize("basis_state", [[0, 0, 0, 0, 0], [1, 0, 1, 1, 0]])
def test_gate_level_qft_matches_default_qubit(basis_state: list[int]):
    """Test the QFT at gate level, as PhaseShift and CNOT, on basis states."""
    num_wires = len(basis_state)

    def prepare_and_transform():
        for wire, bit in enumerate(basis_state):
            if bit:
                qp.PauliX(wire)
        qp.QFT(wires=range(num_wires))

    ops = gate_level(prepare_and_transform)

    def reference():
        prepare_and_transform()
        return measure_qft_outputs(num_wires)

    def gates():
        apply_ops(ops)
        return measure_qft_outputs(num_wires)

    expected = qp.qnode(qp.device("default.qubit", wires=num_wires))(reference)()
    actual = qp.qjit(qp.qnode(LightningDevice(wires=num_wires))(gates), capture=True)()
    for actual_value, expected_value in zip(actual, expected):
        assert np.allclose(actual_value, expected_value, atol=1e-10)


@requires_catalyst
def test_gate_level_toffoli_and_grover_match_default_qubit():
    """Test Toffoli, and a Grover search built from multi-controlled gates, at
    the Clifford+T gate level."""

    def search():
        for wire in range(3):
            qp.Hadamard(wire)
        for _ in range(2):
            qp.FlipSign([1, 0, 1], wires=range(3))
            qp.GroverOperator(wires=range(3))
        qp.Toffoli(wires=[0, 1, 3])

    ops = gate_level(search)
    assert any(op.name == "T" for op in ops)

    def reference():
        search()
        return qp.probs(wires=range(4))

    def gates():
        apply_ops(ops)
        return qp.probs(wires=range(4))

    expected = qp.qnode(qp.device("default.qubit", wires=4))(reference)()
    actual = qp.qjit(qp.qnode(LightningDevice(wires=4))(gates), capture=True)()
    assert np.allclose(actual, expected, atol=1e-10)


@requires_catalyst
def test_near_identity_rotations_are_kept():
    """Test that a rotation near the identity is applied, not snapped to it."""

    @qp.qjit(capture=True)
    @qp.qnode(LightningDevice(wires=1))
    def circuit(angle):
        qp.RX(angle, wires=0)
        return qp.expval(qp.Y(0))

    for angle in (1e-4, 1e-9):
        assert np.isclose(circuit(angle), -np.sin(angle), rtol=1e-9, atol=0)


@requires_catalyst
@pytest.mark.parametrize("seed", [0, 1])
def test_clifford_t_samples_follow_default_qubit_probabilities(seed: int):
    """Test that samples of a Clifford+T circuit follow default.qubit's probabilities."""
    num_wires = 3
    shots = 20000
    circuit = random_clifford_t_circuit(num_wires, depth=30, seed=10 + seed)

    def probs():
        apply_clifford_t_circuit(circuit)
        return qp.probs(wires=range(num_wires))

    def sample():
        apply_clifford_t_circuit(circuit)
        return qp.sample(wires=range(num_wires))

    expected = qp.qnode(qp.device("default.qubit", wires=num_wires))(probs)()
    samples = qp.qjit(
        qp.set_shots(qp.qnode(LightningDevice(wires=num_wires))(sample), shots), capture=True
    )()

    indices = samples.astype(int) @ (2 ** np.arange(num_wires - 1, -1, -1))
    frequencies = np.bincount(indices, minlength=2**num_wires) / shots
    sigma = np.sqrt(expected * (1 - expected) / shots)
    assert np.all(np.abs(frequencies - expected) <= 5 * sigma + 1e-12)


@requires_catalyst
def test_clifford_t_one_shot_mid_circuit_measurements():
    """Test feed-forward from a measurement of a magic state with
    mcm_method="one-shot"."""
    shots = 4000

    @qp.qjit(capture=True)
    @qp.set_shots(shots)
    @qp.qnode(LightningDevice(wires=2), mcm_method="one-shot")
    def circuit():
        qp.RY(1.0, wires=0)
        qp.T(0)
        m = qp.measure(0)
        qp.cond(m, qp.PauliX)(1)
        return qp.expval(qp.Z(1))

    # The outcome of wire 0 is 1 with probability sin^2(1/2).
    assert abs(circuit() - np.cos(1.0)) < 5 / np.sqrt(shots)


@requires_catalyst
def test_dense_register_overflow_fails_loudly():
    """Test that a program whose dense register would pass max_dense_qubits fails with
    an error that says how to run it instead."""

    @qp.qjit(capture=True)
    @qp.qnode(LightningDevice(wires=3, max_dense_qubits=2))
    def circuit():
        for wire in range(3):
            qp.Hadamard(wire)
            qp.T(wire)
        return qp.expval(qp.Z(0))

    with pytest.raises(RuntimeError, match="dense register of 3 qubits.*lightning.qubit"):
        circuit()


@requires_catalyst
def test_state_fails_loudly():
    """Test that the state vector is not available."""

    @qp.qjit(capture=True)
    @qp.qnode(LightningDevice(wires=2))
    def circuit():
        qp.Hadamard(0)
        return qp.state()

    with pytest.raises(RuntimeError, match="cannot return the state vector"):
        circuit()


@requires_catalyst
def test_hermitian_observable_is_rejected():
    """Test that Catalyst does not run a Hermitian observable on the device."""

    @qp.qjit(capture=True)
    @qp.qnode(LightningDevice(wires=1))
    def circuit():
        return qp.expval(qp.Hermitian(np.eye(2), wires=0))

    # Catalyst rejects it while tracing, before the device runs.
    with pytest.raises(Exception):
        circuit()


# The Clifford gates Catalyst's to-ppr converts: it has no SWAP or ISWAP.
PBC_CLIFFORD_GATES = [(gate, n) for gate, n in CLIFFORD_GATES if gate not in (qp.SWAP, qp.ISWAP)]


def pbc_program(
    circuit, measurement, *, decompose_method="pauli-corrected", shots=None, **device_options
):
    """The circuit compiled to Pauli product measurements, run with its feed-forward
    delegated to lightning.cliffish."""
    device = LightningDevice(wires=len(measurement.wires), **device_options)

    @qp.qjit(capture=True)
    @catalyst_passes.ppm_compilation(decompose_method=decompose_method, emit_gadgets=True)
    @qp.qnode(device, shots=shots, mcm_method="device")
    def program():
        apply_clifford_t_circuit(circuit)
        return qp.apply(measurement)

    return program


@requires_delegation
@pytest.mark.parametrize("num_wires", [2, 3, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pbc_programs_match_default_qubit(num_wires: int, seed: int):
    """Test that a Clifford+T circuit compiled to PPMs, with its gadgets collapsed by the
    device, gives the probabilities of the gate-level circuit on default.qubit."""
    rng = np.random.default_rng(seed)
    circuit = []
    for _ in range(24):
        if rng.random() < 0.25:
            gate = qp.T if rng.random() < 0.5 else lambda wires: qp.adjoint(qp.T(wires))
            circuit.append((gate, [int(rng.integers(num_wires))], []))
            continue
        gate, num_gate_wires = PBC_CLIFFORD_GATES[int(rng.integers(len(PBC_CLIFFORD_GATES)))]
        wires = [int(w) for w in rng.choice(num_wires, size=num_gate_wires, replace=False)]
        circuit.append((gate, wires, []))

    @qp.qnode(qp.device("default.qubit", wires=num_wires))
    def reference():
        apply_clifford_t_circuit(circuit)
        return qp.probs(wires=range(num_wires))

    program = pbc_program(circuit, qp.probs(wires=range(num_wires)))
    assert np.allclose(program(), reference(), atol=1e-12)


@requires_delegation
@pytest.mark.parametrize("method", ["auto-corrected", "clifford-corrected"])
def test_pbc_gadget_bugs_are_refused_or_reported(method: str):
    """Test that the device refuses Catalyst's known wrong pi/8 gadgets by default, and with
    gadget_collapse="source" applies the source rotations instead."""
    circuit = [(qp.Hadamard, [0], []), (lambda wires: qp.adjoint(qp.T(wires)), [0], [])]
    circuit += [(qp.CNOT, [0, 1], []), (qp.T, [1], []), (qp.Hadamard, [1], [])]

    @qp.qnode(qp.device("default.qubit", wires=2))
    def reference():
        apply_clifford_t_circuit(circuit)
        return qp.probs(wires=[0, 1])

    measured = pbc_program(circuit, qp.probs(wires=[0, 1]), decompose_method=method)
    with pytest.raises(RuntimeError, match="branch-and-merge|different rotation"):
        measured()

    source = pbc_program(
        circuit, qp.probs(wires=[0, 1]), decompose_method=method, gadget_collapse="source"
    )
    assert np.allclose(source(), reference(), atol=1e-12)


@requires_delegation
def test_pbc_without_collapse_fails_loudly():
    """Test that gadgets that do not collapse fail with a message saying how to run the
    program instead, since the device has no branch-and-merge yet."""
    circuit = [(qp.Hadamard, [0], []), (qp.T, [0], [])]
    program = pbc_program(circuit, qp.probs(wires=[0]), gadget_collapse="off")
    with pytest.raises(RuntimeError, match="mcm_method='one-shot'"):
        program()


@requires_delegation
def test_pbc_samples_at_many_shots():
    """Test the statistics of a wide PPM-compiled program at 10^5 shots: a GHZ state with a T
    on every qubit, undone, so that P(q0 = 1) = sin^2(n pi / 8)."""
    num_wires = 51
    circuit = [(qp.Hadamard, [0], [])]
    circuit += [(qp.CNOT, [i, i + 1], []) for i in range(num_wires - 1)]
    circuit += [(qp.T, [i], []) for i in range(num_wires)]
    circuit += [(qp.CNOT, [i, i + 1], []) for i in reversed(range(num_wires - 1))]
    circuit += [(qp.Hadamard, [0], [])]
    shots = 100_000
    program = pbc_program(circuit, qp.sample(wires=range(num_wires)), shots=shots)
    samples = program()
    exact = np.sin(num_wires * np.pi / 8) ** 2
    assert samples.shape == (shots, num_wires)
    assert abs(samples[:, 0].mean() - exact) < 5 * np.sqrt(exact * (1 - exact) / shots)
