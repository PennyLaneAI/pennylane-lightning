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
    """Test that the TOML file declares Clifford gates, Pauli observables and
    sampling, and nothing differentiable."""
    capabilities = DeviceCapabilities.from_toml_file(LightningDevice.config_filepath)
    assert capabilities.qjit_compatible
    for gate in ("Hadamard", "S", "CNOT", "CZ", "SWAP", "PauliRot", "PauliMeasure"):
        assert gate in capabilities.operations
        assert not capabilities.operations[gate].differentiable
        assert not capabilities.operations[gate].controllable
    assert "T" not in capabilities.operations
    assert "Hermitian" not in capabilities.observables
    assert "StateMP" not in capabilities.measurement_processes
    assert capabilities.supported_mcm_methods == ["one-shot"]


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


@requires_catalyst
@pytest.mark.parametrize(
    "gate, message",
    [
        (lambda: qp.T(0), "cannot apply 'T'"),
        (lambda: qp.RX(0.3, 0), "cannot apply 'RX'"),
        (lambda: qp.RZ(1e-15, 0), "cannot apply 'RZ'"),
    ],
)
def test_non_clifford_operations_fail_loudly(gate: Callable, message: str):
    """Test that non-Clifford operations fail with an error that names them."""

    @qp.qjit(capture=True)
    @qp.qnode(LightningDevice(wires=1))
    def circuit():
        gate()
        return qp.expval(qp.Z(0))

    with pytest.raises(RuntimeError, match=message):
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
