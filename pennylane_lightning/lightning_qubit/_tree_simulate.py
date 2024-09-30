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
"""Simulate a quantum script with channels."""

# pylint: disable=protected-access
from functools import singledispatch, lru_cache
from typing import Optional

import numpy as np

import pennylane as qml
from pennylane.devices.qubit import create_initial_state
from pennylane.measurements import (
    ExpectationMP,
    ProbabilityMP,
)
from pennylane.operation import Channel
from pennylane.typing import Result

from ._measurements import LightningMeasurements
from ._state_vector import LightningStateVector


class TreeTraversalStack:
    """This class is used to record various data used during the
    depth-first tree-traversal procedure for simulating circuits with channels."""

    probs: list
    results: list
    n_kraus: list
    states: list

    def __init__(self, max_depth, n_branches):
        self.n_branches = n_branches
        self.probs = [None] * max_depth
        self.results = [[None]] + [[None] * self.n_branches[d] for d in range(1, max_depth)]
        self.states = [None] * max_depth

    def any_is_empty(self, depth):
        """Return True if any result at ``depth`` is ``None`` and False otherwise."""
        return any(r is None for r in self.results[depth])

    def is_full(self, depth):
        """Return True if the results at ``depth`` are both not ``None`` and False otherwise."""
        return all(r is not None for r in self.results[depth])

    def prune(self, depth):
        """Reset all stack entries at ``depth`` to ``None``."""
        self.probs[depth] = None
        self.results[depth] = [None] * self.n_branches[depth]
        self.states[depth] = None


def tree_simulate(
    circuit: qml.tape.QuantumScript,
    state: LightningStateVector,
    **execution_kwargs,
) -> Result:
    """Simulate a single quantum script using the tree-traversal algorithm.

    The tree-traversal algorithm recursively explores all combinations of Kraus matrices
    outcomes using a depth-first approach. The depth-first approach requires ``n_nodes`` copies
    of the state vector (``n_nodes + 1`` state vectors in total) and records ``n_nodes`` vectors
    of measurements after applying the Kraus matrix for a given branch.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. If None, a random key will be
            generated. Only for simulation using JAX.
        debugger (_Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with

    Returns:
        tuple(TensorLike): The results of the simulation
    """
    #######################
    # main implementation #
    #######################

    ##################
    # Parse node info #
    ##################

    # nodes is the list of all channel operations. nodes[d] is the parent
    # node of a circuit segment (edge) at depth `d`. The first element
    # is None because there is no parent node at depth 0
    nodes: list[Channel] = [None] + [op for op in circuit.operations if isinstance(op, Channel)]
    n_nodes: int = len(nodes) - 1
    n_kraus: list[int] = [None] + [c.num_kraus for c in nodes[1:]]

    #############################
    # Initialize tree-traversal #
    #############################
    # branch_current[:d+1] is the active branch at depth `d`
    # The first entry is always 0 as the first edge does not stem from a channel.
    # For example, if `d = 2` and `branch_current = [0, 1, 1, 0]` we are on the 11-branch,
    # i.e. we're exploring the first two Channels at index 1 of their respective Kraus matrices.
    # The last entry isn't meaningful until we are at depth `d=3`.
    branch_current = qml.math.zeros(n_nodes + 1, dtype=int)
    # Split circuit into segments
    circuits = split_circuit_at_nodes(circuit)
    circuits[0] = prepend_state_prep(circuits[0], None, circuit.wires)
    terminal_measurements = circuits[-1].measurements
    # Initialize stacks
    stack = TreeTraversalStack(n_nodes + 1, n_kraus)
    # The goal is to obtain the measurements of the branches
    # and to combine them into the final result. Exit the loop once the
    # measurements for all branches are available.
    depth = 0

    while stack.any_is_empty(1):

        ###########################################
        # Combine measurements & step up the tree #
        ###########################################

        # Combine two leaves once measurements are available
        if stack.is_full(depth):
            # print(f"full stack at {depth=}")
            # Call `combine_measurements` to count-average measurements
            measurements = combine_measurements(terminal_measurements, stack, depth)
            # print(measurements)
            branch_current[depth:] = 0  # Reset current branch
            stack.prune(depth)  # Clear stacks

            # Go up one level to explore alternate subtree of the same depth
            depth -= 1
            stack.results[depth][branch_current[depth]] = measurements
            branch_current[depth] = (branch_current[depth] + 1) % n_kraus[depth]

            continue

        ###########################################
        # Obtain measurements for the active edge #
        ###########################################

        # Simulate the current depth circuit segment
        # if depth == 0:
        #     initial_state = stack.states[0]
        # else:
        #     initial_state, stack.probs[depth][branch_current[depth]] = branch_state(
        #         stack.states[depth], nodes[depth], branch_current[depth]
        #     )
        initial_state = stack.states[depth]

        circtmp = qml.tape.QuantumScript(
            circuits[depth].operations,
            circuits[depth].measurements,
        )
        # circtmp = prepend_state_prep(circtmp, initial_state, circuit.wires)
        state.reset_state()
        if initial_state is not None:
            state._apply_state_vector(initial_state, circuit.wires)
        if depth > 0:
            state, stack.probs[depth][branch_current[depth]] = apply_channel(state, nodes[depth], branch_current[depth])
        state = state.get_final_state(
            circtmp, mid_measurements=branch_current, **execution_kwargs
        )

        ################################################
        # Update terminal measurements & step sideways #
        ################################################

        if depth == n_nodes:
            # Update measurements and switch to the next branch
            measurements = LightningMeasurements(state).measure_final_state(circtmp)
            # print()
            # print("reached max depth")
            # print(branch_current)
            # print(measurements)
            # print(state.state)
            # print()
            if len(terminal_measurements) == 1:
                measurements = (measurements,)
            stack.results[depth][branch_current[depth]] = measurements
            branch_current[depth] = (branch_current[depth] + 1) % n_kraus[depth]
            continue

        #####################################
        # Update stack & step down the tree #
        #####################################

        # If not at a leaf, project on the zero-branch and increase depth by one
        depth += 1

        if stack.probs[depth] is None:
            # If probs list has not been initialized at the current depth, initialize a list
            # for storing the probabilities of each of the different possible branches at the
            # current depth
            stack.probs[depth] = [None] * n_kraus[depth]

        # Store a copy of the state-vector to project on the next branch
        stack.states[depth] = state.state

    ##################################################
    # Finalize terminal measurements post-processing #
    ##################################################

    results = combine_measurements(terminal_measurements, stack, 1)
    return results


def apply_channel(state: LightningStateVector, channel: Channel, kraus_index: int):
    matrix = _get_kraus_matrices(channel)[kraus_index]
    state._apply_lightning([qml.QubitUnitary(matrix, wires=channel.wires)])

    norm = state.state_vector.normalize()
    return state, norm**2


def split_circuit_at_nodes(circuit):
    """Return a list of circuits segments (one for each channel in the
    original circuit) where the terminal measurements probe the state. Only
    the last segment retains the original terminal measurements.

    Args:
        circuit (QuantumTape): The circuit to simulate

    Returns:
        Sequence[QuantumTape]: Circuit segments.
    """

    split_gen = ((i, op) for i, op in enumerate(circuit) if isinstance(op, Channel))
    circuits = []

    first = 0
    for last, _ in split_gen:
        new_operations = circuit.operations[first:last]
        new_measurements = []
        circuits.append(
            qml.tape.QuantumScript(new_operations, new_measurements, shots=circuit.shots)
        )
        first = last + 1

    last_circuit_operations = circuit.operations[first:]
    last_circuit_measurements = circuit.measurements

    circuits.append(
        qml.tape.QuantumScript(
            last_circuit_operations, last_circuit_measurements, shots=circuit.shots
        )
    )
    return circuits


def prepend_state_prep(circuit, state, wires):
    """Prepend a ``StatePrep`` operation with the prescribed ``wires`` to the circuit.

    ``get_final_state`` executes a circuit on a subset of wires found in operations
    or measurements. This function makes sure that an initial state with the correct size is created
    on the first invocation of ``simulate_tree_mcm``. ``wires`` should be the wires attribute
    of the original circuit (which included all wires)."""
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        return circuit
    state = create_initial_state(wires, None) if state is None else state
    return qml.tape.QuantumScript(
        [qml.StatePrep(qml.math.ravel(state), wires=wires, validate_norm=False)]
        + circuit.operations,
        circuit.measurements,
        shots=circuit.shots,
    )


@lru_cache
def _get_kraus_matrices(op):
    return op.kraus_matrices()


def branch_state(state: LightningStateVector, op, index):
    """Collapse the state on a given branch.

    Args:
        state (TensorLike): The initial state
        op (Channel): Channel being applied to the state
        index (int): The index of the list of kraus matrices

    Returns:
        tuple[TensorLike, float]: The collapsed state and the probability
    """
    matrix = _get_kraus_matrices(op)[index]
    state = state.apply_operation(qml.QubitUnitary(matrix, wires=op.wires), state)

    norm = state.normalize()
    return state, norm**2


def fake_measurements(circuit, state, is_state_batched, **execution_kwargs):
    return np.ones(2) / 2


def combine_measurements(terminal_measurments, stack, depth):
    """Returns combined measurement values of various types."""
    final_measurements = []
    all_probs = stack.probs[depth]
    all_results = stack.results[depth]

    for i, mp in enumerate(terminal_measurments):
        all_mp_results = [res[i] for res in all_results]
        comb_meas = combine_measurements_core(mp, all_probs, all_mp_results)
        final_measurements.append(comb_meas)

    return tuple(final_measurements)


@singledispatch
def combine_measurements_core(
    original_measurement, measures, node_is_mcm
):  # pylint: disable=unused-argument
    """Returns the combined measurement value of a given type."""
    raise TypeError(f"tree_simulate does not support {type(original_measurement).__name__}")


@combine_measurements_core.register
def _(original_measurement: ExpectationMP, probs, results):
    """The expectation value of two branches is a weighted sum of expectation values."""
    return qml.math.dot(probs, results)


@combine_measurements_core.register
def _(original_measurement: ProbabilityMP, probs, results):
    """The combined probability of two branches is a weighted sum of the probabilities.
    Note the implementation is the same as for ``ExpectationMP``."""
    return qml.math.dot(probs, results)
