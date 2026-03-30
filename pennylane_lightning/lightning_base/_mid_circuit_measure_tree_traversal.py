# Copyright 2025 Xanadu Quantum Technologies Inc.

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
This module implements a tree traversal algorithm for simulating quantum circuits with mid-circuit measurements (MCMs) using PennyLane's Lightning backend. The core functionality enables the simulation of dynamic quantum circuits, including support for post-selection, shot vectors, and various measurement types (expectation, probability, sample, counts, and variance).

"""

import numpy as np
import pennylane as qml
from pennylane.devices.qubit.simulate import (
    TreeTraversalStack,
    combine_measurements,
    counts_to_probs,
    get_measurement_dicts,
    insert_mcms,
    prune_mcm_samples,
    samples_to_counts,
    split_circuit_at_mcms,
    update_mcm_samples,
    variance_transform,
)
from pennylane.measurements import MidMeasureMP, find_post_processed_mcms
from pennylane.tape import QuantumScript
from pennylane.typing import Result

from pennylane_lightning.lightning_base._measurements import LightningBaseMeasurements
from pennylane_lightning.lightning_base._state_vector import LightningBaseStateVector


# pylint: disable=too-many-branches, too-many-statements
def mcm_tree_traversal(
    circuit: QuantumScript,
    lightning_state: LightningBaseStateVector,
    lightning_measurement: LightningBaseMeasurements,
    postselect_mode: str = None,
) -> Result:
    """Simulate a single quantum script with native mid-circuit measurements using the tree-traversal algorithm.

    The tree-traversal algorithm recursively explores all combinations of mid-circuit measurement
    outcomes using a depth-first approach. The depth-first approach requires ``n_mcm`` copies
    of the state vector (``n_mcm + 1`` state vectors in total) and records ``n_mcm`` vectors
    of mid-circuit measurement samples. It is generally more efficient than ``one-shot`` because it takes all samples
    at a leaf at once and stops exploring more branches when a single shot is allocated to a sub-tree.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        lightning_state (LightningBaseStateVector): The state vector.
        lightning_measurement (LightningBaseMeasurements): The measurement class used to perform measurements
        postselect_mode (str): Configuration for shots with mid-circuit measurement

    Returns:
        Result: The result of the simulation, which can be a scalar or a tuple of scalars.
    """

    ##########################
    # shot vector processing #
    ##########################
    if circuit.shots.has_partitioned_shots:
        results = []
        for s in circuit.shots:
            aux_circuit = circuit.copy(shots=s)
            lightning_state.reset_state()
            results.append(
                mcm_tree_traversal(
                    aux_circuit, lightning_state, lightning_measurement, postselect_mode
                )
            )
        return tuple(results)

    #######################
    # main implementation #
    #######################

    # `var` measurements cannot be aggregated on the fly as they require the global `expval`
    # variance_transform replaces `var` measurements with `expval` and `expval**2` measurements
    [circuit], variance_post_processing = variance_transform(circuit)
    finite_shots = bool(circuit.shots)

    ##################
    # Parse MCM info #
    ##################

    # mcms is the list of all mid-circuit measurement operations
    # mcms[d] is the parent MCM (node) of a circuit segment (edge) at depth `d`
    # The first element is None because there is no parent MCM at depth 0
    mcms = tuple([None] + [op for op in circuit.operations if isinstance(op, MidMeasureMP)])
    n_mcms = len(mcms) - 1
    # We obtain `measured_mcms_indices`, the list of MCMs which require post-processing:
    # either as requested by terminal measurements or post-selection
    measured_mcms = find_post_processed_mcms(circuit)
    measured_mcms_indices = [i for i, mcm in enumerate(mcms[1:]) if mcm in measured_mcms]
    # `mcm_samples` is a register of MCMs. It is necessary to correctly keep track of
    # correlated MCM values which may be requested by terminal measurements.
    mcm_samples = {
        k + 1: qml.math.empty((circuit.shots.total_shots,), dtype=bool) if finite_shots else None
        for k in measured_mcms_indices
    }

    #############################
    # Initialize tree-traversal #
    #############################

    # mcm_current[:d+1] is the active branch at depth `d`
    # The first entry is always 0 as the first edge does not stem from an MCM.
    # For example, if `d = 2` and `mcm_current = [0, 1, 1, 0]` we are on the 11-branch,
    # i.e. the first two MCMs had outcome 1. The last entry isn't meaningful until we are
    # at depth `d=3`.
    mcm_current = qml.math.zeros(n_mcms + 1, dtype=int)
    # `mid_measurements` maps the elements of `mcm_current` to their respective MCMs
    # This is used by `get_final_state::apply_operation` for `Conditional` operations
    mid_measurements = dict(zip(mcms[1:], mcm_current[1:].tolist()))
    # Split circuit into segments
    circuits = split_circuit_at_mcms(circuit)

    terminal_measurements = circuits[-1].measurements if finite_shots else circuit.measurements
    # Initialize stacks
    cumcounts = [0] * (n_mcms + 1)
    stack = TreeTraversalStack(n_mcms + 1)
    # The goal is to obtain the measurements of the zero-branch and one-branch
    # and to combine them into the final result. Exit the loop once the
    # zero-branch and one-branch measurements are available.
    depth = 0

    while stack.any_is_empty(1):
        ###########################################
        # Combine measurements & step up the tree #
        ###########################################

        # Combine two leaves once measurements are available
        if stack.is_full(depth):
            # Call `combine_measurements` to count-average measurements
            measurement_dicts = get_measurement_dicts(terminal_measurements, stack, depth)
            measurements = combine_measurements(
                terminal_measurements, measurement_dicts, mcm_samples
            )
            mcm_current[depth:] = 0  # Reset current branch
            stack.prune(depth)  # Clear stacks
            # Go up one level to explore alternate subtree of the same depth
            depth -= 1
            if mcm_current[depth] == 1:
                stack.results_1[depth] = measurements
                mcm_current[depth] = 0
            else:
                stack.results_0[depth] = measurements
                mcm_current[depth] = 1
            # Update MCM values
            mid_measurements.update(
                (k, v) for k, v in zip(mcms[depth:], mcm_current[depth:].tolist())
            )
            continue

        ################################################
        # Determine whether to execute the active edge #
        ################################################

        # Parse shots for the current branch
        if finite_shots:
            if stack.counts[depth]:
                shots = stack.counts[depth][mcm_current[depth]]
            else:
                shots = circuits[depth].shots.total_shots
            skip_subtree = not bool(shots)
        else:
            shots = None
            skip_subtree = (
                stack.probs[depth] is not None
                and float(stack.probs[depth][mcm_current[depth]]) <= 0.0  # PROBS_TOL
            )
        # Update active branch dict
        invalid_postselect = (
            depth > 0
            and mcms[depth].postselect is not None
            and mcm_current[depth] != mcms[depth].postselect
        )

        ###########################################
        # Obtain measurements for the active edge #
        ###########################################

        # If num_shots is zero or postselecting on the wrong branch, update measurements with an empty tuple
        if skip_subtree or invalid_postselect:
            # Adjust counts if `invalid_postselect`
            if invalid_postselect:
                if finite_shots:
                    # Bump downstream cumulative counts before zeroing-out counts
                    for d in range(depth + 1, n_mcms + 1):
                        cumcounts[d] += stack.counts[depth][mcm_current[depth]]
                    stack.counts[depth][mcm_current[depth]] = 0
                else:
                    stack.probs[depth][mcm_current[depth]] = 0
            measurements = tuple()
        else:
            # If num_shots is non-zero, simulate the current depth circuit segment

            initial_state = stack.states[depth]  # None
            if depth != 0:
                branch_state(lightning_state, initial_state, mcm_current[depth], mcms[depth])

            circtmp = circuits[depth].copy(shots=qml.measurements.shots.Shots(shots))

            lightning_state = lightning_state.get_final_state(
                circtmp,
                mid_measurements=mid_measurements,
                postselect_mode=postselect_mode,
            )

            measurements = lightning_measurement(lightning_state).measure_final_state(circtmp)

        #####################################
        # Update stack & step down the tree #
        #####################################

        # If not at a leaf, project on the zero-branch and increase depth by one
        if depth < n_mcms and (not skip_subtree and not invalid_postselect):
            depth += 1
            # Update the active branch samples with `update_mcm_samples`
            if finite_shots:
                samples = qml.math.atleast_1d(measurements)
                stack.counts[depth] = samples_to_counts(samples)
                stack.probs[depth] = counts_to_probs(stack.counts[depth])
            else:
                stack.probs[depth] = dict(zip([False, True], measurements))
                samples = None
            # Store a copy of the state-vector to project on the one-branch
            stack.states[depth] = lightning_state.state
            mcm_samples, cumcounts = update_mcm_samples(samples, mcm_samples, depth, cumcounts)
            continue

        ################################################
        # Update terminal measurements & step sideways #
        ################################################

        if not skip_subtree and not invalid_postselect:
            measurements = insert_mcms(circuit, measurements, mid_measurements)

        # If at a zero-branch leaf, update measurements and switch to the one-branch
        if mcm_current[depth] == 0:
            stack.results_0[depth] = measurements
            mcm_current[depth] = True
            mid_measurements[mcms[depth]] = True
            continue
        # If at a one-branch leaf, update measurements
        stack.results_1[depth] = measurements

    ##################################################
    # Finalize terminal measurements post-processing #
    ##################################################

    measurement_dicts = get_measurement_dicts(terminal_measurements, stack, depth)
    if finite_shots:
        terminal_measurements = circuit.measurements
    mcm_samples = {mcms[i]: v for i, v in mcm_samples.items()}
    mcm_samples = prune_mcm_samples(mcm_samples)
    results = combine_measurements(terminal_measurements, measurement_dicts, mcm_samples)
    return variance_post_processing((results,))


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------


def branch_state(
    lightning_state: LightningBaseStateVector, state: np.ndarray, branch: int, mcm: MidMeasureMP
):
    """Collapse the state on a given branch.

    Args:
        lightning_state (LightningBaseStateVector): The state vector class used to handle the state
        state (np.ndarray): The state vector to collapse
        branch (int): The branch on which the state is collapsed
        mcm (MidMeasureMP): Mid-circuit measurement object used to obtain the wires and ``reset``
    """
    # Set the state to the initial state
    lightning_state._apply_state_vector(  # pylint: disable=protected-access
        state, lightning_state.wires
    )

    # Apply the collapse operation
    lightning_state.state_vector.collapse(mcm.wires.tolist()[0], bool(branch))

    if mcm.reset and branch == 1:
        lightning_state._apply_lightning(  # pylint: disable=protected-access
            [qml.PauliX(mcm.wires)]
        )
