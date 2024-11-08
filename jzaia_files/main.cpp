#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <vector>

#include "GateImplementationsLM.hpp"
#include "Gates.hpp"
#include "MeasurementsLQubit.hpp"
#include "ObservablesLQubit.hpp"
#include "StateVectorLQubitManaged.hpp"

using Pennylane::LightningQubit::StateVectorLQubitManaged;
using namespace Pennylane::Gates;
using namespace Pennylane::LightningQubit;
using namespace Pennylane::LightningQubit::Measures;
using namespace Pennylane::LightningQubit::Observables;

// Define values to be selected by each oracle
const std::pair<size_t, std::vector<bool>> oracle1_data = {
    6, std::vector<bool>{true, true, false, true, false}};
const std::pair<size_t, std::vector<bool>> oracle2_data = {
    10, std::vector<bool>{true, false, true, false, true, false, true, false,
                          true}};
const std::pair<size_t, std::vector<bool>> oracle3_data = {
    17, std::vector<bool>{false, false, true, true, false, false, true, true,
                          false, false, true, true, false, false, true, true}};

/**
 * @brief Setup up a circuit for iterations of Grover's search
 *
 * Takes a statevector and places it into uniform superposition. Additionally,
 * Places the ancilla qubit for the oracle in the |-> state such that it can
 * apply phase kickback.
 *
 * @param sv The (presumed all |0>) statevector to apply this operation to
 */
void groversSetup(StateVectorLQubitManaged<double> &sv) {
    const size_t num_qubits = sv.getNumQubits();

    // Place target qubit into |1> state so the Hadamard will take it to |->
    GateImplementationsLM::applyPauliX(sv.getData(), sv.getNumQubits(),
                                       {num_qubits - 1}, false);

    // Set up uniform superposition
    for (size_t i = 0; i < num_qubits; ++i) {
        GateImplementationsLM::applyHadamard(sv.getData(), sv.getNumQubits(),
                                             {i}, false);
    }
}

/**
 * @brief Apply the "Grover's Mirror" reflection to the given statevector
 *
 * Performs a reflection across the vector which represents the uniform
 * superposition. This is used for amplitude-amplification for Grover's
 * algorithm.
 *
 * @param sv The statevector to reflect
 */
void groversMirror(StateVectorLQubitManaged<double> &sv) {
    const size_t num_qubits = sv.getNumQubits();

    // Apply H to all non-ancilla qubits
    for (size_t i = 0; i < num_qubits - 1; ++i) {
        GateImplementationsLM::applyHadamard(sv.getData(), sv.getNumQubits(),
                                             {i}, false);
    }

    std::vector<size_t> controls(num_qubits - 1);
    std::vector<bool> control_values(num_qubits - 1, false);
    std::iota(controls.begin(), controls.end(), 0);

    // Apply an anti-controlled on all search qubits X-gate onto the ancilla
    // qubit
    GateImplementationsLM::applyNCPauliX(sv.getData(), sv.getNumQubits(),
                                         controls, control_values,
                                         {num_qubits - 1}, false);

    // Apply H to all non-ancilla qubits
    for (size_t i = 0; i < num_qubits - 1; ++i) {
        GateImplementationsLM::applyHadamard(sv.getData(), sv.getNumQubits(),
                                             {i}, false);
    }
}

/**
 * @brief A generalized oracle function for Grover's algorithm
 *
 * Applies an oracle for Grover's algorithm. Applies a pauli-X to the
 * rightmost qubit if the leftmost qubits are in the state specified by
 * `control_vals`. Only selects a single state
 *
 * @param sv The statevector to apply the oracle to
 * @param control_vals The state the oracle should select
 */
void oracle(StateVectorLQubitManaged<double> &sv,
            const std::vector<bool> &control_vals) {
    // Sanity check statevector
    assert(sv.getNumQubits() == control_vals.size() + 1);

    // Define controls to be used for applying the X gate
    std::vector<size_t> controls(control_vals.size());
    std::iota(controls.begin(), controls.end(), 0);

    // Apply the X gate to the ancilla, controlled on the chosen bitstring
    GateImplementationsLM::applyNCPauliX(sv.getData(), sv.getNumQubits(),
                                         controls, control_vals,
                                         {control_vals.size()}, false);
}

/**
 * @brief Overall function for running Grover's algorithm on a chosen oracle
 *
 * Run Grover's algorithm from start to finish. Prepares a statevector, and
 * repeats state selection and amplitude-amplification for sqrt(N) iterations
 * (where N = 2^(# of non-ancilla qubits)). This implementation assumes that
 * the oracle always picks precisly 1 state (rather than an arbitrary number).
 *
 * @param num_qubits The number of qubits in the circuit the oracle acts
 *        on (includes the ancilla)
 * @param expected The state which the oracle should apply phase to
 */
void run_experiment(const size_t num_qubits,
                    const std::vector<bool> &expected) {
    // Create the statevector for this circuit
    // NOTE: qubits are numbered left-to-right order
    StateVectorLQubitManaged<double> sv(num_qubits);

    // Set up the circuit by placing into uniform superposition
    groversSetup(sv);

    // Apply (approx) sqrt(N) repetitions of state selection and amp-amp
    const size_t nreps = static_cast<size_t>(sqrt(1llu << (num_qubits - 1)));
    for (size_t reps = 0; reps < nreps; ++reps) {
        // Apply the oracle to apply a phase of -1 to desired state
        oracle(sv, expected);
        // Perform amp-amp by reflecting over |+++...+>
        groversMirror(sv);
    }

    // Set up measurements
    Measurements<StateVectorLQubitManaged<double>> Measurer(sv);
    // Vector to store the most common measurement outcome
    std::vector<size_t> wires(num_qubits - 1);
    std::iota(wires.begin(), wires.end(), 0);
    std::vector<bool> common_result(num_qubits - 1, false);

    // Perform a "measurement" by taking the expected value of this qubit over
    // multiple runs
    std::transform(wires.begin(), wires.end(), common_result.begin(),
                   [&Measurer](size_t wire) {
                       NamedObs<StateVectorLQubitManaged<double>> obs("PauliZ",
                                                                      {wire});
                       return Measurer.expval(obs) < 0;
                   });

    std::cout << "Measured results (most common with expval): " << common_result
              << std::endl;
}

int main(void) {
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

    std::vector<std::pair<size_t, std::vector<bool>>> inputs = {
        // Experiment 1: 11010
        oracle1_data,
        // Experiment 2: 101010101
        oracle2_data,
        // Experiment 3: 0011001100110011
        oracle3_data,
    };

    std::vector<milliseconds> runtimes(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto num_qubits = inputs[i].first;
        auto expected = inputs[i].second;

        std::cout << "Running Oracle " << i + 1 << ". Expected: ";
        std::cout << expected << std::endl;

        auto start_time = high_resolution_clock::now();

        run_experiment(num_qubits, expected);

        runtimes[i] = duration_cast<milliseconds>(high_resolution_clock::now() -
                                                  start_time);

        std::cout << std::endl;
    }

    for (size_t i = 0; i < runtimes.size(); ++i) {
        std::cout << "Time to run oracle " << i + 1 << ": "
                  << runtimes[i].count() << "ms\n";
    }

    return 0;
}
