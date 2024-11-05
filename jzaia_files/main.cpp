#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

#include "Gates.hpp"
#include "GateImplementationsLM.hpp"
#include "MeasurementsLQubit.hpp"
#include "ObservablesLQubit.hpp"
#include "StateVectorLQubitManaged.hpp"


using Pennylane::LightningQubit::StateVectorLQubitManaged;
using namespace Pennylane::Gates;
using namespace Pennylane::LightningQubit;
using namespace Pennylane::LightningQubit::Measures;
using namespace Pennylane::LightningQubit::Observables;

// Define values to be selected by each oracle
#define ORACLE1_QUBITS (6)
/* Oracle 1 selects the string: "11010" */
#define ORACLE1_EXPECTED (std::vector<bool>{true, true, false, true, false})

#define ORACLE2_QUBITS (10)
/* Oracle 2 selects the string: "101010101" */
#define ORACLE2_EXPECTED (std::vector<bool>{true, false, true, false, true, \
                                            false, true, false, true})

#define ORACLE3_QUBITS (17)
/* Oracle 2 selects the string: "0011001100110011" */
#define ORACLE3_EXPECTED (std::vector<bool>{false, false, true, true, false, \
                                            false, true, true, false, false, \
                                            true, true, false, false, true, \
                                            true})


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
    GateImplementationsLM::applyPauliX(sv.getData(),
                                       sv.getNumQubits(),
                                       {num_qubits-1},
                                       false);

    // Set up uniform superposition
    for (size_t i = 0; i < num_qubits; ++i) {
        GateImplementationsLM::applyHadamard(sv.getData(),
                                             sv.getNumQubits(),
                                             {i},
                                             false);
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
    for (size_t i = 0; i < num_qubits-1; ++i) {
        GateImplementationsLM::applyHadamard(sv.getData(),
                                             sv.getNumQubits(),
                                             {i},
                                             false);
    }

    std::vector<size_t> controls(num_qubits-1);
    std::vector<bool> control_values(num_qubits-1, false);
    std::iota(controls.begin(), controls.end(), 0);

    // Apply an anti-controlled on all search qubits X-gate onto the ancilla qubit
    GateImplementationsLM::applyNCPauliX(sv.getData(),
                                         sv.getNumQubits(),
                                         controls,
                                         control_values,
                                         {num_qubits-1},
                                         false);

    // Apply H to all non-ancilla qubits
    for (size_t i = 0; i < num_qubits-1; ++i) {
        GateImplementationsLM::applyHadamard(sv.getData(),
                                             sv.getNumQubits(),
                                             {i},
                                             false);
    }
}

/**
 * @brief The first testing oracle for Grover's
 *
 * A 6-qubit test oracle for Grover's algorithm. Applies a pauli-X to the
 * rightmost qubit if the leftmost 5 qubits are in the state |11010>
 *
 * @param sv The statevector to apply the oracle to. Must be 6 qubits
 */
void oracle1(StateVectorLQubitManaged<double> &sv) {
    // Sanity check statevector
    assert(sv.getNumQubits() == ORACLE1_QUBITS);

    // Define controls to be used for applying the X gate
    std::vector<size_t> controls(ORACLE1_QUBITS-1);
    std::iota(controls.begin(), controls.end(), 0);
    std::vector<bool> control_vals = ORACLE1_EXPECTED;

    // Apply the X gate to the ancilla, controlled on the chosen bitstring
    GateImplementationsLM::applyNCPauliX(sv.getData(),
                                         sv.getNumQubits(),
                                         controls,
                                         control_vals,
                                         {ORACLE1_QUBITS-1},
                                         false);
}

/**
 * @brief The second testing oracle for Grover's
 *
 * A 10-qubit test oracle for Grover's algorithm. Applies a pauli-X to the
 * rightmost qubit if the leftmost 9 qubits are in the state |101010101>
 *
 * @param sv The statevector to apply the oracle to. Must be 10 qubits
 */
void oracle2(StateVectorLQubitManaged<double> &sv) {
    // Sanity check statevector
    assert(sv.getNumQubits() == ORACLE2_QUBITS);

    // Define controls to be used for applying the X gate
    std::vector<size_t> controls(ORACLE2_QUBITS-1);
    std::iota(controls.begin(), controls.end(), 0);
    std::vector<bool> control_vals = ORACLE2_EXPECTED;

    // Apply the X gate to the ancilla, controlled on the chosen bitstring
    GateImplementationsLM::applyNCPauliX(sv.getData(),
                                         sv.getNumQubits(),
                                         controls,
                                         control_vals,
                                         {ORACLE2_QUBITS-1},
                                         false);
}

/**
 * @brief The third testing oracle for Grover's
 *
 * A 17-qubit test oracle for Grover's algorithm. Applies a pauli-X to the
 * rightmost qubit if the leftmost 16 qubits are in the state |0011001100110011>
 *
 * @param sv The statevector to apply the oracle to. Must be 17 qubits
 */
void oracle3(StateVectorLQubitManaged<double> &sv) {
    // Sanity check statevector
    assert(sv.getNumQubits() == ORACLE3_QUBITS);

    // Define controls to be used for applying the X gate
    std::vector<size_t> controls(ORACLE3_QUBITS-1);
    std::iota(controls.begin(), controls.end(), 0);
    std::vector<bool> control_vals = ORACLE3_EXPECTED;

    // Apply the X gate to the ancilla, controlled on the chosen bitstring
    GateImplementationsLM::applyNCPauliX(sv.getData(),
                                         sv.getNumQubits(),
                                         controls,
                                         control_vals,
                                         {ORACLE3_QUBITS-1},
                                         false);
}


/**
 * @brief Overall function for running Grover's algorithm on a chosen oracle
 *
 * Run Grover's algorithm from start to finish. Prepares a statevector, and
 * repeats state selection and amplitude-amplification for sqrt(N) iterations
 * (where N = 2^(# of non-ancilla qubits)). This implementation assumes that
 * the oracle always picks precisly 1 state (rather than an arbitrary number).
 *
 * @param oracle A black-box function that acts on a created statevector
 * @param num_qubits The number of qubits in the circuit the oracle acts
 *        on (includes the ancilla)
 */
void run_experiment(void (*oracle) (StateVectorLQubitManaged<double> &),
                    const size_t num_qubits) {
    // Create the statevector for this circuit
    // NOTE: qubits are numbered left-to-right order
    StateVectorLQubitManaged<double> sv(num_qubits);

    // Set up the circuit by placing into uniform superposition
    groversSetup(sv);

    // Apply (approx) sqrt(N) repetitions of state selection and amp-amp
    const size_t nreps = static_cast<size_t>(sqrt(1llu << (num_qubits-1)));
    for (size_t reps = nreps; reps > 0; --reps) {
        // Apply the oracle to apply a phase of -1 to desired state
        oracle(sv);
        // Perform amp-amp by reflecting over |+++...+>
        groversMirror(sv);
    }

    // Set up measurements
    Measurements<StateVectorLQubitManaged<double>> Measurer(sv);
    // Vector to store the most common measurement outcome
    std::vector<bool> common_result(num_qubits - 1, false);

    // Perform a "measurement" by taking the expected value of this qubit over multiple runs
    for (size_t obs_wire=0; obs_wire < num_qubits - 1; ++obs_wire) {
        NamedObs<StateVectorLQubitManaged<double>> obs("PauliZ", {obs_wire});
        double result = Measurer.expval(obs);
        common_result[obs_wire] = (result < 0);
    }

    // // Commented out: Print the raw statevector output (too large and messy)
    // std::vector<std::size_t> wires(num_qubits-1);
    // for (size_t i = 0; i < num_qubits - 1; ++i) { wires[i] = i; }
    // std::vector<double> probabilities = Measurer.probs(wires);

    // size_t qubitString = 0;
    // for (auto prob : probabilities) {
    //     std::cout << qubitString << ": " << prob << std::endl;
    //     ++qubitString;
    // }

    std::cout << "Measured results (most common with expval): " << common_result << std::endl;
}


int main(void) {
    // Run experiment 1: 11010
    std::cout << "Expected: " << ORACLE1_EXPECTED << std::endl;
    run_experiment(oracle1, ORACLE1_QUBITS);

    // Run experiment 2: 101010101
    std::cout << "Expected: " << ORACLE2_EXPECTED << std::endl;
    run_experiment(oracle2, ORACLE2_QUBITS);

    // Run experiment 3: 0011001100110011
    std::cout << "Expected: " << ORACLE3_EXPECTED << std::endl;
    run_experiment(oracle3, ORACLE3_QUBITS);

    return 0;
}
