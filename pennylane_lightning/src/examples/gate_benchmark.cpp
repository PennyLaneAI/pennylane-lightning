#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include "IndicesUtil.hpp"
#include "StateVectorManaged.hpp"

/**
 * @brief Outputs wall-time for gate-benchmark.
 * @param argc Number of arguments + 1 passed by user.
 * @param argv Binary name followed by number of times gate is repeated and
 * number of qubits.
 * @return Returns 0 if completed successfully.
 */
int main(int argc, char *argv[]) {
    using TestType = double;
    namespace IndicesUtil = Pennylane::IndicesUtil;

    // Handle input
    try {
        if (argc != 3) {
            throw argc;
        }
    } catch (int e) {
        std::cerr << "Wrong number of inputs. User provided " << e - 1
                  << " inputs. "
                  << "Usage: " + std::string(argv[0]) +
                         " $num_gate_reps $num_qubits"
                  << std::endl;
        return -1;
    }
    const size_t num_gate_reps = std::stoi(argv[1]);
    const size_t num_qubits = std::stoi(argv[2]);

    // Generate random values for parametric gates
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<TestType> distr(0.0, 1.0);
    std::vector<std::vector<TestType>> random_parameter_vector(num_gate_reps);
    std::for_each(
        random_parameter_vector.begin(), random_parameter_vector.end(),
        [num_qubits, &eng, &distr](std::vector<TestType> &vec) {
            vec.resize(num_qubits);
            std::for_each(vec.begin(), vec.end(),
                          [&eng, &distr](TestType &val) { val = distr(eng); });
        });

    // Run each gate specified number of times and measure walltime
    Pennylane::StateVectorManaged<TestType> svdat{num_qubits};
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;
    t_start = std::chrono::high_resolution_clock::now();
    for (size_t gate_rep = 0; gate_rep < num_gate_reps; gate_rep++) {
        for (size_t index = 0; index < num_qubits; index++) {
            // Apply single qubit non-parametric operations
            const auto int_idx =
                IndicesUtil::getInternalIndices({index}, num_qubits);
            const auto ext_idx =
                IndicesUtil::getExternalIndices({index}, num_qubits);
            svdat.applyPauliX(int_idx, ext_idx, false);
            svdat.applyPauliY(int_idx, ext_idx, false);
            svdat.applyPauliZ(int_idx, ext_idx, false);
            svdat.applyHadamard(int_idx, ext_idx, false);

            // Apply two qubit non-parametric operations
            const auto two_qubit_int_idx =
                svdat.getInternalIndices({index, (index + 1) % num_qubits});
            const auto two_qubit_ext_idx =
                svdat.getExternalIndices({index, (index + 1) % num_qubits});
            svdat.applyCNOT(two_qubit_int_idx, two_qubit_ext_idx, false);
            svdat.applyCZ(two_qubit_int_idx, two_qubit_ext_idx, false);

            // Apply single qubit parametric operations
            const TestType angle =
                2.0 * M_PI * random_parameter_vector[gate_rep][index];
            svdat.applyRX(int_idx, ext_idx, false, angle);
            svdat.applyRY(int_idx, ext_idx, false, angle);
            svdat.applyRZ(int_idx, ext_idx, false, angle);

            // Apply two qubit parametric operations
            svdat.applyCRX(two_qubit_int_idx, two_qubit_ext_idx, false, angle);
            svdat.applyCRY(two_qubit_int_idx, two_qubit_ext_idx, false, angle);
            svdat.applyCRZ(two_qubit_int_idx, two_qubit_ext_idx, false, angle);
        }
    }
    t_end = std::chrono::high_resolution_clock::now();

    // Output walltime in csv format (Num Qubits, Time (milliseconds))
    const auto walltime =
        0.001 * ((std::chrono::duration_cast<std::chrono::microseconds>(
                      t_end - t_start))
                     .count());
    std::cout << num_qubits << ", "
              << walltime / static_cast<double>(num_gate_reps) << std::endl;

    return 0;
}
