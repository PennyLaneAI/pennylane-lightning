#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <map>

#include "StateVectorManaged.hpp"

std::string_view strip(std::string_view str) {
    auto start = str.find_first_not_of(" \t");
    auto end = str.find_last_not_of(" \t");
    return str.substr(start, end - start + 1);
}

std::vector<std::pair<std::string_view, size_t>> parseGateLists(std::string_view arg) {
    const static std::map<std::string, size_t> available_gates_wires = {
        /* Single-qubit gates */
        {"PauliX", 1},
        {"PauliY", 1},
        {"PauliZ", 1},
        {"Hadamard", 1},
        {"S", 1},
        {"T", 1},
        {"RX", 1},
        {"RY", 1},
        {"RZ", 1},
        {"Rot", 1},
        {"PhaseShift", 1},
        /* Two-qubit gates */
        {"CNOT", 2},
        {"SWAP", 2},
        {"ControlledPhaseShift", 2},
        {"CRX", 2},
        {"CRY", 2},
        {"CRZ", 2},
        {"CRot", 2},
        /* Three-qubit gates */
        {"Toffoli", 3},
        {"CSWAP", 3}
    };

    
    if (arg.empty())
        return std::vector<std::pair<std::string_view, size_t>>(available_gates_wires.begin(),
                available_gates_wires.end());

    std::vector<std::pair<std::string_view, size_t>> ops;

    if (auto pos = arg.find_first_of("["); pos != std::string_view::npos)
    {  
        // arg is a list "[...]"
        auto start = pos + 1;
        auto end = arg.find_last_of("]");
        if (end == std::string_view::npos) {
            throw std::invalid_argument("Argument must contain operators within square brackets [].");
        }
        arg = arg.substr(start, end - start);
    }
        
    size_t start;
    size_t end = 0;
    while ((start = arg.find_first_not_of(",", end)) != string::npos)
    {
        end = arg.find(",", start);
        auto op_name = strip(arg.substr(start, end - start));

        auto iter = available_gates_wires.find(std::string(op_name));

        if (iter == available_gates_wires.end()) {
            std::ostringstream ss;
            ss << "Given gate " << op_name << " is not availabe"; // Change to std::format in C++20
            throw std::invalid_argument(ss.str());
        }
        ops.emplace_back(*iter);
    }
    return ops;
}

/**
 * @brief Outputs wall-time for gate-benchmark.
 * @param argc Number of arguments + 1 passed by user.
 * @param argv Binary name followed by number of times gate is repeated and
 * number of qubits.
 * @return Returns 0 if completed successfully.
 */
int main(int argc, char *argv[]) {
    using TestType = double;

    // Handle input
    if ((argc != 3) && (argc != 4)) {
        std::cerr << "Wrong number of inputs. User provided " << argc - 1
                  << " inputs. "
                  << "Usage: " + std::string(argv[0]) +
                         " num_gate_reps num_qubits [gate_lists]"
                  << std::endl; // Change to std::format in C++20
        return -1;
    }

    size_t num_gate_reps;
    size_t num_qubits;

    try {
        num_gate_reps = std::stoi(argv[1]);
        num_qubits = std::stoi(argv[2]);
    }
    catch(std::exception& e) {
        std::cerr << "Arguements num_gate_reps and num_qubits must be integers." << std::endl;
        return -1;
    }
    
    // Gate list is provided
    std::string_view op_list_s;
    if(argc == 3) {
        op_list_s = "";
    } else { // if argc == 4
        op_list_s = argv[3];
    }

    auto op_list = parseGateLists(op_list_s);
    for(const auto& [op_name, op_wire]: op_list) {
        std::cout << op_name << "\t" << op_wire << std::endl;
    }


    // Generate random values for parametric gates
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<TestType> distr(0.0, 1.0);
    std::vector<TestType> random_parameters(num_gate_reps);

    for(uint32_t k = 0; k < num_gate_reps; ++k)
    {
    }

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
            const auto int_idx = svdat.getInternalIndices({index});
            const auto ext_idx = svdat.getExternalIndices({index});
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
