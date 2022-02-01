#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>

#include "StateVectorManaged.hpp"

using namespace Pennylane;

std::string_view strip(std::string_view str) {
    auto start = str.find_first_not_of(" \t");
    auto end = str.find_last_not_of(" \t");
    return str.substr(start, end - start + 1);
}

struct GateDesc {
    size_t n_wires;  // number of wires the gate applies to
    size_t n_params; // number of parameters the gate requires
};

std::vector<std::pair<std::string_view, GateDesc>>
parseGateLists(std::string_view arg) {
    const static std::map<std::string, GateDesc> available_gates_wires = {
        /* Single-qubit gates */
        {"PauliX", {1, 0}},
        {"PauliY", {1, 0}},
        {"PauliZ", {1, 0}},
        {"Hadamard", {1, 0}},
        {"S", {1, 0}},
        {"T", {1, 0}},
        {"RX", {1, 1}},
        {"RY", {1, 1}},
        {"RZ", {1, 1}},
        {"Rot", {1, 3}},
        {"PhaseShift", {1, 1}},
        /* Two-qubit gates */
        {"CNOT", {2, 0}},
        {"CZ", {2, 0}},
        {"SWAP", {2, 0}},
        {"ControlledPhaseShift", {2, 1}},
        {"CRX", {2, 1}},
        {"CRY", {2, 1}},
        {"CRZ", {2, 1}},
        {"CRot", {2, 3}},
        /* Three-qubit gates */
        {"Toffoli", {3, 0}},
        {"CSWAP", {3, 0}}};

    if (arg.empty()) {
        /*
        return std::vector<std::pair<std::string_view, GateDesc>>(
            available_gates_wires.begin(), available_gates_wires.end());
        */
        return {};
    }

    std::vector<std::pair<std::string_view, GateDesc>> ops;

    if (auto pos = arg.find_first_of('['); pos != std::string_view::npos) {
        // arg is a list "[...]"
        auto start = pos + 1;
        auto end = arg.find_last_of(']');
        if (end == std::string_view::npos) {
            throw std::invalid_argument(
                "Argument must contain operators within square brackets [].");
        }
        arg = arg.substr(start, end - start);
    }

    size_t start;
    size_t end = 0;
    while ((start = arg.find_first_not_of(',', end)) != std::string::npos) {
        end = arg.find(',', start);
        auto op_name = strip(arg.substr(start, end - start));

        auto iter = available_gates_wires.find(std::string(op_name));

        if (iter == available_gates_wires.end()) {
            std::ostringstream ss;
            ss << "Given gate " << op_name
               << " is not availabe"; // TODO: Change to std::format in C++20
            throw std::invalid_argument(ss.str());
        }
        ops.emplace_back(*iter);
    }
    return ops;
}

template <typename RandomEngine>
std::vector<size_t> generateDistinctWires(RandomEngine &re, size_t num_qubits,
                                          size_t num_wires) {
    std::vector<size_t> v(num_qubits, 0);
    std::iota(v.begin(), v.end(), 0);
    shuffle(v.begin(), v.end(), re);
    return std::vector<size_t>(v.begin(), v.begin() + num_wires);
}

template <typename RandomEngine>
std::vector<size_t> generateNeighboringWires(RandomEngine &re,
                                             size_t num_qubits,
                                             size_t num_wires) {
    std::vector<size_t> v;
    v.reserve(num_wires);
    std::uniform_int_distribution<size_t> idist(0, num_qubits - 1);
    size_t start_idx = idist(re);
    for (size_t k = 0; k < num_wires; k++) {
        v.emplace_back((start_idx + k) % num_qubits);
    }
    return v;
}
/**
 * @brief Benchmark Pennylane-Lightning for a given gate set
 *
 * Example usage:
 *
 *     $ gate_benchmark_oplist 10 22 # Benchmark using 10 random gates (sampled
 * evenly from all possible gates) for 22 qubits
 *     $ gate_benchmark_oplist 100 20 [PauliX, CNOT] # Benchmark using 100
 * random gates (where each gate is PauliX or CNOT) for 20 qubits
 *
 * The whole supported gates are PauliX, PauliY, PauliZ, Hadamard, S, T, RX, RY,
 * RZ, Rot, PhaseShift, CNOT, SWAP, ControlledPhaseShift, CRX, CRY, CRZ, CRot,
 * Toffoli and CSWAP.
 *
 * @param argc Number of arguments
 * @param argv Command line arguments
 * @return Returns 0 is completed successfully
 */
int main(int argc, char *argv[]) {
    using TestType = double;

    // Handle input
    if (argc < 4) {
        std::cerr << "Wrong number of inputs. User provided " << argc - 1
                  << " inputs. "
                  << "Usage: " + std::string(argv[0]) +
                         " num_gate_reps num_qubits kernel [gate_lists]\n"
                         "\tExample: "
                  << argv[0] << " 1000 10 PI [PauliX, CNOT]"
                  << std::endl; // Change to std::format in C++20
        return -1;
    }

    size_t num_gate_reps;
    size_t num_qubits;

    try {
        num_gate_reps = std::stoi(argv[1]);
        num_qubits = std::stoi(argv[2]);
    } catch (std::exception &e) {
        std::cerr << "Arguements num_gate_reps and num_qubits must be integers."
                  << std::endl;
        return -1;
    }

    std::string_view kernel_name = argv[3];
    KernelType kernel = string_to_kernel(kernel_name);
    if (kernel == KernelType::Unknown) {
        std::cerr << "Kernel " << kernel_name << " is unknown." << std::endl;
        return 1;
    }

    // Gate list is provided
    std::string op_list_s;
    {
        std::ostringstream ss;
        for (int idx = 4; idx < argc; idx++) {
            ss << argv[idx] << " ";
        }
        op_list_s = ss.str();
    }

    std::vector<std::pair<std::string_view, GateDesc>> op_list;
    try {
        op_list = parseGateLists(op_list_s);
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    if (op_list.empty()) {
        std::cerr << "Please provide a gate list." << std::endl;
        return 1;
    }

    // Generate random gate sequences
    std::random_device rd;
    std::mt19937 re(rd());

    std::vector<std::string_view> random_gate_names;
    std::vector<std::vector<size_t>> random_gate_wires;
    std::vector<bool> random_inverses;
    std::vector<std::vector<TestType>> random_gate_parameters;

    std::uniform_int_distribution<size_t> gate_dist(0, op_list.size() - 1);
    std::uniform_int_distribution<size_t> inverse_dist(0, 1);
    std::uniform_real_distribution<TestType> param_dist(0.0, 2 * M_PI);
    std::uniform_int_distribution<size_t> wire_dist(0, num_qubits - 1);

    auto gen_param = [&param_dist, &re]() { return param_dist(re); };

    for (uint32_t k = 0; k < num_gate_reps; k++) {
        auto [op_name, gate_desc] = op_list[gate_dist(re)];

        std::vector<TestType> gate_params(gate_desc.n_params, 0.0);
        std::generate(gate_params.begin(), gate_params.end(), gen_param);

        random_gate_names.emplace_back(op_name);
        random_inverses.emplace_back(static_cast<bool>(inverse_dist(re)));
        // random_gate_wires.emplace_back(generateDistinctWires(re, num_qubits,
        // gate_desc.n_wires));
        random_gate_wires.emplace_back(
            generateNeighboringWires(re, num_qubits, gate_desc.n_wires));
        random_gate_parameters.emplace_back(std::move(gate_params));
    }

    // Log genereated sequence if LOG is turned on
    const char *env_p = std::getenv("LOG");
    try {
        if (env_p != nullptr && std::stoi(env_p) != 0) {
            for (size_t gate_rep = 0; gate_rep < num_gate_reps; gate_rep++) {
                std::cerr << random_gate_names[gate_rep] << ", "
                          << random_gate_wires[gate_rep] << ", "
                          << random_gate_parameters[gate_rep] << std::endl;
            }
        }
    } catch (std::exception &e) {
        // Just do not print log
    }

    // Run benchmark. Total num_gate_reps number of gates is used.
    Pennylane::StateVectorManaged<TestType> svdat{num_qubits};
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start;
    std::chrono::time_point<std::chrono::high_resolution_clock> t_end;
    t_start = std::chrono::high_resolution_clock::now();

    for (size_t gate_rep = 0; gate_rep < num_gate_reps; gate_rep++) {
        svdat.applyOperation(kernel, std::string(random_gate_names[gate_rep]),
                             random_gate_wires[gate_rep],
                             random_inverses[gate_rep],
                             random_gate_parameters[gate_rep]);
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
