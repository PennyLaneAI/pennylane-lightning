#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>

#include "Constant.hpp"
#include "ExampleUtil.hpp"
#include "StateVectorCPU.hpp"

#ifdef USE_SINGLE_PRECISION
using PrecisionT = float;
#pragma message "Using single precision"
#else
using PrecisionT = double;
#endif

using namespace Pennylane;
using namespace Pennylane::Gates;
using namespace Pennylane::Util;

std::string_view strip(std::string_view str) {
    auto start = str.find_first_not_of(" \t");
    auto end = str.find_last_not_of(" \t");
    return str.substr(start, end - start + 1);
}

template <class RandomEngine>
double benchmark_gate(RandomEngine &re, KernelType kernel,
                      const std::string &gate_name, const size_t num_reps,
                      const size_t num_qubits) {
    const GateOperation gate_op = Util::lookup(
        Util::reverse_pairs(Constant::gate_names), std::string_view(gate_name));
    const size_t num_wires = Util::lookup(Constant::gate_wires, gate_op);
    const size_t num_params = Util::lookup(Constant::gate_num_params, gate_op);

    // Generate random generator sequences
    std::vector<std::vector<size_t>> random_wires;
    std::vector<bool> random_inverses;
    std::vector<std::vector<PrecisionT>> random_params;
    random_wires.reserve(num_reps);
    random_inverses.reserve(num_reps);
    random_params.reserve(num_reps);

    std::uniform_int_distribution<size_t> inverse_dist(0, 1);
    std::uniform_real_distribution<PrecisionT> param_dist(0.0, 2 * M_PI);

    for (uint32_t k = 0; k < num_reps; k++) {
        std::vector<PrecisionT> gate_params;
        gate_params.reserve(num_params);

        random_inverses.emplace_back(static_cast<bool>(inverse_dist(re)));
        random_wires.emplace_back(
            generateNeighboringWires(re, num_qubits, num_wires));

        for (size_t idx = 0; idx < num_params; idx++) {
            gate_params.emplace_back(param_dist(re));
        }
        random_params.emplace_back(std::move(gate_params));
    }

    // Log generated sequence if LOG is turned on
    const char *env_p = std::getenv("LOG");
    try {
        if (env_p != nullptr && std::stoi(env_p) != 0) {
            for (size_t gate_rep = 0; gate_rep < num_reps; gate_rep++) {
                std::cerr << gate_name << ", " << random_wires[gate_rep] << ","
                          << random_inverses[gate_rep] << ","
                          << random_params[gate_rep] << std::endl;
            }
        }
    } catch (std::exception &e) {
        // Just do not print log
    }

    // Run benchmark. Total num_reps number of gates is used.
    StateVectorCPU<PrecisionT> svdat{num_qubits};

    std::chrono::time_point<std::chrono::high_resolution_clock> t_start =
        std::chrono::high_resolution_clock::now();
    for (size_t gate_rep = 0; gate_rep < num_reps; gate_rep++) {
        svdat.applyOperation(kernel, gate_name, random_wires[gate_rep],
                             random_inverses[gate_rep],
                             random_params[gate_rep]);
    }
    std::chrono::time_point<std::chrono::high_resolution_clock> t_end =
        std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

template <class RandomEngine>
double benchmark_generator(RandomEngine &re, KernelType kernel,
                           const std::string &gntr_name, const size_t num_reps,
                           const size_t num_qubits) {
    const auto gntr_name_without_prefix = gntr_name.substr(9);
    const GeneratorOperation gntr_op =
        Util::lookup(Util::reverse_pairs(Constant::generator_names),
                     std::string_view(gntr_name));
    const size_t num_wires = Util::lookup(Constant::generator_wires, gntr_op);

    // Generate random generator sequences
    std::vector<std::vector<size_t>> random_wires;
    std::vector<bool> random_inverses;
    random_wires.reserve(num_reps);
    random_inverses.reserve(num_reps);

    std::uniform_int_distribution<size_t> inverse_dist(0, 1);

    for (uint32_t k = 0; k < num_reps; k++) {
        random_inverses.emplace_back(static_cast<bool>(inverse_dist(re)));
        random_wires.emplace_back(
            generateNeighboringWires(re, num_qubits, num_wires));
    }

    // Log generated sequence if LOG is turned on
    const char *env_p = std::getenv("LOG");
    try {
        if (env_p != nullptr && std::stoi(env_p) != 0) {
            for (size_t gate_rep = 0; gate_rep < num_reps; gate_rep++) {
                std::cerr << gntr_name << ", " << random_wires[gate_rep] << ","
                          << random_inverses[gate_rep] << std::endl;
            }
        }
    } catch (std::exception &e) {
        // Just do not print log
    }

    // Run benchmark. Total num_reps number of gates is used.
    StateVectorCPU<PrecisionT> svdat{num_qubits};

    std::chrono::time_point<std::chrono::high_resolution_clock> t_start =
        std::chrono::high_resolution_clock::now();
    for (size_t gate_rep = 0; gate_rep < num_reps; gate_rep++) {
        [[maybe_unused]] auto scale = svdat.applyGenerator(
            kernel, gntr_name_without_prefix, random_wires[gate_rep],
            random_inverses[gate_rep]);
    }
    std::chrono::time_point<std::chrono::high_resolution_clock> t_end =
        std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

/**
 * @brief Benchmark Pennylane-Lightning for a given generator
 *
 * @param argc Number of arguments
 * @param argv Command line arguments
 * @return Returns 0 is completed successfully
 */
int main(int argc, char *argv[]) {
    // Handle input
    if (argc < 5) { // NOLINT(readability-magic-numbers)
        std::cerr << "Wrong number of inputs. User provided " << argc - 1
                  << " inputs. \n"
                  << "Usage: " + std::string(argv[0]) +
                         " num_reps num_qubits kernel [generator|gate]\n"
                         "Examples: \n"
                         "\t"
                  << argv[0] << " 1000 10 PI GeneratorCRX\n"
                  << "\t" << argv[0] << " 1000 10 LM CRX"
                  << std::endl; // Change to std::format in C++20
        return -1;
    }

    size_t num_reps;
    size_t num_qubits;

    try {
        num_reps = std::stoi(argv[1]);
        num_qubits = std::stoi(argv[2]);
    } catch (std::exception &e) {
        std::cerr << "Arguments num_reps and num_qubits must be integers."
                  << std::endl;
        return -1;
    }

    std::string_view kernel_name = argv[3];
    KernelType kernel = string_to_kernel(kernel_name);
    if (kernel == KernelType::None) {
        std::cerr << "Kernel " << kernel_name << " is unknown." << std::endl;
        return 1;
    }

    const std::string_view gate_or_gntr_name = argv[4];
    const std::string_view generator_prefix = "Generator";

    std::random_device rd;
    std::mt19937 re(rd());

    double walltime;

    if (gate_or_gntr_name.substr(0, generator_prefix.length()) ==
        generator_prefix) { // generators
        walltime = benchmark_generator(
            re, kernel, std::string(gate_or_gntr_name), num_reps, num_qubits);
    } else {
        walltime = benchmark_gate(re, kernel, std::string(gate_or_gntr_name),
                                  num_reps, num_qubits);
    }

    // Output walltime in csv format (Num Qubits, Time (milliseconds))
    std::cout << num_qubits << ", " << walltime / static_cast<double>(num_reps)
              << std::endl;
    return 0;
}
