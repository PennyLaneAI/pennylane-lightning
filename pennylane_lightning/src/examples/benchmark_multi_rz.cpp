#include "ExampleUtil.hpp"
#include "StateVectorManaged.hpp"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>

using namespace Pennylane;
using namespace Pennylane::Gates;

constexpr uint32_t seed = 1337;

int main(int argc, char *argv[]) {
    using TestType = double;

    if (argc != 5) { // NOLINT(readability-magic-numbers)
        std::cout << "Usage: " << argv[0]
                  << " num_gate_reps num_qubits num_wires kernel" << std::endl;
        return 1;
    }

    size_t num_gate_reps;
    size_t num_qubits;
    size_t num_wires;

    try {
        num_gate_reps = std::stoi(argv[1]);
        num_qubits = std::stoi(argv[2]);
        num_wires = std::stoi(argv[3]);
    } catch (std::exception &e) {
        std::cerr << "Arguments must be integers." << std::endl;
        return 1;
    }

    std::string_view kernel_name = argv[4];
    KernelType kernel = string_to_kernel(kernel_name);
    if (kernel == KernelType::None) {
        std::cerr << "Kernel " << kernel_name << " is unknown." << std::endl;
        return 1;
    }

    std::mt19937 re{seed}; // NOLINT(readability-magic-number)
    std::uniform_real_distribution<double> param_dist(-M_PI, M_PI);

    std::vector<std::vector<size_t>> wires;
    std::vector<double> params;

    wires.reserve(num_gate_reps);
    params.reserve(num_gate_reps);

    for (size_t gate_rep = 0; gate_rep < num_gate_reps; gate_rep++) {
        wires.emplace_back(generateDistinctWires(re, num_qubits, num_wires));
        params.emplace_back(param_dist(re));
    }

    StateVectorManaged<TestType> sv{num_qubits};

    std::chrono::time_point<std::chrono::high_resolution_clock> t_start =
        std::chrono::high_resolution_clock::now();

    for (size_t gate_rep = 0; gate_rep < num_gate_reps; gate_rep++) {
        sv.applyOperation(kernel, "MultiRZ", wires[gate_rep], false,
                          {params[gate_rep]});
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> t_end =
        std::chrono::high_resolution_clock::now();
    const auto walltime =
        0.001 * ((std::chrono::duration_cast<std::chrono::microseconds>(
                      t_end - t_start))
                     .count());
    std::cout << num_qubits << ", "
              << walltime / static_cast<double>(num_gate_reps) << std::endl;

    return 0;
}
