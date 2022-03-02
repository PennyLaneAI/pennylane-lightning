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
#include "LinearAlgebra.hpp"
#include "StateVectorManagedCPU.hpp"

#ifdef USE_SINGLE_PRECISION
using PrecisionT = float;
#pragma message "Using single precision"
#else
using PrecisionT = double;
#endif

using namespace Pennylane;
using namespace Pennylane::Gates;
using namespace Pennylane::Util;

struct MatOpDesc {
    std::vector<size_t> wires;
    bool inverse;
    std::vector<std::complex<PrecisionT>> mat;

    template <typename Arg0, typename Arg1, typename Arg2>
    MatOpDesc(Arg0 &&arg0, Arg1 &&arg1, Arg2 &&arg2)
        : wires{std::forward<Arg0>(arg0)}, inverse{std::forward<Arg1>(arg1)},
          mat{std::forward<Arg2>(arg2)} {}
};

template <class RandomEngine>
auto generateMatrixSequence(RandomEngine &re, const size_t num_reps,
                            const size_t num_qubits, const size_t num_wires)
    -> std::vector<MatOpDesc> {

    std::vector<MatOpDesc> matrix_seq;
    matrix_seq.reserve(num_reps);
    std::uniform_int_distribution<size_t> inverse_dist(0, 1);
    for (uint32_t k = 0; k < num_reps; k++) {
        bool inverse = static_cast<bool>(inverse_dist(re));
        auto wires = generateNeighboringWires(re, num_qubits, num_wires);

        matrix_seq.emplace_back(std::move(wires), inverse,
                                Util::randomUnitary<PrecisionT>(re, num_wires));
    }
    return matrix_seq;
}

double benchmarkMatrix(KernelType kernel, const size_t num_qubits,
                       const std::vector<MatOpDesc> &mat_seq) {
    // Run benchmark. Total num_reps number of gates is used.
    StateVectorManagedCPU<PrecisionT> svdat{num_qubits};

    std::chrono::time_point<std::chrono::high_resolution_clock> t_start =
        std::chrono::high_resolution_clock::now();
    for (const auto &mat_desc : mat_seq) {
        svdat.applyMatrix(kernel, mat_desc.mat.data(), mat_desc.wires,
                          mat_desc.inverse);
    }
    std::chrono::time_point<std::chrono::high_resolution_clock> t_end =
        std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

template <typename RandomEngine>
double runBenchmarkMatrix(RandomEngine &re, KernelType kernel, size_t num_reps,
                          size_t num_qubits, size_t num_wires) {
    auto mat_seq = generateMatrixSequence(re, num_reps, num_qubits, num_wires);

    // Log generated sequence if LOG is turned on
    const char *env_p = std::getenv("LOG");
    try {
        if (env_p != nullptr && std::stoi(env_p) != 0) {
            for (const auto &mat_desc : mat_seq) {
                std::cerr << mat_desc.wires << ", " << mat_desc.inverse << ", "
                          << mat_desc.mat << std::endl;
            }
        }
    } catch (std::exception &e) {
        // Just do not print log
    }

    return benchmarkMatrix(kernel, num_qubits, mat_seq);
}

/**
 * @brief Benchmark Pennylane-Lightning for a given generator
 *
 * @param argc Number of arguments
 * @param argv Command line arguments
 * @return Returns 0 is completed successfully
 */
int main(int argc, char *argv[]) {
    namespace Constant = Gates::Constant;
    // Handle input
    if (argc != 5) { // NOLINT(readability-magic-numbers)
        std::cerr << "Wrong number of inputs. User provided " << argc - 1
                  << " inputs. \n"
                  << "Usage: " + std::string(argv[0]) +
                         " num_reps num_qubits kernel num_wires\n"
                         "Examples: \n"
                  << "\t" << argv[0] << " 1000 10 PI 4\n";
        return -1;
    }

    size_t num_reps;
    size_t num_qubits;
    size_t num_wires;

    try {
        num_reps = std::stoi(argv[1]);
        num_qubits = std::stoi(argv[2]);
        num_wires = std::stoi(argv[4]);
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

    std::random_device rd;
    std::mt19937 re(rd());

    double walltime =
        runBenchmarkMatrix(re, kernel, num_reps, num_qubits, num_wires);

    // Output walltime in csv format (Num Qubits, Time (milliseconds))
    std::cout << num_qubits << ", " << walltime / static_cast<double>(num_reps)
              << std::endl;
    return 0;
}
