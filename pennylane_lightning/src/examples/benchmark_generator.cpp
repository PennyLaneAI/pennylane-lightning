#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>

#include "Constant.hpp"
#include "DynamicDispatcher.hpp"
#include "ExampleUtil.hpp"
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

auto generatorOp(const std::string_view &name) -> Gates::GeneratorOperation {
    auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
    return dispatcher.strToGeneratorOp(std::string(name));
}

struct GeneratorDesc {
    std::string name;
    std::vector<size_t> wires;
    bool inverse;

    template <typename Arg0, typename Arg1, typename Arg2>
    GeneratorDesc(Arg0 &&arg0, Arg1 &&arg1, Arg2 &&arg2)
        : name{std::forward<Arg0>(arg0)}, wires{std::forward<Arg1>(arg1)},
          inverse{std::forward<Arg2>(arg2)} {}
};

std::ostream &operator<<(std::ostream &os, GeneratorDesc &desc) {
    os << desc.name << ", " << desc.wires << "," << desc.inverse << std::endl;
    return os;
}

template <class RandomEngine>
auto generateGeneratorSequence(RandomEngine &re,
                               const GeneratorOperation gntr_op,
                               const size_t num_reps, const size_t num_qubits,
                               const size_t num_wires_for_multi_qubit)
    -> std::vector<GeneratorDesc> {
    namespace Constant = Gates::Constant;
    using Gates::GeneratorOperation;

    const auto gntr_name =
        Util::lookup(Constant::generator_names, gntr_op).substr(9);

    const size_t num_wires = [=]() {
        if (Util::array_has_elt(Constant::multi_qubit_generators, gntr_op)) {
            // if multi qubit gate
            return num_wires_for_multi_qubit;
        }
        return Util::lookup(Constant::generator_wires, gntr_op);
    }();

    std::vector<GeneratorDesc> gntr_seq;
    std::uniform_int_distribution<size_t> inverse_dist(0, 1);

    for (uint32_t k = 0; k < num_reps; k++) {

        bool inverse = static_cast<bool>(inverse_dist(re));
        auto wires = generateNeighboringWires(re, num_qubits, num_wires);

        gntr_seq.emplace_back(gntr_name, std::move(wires), inverse);
    }
    return gntr_seq;
}

double benchmarkGenerator(KernelType kernel, const size_t num_qubits,
                          const std::vector<GeneratorDesc> &gntr_seq) {
    // Run benchmark. Total num_reps number of gates is used.
    StateVectorManagedCPU<PrecisionT> svdat{num_qubits};

    std::chrono::time_point<std::chrono::high_resolution_clock> t_start =
        std::chrono::high_resolution_clock::now();
    for (const auto &gntr : gntr_seq) {
        [[maybe_unused]] PrecisionT scale =
            svdat.applyGenerator(kernel, gntr.name, gntr.wires, gntr.inverse);
    }
    std::chrono::time_point<std::chrono::high_resolution_clock> t_end =
        std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

template <typename RandomEngine>
double runBenchmarkGenerator(RandomEngine &re, KernelType kernel,
                             const GeneratorOperation gntr_op, size_t num_reps,
                             size_t num_qubits,
                             size_t num_wires_for_multi_qubit) {
    auto gntr_seq = generateGeneratorSequence(re, gntr_op, num_reps, num_qubits,
                                              num_wires_for_multi_qubit);

    // Log generated sequence if LOG is turned on
    const char *env_p = std::getenv("LOG");
    try {
        if (env_p != nullptr && std::stoi(env_p) != 0) {
            for (const auto &gntr : gntr_seq) {
                std::cerr << gntr.name << ", " << gntr.wires << ","
                          << gntr.inverse << std::endl;
            }
        }
    } catch (std::exception &e) {
        // Just do not print log
    }

    return benchmarkGenerator(kernel, num_qubits, gntr_seq);
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
    if (argc != 5 && argc != 6) { // NOLINT(readability-magic-numbers)
        std::cerr
            << "Wrong number of inputs. User provided " << argc - 1
            << " inputs. \n"
            << "Usage: " + std::string(argv[0]) +
                   " num_reps num_qubits kernel [generator|gate] [num_wires]\n"
                   "Examples: \n"
            << "\t" << argv[0] << " 1000 10 PI GeneratorCRX\n"
            << "\t" << argv[0] << " 1000 10 LM CRX\n"
            << "\t" << argv[0] << " 1000 10 LM MutliRZ 3\n";
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

    std::string_view gntr_name = argv[4];
    Gates::GeneratorOperation gntr_op;

    try {
        gntr_op = generatorOp(gntr_name);
    } catch (std::exception &e) {
        std::cout << "Unknown generator " + std::string(gntr_name) + " provided"
                  << std::endl;
        return 1;
    }

    size_t num_wires_for_multi_qubit = 0;
    if (Util::array_has_elt(Constant::multi_qubit_generators, gntr_op)) {
        // User provided a multi-qubit gates
        if (argc != 6) { // NOLINT(readability-magic-numbers)
            std::cerr << "One should provide the number of wires when using "
                         "multi qubit generators."
                      << std::endl;
            return 1;
        }

        try {
            // NOLINTNEXTLINE(readability-magic-numbers)
            num_wires_for_multi_qubit = std::stoi(argv[5]);
        } catch (std::exception &e) {
            std::cerr << "Number of wires must be an integer" << std::endl;
            return 1;
        }
    }

    std::random_device rd;
    std::mt19937 re(rd());

    double walltime = runBenchmarkGenerator(
        re, kernel, gntr_op, num_reps, num_qubits, num_wires_for_multi_qubit);

    // Output walltime in csv format (Num Qubits, Time (milliseconds))
    std::cout << num_qubits << ", " << walltime / static_cast<double>(num_reps)
              << std::endl;
    return 0;
}
