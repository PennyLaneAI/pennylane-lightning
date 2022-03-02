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

struct GateDesc {
    std::string name;
    std::vector<size_t> wires;
    bool inverse;
    std::vector<PrecisionT> params;

    template <typename Arg0, typename Arg1, typename Arg2, typename Arg3>
    GateDesc(Arg0 &&arg0, Arg1 &&arg1, Arg2 &&arg2, Arg3 &&arg3)
        : name{std::forward<Arg0>(arg0)}, wires{std::forward<Arg1>(arg1)},
          inverse{std::forward<Arg2>(arg2)}, params{std::forward<Arg3>(arg3)} {}
};

std::ostream &operator<<(std::ostream &os, GateDesc &desc) {
    os << desc.name << ", " << desc.wires << "," << desc.inverse << ","
       << desc.params << std::endl;
    return os;
}

template <class RandomEngine>
auto generateGateSequence(RandomEngine &re, const std::string &gate_name,
                          const size_t num_reps, const size_t num_qubits,
                          const size_t num_wires_for_multi_qubit)
    -> std::vector<GateDesc> {
    using Gates::Constant::multi_qubit_gates;

    const GateOperation gate_op = Util::lookup(
        Util::reverse_pairs(Constant::gate_names), std::string_view(gate_name));
    const size_t num_wires = [=]() {
        if (Util::array_has_elt(multi_qubit_gates, gate_op)) {
            // if multi qubit gate
            return num_wires_for_multi_qubit;
        }
        return Util::lookup(Constant::gate_wires, gate_op);
    }();
    const size_t num_params = Util::lookup(Constant::gate_num_params, gate_op);

    std::vector<GateDesc> gate_seq;
    std::uniform_int_distribution<size_t> inverse_dist(0, 1);
    std::uniform_real_distribution<PrecisionT> param_dist(0.0, 2 * M_PI);

    for (uint32_t k = 0; k < num_reps; k++) {
        std::vector<PrecisionT> params;
        params.reserve(num_params);

        bool inverse = static_cast<bool>(inverse_dist(re));
        auto wires = generateNeighboringWires(re, num_qubits, num_wires);

        for (size_t idx = 0; idx < num_params; idx++) {
            params.emplace_back(param_dist(re));
        }

        gate_seq.emplace_back(gate_name, std::move(wires), inverse,
                              std::move(params));
    }
    return gate_seq;
}

double benchmarkGate(KernelType kernel, const size_t num_qubits,
                     const std::vector<GateDesc> &gate_seq) {
    // Run benchmark. Total num_reps number of gates is used.
    StateVectorManagedCPU<PrecisionT> svdat{num_qubits};

    std::chrono::time_point<std::chrono::high_resolution_clock> t_start =
        std::chrono::high_resolution_clock::now();
    for (const auto &gate : gate_seq) {
        svdat.applyOperation(kernel, gate.name, gate.wires, gate.inverse,
                             gate.params);
    }
    std::chrono::time_point<std::chrono::high_resolution_clock> t_end =
        std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

template <typename RandomEngine>
double runBenchmarkGate(RandomEngine &re, KernelType kernel,
                        const std::string &gate_name, size_t num_reps,
                        size_t num_qubits, size_t num_wires_for_multi_qubit) {
    auto gate_seq = generateGateSequence(re, gate_name, num_reps, num_qubits,
                                         num_wires_for_multi_qubit);

    // Log generated sequence if LOG is turned on
    const char *env_p = std::getenv("LOG");
    try {
        if (env_p != nullptr && std::stoi(env_p) != 0) {
            for (const auto &gate : gate_seq) {
                std::cerr << gate.name << ", " << gate.wires << ","
                          << gate.inverse << "," << gate.params << std::endl;
            }
        }
    } catch (std::exception &e) {
        // Just do not print log
    }

    return benchmarkGate(kernel, num_qubits, gate_seq);
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

    std::string_view gate_name = argv[4];
    if (!Util::array_has_elt(Util::second_elts_of(Constant::gate_names),
                             gate_name)) {
        std::cerr << "Unknown gate name " << gate_name << " is provided"
                  << std::endl;
        return 1;
    }

    Gates::GateOperation gate_op =
        Util::lookup(Util::reverse_pairs(Constant::gate_names), gate_name);

    size_t num_wires_for_multi_qubit = 0;
    if (Util::array_has_elt(Constant::multi_qubit_gates, gate_op)) {
        // User provided a multi-qubit gates
        if (argc != 6) { // NOLINT(readability-magic-numbers)
            std::cerr << "One should provide the number of wires when using "
                         "multi qubit gates."
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

    double walltime =
        runBenchmarkGate(re, kernel, std::string(gate_name), num_reps,
                         num_qubits, num_wires_for_multi_qubit);

    // Output walltime in csv format (Num Qubits, Time (milliseconds))
    std::cout << num_qubits << ", " << walltime / static_cast<double>(num_reps)
              << std::endl;
    return 0;
}
