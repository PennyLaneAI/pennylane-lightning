#include <algorithm>
#include <chrono>
#include <complex>
#include <iostream>
#include <numeric>
#include <vector>

#include "StateVectorKokkos.hpp"
// #include "StateVectorKokkosMPI.hpp"

#include "output_utils.hpp"

namespace {
    using namespace Pennylane;
    using namespace Pennylane::LightningKokkos;
    using namespace Pennylane::Gates;
    using t_scale = std::milli;
    using namespace BMUtils;
    // using SVMPI = StateVectorKokkosMPI;
} // namespace

int main(int argc, char *argv[]) {
    auto indices = prep_input_1q<unsigned int>(argc, argv);
    constexpr std::size_t run_avg = 1;
    std::string gate = "Hadamard";

    // Create PennyLane Lightning statevector
    StateVectorKokkos<double> sv(indices.q);

    // Create vector for run-times to average
    std::vector<double> times;
    times.reserve(run_avg);
    std::vector<std::size_t> targets{indices.t};

    // Apply the gates `run_avg` times on the indicated targets
    for (std::size_t i = 0; i < run_avg; i++) {
        TIMING(sv.applyOperation(gate, targets));
    }

    CSVOutput<decltype(indices), t_scale> csv(indices, gate,
                                              average_times(times));
    std::cout << csv << std::endl;
    return 0;
}
