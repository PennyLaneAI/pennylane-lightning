#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <string_view>
#include <vector>

#include <mpi.h>

#include "Constant.hpp"
#include "ConstantUtil.hpp"
#include "Error.hpp"
#include "MeasurementsKokkos.hpp"
#include "MeasurementsKokkosMPI.hpp"
#include "ObservablesKokkos.hpp"
#include "StateVectorKokkos.hpp"
#include "StateVectorKokkosMPI.hpp"
#include <iostream>

namespace {
using namespace Pennylane;
using namespace Pennylane::Gates;
using namespace Pennylane::Gates::Constant;
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::LightningKokkos::Measures;
using namespace Pennylane::LightningKokkos::Observables;
using namespace Pennylane::Util;
using t_scale = std::milli;
} // namespace

std::pair<std::size_t, std::size_t> prep_input_1q(int argc, char *argv[]) {
    if (argc <= 2) {
        std::cout << "Please ensure you specify the following arguments: "
                     "total_qubits swapping_global_qubit"
                  << std::endl;
        std::exit(-1);
    }
    std::string arg_qubits = argv[1];
    std::string arg_swapping_global_qubit = argv[2];
    std::size_t qubits = std::stoi(arg_qubits);
    std::size_t swapping_global_qubit = std::stoi(arg_swapping_global_qubit);

    return {qubits, swapping_global_qubit};
}

int main(int argc, char *argv[]) {
    if (argc <= 3) {
        std::cout << "Please ensure you specify the following arguments: "
                     "total_qubits swapping_global_qubit"
                  << std::endl;
        std::exit(-1);
    }

    std::string arg_qubits = argv[1];
    std::string arg_swapping_global_qubit = argv[2];
    std::string arg_swapping_local_qubit = argv[3];

    std::size_t nq = std::stoi(arg_qubits);
    std::size_t swapping_global_qubit = std::stoi(arg_swapping_global_qubit);
    std::size_t swapping_local_qubit = std::stoi(arg_swapping_local_qubit);

    // Create PennyLane Lightning statevector
    StateVectorKokkosMPI<double> svmpi(nq);
    std::size_t repeats = 4;

    svmpi.swapGlobalLocalWires({swapping_global_qubit}, {swapping_local_qubit});
    svmpi.swapGlobalLocalWires({swapping_local_qubit}, {swapping_global_qubit});

    const auto t_start = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < repeats; i++) {
        svmpi.swapGlobalLocalWires({swapping_global_qubit},
                                   {swapping_local_qubit});
        svmpi.swapGlobalLocalWires({swapping_local_qubit},
                                   {swapping_global_qubit});
    }
    const auto t_end = std::chrono::high_resolution_clock::now();
    const double t_duration =
        std::chrono::duration<double, t_scale>(t_end - t_start).count();
    double average_time = t_duration / (2.0 * repeats);
    double data_sent_GB =
        exp2(svmpi.getNumLocalWires() - 1) * 128 / 8 / 1024 / 1024 / 1024;
    std::cout << "Average time for swapping " << average_time << " ms"
              << std::endl;
    std::cout << "Data sent = Data received = " << data_sent_GB << " GB"
              << std::endl;
    std::cout << "Effective single direction bandwidth = "
              << data_sent_GB / average_time * 1000.0
              << " GB/s   - time include copying to/from buffer" << std::endl;

    int finflag;
    MPI_Finalized(&finflag);
    if (!finflag) {
        MPI_Finalize();
    }

    return 0;
}
