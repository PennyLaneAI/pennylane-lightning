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
                     "total_qubits xor_rank"
                  << std::endl;
        std::exit(-1);
    }
    std::string arg_qubits = argv[1];
    std::string arg_xor_rank = argv[2];
    std::size_t qubits = std::stoi(arg_qubits);
    std::size_t xor_rank = std::stoi(arg_xor_rank);

    return {qubits, xor_rank};
}

int main(int argc, char *argv[]) {

    auto inputs = prep_input_1q(argc, argv);
    std::size_t nq = inputs.first;
    std::size_t xor_rank = inputs.second;

    // Create PennyLane Lightning statevector
    StateVectorKokkosMPI<double> svmpi(nq);
    std::size_t repeats = 8;
    std::size_t send_size = exp2(svmpi.get_num_local_wires() - 1);

    std::size_t my_rank = svmpi.get_mpi_rank();
    std::size_t dest_rank = my_rank ^ xor_rank;

    // Warmup
    svmpi.mpi_sendrecv(dest_rank, dest_rank, send_size, xor_rank);

    const auto t_start = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < repeats; i++) {
        svmpi.mpi_sendrecv(dest_rank, dest_rank, send_size, xor_rank);
    }
    const auto t_end = std::chrono::high_resolution_clock::now();
    const double t_duration =
        std::chrono::duration<double, t_scale>(t_end - t_start).count();
    double average_time = t_duration / (repeats);
    double data_sent_GB = send_size * 128 / 8 / 1024 / 1024 / 1024;
    std::cout << "Average time for swapping " << average_time
              << " ms (My rank:" << my_rank << " dest rank:" << dest_rank << ")"
              << std::endl;
    std::cout << "Data sent = Data received = " << data_sent_GB
              << " GB (My rank:" << my_rank << " dest rank:" << dest_rank << ")"
              << std::endl;
    std::cout << "Effective single direction bandwidth = "
              << data_sent_GB / average_time * 1000.0
              << " GB/s (My rank:" << my_rank << " dest rank:" << dest_rank
              << ")" << std::endl;

    int finflag;
    MPI_Finalized(&finflag);
    if (!finflag) {
        MPI_Finalize();
    }

    return 0;
}
