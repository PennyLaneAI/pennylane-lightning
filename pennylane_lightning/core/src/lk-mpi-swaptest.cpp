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
}


std::vector<std::complex<double>> get_ascend_vector(const std::size_t nq) {
    constexpr std::size_t one{1U};
    std::size_t nsv = one << nq;
    std::vector<std::complex<double>> vec(nsv);
    for (std::size_t i = 0; i < vec.size(); i++) {
        vec[i] = std::complex<double>{static_cast<double>(i + 1)};
    }
    // normalize(vec);
    return vec;
}

std::pair<std::size_t, std::size_t>  prep_input_1q(int argc, char *argv[]) {
    if (argc <= 2) {
        std::cout << "Please ensure you specify the following arguments: "
                     "total_qubits swapping_qubit"
                  << std::endl;
        std::exit(-1);
    }
    std::string arg_qubits = argv[1];
    std::string arg_swapping_qubit = argv[2];
    std::size_t qubits = std::stoi(arg_qubits);
    std::size_t swapping_qubit = std::stoi(arg_swapping_qubit);

    return {qubits, swapping_qubit};
}




int main(int argc, char *argv[]) {


    auto inputs = prep_input_1q(argc, argv);
    std::size_t nq = inputs.first;
    std::size_t swapping = inputs.second;

    // Create PennyLane Lightning statevector
    StateVectorKokkosMPI<double> svmpi(nq);
    std::size_t repeats = 4;

    svmpi.swap_global_local_wires({swapping}, {nq - 1});
    svmpi.swap_global_local_wires({nq-1}, {swapping});

    
    const auto t_start = std::chrono::high_resolution_clock::now();   

    for (std::size_t i = 0; i < repeats; i++) {
        svmpi.swap_global_local_wires({swapping}, {nq - 1});
        svmpi.swap_global_local_wires({nq-1}, {swapping});
    }
    const auto t_end = std::chrono::high_resolution_clock::now();   
    const double t_duration = std::chrono::duration<double, t_scale>(t_end - t_start).count();  
    double average_time = t_duration / (2.0 * repeats); 
    double data_sent_GB = std::pow(2, svmpi.get_num_local_wires()) * 128 / 8 / 2 / 1024 / 1024 / 1024;
    std::cout << "Average time for swapping "  << average_time << " ms" << std::endl;  
    std::cout << "Data sent = Data received = " << data_sent_GB << " GB" << std::endl;
    std::cout << "Effective single direction bandwidth = " << data_sent_GB/average_time*1000.0 << " GB/s   - time include copying to/from buffer" << std::endl;
    





    int finflag;
    MPI_Finalized(&finflag);
    if (!finflag) {
        MPI_Finalize();
    }

    return 0;
}
