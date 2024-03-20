#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <vector>

#include <mpi.h>

#include "StateVectorKokkos.hpp"
#include "StateVectorKokkosMPI.hpp"

#include "output_utils.hpp"

namespace {
using namespace Pennylane;
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::Gates;
using t_scale = std::milli;
using namespace BMUtils;

void normalize(std::vector<std::complex<double>> &vec) {
    double sum{0.0};
    for (std::size_t i = 0; i < vec.size(); i++) {
        sum += norm(vec[i]);
    }
    sum = std::sqrt(sum);
    for (std::size_t i = 0; i < vec.size(); i++) {
        vec[i] /= sum;
    }
}

std::vector<std::complex<double>> get_ascend_vector(const std::size_t nq) {
    constexpr std::size_t one{1U};
    std::size_t nsv = one << nq;
    std::vector<std::complex<double>> vec(nsv);
    for (std::size_t i = 0; i < vec.size(); i++) {
        vec[i] = std::complex<double>{static_cast<double>(i + 1)};
    }
    normalize(vec);
    return vec;
}

template <class ComplexT>
void print(const std::vector<ComplexT> &vec,
           const std::string &name = "vector") {
    std::cout << "Vector : " << name << " = np.array([" << std::endl;
    for (auto &e : vec) {
        std::cout << real(e) << "+ 1j * " << imag(e) << std::endl;
    }
    std::cout << "])" << std::endl;
}

// void print_basis_states(const std::size_t n) {
//     constexpr std::size_t one{1U};
//     for (std::size_t i = 0; i < one << n; i++) {
//         StateVectorKokkosMPI<double> sv(n);
//         sv.setBasisState(i);
//         for (std::size_t rank = 0; rank < sv.get_mpi_size(); i++) {
//             if (rank == sv.get_mpi_rank()) {
//                 print(sv.getDataVector(), "basis-"+std::to_string(rank)+"-"+std::to_string(i));
//             }
//             sv.mpi_barrier();
//         }
//     }
// }

} // namespace

int main(int argc, char *argv[]) {
printf("line_96\n");    auto indices = prep_input_1q<unsigned int>(argc, argv);
printf("line_95\n");    constexpr std::size_t run_avg = 1;
printf("line_94\n");    std::string gate = "Hadamard";
printf("line_93\n");    std::size_t nq = indices.q;
printf("line_92\n");    std::vector<std::complex<double>> sv_data = get_ascend_vector(nq);
printf("line_91\n");
printf("line_90\n");    // Create PennyLane Lightning statevector
printf("line_72\n");    StateVectorKokkos<double> sv(sv_data);
printf("line_73\n");    StateVectorKokkosMPI<double> svmpi(indices.q);
printf("line_74\n");
printf("line_75\n");    // print_basis_states(indices.q);
printf("line_76\n");
printf("line_77\n");    // Create vector for run-times to average
printf("line_78\n");    std::vector<double> times;
printf("line_79\n");    times.reserve(run_avg);
printf("line_80\n");    std::vector<std::size_t> targets{indices.t};
printf("line_81\n");
printf("line_82\n");    // Apply the gates `run_avg` times on the indicated targets
printf("line_83\n");    for (std::size_t i = 0; i < run_avg; i++) {
printf("line_84\n");        TIMING(sv.applyOperation(gate, targets));
printf("line_85\n");    }
printf("line_86\n");
printf("line_87\n");    CSVOutput<decltype(indices), t_scale> csv(indices, gate,
                                              average_times(times));
printf("line_89\n");    std::cout << csv << std::endl;
    return 0;
}
