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
#include "StateVectorKokkos.hpp"
#include "StateVectorKokkosMPI.hpp"

#include "output_utils.hpp"

namespace {
using namespace Pennylane;
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::Gates;
using namespace Pennylane::Gates::Constant;
using namespace Pennylane::Util;
using namespace BMUtils;
using t_scale = std::milli;

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
[[maybe_unused]] void print(const std::vector<ComplexT> &vec,
                            const std::string &name = "vector") {
    std::cout << "Vector : " << name << " = np.array([" << std::endl;
    for (auto &e : vec) {
        std::cout << real(e) << " + 1j * " << imag(e) << std::endl;
    }
    std::cout << "])" << std::endl;
}

[[maybe_unused]] void print(StateVectorKokkosMPI<double> sv,
                            const std::string &name = "statevector") {
    auto data = sv.getDataVector();
    if (0 == sv.get_mpi_rank()) {
        print(data, name);
    }
}

[[maybe_unused]] void print_basis_states(const std::size_t n) {
    constexpr std::size_t one{1U};
    for (std::size_t i = 0; i < one << n; i++) {
        StateVectorKokkosMPI<double> sv(n);
        sv.setBasisState(i);
        print(sv, "basis-" + std::to_string(i));
    }
}

[[maybe_unused]] void print_local_wires(const std::size_t n) {
    for (std::size_t i = 0; i < n; i++) {
        StateVectorKokkosMPI<double> sv(n);
        if (sv.get_mpi_rank() == 0) {
            std::cout << "wire-" << std::to_string(i)
                      << " locality = " << sv.is_wires_local({i}) << std::endl;
        }
    }
}

[[maybe_unused]] void allclose(StateVectorKokkosMPI<double> svmpi,
                               StateVectorKokkos<double> sv) {
    [[maybe_unused]] constexpr double tol = 1.0e-6;
    auto dat = svmpi.getDataVector();
    if (svmpi.get_mpi_rank() == 0) {
        auto ref = sv.getDataVector();
        PL_ABORT_IF_NOT(dat.size() == ref.size(), "Wrong statevector size.");
        for (std::size_t i = 0; i < ref.size(); i++) {
            auto diff = dat[i] - ref[i];
            [[maybe_unused]] double err =
                norm(std::complex<double>{real(diff), imag(diff)});
            // std::cout << err << std::endl;
            PL_ABORT_IF_NOT(err < tol, "Wrong statevector entry.");
        }
    }
    svmpi.barrier();
}

} // namespace

int main(int argc, char *argv[]) {
    auto indices = prep_input_1q<unsigned int>(argc, argv);
    constexpr std::size_t run_avg = 1;
    std::vector<std::string> gates = {
        "Identity", "PauliX",     "PauliY", "PauliZ", "Hadamard", "S",
        "T",        "PhaseShift", "RX",     "RY",     "RZ",       "Rot"};
    std::size_t nq = indices.q;
    std::vector<std::complex<double>> sv_data = get_ascend_vector(nq);

    // Create PennyLane Lightning statevector
    StateVectorKokkos<double> sv(sv_data);
    StateVectorKokkosMPI<double> svmpi(sv_data);
    allclose(svmpi, sv);
    // print(svmpi);
    // print_basis_states(indices.q);
    print_local_wires(indices.q);

    // Create vector for run-times to average
    std::vector<double> times;
    times.reserve(run_avg);
    // std::vector<std::size_t> targets{indices.t};

    // Test 1q-gates
    for (auto &gate : gates) {
        for (auto inverse : std::vector<bool>({false, true})) {
            for (std::size_t target = 0; target < indices.q; target++) {
                auto gate_op =
                    reverse_lookup(gate_names, std::string_view{gate});
                auto npar = lookup(gate_num_params, gate_op);
                std::vector<double> params(npar, 0.1);
                TIMING(sv.applyOperation(gate, {target}, inverse, params));
                TIMING(svmpi.applyOperation(gate, {target}, inverse, params));
                allclose(svmpi, sv);
            }
            if (svmpi.get_mpi_rank() == 0) {
                CSVOutput<decltype(indices), t_scale> csv(indices, gate,
                                                          average_times(times));
                std::cout << csv << std::endl;
            }
        }
    }
    // Test 1q-unitary
    for (auto inverse : std::vector<bool>({false, true})) {
        for (std::size_t target = 0; target < indices.q; target++) {
            std::vector<Kokkos::complex<double>> matrix = {
                {0.97517033, 0.19767681},
                {-0.09933467, 0.00996671},
                {0.09933467, 0.00996671},
                {0.97517033, 0.19767681}}; // qml.Rot(0.1,0.2,0.3,wires=[0])
            TIMING(sv.applyOperation("Matrix", {target}, inverse, {}, matrix));
            TIMING(
                svmpi.applyOperation("Matrix", {target}, inverse, {}, matrix));
            allclose(svmpi, sv);
        }
        if (svmpi.get_mpi_rank() == 0) {
            CSVOutput<decltype(indices), t_scale> csv(indices, "Matrix",
                                                      average_times(times));
            std::cout << csv << std::endl;
        }
    }
    svmpi.barrier();
    int finflag;
    MPI_Finalized(&finflag);
    if (!finflag) {
        MPI_Finalize();
    }

    return 0;
}
