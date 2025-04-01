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

#include <iostream>
#include "Constant.hpp"
#include "ConstantUtil.hpp"
#include "Error.hpp"
#include "MeasurementsKokkos.hpp"
//#include "MeasurementsKokkosMPI.hpp"
#include "ObservablesKokkos.hpp"
#include "StateVectorKokkos.hpp"
#include "StateVectorKokkosMPI.hpp"


namespace {
using namespace Pennylane;
using namespace Pennylane::Gates;
using namespace Pennylane::Gates::Constant;
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::LightningKokkos::Measures;
using namespace Pennylane::LightningKokkos::Observables;
using namespace Pennylane::Util;
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
    auto data = sv.getDataVector(0);
    if (0 == sv.get_mpi_rank()) {
        print(data, name);
    }
}

/* [[maybe_unused]] void print_basis_states(const std::size_t n) {
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
} */

[[maybe_unused]] void allclose(StateVectorKokkosMPI<double> svmpi,
                               StateVectorKokkos<double> sv) {
    [[maybe_unused]] constexpr double tol = 1.0e-6;
    auto dat = svmpi.getDataVector(0);
    if (svmpi.get_mpi_rank() == 0) {
        auto ref = sv.getDataVector();
        PL_ABORT_IF_NOT(dat.size() == ref.size(), "Wrong statevector size.");
        for (std::size_t i = 0; i < ref.size(); i++) {
            auto diff = dat[i] - ref[i];
            [[maybe_unused]] double err =
                norm(std::complex<double>{real(diff), imag(diff)});
            if (err > tol) {
                std::cout << err << std::endl;
            }
            PL_ABORT_IF_NOT(err < tol, "Wrong statevector entry.");
        }
    }
    svmpi.barrier();
}

template <typename T> [[maybe_unused]] void allclose(T res, T ref) {
    [[maybe_unused]] constexpr double tol = 1.0e-6;
    auto err = std::abs(res - ref);
    if (err > tol) {
        std::cout << err << std::endl;
    }
    PL_ABORT_IF_NOT(err < tol, "Wrong result.");
}

} // namespace

template <class T> struct CtrlTgtQubitIndices {
    T c;
    T t;
    T q;
    CtrlTgtQubitIndices(T ctrl, T tgt, T qubits)
        : c{ctrl}, t{tgt}, q{qubits} {};

    friend std::ostream &operator<<(std::ostream &os, CtrlTgtQubitIndices idx) {
        return os << "ctrl=" << idx.c << ",tgt=" << idx.t
                  << ",qubits=" << idx.q;
    }
};

/**
 * @brief Utility struct to track qubit count and gate indices for 1 qubit
 * gates.
 *
 * @tparam T
 */
template <class T> struct TgtQubitIndices {
    T t;
    T q;
    TgtQubitIndices(T tgt, T qubits) : t{tgt}, q{qubits} {};

    friend std::ostream &operator<<(std::ostream &os, TgtQubitIndices idx) {
        return os << "tgt=" << idx.t << ",qubits=" << idx.q;
    }
};

template <class T> TgtQubitIndices<T> prep_input_1q(int argc, char *argv[]) {
    if (argc <= 2) {
        std::cout << "Please ensure you specify the following arguments: "
                     "target_qubit total_qubits"
                  << std::endl;
        std::exit(-1);
    }
    std::string arg_idx0 = argv[1];
    std::string arg_qubits = argv[2];
    T index_tgt = std::stoi(arg_idx0);
    T qubits = std::stoi(arg_qubits);

    return {index_tgt, qubits};
}

int main(int argc, char *argv[]) {
    auto indices = prep_input_1q<unsigned int>(argc, argv);
    constexpr std::size_t run_avg = 1;
    std::vector<std::string> gates_1q = {
        "Identity", "PauliX",     "PauliY", "PauliZ", "Hadamard", "S",
        "T",        "PhaseShift", "RX",     "RY",     "RZ",       "Rot"};
    std::vector<std::string> gates_2q = {"CNOT",
                                         "CY",
                                         "CZ",
                                         "SWAP",
                                         "IsingXX",
                                         "IsingXY",
                                         "IsingYY",
                                         "IsingZZ",
                                         "ControlledPhaseShift",
                                         "CRX",
                                         "CRY",
                                         "CRZ",
                                         "CRot",
                                         "SingleExcitation",
                                         "SingleExcitationMinus",
                                         "SingleExcitationPlus"};
    std::vector<std::string> gens = {"PhaseShift",
                                     "RX",
                                     "RY",
                                     "RZ",
                                     "IsingXX",
                                     "IsingXY",
                                     "IsingYY",
                                     "IsingZZ",
                                     "ControlledPhaseShift",
                                     "CRX",
                                     "CRY",
                                     "CRZ",
                                     "SingleExcitation",
                                     "SingleExcitationMinus",
                                     "SingleExcitationPlus"};
    std::size_t nq = indices.q;
    std::vector<std::complex<double>> sv_data = get_ascend_vector(nq);

    // Create PennyLane Lightning statevector
    StateVectorKokkos<double> sv(sv_data);
    StateVectorKokkosMPI<double> svmpi(sv_data);
    [[maybe_unused]] auto nglobal = svmpi.get_num_global_wires();
    // print(svmpi);
    // print_basis_states(indices.q);
//    print_local_wires(indices.q);

    // Create vector for run-times to average
    std::vector<double> times;
    times.reserve(run_avg);
    // std::vector<std::size_t> targets{indices.t};

    // Test 1q-gates
    for (auto &gate : gates_1q) {
        for (auto inverse : std::vector<bool>({false, true})) {
            for (std::size_t target = 0; target < nq; target++) {
                if (svmpi.get_mpi_rank() == 0) {
                    std::cout << "Press Enter to continue.\n";
        std::ignore = std::getchar();
                    std::cout << "Testing " << gate << " with : "
                              << "(inv, targets) = (" << inverse << ", "
                              << target << ")" << std::endl;
                }
                auto gate_op =
                    reverse_lookup(gate_names, std::string_view{gate});
                auto npar = lookup(gate_num_params, gate_op);
                std::vector<double> params(npar, 0.1);
                sv.applyOperation(gate, {target}, inverse, params);
                svmpi.applyOperation(gate, {target}, inverse, params);
                allclose(svmpi, sv);
            }
            if (svmpi.get_mpi_rank() == 0) {
                std::cout << "OK" << std::endl;
            }
        }
    }
    // Test 1q-unitary
    for (auto inverse : std::vector<bool>({false, true})) {
        for (std::size_t target = 0; target < nq; target++) {
            if (svmpi.get_mpi_rank() == 0) {
                std::cout << "Testing Matrix with :(inv, targets) = ("
                          << inverse << ", " << target << ")" << std::endl;
            }
            std::vector<Kokkos::complex<double>> matrix = {
                {0.97517033, 0.19767681},
                {-0.09933467, 0.00996671},
                {0.09933467, 0.00996671},
                {0.97517033, 0.19767681}}; // qml.Rot(0.1,0.2,0.3,wires=[0])
            sv.applyOperation("Matrix", {target}, inverse, {}, matrix);
            
                svmpi.applyOperation("Matrix", {target}, inverse, {}, matrix);
            allclose(svmpi, sv);
        }
        if (svmpi.get_mpi_rank() == 0) {
            std::cout << "matrix!" << std::endl;
        }
    }
    // Test 1q-gates
    for (auto &gen : gens) {
        for (auto inverse : std::vector<bool>({false, true})) {
            for (std::size_t target0 = 0; target0 < nq; target0++) {
                for (std::size_t target1 = 0; target1 < nq; target1++) {
                    if (target0 == target1) {
                        continue;
                    }
                    auto gen_op =
                        reverse_lookup(generator_names, std::string_view{gen});
                    auto nwires = lookup(generator_wires, gen_op);
                    if (nwires == 1 && target1 != 0) {
                        continue;
                    }
                    if (svmpi.get_mpi_rank() == 0) {
                        std::cout << "Testing generator " << gen << " with : "
                                  << "(inv, targets) = (" << inverse << ", "
                                  << target0 << ", " << target1 << ")"
                                  << std::endl;
                    }
                    std::vector<std::size_t> wires =
                        (nwires == 1)
                            ? std::vector<std::size_t>{target0}
                            : std::vector<std::size_t>{target0, target1};
                    std::vector<double> params(0, 0.1);
                    //sv.applyGenerator(gen, wires, inverse, params);
                    //svmpi.applyGenerator(gen, wires, inverse, params);
                    allclose(svmpi, sv);
                }
                if (svmpi.get_mpi_rank() == 0) {
                    std::cout << "gen!" << std::endl;
                }
            }
        }
    }
    // Test 2q-gates
    for (auto &gate : gates_2q) {
        for (auto inverse : std::vector<bool>({false, true})) {
            for (std::size_t target0 = 0; target0 < nq; target0++) {
                for (std::size_t target1 = 0; target1 < nq; target1++) {
                    if (target0 == target1) {
                        continue;
                    }
                    if (svmpi.get_mpi_rank() == 0) {
                        std::cout << "Testing " << gate << " with : "
                                  << "(inv, targets) = (" << inverse << ", "
                                  << target0 << ", " << target1 << ")"
                                  << std::endl;
                    }
                    auto gate_op =
                        reverse_lookup(gate_names, std::string_view{gate});
                    auto npar = lookup(gate_num_params, gate_op);
                    std::vector<double> params(npar, 0.1);
                    sv.applyOperation(gate, {target0, target1}, inverse,
                                             params);
                    svmpi.applyOperation(gate, {target0, target1},
                                                inverse, params);
                    allclose(svmpi, sv);
                }
            }
            if (svmpi.get_mpi_rank() == 0) {
                std::cout << "op!" << std::endl;
            }
        }
    }
    // Test named gates
    std::vector<std::string> named_obs = {"Identity", "PauliX", "PauliY",
                                          "PauliZ", "Hadamard"};
    for (auto &gate : named_obs) {
        for (std::size_t target = 0; target < nq; target++) {
            if (svmpi.get_mpi_rank() == 0) {
                std::cout << "Testing obs " << gate << " with : "
                          << "(targets) = (" << target << ")" << std::endl;
            }
            auto ob = NamedObs<decltype(sv)>(gate, {target});
            auto obmpi = NamedObs<decltype(svmpi)>(gate, {target});
            Measurements measure{sv};
            //MeasurementsMPI measurempi{svmpi};
            [[maybe_unused]] auto res = measure.expval(ob);
            //auto resmpi = measurempi.expval(obmpi);
            //allclose(resmpi, res);
        }
    }
    {
        const std::vector<Kokkos::complex<double>> matrix = {
            {2.0, 0.0},
            {0.09933467, -0.00996671},
            {0.09933467, 0.00996671},
            {-1.0, 0.0}};
        for (std::size_t target = 0; target < nq; target++) {
            if (svmpi.get_mpi_rank() == 0) {
                std::cout << "Testing Hermitian obs with : "
                          << "(targets) = (" << target << ")" << std::endl;
            }
            auto ob = HermitianObs<decltype(sv)>(matrix, {target});
            auto obmpi = HermitianObs<decltype(svmpi)>(matrix, {target});
            Measurements measure{sv};
            //MeasurementsMPI measurempi{svmpi};
            [[maybe_unused]]auto res = measure.expval(ob);
            //auto resmpi = measurempi.expval(obmpi);
            //allclose(resmpi, res);
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
