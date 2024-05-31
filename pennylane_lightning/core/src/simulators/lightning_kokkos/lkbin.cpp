#include <iostream>

#include "StateVectorKokkos.hpp"

namespace {
using namespace Pennylane::LightningKokkos;

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

/**
 * @brief Utility method to read and make use of the binary input arguments for
 * 1 qubit gates.
 *
 * @tparam T Integer type used for indexing.
 * @param argc
 * @param argv
 * @return TgtQubitIndices<T>
 */
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

} // namespace

int main(int argc, char *argv[]) {
    constexpr std::size_t nrepeat = 200;
    auto indices = prep_input_1q<unsigned int>(argc, argv);
    std::size_t nq = indices.q;
    StateVectorKokkos<double> sv(nq);
    for (std::size_t i = 0; i < nrepeat; i++) {
        sv.applyOperation("Hadamard", {indices.t});
    }
    return 0;
}
