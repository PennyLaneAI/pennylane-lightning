#include <cstdlib>
#include <iostream>
#include <ratio>
#include <vector>

namespace BMUtils {

/**
 * @brief Utility struct to track qubit count and gate indices for 2 qubit
 * gates.
 *
 * @tparam T Integer index type.
 */
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

/**
 * @brief Utility method to read and make use of the binary input arguments for
 * 2 qubit gates.
 *
 * @tparam T Integer index type
 * @param argc
 * @param argv
 * @return CtrlTgtQubitIndices<T>
 */
template <class T>
CtrlTgtQubitIndices<T> prep_input_2q(int argc, char *argv[]) {
    if (argc <= 3) {
        std::cout << "Please ensure you specify the following arguments: "
                     "control_qubit target_qubit total_qubits"
                  << std::endl;
        std::exit(-1);
    }

    std::string arg_idx0 = argv[1];
    std::string arg_idx1 = argv[2];
    std::string arg_qubits = argv[3];
    T index_ctrl = std::stoi(arg_idx0);
    T index_tgt = std::stoi(arg_idx1);
    T qubits = std::stoi(arg_qubits);

    return {index_ctrl, index_tgt, qubits};
}

/**
 * @brief To to collect and format the output of CSV data from gates.
 *
 * @tparam T
 * @tparam U
 */
template <class T, class U = std::milli> struct CSVOutput {
    const T indices_;
    const std::string gate_name_;
    const double t_avg_;

    CSVOutput(T indices, std::string gate_name, double t_avg)
        : indices_{indices}, gate_name_{gate_name}, t_avg_{t_avg} {};

    friend std::ostream &operator<<(std::ostream &os, CSVOutput idx) {
        return os << idx.indices_ << "," << idx.gate_name_
                  << ",t_avg=" << idx.t_avg_ << ",t_ratio=" << U::num << "/"
                  << U::den;
    }
};

/**
 * @brief Utility function to determine the average time over a give number of
 * runs.
 *
 * @param times Vector data of multiple runs.
 * @return double
 */
static inline double average_times(std::vector<double> &times) {
    return std::accumulate(times.begin(), times.end(), 0.0,
                           std::plus<double>()) /
           times.size();
}

// Utility macro to time the execution of the provided FUNC.
#define TIMING(FUNC)                                                           \
    {                                                                          \
        const auto t_start = std::chrono::high_resolution_clock::now();        \
        FUNC;                                                                  \
        const auto t_end = std::chrono::high_resolution_clock::now();          \
        const double t_duration =                                              \
            std::chrono::duration<double, t_scale>(t_end - t_start).count();   \
        times.push_back(t_duration);                                           \
    }

} // namespace BMUtils