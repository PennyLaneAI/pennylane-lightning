#include <algorithm>
#include <complex>
#include <random>
#include <vector>

#include "GateOperations.hpp"

#include <catch2/catch.hpp>

/**
 * @brief Utility function to compare complex statevector data.
 *
 * @tparam Data_t Floating point data-type.
 * @param data1 StateVector data 1.
 * @param data2 StateVector data 2.
 * @return true Data are approximately equal.
 * @return false Data are not approximately equal.
 */
template <class Data_t>
inline bool isApproxEqual(
    const std::vector<Data_t> &data1, const std::vector<Data_t> &data2,
    const typename Data_t::value_type eps =
        std::numeric_limits<typename Data_t::value_type>::epsilon() * 100) {
    if (data1.size() != data2.size()) {
        return false;
    }

    for (size_t i = 0; i < data1.size(); i++) {
        if (data1[i].real() != Approx(data2[i].real()).epsilon(eps) ||
            data1[i].imag() != Approx(data2[i].imag()).epsilon(eps)) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Utility function to compare complex statevector data.
 *
 * @tparam Data_t Floating point data-type.
 * @param data1 StateVector data 1.
 * @param data2 StateVector data 2.
 * @return true Data are approximately equal.
 * @return false Data are not approximately equal.
 */
template <class Data_t>
inline bool
isApproxEqual(const Data_t &data1, const Data_t &data2,
              typename Data_t::value_type eps =
                  std::numeric_limits<typename Data_t::value_type>::epsilon() *
                  100) {
    return !(data1.real() != Approx(data2.real()).epsilon(eps) ||
             data1.imag() != Approx(data2.imag()).epsilon(eps));
}

/**
 * @brief Multiplies every value in a dataset by a given complex scalar value.
 *
 * @tparam Data_t Precision of complex data type. Supports float and double
 * data.
 * @param data Data to be scaled.
 * @param scalar Scalar value.
 */
template <class Data_t>
void scaleVector(std::vector<std::complex<Data_t>> &data,
                 std::complex<Data_t> scalar) {
    std::transform(
        data.begin(), data.end(), data.begin(),
        [scalar](const std::complex<Data_t> &c) { return c * scalar; });
}

/**
 * @brief create |0>^N
 */
template <typename fp_t>
auto create_zero_state(size_t num_qubits) -> std::vector<std::complex<fp_t>> {
    std::vector<std::complex<fp_t>> res(1U << num_qubits, {0.0, 0.0});
    res[0] = std::complex<fp_t>{1.0, 0.0};
    return res;
}

/**
 * @brief create |+>^N
 */
template <typename fp_t>
auto create_plus_state(size_t num_qubits) -> std::vector<std::complex<fp_t>> {
    std::vector<std::complex<fp_t>> res(1U << num_qubits, {1.0, 0.0});
    for (auto &elt : res) {
        elt /= std::sqrt(1U << num_qubits);
    }
    return res;
}

/**
 * @brief create a random state
 */
template <typename fp_t, class RandomEngine>
auto create_random_state(RandomEngine &re, size_t num_qubits)
    -> std::vector<std::complex<fp_t>> {
    std::vector<std::complex<fp_t>> res(1U << num_qubits, {0.0, 0.0});
    std::uniform_real_distribution<fp_t> dist;
    for (size_t idx = 0; idx < (1U << num_qubits); idx++) {
        res[idx] = {dist(re), dist(re)};
    }

    fp_t squared_norm = std::transform_reduce(
        std::cbegin(res), std::cend(res), fp_t{}, std::plus<fp_t>(),
        static_cast<fp_t (*)(const std::complex<fp_t> &)>(&std::norm<fp_t>));

    scaleVector(res, std::complex<fp_t>{1.0, 0.0} / std::sqrt(squared_norm));
    return res;
}
