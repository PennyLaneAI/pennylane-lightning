#include <algorithm>
#include <complex>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "Constant.hpp"
#include "GateOperation.hpp"
#include "Error.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

namespace TestHelper {
/**
 * @brief Custom catch matcher for std::vector<std::complex<T>>
 */
template <typename T, typename AllocComp, typename AllocMatch>
struct ComplexApproxMatcher
    : Catch::MatcherBase<std::vector<std::complex<T>, AllocMatch>> {

    ComplexApproxMatcher(
        const std::vector<std::complex<T>, AllocComp> &comparator)
        : m_comparator(comparator) {}

    bool
    match(std::vector<std::complex<T>, AllocMatch> const &v) const override {
        if (m_comparator.size() != v.size()) {
            return false;
        }
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (std::real(m_comparator[i]) != approx(std::real(v[i])) ||
                std::imag(m_comparator[i]) != approx(std::imag(v[i]))) {
                return false;
            }
        }
        return true;
    }
    std::string describe() const override {
        return "is approx: " + ::Catch::Detail::stringify(m_comparator);
    }
    template <typename = std::enable_if_t<std::is_constructible_v<double, T>>>
    ComplexApproxMatcher &epsilon(const T &newEpsilon) {
        approx.epsilon(newEpsilon);
        return *this;
    }
    template <typename = std::enable_if_t<std::is_constructible_v<double, T>>>
    ComplexApproxMatcher &margin(const T &newMargin) {
        approx.margin(newMargin);
        return *this;
    }
    template <typename = std::enable_if_t<std::is_constructible_v<double, T>>>
    ComplexApproxMatcher &scale(const T &newScale) {
        approx.scale(newScale);
        return *this;
    }

    const std::vector<std::complex<T>, AllocComp> &m_comparator;
    mutable Catch::Detail::Approx approx = Catch::Detail::Approx::custom();
};

template <typename T, typename AllocComp = std::allocator<std::complex<T>>,
          typename AllocMatch = AllocComp>
ComplexApproxMatcher<T, AllocComp, AllocMatch>
Approx(std::vector<std::complex<T>, AllocComp> const &comparator) {
    return ComplexApproxMatcher<T, AllocComp, AllocMatch>{comparator};
}
} // namespace TestHelper

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
template <typename PrecisionT>
auto create_zero_state(size_t num_qubits) -> std::vector<std::complex<PrecisionT>> {
    std::vector<std::complex<PrecisionT>> res(1U << num_qubits, {0.0, 0.0});
    res[0] = std::complex<PrecisionT>{1.0, 0.0};
    return res;
}

/**
 * @brief create |+>^N
 */
template <typename PrecisionT>
auto create_plus_state(size_t num_qubits) -> std::vector<std::complex<PrecisionT>> {
    std::vector<std::complex<PrecisionT>> res(1U << num_qubits, {1.0, 0.0});
    for (auto &elt : res) {
        elt /= std::sqrt(1U << num_qubits);
    }
    return res;
}

/**
 * @brief create a random state
 */
template <typename PrecisionT, class RandomEngine>
auto create_random_state(RandomEngine &re, size_t num_qubits)
    -> std::vector<std::complex<PrecisionT>> {
    std::vector<std::complex<PrecisionT>> res(1U << num_qubits, {0.0, 0.0});
    std::uniform_real_distribution<PrecisionT> dist;
    for (size_t idx = 0; idx < (1U << num_qubits); idx++) {
        res[idx] = {dist(re), dist(re)};
    }

    PrecisionT squared_norm = std::transform_reduce(
        std::cbegin(res), std::cend(res), PrecisionT{}, std::plus<PrecisionT>(),
        static_cast<PrecisionT (*)(const std::complex<PrecisionT> &)>(&std::norm<PrecisionT>));

    scaleVector(res, std::complex<PrecisionT>{1.0, 0.0} / std::sqrt(squared_norm));
    return res;
}

/**
 * @brief Create an arbitrary product state in X- or Z-basis.
 *
 * Example: create_product_state("+01") will produce |+01> state.
 */
template <typename PrecisionT>
auto create_product_state(std::string_view str) {
    using Pennylane::Util::INVSQRT2;
    std::vector<std::complex<PrecisionT>> st;
    st.resize(1U << str.length());

    std::vector<PrecisionT> zero{1.0, 0.0};
    std::vector<PrecisionT> one{0.0, 1.0};

    std::vector<PrecisionT> plus{INVSQRT2<PrecisionT>(), INVSQRT2<PrecisionT>()};
    std::vector<PrecisionT> minus{INVSQRT2<PrecisionT>(), -INVSQRT2<PrecisionT>()};

    for(size_t k = 0; k < (1U << str.length()); k++) {
        PrecisionT elt = 1.0;
        for(size_t n = 0; n < str.length(); n++) {
            char c = str[n];
            const size_t wire = str.length() - 1 - n;
            switch(c) {
            case '0':
                elt *= zero[(k >> wire) & 1U];
                break;
            case '1':
                elt *= one[(k >> wire) & 1U];
                break;
            case '+':
                elt *= plus[(k >> wire) & 1U];
                break;
            case '-':
                elt *= minus[(k >> wire) & 1U];
                break;
            default:
                PL_ABORT("Unknown character in the argument.");
            }
        }
        st[k] = elt;
    }
    return st;
}

inline std::vector<size_t> createWires(Pennylane::GateOperation op) {
    if (Pennylane::Util::array_has_elt(Pennylane::Constant::multi_qubit_gates, op)) {
        // if multi-qubit gates
        return {0, 1, 2};
    }
    switch (Pennylane::Util::lookup(Pennylane::Constant::gate_wires, op)) {
    case 1:
        return {0};
    case 2:
        return {0, 1};
    case 3:
        return {0, 1, 2};
    default:
        PL_ABORT("The number of wires for a given gate is unknown.");
    }
}

template <class PrecisionT>
std::vector<PrecisionT> createParams(Pennylane::GateOperation op) {
    switch (Pennylane::Util::lookup(Pennylane::Constant::gate_num_params, op)) {
    case 0:
        return {};
    case 1:
        return {0.312};
    case 3:
        return {0.128, -0.563, 1.414};
    default:
        PL_ABORT("The number of parameters for a given gate is unknown.");
    }
}
