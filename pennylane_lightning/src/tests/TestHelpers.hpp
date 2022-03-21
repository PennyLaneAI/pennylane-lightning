#include <algorithm>
#include <complex>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "Constant.hpp"
#include "ConstantUtil.hpp"
#include "Error.hpp"
#include "GateOperation.hpp"
#include "LinearAlgebra.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

namespace Pennylane {
template <class T, class Alloc = std::allocator<T>> struct PLApprox {
    const std::vector<T, Alloc> &comp_;

    explicit PLApprox(const std::vector<T, Alloc> &comp) : comp_{comp} {}

    Util::remove_complex_t<T> margin_{};
    Util::remove_complex_t<T> epsilon_ =
        std::numeric_limits<float>::epsilon() * 100;

    template <class AllocA>
    [[nodiscard]] bool compare(const std::vector<T, AllocA> &lhs) const {
        if (lhs.size() != comp_.size()) {
            return false;
        }

        for (size_t i = 0; i < lhs.size(); i++) {
            if constexpr (Util::is_complex_v<T>) {
                if (lhs[i].real() != Approx(comp_[i].real())
                                         .epsilon(epsilon_)
                                         .margin(margin_) ||
                    lhs[i].imag() != Approx(comp_[i].imag())
                                         .epsilon(epsilon_)
                                         .margin(margin_)) {
                    return false;
                }
            } else {
                if (lhs[i] !=
                    Approx(comp_[i]).epsilon(epsilon_).margin(margin_)) {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] std::string describe() const {
        std::ostringstream ss;
        ss << "is Approx to {";
        for (const auto &elt : comp_) {
            ss << elt << ", ";
        }
        ss << "}" << std::endl;
        return ss.str();
    }

    PLApprox &epsilon(Util::remove_complex_t<T> eps) {
        epsilon_ = eps;
        return *this;
    }
    PLApprox &margin(Util::remove_complex_t<T> m) {
        margin_ = m;
        return *this;
    }
};

/**
 * @brief Simple helper for PLApprox for the cases when the class template
 * deduction does not work well.
 */
template <typename T, class Alloc>
PLApprox<T, Alloc> approx(const std::vector<T, Alloc> &vec) {
    return PLApprox<T, Alloc>(vec);
}

template <typename T, class Alloc>
std::ostream &operator<<(std::ostream &os, const PLApprox<T, Alloc> &approx) {
    os << approx.describe();
    return os;
}
template <class T, class AllocA, class AllocB>
bool operator==(const std::vector<T, AllocA> &lhs,
                const PLApprox<T, AllocB> &rhs) {
    return rhs.compare(lhs);
}
template <class T, class AllocA, class AllocB>
bool operator!=(const std::vector<T, AllocA> &lhs,
                const PLApprox<T, AllocB> &rhs) {
    return !rhs.compare(lhs);
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
template <class Data_t, class AllocA, class AllocB>
inline bool
isApproxEqual(const std::vector<Data_t, AllocA> &data1,
              const std::vector<Data_t, AllocB> &data2,
              const typename Data_t::value_type eps =
                  std::numeric_limits<typename Data_t::value_type>::epsilon() *
                  100) {
    return data1 == PLApprox<Data_t, AllocB>(data2).epsilon(eps);
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
 * @brief Multiplies every value in a dataset by a given complex scalar value.
 *
 * @tparam Data_t Precision of complex data type. Supports float and double
 * data.
 * @param data Data to be scaled.
 * @param scalar Scalar value.
 */
template <class Data_t>
void scaleVector(std::vector<std::complex<Data_t>> &data, Data_t scalar) {
    std::transform(
        data.begin(), data.end(), data.begin(),
        [scalar](const std::complex<Data_t> &c) { return c * scalar; });
}

/**
 * @brief create |0>^N
 */
template <typename PrecisionT>
auto createZeroState(size_t num_qubits)
    -> std::vector<std::complex<PrecisionT>> {
    std::vector<std::complex<PrecisionT>> res(size_t{1U} << num_qubits,
                                              {0.0, 0.0});
    res[0] = std::complex<PrecisionT>{1.0, 0.0};
    return res;
}

/**
 * @brief create |+>^N
 */
template <typename PrecisionT>
auto createPlusState(size_t num_qubits)
    -> std::vector<std::complex<PrecisionT>> {
    std::vector<std::complex<PrecisionT>> res(size_t{1U} << num_qubits,
                                              {1.0, 0.0});
    for (auto &elt : res) {
        elt /= std::sqrt(1U << num_qubits);
    }
    return res;
}

/**
 * @brief create a random state
 */
template <typename PrecisionT, class RandomEngine>
auto createRandomState(RandomEngine &re, size_t num_qubits)
    -> std::vector<std::complex<PrecisionT>> {
    std::vector<std::complex<PrecisionT>> res(size_t{1U} << num_qubits,
                                              {0.0, 0.0});
    std::uniform_real_distribution<PrecisionT> dist;
    for (size_t idx = 0; idx < (size_t{1U} << num_qubits); idx++) {
        res[idx] = {dist(re), dist(re)};
    }

    scaleVector(res, std::complex<PrecisionT>{1.0, 0.0} /
                         std::sqrt(Util::squaredNorm(res.data(), res.size())));
    return res;
}

/**
 * @brief Create an arbitrary product state in X- or Z-basis.
 *
 * Example: createProductState("+01") will produce |+01> state.
 */
template <typename PrecisionT> auto createProductState(std::string_view str) {
    using Pennylane::Util::INVSQRT2;
    std::vector<std::complex<PrecisionT>> st;
    st.resize(size_t{1U} << str.length());

    std::vector<PrecisionT> zero{1.0, 0.0};
    std::vector<PrecisionT> one{0.0, 1.0};

    std::vector<PrecisionT> plus{INVSQRT2<PrecisionT>(),
                                 INVSQRT2<PrecisionT>()};
    std::vector<PrecisionT> minus{INVSQRT2<PrecisionT>(),
                                  -INVSQRT2<PrecisionT>()};

    for (size_t k = 0; k < (size_t{1U} << str.length()); k++) {
        PrecisionT elt = 1.0;
        for (size_t n = 0; n < str.length(); n++) {
            char c = str[n];
            const size_t wire = str.length() - 1 - n;
            switch (c) {
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

inline auto createWires(Gates::GateOperation op) -> std::vector<size_t> {
    if (Pennylane::Util::array_has_elt(Gates::Constant::multi_qubit_gates,
                                       op)) {
        // if multi-qubit gates
        return {0, 1, 2};
    }
    switch (Pennylane::Util::lookup(Gates::Constant::gate_wires, op)) {
    case 1:
        return {0};
    case 2:
        return {0, 1};
    case 3:
        return {0, 1, 2};
    default:
        PL_ABORT("The number of wires for a given gate is unknown.");
    }
    return {};
}

template <class PrecisionT>
auto createParams(Gates::GateOperation op) -> std::vector<PrecisionT> {
    switch (Pennylane::Util::lookup(Gates::Constant::gate_num_params, op)) {
    case 0:
        return {};
    case 1:
        return {PrecisionT{0.312}};
    case 3:
        return {PrecisionT{0.128}, PrecisionT{-0.563}, PrecisionT{1.414}};
    default:
        PL_ABORT("The number of parameters for a given gate is unknown.");
    }
    return {};
}

template <class PrecisionT> struct PrecisionToName;

template <> struct PrecisionToName<float> {
    constexpr static auto value = "float";
};
template <> struct PrecisionToName<double> {
    constexpr static auto value = "double";
};
} // namespace Pennylane
