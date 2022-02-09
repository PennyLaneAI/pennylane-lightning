#include <algorithm>
#include <complex>
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
template <typename T> struct remove_complex { using type = T; };
template <typename T> struct remove_complex<std::complex<T>> {
    using type = T;
};
template <typename T> using remove_complex_t = typename remove_complex<T>::type;

template <typename T> struct is_complex : std::false_type {};

template <typename T> struct is_complex<std::complex<T>> : std::true_type {};

template <typename T> constexpr bool is_complex_v = is_complex<T>::value;

template <class T, class Alloc> struct PLApprox {
    const std::vector<T, Alloc> &comp_;

    explicit PLApprox(const std::vector<T, Alloc> &comp) : comp_{comp} {}

    remove_complex_t<T> margin_{};
    remove_complex_t<T> epsilon_ = std::numeric_limits<float>::epsilon() * 100;

    template <class AllocA>
    [[nodiscard]] bool compare(const std::vector<T, AllocA> &lhs) const {
        if (lhs.size() != comp_.size()) {
            return false;
        }

        for (size_t i = 0; i < lhs.size(); i++) {
            if constexpr (is_complex_v<T>) {
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
    PLApprox &epsilon(remove_complex_t<T> eps) {
        epsilon_ = eps;
        return *this;
    }
    PLApprox &margin(remove_complex_t<T> m) {
        margin_ = m;
        return *this;
    }
};
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
    return data1 == PLApprox(data2);
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
    std::vector<std::complex<PrecisionT>> res(1U << num_qubits, {0.0, 0.0});
    res[0] = std::complex<PrecisionT>{1.0, 0.0};
    return res;
}

/**
 * @brief create |+>^N
 */
template <typename PrecisionT>
auto createPlusState(size_t num_qubits)
    -> std::vector<std::complex<PrecisionT>> {
    std::vector<std::complex<PrecisionT>> res(1U << num_qubits, {1.0, 0.0});
    for (auto &elt : res) {
        elt /= std::sqrt(1U << num_qubits);
    }
    return res;
}

/**
 * @brief Calculate the squared norm of a vector
 */
template <typename PrecisionT>
auto squaredNorm(const std::complex<PrecisionT> *data, size_t data_size)
    -> PrecisionT {
    return std::transform_reduce(
        data, data + data_size, PrecisionT{}, std::plus<PrecisionT>(),
        static_cast<PrecisionT (*)(const std::complex<PrecisionT> &)>(
            &std::norm<PrecisionT>));
}

/**
 * @brief create a random state
 */
template <typename PrecisionT, class RandomEngine>
auto createRandomState(RandomEngine &re, size_t num_qubits)
    -> std::vector<std::complex<PrecisionT>> {
    std::vector<std::complex<PrecisionT>> res(1U << num_qubits, {0.0, 0.0});
    std::uniform_real_distribution<PrecisionT> dist;
    for (size_t idx = 0; idx < (1U << num_qubits); idx++) {
        res[idx] = {dist(re), dist(re)};
    }

    scaleVector(res, std::complex<PrecisionT>{1.0, 0.0} /
                         std::sqrt(squaredNorm(res.data(), res.size())));
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
    st.resize(1U << str.length());

    std::vector<PrecisionT> zero{1.0, 0.0};
    std::vector<PrecisionT> one{0.0, 1.0};

    std::vector<PrecisionT> plus{INVSQRT2<PrecisionT>(),
                                 INVSQRT2<PrecisionT>()};
    std::vector<PrecisionT> minus{INVSQRT2<PrecisionT>(),
                                  -INVSQRT2<PrecisionT>()};

    for (size_t k = 0; k < (1U << str.length()); k++) {
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
        return {0.312};
    case 3:
        return {0.128, -0.563, 1.414};
    default:
        PL_ABORT("The number of parameters for a given gate is unknown.");
    }
    return {};
}
/**
 * @brief Generate random unitary matrix
 *
 * @return Generated unitary matrix in row-major format
 */
template <typename PrecisionT, class RandomEngine>
auto randomUnitary(RandomEngine &re, size_t num_qubits)
    -> std::vector<std::complex<PrecisionT>> {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t dim = (1U << num_qubits);
    std::vector<ComplexPrecisionT> res(dim * dim, ComplexPrecisionT{});

    std::normal_distribution<PrecisionT> dist;

    auto generator = [&dist, &re]() -> ComplexPrecisionT {
        return ComplexPrecisionT{dist(re), dist(re)};
    };

    std::generate(res.begin(), res.end(), generator);

    // Simple algorithm to make rows orthogonal with Gram-Schmidt
    // This algorithm is unstable but works for a small matrix.
    // Use QR decomposition when we have LAPACK support.

    for (size_t row2 = 0; row2 < dim; row2++) {
        ComplexPrecisionT *row2_p = res.data() + row2 * dim;
        for (size_t row1 = 0; row1 < row2; row1++) {
            const ComplexPrecisionT *row1_p = res.data() + row1 * dim;
            ComplexPrecisionT dot12 = Util::innerProdC(row1_p, row2_p, dim);
            ComplexPrecisionT dot11 = squaredNorm(row1_p, dim);

            // orthogonalize row2
            std::transform(
                row2_p, row2_p + dim, row1_p, row2_p,
                [scale = dot12 / dot11](auto &elt2, const auto &elt1) {
                    return elt2 - scale * elt1;
                });
        }
    }

    // Normalize each row
    for (size_t row = 0; row < dim; row++) {
        ComplexPrecisionT *row_p = res.data() + row * dim;
        PrecisionT norm2 = std::sqrt(squaredNorm(row_p, dim));

        // noramlize row2
        std::transform(row_p, row_p + dim, row_p, [norm2](const auto c) {
            return (static_cast<PrecisionT>(1.0) / norm2) * c;
        });
    }
    return res;
}

template <class PrecisionT> struct PrecisionToName;

template <> struct PrecisionToName<float> {
    constexpr static auto value = "float";
};
template <> struct PrecisionToName<double> {
    constexpr static auto value = "double";
};
} // namespace Pennylane
