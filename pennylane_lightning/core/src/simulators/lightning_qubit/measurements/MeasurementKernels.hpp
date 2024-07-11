#include <complex>
#include <vector>

#include "Util.hpp"

/// @cond DEV
namespace {
namespace PUtil = Pennylane::Util;
} // namespace
/// @endcond

#define PROBS_CORE_DECLARE(n)                                                  \
    std::size_t rev_wires_##n;                                                 \
    if constexpr (n_wires > n) {                                               \
        rev_wires_##n = rev_wires[n];                                          \
    }
#define PROBS_CORE_SHIFT(n)                                                    \
    if constexpr (n_wires > n) {                                               \
        pindex |= ((svindex & (one << rev_wires_##n)) >> rev_wires_##n)        \
                  << (n_wires - 1 - n);                                        \
    }

#define PROBS_SPECIAL_CASE(n)                                                  \
    if (n_wires == n) {                                                        \
        return Pennylane::LightningQubit::Measures::probs_core<PrecisionT, n>( \
            exp2(num_qubits), arr_data, rev_wires);                            \
    }

#define PROBS_SPECIAL_CASES                                                    \
    PROBS_SPECIAL_CASE(1)                                                      \
    PROBS_SPECIAL_CASE(2)                                                      \
    PROBS_SPECIAL_CASE(3)                                                      \
    PROBS_SPECIAL_CASE(4)                                                      \
    PROBS_SPECIAL_CASE(5)                                                      \
    PROBS_SPECIAL_CASE(6)                                                      \
    PROBS_SPECIAL_CASE(7)                                                      \
    PROBS_SPECIAL_CASE(8)                                                      \
    PROBS_SPECIAL_CASE(9)                                                      \
    PROBS_SPECIAL_CASE(10)                                                     \
    PROBS_SPECIAL_CASE(11)                                                     \
    PROBS_SPECIAL_CASE(12)                                                     \
    PROBS_SPECIAL_CASE(13)                                                     \
    PROBS_SPECIAL_CASE(14)                                                     \
    PROBS_SPECIAL_CASE(15)                                                     \
    PROBS_SPECIAL_CASE(16)                                                     \
    PROBS_SPECIAL_CASE(17)                                                     \
    PROBS_SPECIAL_CASE(18)                                                     \
    PROBS_SPECIAL_CASE(19)                                                     \
    PROBS_SPECIAL_CASE(20)                                                     \
    PROBS_SPECIAL_CASE(21)                                                     \
    PROBS_SPECIAL_CASE(22)                                                     \
    PROBS_SPECIAL_CASE(23)                                                     \
    PROBS_SPECIAL_CASE(24)                                                     \
    PROBS_SPECIAL_CASE(25)                                                     \
    PROBS_SPECIAL_CASE(26)                                                     \
    PROBS_SPECIAL_CASE(27)                                                     \
    PROBS_SPECIAL_CASE(28)                                                     \
    PROBS_SPECIAL_CASE(29)                                                     \
    PROBS_SPECIAL_CASE(30)

namespace Pennylane::LightningQubit::Measures {

template <class PrecisionT, std::size_t n_wires>
auto probs_core(const std::size_t num_data,
                const std::complex<PrecisionT> *arr_data,
                const std::vector<std::size_t> &rev_wires)
    -> std::vector<PrecisionT> {
    constexpr std::size_t one{1};
    std::vector<PrecisionT> probs(PUtil::exp2(n_wires), 0);
    PROBS_CORE_DECLARE(0)
    PROBS_CORE_DECLARE(1)
    PROBS_CORE_DECLARE(2)
    PROBS_CORE_DECLARE(3)
    PROBS_CORE_DECLARE(4)
    PROBS_CORE_DECLARE(5)
    PROBS_CORE_DECLARE(6)
    PROBS_CORE_DECLARE(7)
    PROBS_CORE_DECLARE(8)
    PROBS_CORE_DECLARE(9)
    PROBS_CORE_DECLARE(10)
    PROBS_CORE_DECLARE(11)
    PROBS_CORE_DECLARE(12)
    PROBS_CORE_DECLARE(13)
    PROBS_CORE_DECLARE(14)
    for (std::size_t svindex = 0; svindex < num_data; svindex++) {
        std::size_t pindex{0U};
        PROBS_CORE_SHIFT(0)
        PROBS_CORE_SHIFT(1)
        PROBS_CORE_SHIFT(2)
        PROBS_CORE_SHIFT(3)
        PROBS_CORE_SHIFT(4)
        PROBS_CORE_SHIFT(5)
        PROBS_CORE_SHIFT(6)
        PROBS_CORE_SHIFT(7)
        PROBS_CORE_SHIFT(8)
        PROBS_CORE_SHIFT(9)
        PROBS_CORE_SHIFT(10)
        PROBS_CORE_SHIFT(11)
        PROBS_CORE_SHIFT(12)
        PROBS_CORE_SHIFT(13)
        PROBS_CORE_SHIFT(14)
        probs[pindex] += std::norm(arr_data[svindex]);
    }
    return probs;
}
} // namespace Pennylane::LightningQubit::Measures