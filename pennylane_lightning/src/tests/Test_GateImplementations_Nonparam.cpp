#include "CompareVector.hpp"
#include "TestHelpers.hpp"
#include "TestKernels.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file Test_GateImplementations_Nonparam.cpp
 *
 * This file contains tests for non-parameterized gates. List of such
 * gates are [PauliX, PauliY, PauliZ, Hadamard, S, T, CNOT, SWAP, CZ, Toffoli,
 * CSWAP].
 */
using namespace Pennylane;

/**
 * @brief Run test suit only when the gate is defined
 */
#define PENNYLANE_RUN_TEST(GATE_NAME)                                          \
    template <typename PrecisionT, class GateImplementation,                   \
              typename U = void>                                               \
    struct Apply##GATE_NAME##IsDefined {                                       \
        constexpr static bool value = false;                                   \
    };                                                                         \
    template <typename PrecisionT, class GateImplementation>                   \
    struct Apply##GATE_NAME##IsDefined<                                        \
        PrecisionT, GateImplementation,                                        \
        std::enable_if_t<std::is_pointer_v<                                    \
            decltype(&GateImplementation::template apply##GATE_NAME<           \
                     PrecisionT>)>>> {                                         \
        constexpr static bool value = true;                                    \
    };                                                                         \
    template <typename PrecisionT, typename TypeList>                          \
    void testApply##GATE_NAME##ForKernels() {                                  \
        if constexpr (!std::is_same_v<TypeList, void>) {                       \
            using GateImplementation = typename TypeList::Type;                \
            if constexpr (Apply##GATE_NAME##IsDefined<                         \
                              PrecisionT, GateImplementation>::value) {        \
                testApply##GATE_NAME<PrecisionT, GateImplementation>();        \
            } else {                                                           \
                SUCCEED("Member function apply" #GATE_NAME                     \
                        " is not defined for kernel "                          \
                        << GateImplementation::name);                          \
            }                                                                  \
            testApply##GATE_NAME##ForKernels<PrecisionT,                       \
                                             typename TypeList::Next>();       \
        }                                                                      \
    }                                                                          \
    TEMPLATE_TEST_CASE("GateImplementation::apply" #GATE_NAME,                 \
                       "[GateImplementations_Nonparam]", float, double) {      \
        using PrecisionT = TestType;                                           \
        testApply##GATE_NAME##ForKernels<PrecisionT, TestKernels>();           \
    }                                                                          \
    static_assert(true, "Require semicolon")

/*******************************************************************************
 * Single-qubit gates
 ******************************************************************************/
template <typename PrecisionT, class GateImplementation>
void testApplyIdentity() {
    const size_t num_qubits = 3;
    for (size_t index = 0; index < num_qubits; index++) {
        auto st_pre = createZeroState<PrecisionT>(num_qubits);
        auto st_post = createZeroState<PrecisionT>(num_qubits);

        GateImplementation::applyIdentity(st_pre.data(), num_qubits, {index},
                                          false);
        CHECK(std::equal(st_pre.begin(), st_pre.end(), st_post.begin()));
    }
    for (size_t index = 0; index < num_qubits; index++) {
        auto st_pre = createZeroState<PrecisionT>(num_qubits);
        auto st_post = createZeroState<PrecisionT>(num_qubits);
        GateImplementation::applyHadamard(st_pre.data(), num_qubits, {index},
                                          false);
        GateImplementation::applyHadamard(st_post.data(), num_qubits, {index},
                                          false);

        GateImplementation::applyIdentity(st_pre.data(), num_qubits, {index},
                                          false);
        CHECK(std::equal(st_pre.begin(), st_pre.end(), st_post.begin()));
    }
}
PENNYLANE_RUN_TEST(Identity);

template <typename PrecisionT, class GateImplementation>
void testApplyPauliX() {
    const size_t num_qubits = 3;
    DYNAMIC_SECTION(GateImplementation::name
                    << ", PauliX - " << PrecisionToName<PrecisionT>::value) {
        for (size_t index = 0; index < num_qubits; index++) {
            auto st = createZeroState<PrecisionT>(num_qubits);

            GateImplementation::applyPauliX(st.data(), num_qubits, {index},
                                            false);

            std::string expected_str("000");
            expected_str[index] = '1';
            REQUIRE(st == approx(createProductState<PrecisionT>(expected_str)));
        }
    }
}
PENNYLANE_RUN_TEST(PauliX);

template <typename PrecisionT, class GateImplementation>
void testApplyPauliY() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;

    constexpr ComplexPrecisionT p =
        Util::ConstMult(static_cast<PrecisionT>(0.5),
                        Util::ConstMult(Util::INVSQRT2<PrecisionT>(),
                                        Util::IMAG<PrecisionT>()));
    constexpr ComplexPrecisionT m = Util::ConstMult(-1, p);

    const std::vector<std::vector<ComplexPrecisionT>> expected_results = {
        {m, m, m, m, p, p, p, p},
        {m, m, p, p, m, m, p, p},
        {m, p, m, p, m, p, m, p}};

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = createPlusState<PrecisionT>(num_qubits);

        GateImplementation::applyPauliY(st.data(), num_qubits, {index}, false);

        CHECK(st == approx(expected_results[index]));
    }
}
PENNYLANE_RUN_TEST(PauliY);

template <typename PrecisionT, class GateImplementation>
void testApplyPauliZ() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;

    constexpr ComplexPrecisionT p(static_cast<PrecisionT>(0.5) *
                                  Util::INVSQRT2<PrecisionT>());
    constexpr ComplexPrecisionT m(Util::ConstMult(-1, p));

    const std::vector<std::vector<ComplexPrecisionT>> expected_results = {
        {p, p, p, p, m, m, m, m},
        {p, p, m, m, p, p, m, m},
        {p, m, p, m, p, m, p, m}};

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = createPlusState<PrecisionT>(num_qubits);
        GateImplementation::applyPauliZ(st.data(), num_qubits, {index}, false);

        CHECK(st == approx(expected_results[index]));
    }
}
PENNYLANE_RUN_TEST(PauliZ);

template <typename PrecisionT, class GateImplementation>
void testApplyHadamard() {
    const size_t num_qubits = 3;
    for (size_t index = 0; index < num_qubits; index++) {
        auto st = createZeroState<PrecisionT>(num_qubits);

        GateImplementation::applyHadamard(st.data(), num_qubits, {index},
                                          false);

        std::vector<char> expected_string;
        expected_string.resize(num_qubits);
        std::fill(expected_string.begin(), expected_string.end(), '0');
        expected_string[index] = '+';
        const auto expected = createProductState<PrecisionT>(
            std::string_view{expected_string.data(), num_qubits});
        CHECK(expected == approx(st));
    }
}
PENNYLANE_RUN_TEST(Hadamard);

template <typename PrecisionT, class GateImplementation> void testApplyS() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;

    constexpr ComplexPrecisionT r(static_cast<PrecisionT>(0.5) *
                                  Util::INVSQRT2<PrecisionT>());
    constexpr ComplexPrecisionT i(Util::ConstMult(r, Util::IMAG<PrecisionT>()));

    const std::vector<std::vector<ComplexPrecisionT>> expected_results = {
        {r, r, r, r, i, i, i, i},
        {r, r, i, i, r, r, i, i},
        {r, i, r, i, r, i, r, i}};

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = createPlusState<PrecisionT>(num_qubits);

        GateImplementation::applyS(st.data(), num_qubits, {index}, false);

        CHECK(st == approx(expected_results[index]));
    }
}
PENNYLANE_RUN_TEST(S);

template <typename PrecisionT, class GateImplementation> void testApplyT() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;
    // Test using |+++> state

    ComplexPrecisionT r(1.0 / (2.0 * std::sqrt(2)), 0);
    ComplexPrecisionT i(1.0 / 4, 1.0 / 4);

    const std::vector<std::vector<ComplexPrecisionT>> expected_results = {
        {r, r, r, r, i, i, i, i},
        {r, r, i, i, r, r, i, i},
        {r, i, r, i, r, i, r, i}};

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = createPlusState<PrecisionT>(num_qubits);

        GateImplementation::applyT(st.data(), num_qubits, {index}, false);

        CHECK(st == approx(expected_results[index]));
    }
}
PENNYLANE_RUN_TEST(T);
/*******************************************************************************
 * Two-qubit gates
 ******************************************************************************/

template <typename PrecisionT, class GateImplementation> void testApplyCNOT() {
    const size_t num_qubits = 3;

    SECTION("CNOT0,1 |000> = |000>") {
        const auto ini_st = createProductState<PrecisionT>("000");
        auto st = ini_st;
        GateImplementation::applyCNOT(st.data(), num_qubits, {0, 1}, false);
        CHECK(st == ini_st);
    }

    SECTION("CNOT0,1 |100> = |110>") {
        const auto ini_st = createProductState<PrecisionT>("100");
        auto st = ini_st;
        GateImplementation::applyCNOT(st.data(), num_qubits, {0, 1}, false);
        CHECK(st == approx(createProductState<PrecisionT>("110")).margin(1e-7));
    }
    SECTION("CNOT1,2 |110> = |111>") {
        const auto ini_st = createProductState<PrecisionT>("110");
        auto st = ini_st;
        GateImplementation::applyCNOT(st.data(), num_qubits, {1, 2}, false);
        CHECK(st == approx(createProductState<PrecisionT>("111")).margin(1e-7));
    }

    SECTION("Generate GHZ state") {
        auto st = createProductState<PrecisionT>("+00");

        // Test using |+00> state to generate 3-qubit GHZ state
        for (size_t index = 1; index < num_qubits; index++) {
            GateImplementation::applyCNOT(st.data(), num_qubits,
                                          {index - 1, index}, false);
        }
        CHECK(st.front() == Util::INVSQRT2<PrecisionT>());
        CHECK(st.back() == Util::INVSQRT2<PrecisionT>());
    }
}
PENNYLANE_RUN_TEST(CNOT);

// NOLINTNEXTLINE: Avoiding complexity errors
template <typename PrecisionT, class GateImplementation> void testApplyCY() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;
    auto ini_st =
        createProductState<PrecisionT>("+10"); // Test using |+10> state

    CHECK(ini_st == std::vector<ComplexPrecisionT>{
                        Util::ZERO<PrecisionT>(), Util::ZERO<PrecisionT>(),
                        std::complex<PrecisionT>(1.0 / sqrt(2), 0),
                        Util::ZERO<PrecisionT>(), Util::ZERO<PrecisionT>(),
                        Util::ZERO<PrecisionT>(),
                        std::complex<PrecisionT>(1.0 / sqrt(2), 0),
                        Util::ZERO<PrecisionT>()});

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CY 0,1 |+10> -> i|100> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexPrecisionT> expected{
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(1.0 / sqrt(2), 0),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(0, -1 / sqrt(2)),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>()};

        auto sv01 = ini_st;
        GateImplementation::applyCY(sv01.data(), num_qubits, {0, 1}, false);
        CHECK(sv01 == expected);
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CY 0,2 |+10> -> |010> + i |111> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexPrecisionT> expected{
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(1.0 / sqrt(2), 0.0),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(0.0, 1 / sqrt(2))};

        auto sv02 = ini_st;

        GateImplementation::applyCY(sv02.data(), num_qubits, {0, 2}, false);
        CHECK(sv02 == expected);
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", CY 1,2 |+10> -> i|+11> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexPrecisionT> expected{
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(0.0, 1.0 / sqrt(2)),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(0.0, 1 / sqrt(2))};

        auto sv12 = ini_st;

        GateImplementation::applyCY(sv12.data(), num_qubits, {1, 2}, false);
        CHECK(sv12 == expected);
    }
}
PENNYLANE_RUN_TEST(CY);

// NOLINTNEXTLINE: Avoiding complexity errors
template <typename PrecisionT, class GateImplementation> void testApplyCZ() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;

    auto ini_st = createProductState<PrecisionT>("+10");

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CZ0,1 |+10> -> |-10> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexPrecisionT> expected{
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(1.0 / sqrt(2), 0),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(-1 / sqrt(2), 0),
            Util::ZERO<PrecisionT>()};

        auto sv01 = ini_st;
        auto sv10 = ini_st;

        GateImplementation::applyCZ(sv01.data(), num_qubits, {0, 1}, false);
        GateImplementation::applyCZ(sv10.data(), num_qubits, {1, 0}, false);

        CHECK(sv01 == expected);
        CHECK(sv10 == expected);
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CZ0,2 |+10> -> |+10> - "
                    << PrecisionToName<PrecisionT>::value) {
        const auto &expected = ini_st;

        auto sv02 = ini_st;
        auto sv20 = ini_st;

        GateImplementation::applyCZ(sv02.data(), num_qubits, {0, 2}, false);
        GateImplementation::applyCZ(sv20.data(), num_qubits, {2, 0}, false);

        CHECK(sv02 == expected);
        CHECK(sv20 == expected);
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", CZ1,2 |+10> -> |+10> - "
                    << PrecisionToName<PrecisionT>::value) {
        const auto &expected = ini_st;

        auto sv12 = ini_st;
        auto sv21 = ini_st;

        GateImplementation::applyCZ(sv12.data(), num_qubits, {1, 2}, false);
        GateImplementation::applyCZ(sv21.data(), num_qubits, {2, 1}, false);

        CHECK(sv12 == expected);
        CHECK(sv21 == expected);
    }
}
PENNYLANE_RUN_TEST(CZ);

// NOLINTNEXTLINE: Avoiding complexity errors
template <typename PrecisionT, class GateImplementation> void testApplySWAP() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;
    auto ini_st = createProductState<PrecisionT>("+10");

    // Test using |+10> state

    CHECK(ini_st == std::vector<ComplexPrecisionT>{
                        Util::ZERO<PrecisionT>(), Util::ZERO<PrecisionT>(),
                        Util::INVSQRT2<PrecisionT>(), Util::ZERO<PrecisionT>(),
                        Util::ZERO<PrecisionT>(), Util::ZERO<PrecisionT>(),
                        Util::INVSQRT2<PrecisionT>(),
                        Util::ZERO<PrecisionT>()});

    DYNAMIC_SECTION(GateImplementation::name
                    << ", SWAP0,1 |+10> -> |1+0> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexPrecisionT> expected{
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(1.0 / sqrt(2), 0),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(1.0 / sqrt(2), 0),
            Util::ZERO<PrecisionT>()};
        auto sv01 = ini_st;
        auto sv10 = ini_st;

        GateImplementation::applySWAP(sv01.data(), num_qubits, {0, 1}, false);
        GateImplementation::applySWAP(sv10.data(), num_qubits, {1, 0}, false);

        CHECK(sv01 == expected);
        CHECK(sv10 == expected);
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", SWAP0,2 |+10> -> |01+> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexPrecisionT> expected{
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(1.0 / sqrt(2), 0),
            std::complex<PrecisionT>(1.0 / sqrt(2), 0),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>()};

        auto sv02 = ini_st;
        auto sv20 = ini_st;

        GateImplementation::applySWAP(sv02.data(), num_qubits, {0, 2}, false);
        GateImplementation::applySWAP(sv20.data(), num_qubits, {2, 0}, false);

        CHECK(sv02 == expected);
        CHECK(sv20 == expected);
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SWAP1,2 |+10> -> |+01> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexPrecisionT> expected{
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(1.0 / sqrt(2), 0),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(1.0 / sqrt(2), 0),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>()};

        auto sv12 = ini_st;
        auto sv21 = ini_st;

        GateImplementation::applySWAP(sv12.data(), num_qubits, {1, 2}, false);
        GateImplementation::applySWAP(sv21.data(), num_qubits, {2, 1}, false);

        CHECK(sv12 == expected);
        CHECK(sv21 == expected);
    }
}
PENNYLANE_RUN_TEST(SWAP);

/*******************************************************************************
 * Three-qubit gates
 ******************************************************************************/
template <typename PrecisionT, class GateImplementation>
void testApplyToffoli() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;
    auto ini_st = createProductState<PrecisionT>("+10");

    // Test using |+10> state
    DYNAMIC_SECTION(GateImplementation::name
                    << ", Toffoli 0,1,2 |+10> -> |010> + |111> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexPrecisionT> expected{
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(1.0 / sqrt(2), 0),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(1.0 / sqrt(2), 0)};

        auto sv012 = ini_st;

        GateImplementation::applyToffoli(sv012.data(), num_qubits, {0, 1, 2},
                                         false);

        CHECK(sv012 == expected);
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", Toffoli 1,0,2 |+10> -> |010> + |111> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexPrecisionT> expected{
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(1.0 / sqrt(2), 0),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(1.0 / sqrt(2), 0)};

        auto sv102 = ini_st;

        GateImplementation::applyToffoli(sv102.data(), num_qubits, {1, 0, 2},
                                         false);

        CHECK(sv102 == expected);
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", Toffoli 0,2,1 |+10> -> |+10> - "
                    << PrecisionToName<PrecisionT>::value) {
        const auto &expected = ini_st;

        auto sv021 = ini_st;

        GateImplementation::applyToffoli(sv021.data(), num_qubits, {0, 2, 1},
                                         false);

        CHECK(sv021 == expected);
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", Toffoli 1,2,0 |+10> -> |+10> - "
                    << PrecisionToName<PrecisionT>::value) {
        const auto &expected = ini_st;

        auto sv120 = ini_st;
        GateImplementation::applyToffoli(sv120.data(), num_qubits, {1, 2, 0},
                                         false);
        CHECK(sv120 == expected);
    }
}
PENNYLANE_RUN_TEST(Toffoli);

template <typename PrecisionT, class GateImplementation> void testApplyCSWAP() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;

    auto ini_st =
        createProductState<PrecisionT>("+10"); // Test using |+10> state

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CSWAP 0,1,2 |+10> -> |010> + |101> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexPrecisionT> expected{
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(1.0 / sqrt(2), 0),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(1.0 / sqrt(2), 0),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>()};

        auto sv012 = ini_st;
        GateImplementation::applyCSWAP(sv012.data(), num_qubits, {0, 1, 2},
                                       false);
        CHECK(sv012 == expected);
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CSWAP 1,0,2 |+10> -> |01+> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexPrecisionT> expected{
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            std::complex<PrecisionT>(1.0 / sqrt(2), 0),
            std::complex<PrecisionT>(1.0 / sqrt(2), 0),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>(),
            Util::ZERO<PrecisionT>()};

        auto sv102 = ini_st;
        GateImplementation::applyCSWAP(sv102.data(), num_qubits, {1, 0, 2},
                                       false);
        CHECK(sv102 == expected);
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", CSWAP 2,1,0 |+10> -> |+10> - "
                    << PrecisionToName<PrecisionT>::value) {
        const auto &expected = ini_st;

        auto sv210 = ini_st;
        GateImplementation::applyCSWAP(sv210.data(), num_qubits, {2, 1, 0},
                                       false);
        CHECK(sv210 == expected);
    }
}
PENNYLANE_RUN_TEST(CSWAP);
