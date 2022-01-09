#include "GateOperationsLM.hpp"
#include "GateOperationsPI.hpp"
#include "Gates.hpp"
#include "TestHelpers.hpp"
#include "TestMacros.hpp"
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
 * @file This file contains tests for non-parameterized gates. List of such
 * gates are [PauliX, PauliY, PauliZ, Hadamard, S, T, CNOT, SWAP, CZ, Toffoli,
 * CSWAP].
 */
using namespace Pennylane;

namespace {
using std::vector;
}

/*******************************************************************************
 * Single-qubit gates
 ******************************************************************************/

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyPauliX",
                           "[GateOperations_Nonparam],[single-qubit]",
                           ALL_GATE_OPERATIONS, (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;
    for (size_t index = 0; index < num_qubits; index++) {
        auto st = create_zero_state<fp_t>(num_qubits);
        CHECK(st[0] == Util::ONE<fp_t>());

        TestType::applyPauliX(st.data(), num_qubits, {index}, false);
        CHECK(st[0] == Util::ZERO<fp_t>());
        CHECK(st[0b1 << (num_qubits - index - 1)] == Util::ONE<fp_t>());
    }
}

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyPauliY",
                           "[GateOperations_Nonparam],[single-qubit]",
                           ALL_GATE_OPERATIONS, (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;

    constexpr CFP_t p = Util::ConstMult(
        static_cast<fp_t>(0.5),
        Util::ConstMult(Util::INVSQRT2<fp_t>(), Util::IMAG<fp_t>()));
    constexpr CFP_t m = Util::ConstMult(-1, p);

    const std::vector<std::vector<CFP_t>> expected_results = {
        {m, m, m, m, p, p, p, p},
        {m, m, p, p, m, m, p, p},
        {m, p, m, p, m, p, m, p}};

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = create_plus_state<fp_t>(num_qubits);

        TestType::applyPauliY(st.data(), num_qubits, {index}, false);

        CHECK(isApproxEqual(st, expected_results[index]));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyPauliZ",
                           "[GateOperations_Nonparam],[single-qubit]",
                           ALL_GATE_OPERATIONS, (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;

    constexpr CFP_t p(static_cast<fp_t>(0.5) * Util::INVSQRT2<fp_t>());
    constexpr CFP_t m(Util::ConstMult(-1, p));

    const std::vector<std::vector<CFP_t>> expected_results = {
        {p, p, p, p, m, m, m, m},
        {p, p, m, m, p, p, m, m},
        {p, m, p, m, p, m, p, m}};

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = create_plus_state<fp_t>(num_qubits);
        TestType::applyPauliZ(st.data(), num_qubits, {index}, false);

        CHECK(isApproxEqual(st, expected_results[index]));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyHadamard",
                           "[GateOperations_Nonparam],[single-qubit]",
                           (GateOperationsLM, GateOperationsPI),
                           (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;
    for (size_t index = 0; index < num_qubits; index++) {
        auto st = create_zero_state<fp_t>(num_qubits);

        CHECK(st[0] == CFP_t{1, 0});
        TestType::applyHadamard(st.data(), num_qubits, {index}, false);

        CFP_t expected(1 / std::sqrt(2), 0);
        CHECK(expected.real() == Approx(st[0].real()));
        CHECK(expected.imag() == Approx(st[0].imag()));

        CHECK(expected.real() ==
              Approx(st[0b1 << (num_qubits - index - 1)].real()));
        CHECK(expected.imag() ==
              Approx(st[0b1 << (num_qubits - index - 1)].imag()));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyS",
                           "[GateOperations_Nonparam],[single-qubit]",
                           ALL_GATE_OPERATIONS, (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;

    constexpr CFP_t r(static_cast<fp_t>(0.5) * Util::INVSQRT2<fp_t>());
    constexpr CFP_t i(Util::ConstMult(r, Util::IMAG<fp_t>()));

    const std::vector<std::vector<CFP_t>> expected_results = {
        {r, r, r, r, i, i, i, i},
        {r, r, i, i, r, r, i, i},
        {r, i, r, i, r, i, r, i}};

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = create_plus_state<fp_t>(num_qubits);

        TestType::applyS(st.data(), num_qubits, {index}, false);

        CHECK(isApproxEqual(st, expected_results[index]));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyT",
                           "[GateOperations_Nonparam],[single-qubit]",
                           ALL_GATE_OPERATIONS, (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;
    // Test using |+++> state

    CFP_t r(1.0 / (2.0 * std::sqrt(2)), 0);
    CFP_t i(1.0 / 4, 1.0 / 4);

    const std::vector<std::vector<CFP_t>> expected_results = {
        {r, r, r, r, i, i, i, i},
        {r, r, i, i, r, r, i, i},
        {r, i, r, i, r, i, r, i}};

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = create_plus_state<fp_t>(num_qubits);

        TestType::applyT(st.data(), num_qubits, {index}, false);

        CHECK(isApproxEqual(st, expected_results[index]));
    }
}

/*******************************************************************************
 * Two-qubit gates
 ******************************************************************************/

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyCNOT",
                           "[GateOperations_Nonparam],[two-qubit]",
                           ALL_GATE_OPERATIONS, (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;
    auto st = create_zero_state<fp_t>(num_qubits);

    // Test using |+00> state to generate 3-qubit GHZ state
    TestType::applyHadamard(st.data(), num_qubits, {0}, false);

    if constexpr (array_has_elt(TestType::implemented_gates,
                                GateOperations::CNOT)) {
        for (size_t index = 1; index < num_qubits; index++) {
            TestType::applyCNOT(st.data(), num_qubits, {index - 1, index},
                                false);
        }
        CHECK(st.front() == Util::INVSQRT2<fp_t>());
        CHECK(st.back() == Util::INVSQRT2<fp_t>());
    }
}

// NOLINTNEXTLINE: Avoiding complexity errors
TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applySWAP",
                           "[GateOperations_Nonparam],[two-qubit]",
                           ALL_GATE_OPERATIONS, (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;
    auto ini_st = create_zero_state<fp_t>(num_qubits);

    // Test using |+10> state
    TestType::applyHadamard(ini_st.data(), num_qubits, {0}, false);
    TestType::applyPauliX(ini_st.data(), num_qubits, {1}, false);

    if constexpr (array_has_elt(TestType::implemented_gates,
                                GateOperations::SWAP)) {
        CHECK(ini_st ==
              std::vector<CFP_t>{Util::ZERO<fp_t>(), Util::ZERO<fp_t>(),
                                 Util::INVSQRT2<fp_t>(), Util::ZERO<fp_t>(),
                                 Util::ZERO<fp_t>(), Util::ZERO<fp_t>(),
                                 Util::INVSQRT2<fp_t>(), Util::ZERO<fp_t>()});

        SECTION("SWAP0,1 |+10> -> |1+0>") {
            std::vector<CFP_t> expected{Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        std::complex<fp_t>(1.0 / sqrt(2), 0),
                                        Util::ZERO<fp_t>(),
                                        std::complex<fp_t>(1.0 / sqrt(2), 0),
                                        Util::ZERO<fp_t>()};
            auto sv01 = ini_st;
            auto sv10 = ini_st;

            TestType::applySWAP(sv01.data(), num_qubits, {0, 1}, false);
            TestType::applySWAP(sv10.data(), num_qubits, {1, 0}, false);

            CHECK(sv01 == expected);
            CHECK(sv10 == expected);
        }

        SECTION("SWAP0,2 |+10> -> |01+>") {
            std::vector<CFP_t> expected{Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        std::complex<fp_t>(1.0 / sqrt(2), 0),
                                        std::complex<fp_t>(1.0 / sqrt(2), 0),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>()};

            auto sv02 = ini_st;
            auto sv20 = ini_st;

            TestType::applySWAP(sv02.data(), num_qubits, {0, 2}, false);
            TestType::applySWAP(sv20.data(), num_qubits, {2, 0}, false);

            CHECK(sv02 == expected);
            CHECK(sv20 == expected);
        }
        SECTION("SWAP1,2 |+10> -> |+01>") {
            std::vector<CFP_t> expected{
                Util::ZERO<fp_t>(), std::complex<fp_t>(1.0 / sqrt(2), 0),
                Util::ZERO<fp_t>(), Util::ZERO<fp_t>(),
                Util::ZERO<fp_t>(), std::complex<fp_t>(1.0 / sqrt(2), 0),
                Util::ZERO<fp_t>(), Util::ZERO<fp_t>()};

            auto sv12 = ini_st;
            auto sv21 = ini_st;

            TestType::applySWAP(sv12.data(), num_qubits, {1, 2}, false);
            TestType::applySWAP(sv21.data(), num_qubits, {2, 1}, false);

            CHECK(sv12 == expected);
            CHECK(sv21 == expected);
        }
    }
}

// NOLINTNEXTLINE: Avoiding complexity errors
TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyCY",
                           "[GateOperations_Nonparam],[two-qubit]",
                           ALL_GATE_OPERATIONS, (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;
    auto ini_st = create_zero_state<fp_t>(num_qubits);

    // Test using |+10> state
    TestType::applyHadamard(ini_st.data(), num_qubits, {0}, false);
    TestType::applyPauliX(ini_st.data(), num_qubits, {1}, false);

    if constexpr (array_has_elt(TestType::implemented_gates,
                                GateOperations::CY)) {
        CHECK(ini_st ==
              std::vector<CFP_t>{
                  Util::ZERO<fp_t>(), Util::ZERO<fp_t>(),
                  std::complex<fp_t>(1.0 / sqrt(2), 0), Util::ZERO<fp_t>(),
                  Util::ZERO<fp_t>(), Util::ZERO<fp_t>(),
                  std::complex<fp_t>(1.0 / sqrt(2), 0), Util::ZERO<fp_t>()});

        SECTION("CY 0,1 |+10> -> i|100>") {
            std::vector<CFP_t> expected{Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        std::complex<fp_t>(1.0 / sqrt(2), 0),
                                        Util::ZERO<fp_t>(),
                                        std::complex<fp_t>(0, -1 / sqrt(2)),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>()};

            auto sv01 = ini_st;
            TestType::applyCY(sv01.data(), num_qubits, {0, 1}, false);
            CHECK(sv01 == expected);
        }

        SECTION("CY 0,2 |+10> -> |010> + i |111>") {
            std::vector<CFP_t> expected{Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        std::complex<fp_t>(1.0 / sqrt(2), 0.0),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        std::complex<fp_t>(0.0, 1 / sqrt(2))};

            auto sv02 = ini_st;

            TestType::applyCY(sv02.data(), num_qubits, {0, 2}, false);
            CHECK(sv02 == expected);
        }
        SECTION("CY 1,2 |+10> -> i|+11>") {
            std::vector<CFP_t> expected{
                Util::ZERO<fp_t>(), Util::ZERO<fp_t>(),
                Util::ZERO<fp_t>(), std::complex<fp_t>(0.0, 1.0 / sqrt(2)),
                Util::ZERO<fp_t>(), Util::ZERO<fp_t>(),
                Util::ZERO<fp_t>(), std::complex<fp_t>(0.0, 1 / sqrt(2))};

            auto sv12 = ini_st;

            TestType::applyCY(sv12.data(), num_qubits, {1, 2}, false);
            CHECK(sv12 == expected);
        }
    }
}

// NOLINTNEXTLINE: Avoiding complexity errors
TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyCZ",
                           "[GateOperations_Nonparam],[two-qubit]",
                           ALL_GATE_OPERATIONS, (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;

    auto ini_st = create_zero_state<fp_t>(num_qubits);

    // Test using |+10> state
    TestType::applyHadamard(ini_st.data(), num_qubits, {0}, false);
    TestType::applyPauliX(ini_st.data(), num_qubits, {1}, false);

    if constexpr (array_has_elt(TestType::implemented_gates,
                                GateOperations::CZ)) {
        auto st = ini_st;
        CHECK(st == std::vector<CFP_t>{Util::ZERO<fp_t>(), Util::ZERO<fp_t>(),
                                       std::complex<fp_t>(1.0 / sqrt(2), 0),
                                       Util::ZERO<fp_t>(), Util::ZERO<fp_t>(),
                                       Util::ZERO<fp_t>(),
                                       std::complex<fp_t>(1.0 / sqrt(2), 0),
                                       Util::ZERO<fp_t>()});

        SECTION("CZ0,1 |+10> -> |-10>") {
            std::vector<CFP_t> expected{Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        std::complex<fp_t>(1.0 / sqrt(2), 0),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        std::complex<fp_t>(-1 / sqrt(2), 0),
                                        Util::ZERO<fp_t>()};

            auto sv01 = ini_st;
            auto sv10 = ini_st;

            TestType::applyCZ(sv01.data(), num_qubits, {0, 1}, false);
            TestType::applyCZ(sv10.data(), num_qubits, {1, 0}, false);

            CHECK(sv01 == expected);
            CHECK(sv10 == expected);
        }

        SECTION("CZ0,2 |+10> -> |+10>") {
            const std::vector<CFP_t> &expected{ini_st};

            auto sv02 = ini_st;
            auto sv20 = ini_st;

            TestType::applyCZ(sv02.data(), num_qubits, {0, 2}, false);
            TestType::applyCZ(sv20.data(), num_qubits, {2, 0}, false);

            CHECK(sv02 == expected);
            CHECK(sv20 == expected);
        }
        SECTION("CZ1,2 |+10> -> |+10>") {
            const std::vector<CFP_t> &expected{ini_st};

            auto sv12 = ini_st;
            auto sv21 = ini_st;

            TestType::applyCZ(sv12.data(), num_qubits, {1, 2}, false);
            TestType::applyCZ(sv21.data(), num_qubits, {2, 1}, false);

            CHECK(sv12 == expected);
            CHECK(sv21 == expected);
        }
    }
}

/*******************************************************************************
 * Three-qubit gates
 ******************************************************************************/

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyToffoli",
                           "[GateOperations_Nonparam],[three-qubit]",
                           ALL_GATE_OPERATIONS, (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;
    auto ini_st = create_zero_state<fp_t>(num_qubits);

    // Test using |+10> state
    TestType::applyHadamard(ini_st.data(), num_qubits, {0}, false);
    TestType::applyPauliX(ini_st.data(), num_qubits, {1}, false);

    if constexpr (array_has_elt(TestType::implemented_gates,
                                GateOperations::Toffoli)) {
        SECTION("Toffoli 0,1,2 |+10> -> |010> + |111>") {
            std::vector<CFP_t> expected{Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        std::complex<fp_t>(1.0 / sqrt(2), 0),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        std::complex<fp_t>(1.0 / sqrt(2), 0)};

            auto sv012 = ini_st;

            TestType::applyToffoli(sv012.data(), num_qubits, {0, 1, 2}, false);

            CHECK(sv012 == expected);
        }

        SECTION("Toffoli 1,0,2 |+10> -> |010> + |111>") {
            std::vector<CFP_t> expected{Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        std::complex<fp_t>(1.0 / sqrt(2), 0),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        std::complex<fp_t>(1.0 / sqrt(2), 0)};

            auto sv102 = ini_st;

            TestType::applyToffoli(sv102.data(), num_qubits, {1, 0, 2}, false);

            CHECK(sv102 == expected);
        }

        SECTION("Toffoli 0,2,1 |+10> -> |+10>") {
            const auto &expected = ini_st;

            auto sv021 = ini_st;

            TestType::applyToffoli(sv021.data(), num_qubits, {0, 2, 1}, false);

            CHECK(sv021 == expected);
        }

        SECTION("Toffoli 1,2,0 |+10> -> |+10>") {
            const auto &expected = ini_st;

            auto sv120 = ini_st;
            TestType::applyToffoli(sv120.data(), num_qubits, {1, 2, 0}, false);
            CHECK(sv120 == expected);
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyCSWAP",
                           "[GateOperations_Nonparam],[three-qubit]",
                           ALL_GATE_OPERATIONS, (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;

    auto ini_st = create_zero_state<fp_t>(num_qubits);

    // Test using |+10> state
    TestType::applyHadamard(ini_st.data(), num_qubits, {0}, false);
    TestType::applyPauliX(ini_st.data(), num_qubits, {1}, false);

    if constexpr (array_has_elt(TestType::implemented_gates,
                                GateOperations::CSWAP)) {
        SECTION("CSWAP 0,1,2 |+10> -> |010> + |101>") {
            std::vector<CFP_t> expected{Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        std::complex<fp_t>(1.0 / sqrt(2), 0),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        std::complex<fp_t>(1.0 / sqrt(2), 0),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>()};

            auto sv012 = ini_st;
            TestType::applyCSWAP(sv012.data(), num_qubits, {0, 1, 2}, false);
            CHECK(sv012 == expected);
        }

        SECTION("CSWAP 1,0,2 |+10> -> |01+>") {
            std::vector<CFP_t> expected{Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        std::complex<fp_t>(1.0 / sqrt(2), 0),
                                        std::complex<fp_t>(1.0 / sqrt(2), 0),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>(),
                                        Util::ZERO<fp_t>()};

            auto sv102 = ini_st;
            TestType::applyCSWAP(sv102.data(), num_qubits, {1, 0, 2}, false);
            CHECK(sv102 == expected);
        }
        SECTION("CSWAP 2,1,0 |+10> -> |+10>") {
            const auto &expected = ini_st;

            auto sv210 = ini_st;
            TestType::applyCSWAP(sv210.data(), num_qubits, {2, 1, 0}, false);
            CHECK(sv210 == expected);
        }
    }
}
