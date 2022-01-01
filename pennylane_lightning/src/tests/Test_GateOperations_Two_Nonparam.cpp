#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "GateOperationsLM.hpp"
#include "GateOperationsPI.hpp"
#include "Gates.hpp"
#include "Util.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane;

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyCNOT",
                           "[GateOperations_Two_Nonparam]",
                           (GateOperationsPI, GateOperationsLM),
                           (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;
    auto st = create_zero_state<fp_t>(num_qubits);

    // Test using |+00> state to generate 3-qubit GHZ state
    TestType::applyHadamard(st.data(), num_qubits, {0}, false);

    // correct this when LM kernels are fully developed
    if constexpr (array_has(TestType::implemented_gates, GateOperations::CNOT)) {
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
                           "[GateOperations_Two_Nonparam]",
                           (GateOperationsPI, GateOperationsLM),
                           (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;
    auto ini_st = create_zero_state<fp_t>(num_qubits);

    // Test using |+10> state
    TestType::applyHadamard(ini_st.data(), num_qubits, {0}, false);
    TestType::applyPauliX(ini_st.data(), num_qubits, {1}, false);

    // correct this when LM kernels are fully developed
    if constexpr (array_has(TestType::implemented_gates, GateOperations::SWAP)) {
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
TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyCZ",
                           "[GateOperations_Two_Nonparam]",
                           (GateOperationsPI, GateOperationsLM),
                           (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;

    auto ini_st = create_zero_state<fp_t>(num_qubits);

    // Test using |+10> state
    TestType::applyHadamard(ini_st.data(), num_qubits, {0}, false);
    TestType::applyPauliX(ini_st.data(), num_qubits, {1}, false);

    // correct this when LM kernels are fully developed
    if constexpr (array_has(TestType::implemented_gates, GateOperations::CZ)) {
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
