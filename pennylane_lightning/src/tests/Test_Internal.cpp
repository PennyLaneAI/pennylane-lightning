#include "TestHelpers.hpp"
#include "GateImplementationsPI.hpp"

#include <catch2/catch.hpp>

/**
 * We test internal functions for test suite.
 */

TEMPLATE_TEST_CASE("create_product_state", "[Test_Internal]",
        float, double) {
    using PrecisionT = TestType;
    using TestHelper::Approx;
    using Pennylane::GateImplementationsPI;

    SECTION("create_product_state(\"+-0\") == |+-0> ") {
        const auto st = create_product_state<PrecisionT>("+-0");

        auto expected = create_zero_state<PrecisionT>(3);
        GateImplementationsPI::applyHadamard(expected.data(), 3, {0}, false);

        GateImplementationsPI::applyPauliX(expected.data(), 3, {1}, false);
        GateImplementationsPI::applyHadamard(expected.data(), 3, {1}, false);
        
        REQUIRE_THAT(st, Approx(expected).margin(1e-7));
    }
    SECTION("create_product_state(\"+-0\") == |+-1> ") {
        const auto st = create_product_state<PrecisionT>("+-0");

        auto expected = create_zero_state<PrecisionT>(3);
        GateImplementationsPI::applyHadamard(expected.data(), 3, {0}, false);

        GateImplementationsPI::applyPauliX(expected.data(), 3, {1}, false);
        GateImplementationsPI::applyHadamard(expected.data(), 3, {1}, false);

        GateImplementationsPI::applyPauliX(expected.data(), 3, {2}, false);

        
        REQUIRE_THAT(st, !Approx(expected).margin(1e-7));
    }
}
