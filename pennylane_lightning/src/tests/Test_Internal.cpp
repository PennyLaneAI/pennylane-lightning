#include "GateImplementationsPI.hpp"
#include "TestHelpers.hpp"

#include <catch2/catch.hpp>

/**
 * We test internal functions for test suite.
 */

TEMPLATE_TEST_CASE("createProductState", "[Test_Internal]", float, double) {
    using PrecisionT = TestType;
    using Pennylane::GateImplementationsPI;
    using TestHelper::Approx;

    SECTION("createProductState(\"+-0\") == |+-0> ") {
        const auto st = createProductState<PrecisionT>("+-0");

        auto expected = createZeroState<PrecisionT>(3);
        GateImplementationsPI::applyHadamard(expected.data(), 3, {0}, false);

        GateImplementationsPI::applyPauliX(expected.data(), 3, {1}, false);
        GateImplementationsPI::applyHadamard(expected.data(), 3, {1}, false);

        REQUIRE_THAT(st, Approx(expected).margin(1e-7));
    }
    SECTION("createProductState(\"+-0\") == |+-1> ") {
        const auto st = createProductState<PrecisionT>("+-0");

        auto expected = createZeroState<PrecisionT>(3);
        GateImplementationsPI::applyHadamard(expected.data(), 3, {0}, false);

        GateImplementationsPI::applyPauliX(expected.data(), 3, {1}, false);
        GateImplementationsPI::applyHadamard(expected.data(), 3, {1}, false);

        GateImplementationsPI::applyPauliX(expected.data(), 3, {2}, false);

        REQUIRE_THAT(st, !Approx(expected).margin(1e-7));
    }
}
