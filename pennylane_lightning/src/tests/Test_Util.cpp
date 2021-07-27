#include <algorithm>
#include <complex>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "Util.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane;

/**
 * @brief This tests the compile-time calculation of a given multiplication.
 */
TEMPLATE_TEST_CASE("Util::ConstMult", "[StateVector]", float, double) {
    constexpr TestType r_val = 0.679;
    constexpr std::complex<TestType> c0_val{1.321, -0.175};
    constexpr std::complex<TestType> c1_val{0.579, 1.334};

    SECTION("Real times Complex") {
        constexpr std::complex<TestType> result =
            Util::ConstMult(r_val, c0_val);
        const std::complex<TestType> expected = r_val * c0_val;
        CHECK(isApproxEqual(result, expected));
    }
    SECTION("Complex times Complex") {
        constexpr std::complex<TestType> result =
            Util::ConstMult(c0_val, c1_val);
        const std::complex<TestType> expected = c0_val * c1_val;
        CHECK(isApproxEqual(result, expected));
    }
}

TEMPLATE_TEST_CASE("Constant values", "[StateVector]", float, double) {
    SECTION("One"){
        CHECK(Util::ONE<TestType>() == std::complex<TestType>{1,0});
    }
    SECTION("Zero"){
        CHECK(Util::ZERO<TestType>() == std::complex<TestType>{0,0});
    }
    SECTION("Imag"){
        CHECK(Util::IMAG<TestType>() == std::complex<TestType>{0,1});
    }
    SECTION("Sqrt2"){
        CHECK(Util::SQRT2<TestType>() == std::sqrt(static_cast<TestType>(2)));
    }
    SECTION("Inverse Sqrt2"){
        CHECK(Util::INVSQRT2<TestType>() == static_cast<TestType>(1/std::sqrt(2)));
    }
    SECTION("2^n"){
        for (size_t i = 0; i < 10; i++){
            CHECK(Util::exp2(i) == static_cast<size_t>(std::pow(2,i)));
        }
    }
}