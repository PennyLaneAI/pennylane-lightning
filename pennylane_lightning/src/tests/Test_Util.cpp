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
 * @brief This tests the compile-time calculation of a given scalar
 * multiplication.
 */
TEMPLATE_TEST_CASE("Util::ConstMult", "[Util]", float, double) {
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

TEMPLATE_TEST_CASE("Constant values", "[Util]", float, double) {
    SECTION("One") {
        CHECK(Util::ONE<TestType>() == std::complex<TestType>{1, 0});
    }
    SECTION("Zero") {
        CHECK(Util::ZERO<TestType>() == std::complex<TestType>{0, 0});
    }
    SECTION("Imag") {
        CHECK(Util::IMAG<TestType>() == std::complex<TestType>{0, 1});
    }
    SECTION("Sqrt2") {
        CHECK(Util::SQRT2<TestType>() == std::sqrt(static_cast<TestType>(2)));
    }
    SECTION("Inverse Sqrt2") {
        CHECK(Util::INVSQRT2<TestType>() ==
              static_cast<TestType>(1 / std::sqrt(2)));
    }
}

TEMPLATE_TEST_CASE("Utility math functions", "[Util]", float, double) {
    SECTION("exp2: 2^n") {
        for (size_t i = 0; i < 10; i++) {
            CHECK(Util::exp2(i) == static_cast<size_t>(std::pow(2, i)));
        }
    }
    SECTION("maxDecimalForQubit") {
        for (size_t num_qubits = 0; num_qubits < 4; num_qubits++) {
            for (size_t index = 0; index < num_qubits; index++) {
                CHECK(Util::maxDecimalForQubit(index, num_qubits) ==
                      static_cast<size_t>(std::pow(2, num_qubits - index - 1)));
            }
        }
    }
    SECTION("dimSize") {
        using namespace Catch::Matchers;
        for (size_t i = 0; i < 64; i++) {
            std::vector<size_t> data(i);
            TestType rem;
            TestType f2 = std::modf(sqrt(i), &rem);
            if (i < 4) {
                CHECK_THROWS_AS(Util::dimSize(data), std::invalid_argument);
                CHECK_THROWS_WITH(
                    Util::dimSize(data),
                    Contains("The dataset must be at least 2x2"));
            } else if (rem != 0.0 && i >= 4 && (i & (i - 1))) {
                CHECK_THROWS_AS(Util::dimSize(data), std::invalid_argument);
                CHECK_THROWS_WITH(Util::dimSize(data),
                                  Contains("The dataset must be a power of 2"));
            } else if (std::sqrt(i) * std::sqrt(i) != i) {
                CHECK_THROWS_AS(Util::dimSize(data), std::invalid_argument);
                CHECK_THROWS_WITH(
                    Util::dimSize(data),
                    Contains("The dataset must be a perfect square"));
            } else {
                CHECK(Util::dimSize(data) == std::log2(std::sqrt(i)));
            }
        }
    }
    SECTION("innerProd") {
        SECTION("Iterative increment") {
            for (size_t i = 0; i < 12; i++) {
                std::vector<std::complex<double>> data1(1UL << i, {1, 1});
                std::vector<std::complex<double>> data2(1UL << i, {1, 1});
                std::complex<double> expected_result(0, 1UL << (i + 1));
                std::complex<double> result = Util::innerProd(data1, data2);
                CHECK(isApproxEqual(result, expected_result));
            }
        }
        SECTION("Random complex") {
            std::vector<std::complex<double>> data1{
                {0.326417, 0},  {-0, 0.343918}, {0, 0.508364}, {-0.53562, -0},
                {0, -0.178322}, {0.187883, -0}, {0.277721, 0}, {-0, 0.292611}};
            std::vector<std::complex<double>> data2{
                {0, -0.479426}, {0, 0}, {2.77556e-17, 0}, {0, 0},
                {0.877583, 0},  {0, 0}, {0, 0},           {0, 0}};
            std::complex<double> expected_result(0, -0.312985152368);
            std::complex<double> result = Util::innerProd(data1, data2);
            CHECK(isApproxEqual(result, expected_result));
        }
    }
    SECTION("innerProdC") {
        SECTION("Iterative increment") {
            for (size_t i = 0; i < 12; i++) {
                std::vector<std::complex<double>> data1(1UL << i, {1, 1});
                std::vector<std::complex<double>> data2(1UL << i, {1, 1});
                std::complex<double> expected_result(1UL << (i + 1), 0);
                std::complex<double> result = Util::innerProdC(data1, data2);
                CAPTURE(result);
                CAPTURE(expected_result);
                CHECK(isApproxEqual(result, expected_result));
            }
        }
        SECTION("Random complex") {
            std::vector<std::complex<double>> data2{
                {0, -0.479426}, {0, 0}, {2.77556e-17, 0}, {0, 0},
                {0.877583, 0},  {0, 0}, {0, 0},           {0, 0}};
            std::vector<std::complex<double>> data1{
                {0.326417, 0},  {-0, 0.343918}, {0, 0.508364}, {-0.53562, -0},
                {0, -0.178322}, {0.187883, -0}, {0.277721, 0}, {-0, 0.292611}};
            std::complex<double> expected_result(0, -4.40916e-7);
            std::complex<double> result = Util::innerProdC(data1, data2);
            CAPTURE(result);
            CAPTURE(expected_result);
            CHECK(isApproxEqual(result, expected_result, 1e-5));
        }
    }
}
