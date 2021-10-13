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
                CHECK_THROWS_WITH(Util::dimSize(data),
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
    SECTION("matrixVecProd") {
        SECTION("Simple Iterative") {
            for (size_t m = 2; m < 8; ++m) {
                std::vector<std::complex<double>> mat(m * m, {1, 1});
                std::vector<std::complex<double>> v_in(m, {1, 1});
                std::vector<std::complex<double>> v_expected(
                    m, {0, static_cast<double>(2 * m)});
                std::vector<std::complex<double>> v_out =
                    Util::matrixVecProd(mat, v_in, m, m);
                for (size_t i = 0; i < m; ++i)
                    CHECK(isApproxEqual(v_out[i], v_expected[i]));
            }
        }
        SECTION("Random Complex") {
            std::vector<std::complex<double>> mat{
                {0.417876, 0.27448},   {0.601209, 0.723548},
                {0.781624, 0.538222},  {0.0597232, 0.27755},
                {0.0431741, 0.593319}, {0.224124, 0.130335},
                {0.237877, 0.01557},   {0.931634, 0.786367},
                {0.378397, 0.894381},  {0.840747, 0.889789},
                {0.530623, 0.463644},  {0.868736, 0.760685},
                {0.258175, 0.836569},  {0.495012, 0.667726},
                {0.298962, 0.384992},  {0.659472, 0.232696}};
            std::vector<std::complex<double>> v_in{{0.417876, 0.27448},
                                                   {0.601209, 0.723548},
                                                   {0.781624, 0.538222},
                                                   {0.0597232, 0.27755}};
            std::vector<std::complex<double>> v_expected{{0.184998, 1.97393},
                                                         {-0.0894368, 0.946047},
                                                         {-0.219747, 2.55541},
                                                         {-0.305997, 1.83881}};
            std::vector<std::complex<double>> v_out =
                Util::matrixVecProd(mat, v_in, 4, 4);
            for (size_t i = 0; i < 4; ++i)
                CHECK(isApproxEqual(v_out[i], v_out[i]));
        }
    }
    SECTION("Transpose") {
        SECTION("Simple Matrix") {
            for (size_t m = 2; m < 4; ++m) {
                std::vector<std::complex<double>> mat(m * m, {0, 0});
                for (size_t i = 0; i < m; ++i)
                    mat[i * m + i] = {1, 1};
                std::vector<std::complex<double>> mat_t =
                    Util::Transpose(mat, m, m);
                for (size_t i = 0; i < m * m; ++i)
                    CHECK(isApproxEqual(mat[i], mat_t[i]));
            }
        }
        SECTION("Random Complex") {
            std::vector<std::complex<double>> mat{
                {0.417876, 0.27448},   {0.601209, 0.723548},
                {0.781624, 0.538222},  {0.0597232, 0.27755},
                {0.0431741, 0.593319}, {0.224124, 0.130335},
                {0.237877, 0.01557},   {0.931634, 0.786367},
                {0.378397, 0.894381},  {0.840747, 0.889789},
                {0.530623, 0.463644},  {0.868736, 0.760685},
                {0.258175, 0.836569},  {0.495012, 0.667726},
                {0.298962, 0.384992},  {0.659472, 0.232696}};
            std::vector<std::complex<double>> mat_t_expected{
                {0.417876, 0.27448},  {0.0431741, 0.593319},
                {0.378397, 0.894381}, {0.258175, 0.836569},
                {0.601209, 0.723548}, {0.224124, 0.130335},
                {0.840747, 0.889789}, {0.495012, 0.667726},
                {0.781624, 0.538222}, {0.237877, 0.01557},
                {0.530623, 0.463644}, {0.298962, 0.384992},
                {0.0597232, 0.27755}, {0.931634, 0.786367},
                {0.868736, 0.760685}, {0.659472, 0.232696}};
            std::vector<std::complex<double>> mat_t =
                Util::Transpose(mat, 4, 4);
            for (size_t i = 0; i < 16; ++i)
                CHECK(isApproxEqual(mat_t[i], mat_t_expected[i]));
        }
    }
    SECTION("matrixMatProd") {
        SECTION("Simple Iterative") {
            for (size_t m = 2; m < 5; ++m) {
                std::vector<std::complex<double>> m_left(m * m, {1, 1});
                std::vector<std::complex<double>> m_right(m * m, {1, 1});
                std::vector<std::complex<double>> m_out_expected(
                    m * m, {0, static_cast<double>(2*m)});
                std::vector<std::complex<double>> m_out =
                    Util::matrixMatProd(m_left, m_right, m, m, m, 1, true);
                for (size_t i = 0; i < m * m; ++i)
                    CHECK(isApproxEqual(m_out[i], m_out_expected[i]));
            }
        }
        SECTION("Random Complex") {
            std::vector<std::complex<double>> m_left{
                (0.94007, 0.424517),  (0.256163, 0.0615097),
                (0.505297, 0.343107), (0.729021, 0.241991),
                (0.860825, 0.633264), (0.987668, 0.195166),
                (0.606897, 0.144482), (0.0183697, 0.375071),
                (0.355853, 0.152383), (0.985341, 0.0888863),
                (0.608352, 0.653375), (0.268477, 0.58398),
                (0.960381, 0.786669), (0.498357, 0.185307),
                (0.283511, 0.844801), (0.269318, 0.792981)};
            std::vector<std::complex<double>> m_right{
                (0.94007, 0.424517),  (0.256163, 0.0615097),
                (0.505297, 0.343107), (0.729021, 0.241991),
                (0.860825, 0.633264), (0.987668, 0.195166),
                (0.606897, 0.144482), (0.0183697, 0.375071),
                (0.355853, 0.152383), (0.985341, 0.0888863),
                (0.608352, 0.653375), (0.268477, 0.58398),
                (0.960381, 0.786669), (0.498357, 0.185307),
                (0.283511, 0.844801), (0.269318, 0.792981)};
            std::vector<std::complex<double>> m_out_1 =
                Util::matrixMatProd(m_left, m_right, 4, 4, 4, 1, true);
            std::vector<std::complex<double>> m_out_2 =
                Util::matrixMatProd(m_left, m_right, 4, 4, 4, 1, false);
            for (size_t i = 0; i < 16; ++i) {
                CHECK(isApproxEqual(m_out_1[i], m_out_2[i]));
            }
        }
    }
}
