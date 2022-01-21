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

// NOLINTNEXTLINE: Avoid complexity errors
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
            for (size_t m = 2; m < 8; m++) {
                std::vector<std::complex<double>> mat(m * m, {1, 1});
                std::vector<std::complex<double>> v_in(m, {1, 1});
                std::vector<std::complex<double>> v_expected(
                    m, {0, static_cast<double>(2 * m)});
                std::vector<std::complex<double>> v_out =
                    Util::matrixVecProd(mat, v_in, m, m);
                CAPTURE(v_out);
                CAPTURE(v_expected);
                for (size_t i = 0; i < m; i++) {
                    CHECK(isApproxEqual(v_out[i], v_expected[i]));
                }
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
            CAPTURE(v_out);
            for (size_t i = 0; i < 4; i++) {
                CHECK(isApproxEqual(v_out[i], v_out[i]));
            }
        }
        SECTION("Invalid Arguments") {
            using namespace Catch::Matchers;
            std::vector<std::complex<double>> mat(2 * 3, {1, 1});
            std::vector<std::complex<double>> v_in(2, {1, 1});
            CHECK_THROWS_AS(Util::matrixVecProd(mat, v_in, 2, 3),
                            std::invalid_argument);
            CHECK_THROWS_WITH(Util::matrixVecProd(mat, v_in, 2, 3),
                              Contains("Invalid size for the input vector"));
            CHECK_THROWS_AS(Util::matrixVecProd(mat, v_in, 2, 2),
                            std::invalid_argument);
            CHECK_THROWS_WITH(
                Util::matrixVecProd(mat, v_in, 2, 2),
                Contains(
                    "Invalid number of rows and columns for the input matrix"));
        }
    }
    SECTION("vecMatrixProd") {
        SECTION("Simple Iterative") {
            for (size_t m = 2; m < 8; m++) {
                std::vector<double> mat(m * m, 1);
                std::vector<double> v_in(m, 1);
                std::vector<double> v_expected(m, m);
                std::vector<double> v_out =
                    Util::vecMatrixProd(v_in, mat, m, m);
                CAPTURE(v_out);
                CAPTURE(v_expected);
                for (size_t i = 0; i < m; i++) {
                    CHECK(v_out[i] == v_expected[i]);
                }
            }
        }
        SECTION("Zero Vector") {
            for (size_t m = 2; m < 8; m++) {
                std::vector<double> mat(m * m, 1);
                std::vector<double> v_in(m, 0);
                std::vector<double> v_expected(m, 0);
                std::vector<double> v_out =
                    Util::vecMatrixProd(v_in, mat, m, m);
                CAPTURE(v_out);
                CAPTURE(v_expected);
                for (size_t i = 0; i < m; i++) {
                    CHECK(v_out[i] == v_expected[i]);
                }
            }
        }
        SECTION("Random Matrix") {
            std::vector<float> v_in{1.0, 2.0, 3.0, 4.0};
            std::vector<float> mat{1.0, 0.1,  0.2, 0.2,  0.6,  0.1,
                                   0.4, -0.7, 1.2, -0.5, -0.6, 0.7};
            std::vector<float> v_expected{0.6, -3.2, 6.8};
            std::vector<float> v_out = Util::vecMatrixProd(v_in, mat, 4, 3);
            CAPTURE(v_out);
            CAPTURE(v_expected);
            for (size_t i = 0; i < 3; i++) {
                CHECK(std::abs(v_out[i] - v_expected[i]) < 0.000001);
            }
        }
    }
    SECTION("Transpose") {
        SECTION("Simple Matrix") {
            for (size_t m = 2; m < 8; m++) {
                std::vector<std::complex<double>> mat(m * m, {0, 0});
                for (size_t i = 0; i < m; i++) {
                    mat[i * m + i] = {1, 1};
                }
                std::vector<std::complex<double>> mat_t =
                    Util::Transpose(mat, m, m);
                CAPTURE(mat_t);
                CAPTURE(mat);
                for (size_t i = 0; i < m * m; i++) {
                    CHECK(isApproxEqual(mat[i], mat_t[i]));
                }
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
            std::vector<std::complex<double>> mat_t_exp{
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
            CAPTURE(mat_t);
            CAPTURE(mat_t_exp);
            for (size_t i = 0; i < 16; i++) {
                CHECK(isApproxEqual(mat_t[i], mat_t_exp[i]));
            }
        }
        SECTION("Invalid Arguments") {
            using namespace Catch::Matchers;
            std::vector<std::complex<double>> mat(2 * 3, {1, 1});
            CHECK_THROWS_AS(Util::Transpose(mat, 2, 2), std::invalid_argument);
            CHECK_THROWS_WITH(
                Util::Transpose(mat, 2, 2),
                Contains(
                    "Invalid number of rows and columns for the input matrix"));
        }
    }
    SECTION("matrixMatProd") {
        SECTION("Simple Iterative") {
            for (size_t m = 2; m < 8; m++) {
                std::vector<std::complex<double>> m_left(m * m, {1, 1});
                std::vector<std::complex<double>> m_right(m * m, {1, 1});
                std::vector<std::complex<double>> m_out_exp(
                    m * m, {0, static_cast<double>(2 * m)});
                std::vector<std::complex<double>> m_out =
                    Util::matrixMatProd(m_left, m_right, m, m, m, true);
                CAPTURE(m_out);
                CAPTURE(m_out_exp);
                for (size_t i = 0; i < m * m; i++) {
                    CHECK(isApproxEqual(m_out[i], m_out_exp[i]));
                }
            }
        }
        SECTION("Random Complex") {
            std::vector<std::complex<double>> m_left{
                {0.94007, 0.424517},  {0.256163, 0.0615097},
                {0.505297, 0.343107}, {0.729021, 0.241991},
                {0.860825, 0.633264}, {0.987668, 0.195166},
                {0.606897, 0.144482}, {0.0183697, 0.375071},
                {0.355853, 0.152383}, {0.985341, 0.0888863},
                {0.608352, 0.653375}, {0.268477, 0.58398},
                {0.960381, 0.786669}, {0.498357, 0.185307},
                {0.283511, 0.844801}, {0.269318, 0.792981}};
            std::vector<std::complex<double>> m_right{
                {0.94007, 0.424517},  {0.256163, 0.0615097},
                {0.505297, 0.343107}, {0.729021, 0.241991},
                {0.860825, 0.633264}, {0.987668, 0.195166},
                {0.606897, 0.144482}, {0.0183697, 0.375071},
                {0.355853, 0.152383}, {0.985341, 0.0888863},
                {0.608352, 0.653375}, {0.268477, 0.58398},
                {0.960381, 0.786669}, {0.498357, 0.185307},
                {0.283511, 0.844801}, {0.269318, 0.792981}};
            std::vector<std::complex<double>> m_right_tp{
                {0.94007, 0.424517},   {0.860825, 0.633264},
                {0.355853, 0.152383},  {0.960381, 0.786669},
                {0.256163, 0.0615097}, {0.987668, 0.195166},
                {0.985341, 0.0888863}, {0.498357, 0.185307},
                {0.505297, 0.343107},  {0.606897, 0.144482},
                {0.608352, 0.653375},  {0.283511, 0.844801},
                {0.729021, 0.241991},  {0.0183697, 0.375071},
                {0.268477, 0.58398},   {0.269318, 0.792981}};
            std::vector<std::complex<double>> m_out_exp{
                {1.522375435807200, 2.018315393556500},
                {1.241561065671800, 0.915996420839700},
                {0.561409446565600, 1.834755796266900},
                {0.503973820211400, 1.664651528374090},
                {1.183556828429700, 2.272762769584300},
                {1.643767359748500, 0.987318478828500},
                {0.752063484100700, 1.482770126810700},
                {0.205343773497200, 1.552791421044900},
                {0.977117116888800, 2.092066653216500},
                {1.604565422784600, 1.379671036009100},
                {0.238648365886400, 1.582741563052100},
                {-0.401698027789600, 1.469264325654110},
                {0.487510164243000, 2.939585667799000},
                {0.845207296911400, 1.843583823364000},
                {-0.482010055957000, 2.062995137499000},
                {-0.524094900662100, 1.815727577737900}};
            std::vector<std::complex<double>> m_out_1 =
                Util::matrixMatProd(m_left, m_right_tp, 4, 4, 4, true);
            std::vector<std::complex<double>> m_out_2 =
                Util::matrixMatProd(m_left, m_right, 4, 4, 4, false);
            CAPTURE(m_out_1);
            CAPTURE(m_out_2);
            CAPTURE(m_out_exp);
            for (size_t i = 0; i < 16; i++) {
                CHECK(isApproxEqual(m_out_1[i], m_out_2[i]));
            }
            for (size_t i = 0; i < 16; i++) {
                CHECK(isApproxEqual(m_out_1[i], m_out_exp[i]));
            }
        }
        SECTION("Invalid Arguments") {
            using namespace Catch::Matchers;
            std::vector<std::complex<double>> m_left(2 * 3, {1, 1});
            std::vector<std::complex<double>> m_right(3 * 4, {1, 1});
            CHECK_THROWS_AS(Util::matrixMatProd(m_left, m_right, 2, 3, 4),
                            std::invalid_argument);
            CHECK_THROWS_WITH(Util::matrixMatProd(m_left, m_right, 2, 3, 4),
                              Contains("Invalid number of rows and columns for "
                                       "the input left matrix"));
            CHECK_THROWS_AS(Util::matrixMatProd(m_left, m_right, 2, 3, 3),
                            std::invalid_argument);
            CHECK_THROWS_WITH(Util::matrixMatProd(m_left, m_right, 2, 3, 3),
                              Contains("Invalid number of rows and columns for "
                                       "the input right matrix"));
        }
    }
}

/**
 * @brief Count number of 1s in the binary representation of x
 *
 * This is a slow version of countBit1 defined in Util.hpp
 */
int popcount_slow(uint64_t x) {
    int c = 0;
    for (; x != 0; x >>= 1) {
        if ((x & 1U) != 0U) {
            c++;
        }
    }
    return c;
}

/**
 * @brief Count number of trailing zeros in the binary representation of x
 *
 * This is a slow version of countTrailing0 defined in Util.hpp
 */
int ctz_slow(uint64_t x) {
    int c = 0;
    while ((x & 1) == 0) {
        x >>= 1;
        c++;
    }
    return c;
}
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("Utility bit operations", "[Util]") {
    SECTION("Internal::countBit1Fast") {
        { // for uint32_t
            uint32_t n = 0;
            CHECK(Util::Internal::countBit1(n) == 0);
            for (uint32_t k = 0; k < 100; k++) {
                n <<= 1U;
                n ^= 1U;
                CHECK(Util::Internal::countBit1(n) == popcount_slow(n));
            }
        }
        { // for uint64_t
            uint64_t n = 0;
            CHECK(Util::Internal::countBit1(n) == 0);
            for (uint32_t k = 0; k < 100; k++) {
                n <<= 1U;
                n ^= 1U;
                CHECK(Util::Internal::countBit1(n) == popcount_slow(n));
            }
        }
    }

    SECTION("isPerfectPowerOf2") {
        size_t n = 1U;
        CHECK(Util::isPerfectPowerOf2(n));
        for (size_t k = 0; k < sizeof(size_t) - 2; k++) {
            n *= 2;
            CHECK(Util::isPerfectPowerOf2(n));
            CHECK(!Util::isPerfectPowerOf2(n + 1));
        }

        CHECK(!Util::isPerfectPowerOf2(0U));
        CHECK(!Util::isPerfectPowerOf2(124U));
        CHECK(!Util::isPerfectPowerOf2(1077U));
        CHECK(!Util::isPerfectPowerOf2(1000000000U));

        if constexpr (sizeof(size_t) == 8) {
            // if size_t is uint64_t
            CHECK(!Util::isPerfectPowerOf2(1234556789012345678U));
        }
    }

    SECTION("Internal::countTrailing0") {
        { // for uint32_t
            for (uint32_t c = 0; c < 31; c++) {
                uint32_t n = static_cast<uint32_t>(1U)
                             << static_cast<uint32_t>(c);
                CHECK(Util::Internal::countTrailing0(n) == c);
                CHECK(Util::Internal::countTrailing0(n | (1U << 31U)) == c);
            }
        }
        { // for uint64_t
            for (uint32_t c = 0; c < 63; c++) {
                uint64_t n = static_cast<uint64_t>(1U)
                             << static_cast<uint64_t>(c);
                CHECK(Util::Internal::countTrailing0(n) == c);
                CHECK(Util::Internal::countTrailing0(n | (1UL << 63U)) == c);
            }
        }
    }

    SECTION("log2PerfectPower") {
        { // for uint32_t
            for (uint32_t c = 0; c < 32; c++) {
                uint32_t n = static_cast<uint32_t>(1U)
                             << static_cast<uint64_t>(c);
                CHECK(Util::log2PerfectPower(n) == c);
            }
        }
        { // for uint64_t
            for (uint32_t c = 0; c < 32; c++) {
                uint32_t n = static_cast<uint64_t>(1U)
                             << static_cast<uint64_t>(c);
                CHECK(Util::log2PerfectPower(n) == c);
            }
        }
    }
}
