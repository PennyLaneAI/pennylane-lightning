// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <algorithm>
#include <complex>
#include <vector>

#include <catch2/catch.hpp>

#include "LinearAlgebra.hpp"
#include "TestHelpers.hpp"
#include "Util.hpp"

#if defined(_MSC_VER)
#pragma warning(disable : 4305)
#endif

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using Pennylane::Util::randomUnitary;
} // namespace
/// @endcond

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEMPLATE_TEST_CASE("Inner product", "[Util][LinearAlgebra]", float, double) {
    SECTION("innerProdC") {
        SECTION("Iterative increment") {
            for (std::size_t i = 0; i < 12; i++) {
                auto sz = static_cast<std::size_t>(1U << i);
                std::vector<std::complex<TestType>> data1(sz, {1.0, 1.0});
                std::vector<std::complex<TestType>> data2(sz, {1.0, 1.0});
                std::complex<TestType> expected_result(
                    std::size_t{1U} << (i + 1), 0);
                std::complex<TestType> result = Util::innerProdC(data1, data2);
                CAPTURE(result);
                CAPTURE(expected_result);
                CHECK(isApproxEqual(result, expected_result));
            }
        }
        SECTION("Random complex") {
            std::vector<std::complex<TestType>> data2{
                {0, -0.479426}, {0, 0}, {2.77556e-17, 0}, {0, 0},
                {0.877583, 0},  {0, 0}, {0, 0},           {0, 0}};
            std::vector<std::complex<TestType>> data1{
                {0.326417, 0},  {-0, 0.343918}, {0, 0.508364}, {-0.53562, -0},
                {0, -0.178322}, {0.187883, -0}, {0.277721, 0}, {-0, 0.292611}};
            std::complex<TestType> expected_result(0, -4.40916e-7);
            std::complex<TestType> result = Util::innerProdC(data1, data2);
            CAPTURE(result);
            CAPTURE(expected_result);
            CHECK(real(result) == Approx(real(expected_result)).margin(1e-7));
            CHECK(imag(result) == Approx(imag(expected_result)).margin(1e-7));
        }
    }
    SECTION("innerProdC-inline") {
        SECTION("Iterative increment") {
            for (std::size_t i = 0; i < 12; i++) {
                auto sz = static_cast<std::size_t>(1U << i);
                std::vector<std::complex<TestType>> data1(sz, {1.0, 1.0});
                std::vector<std::complex<TestType>> data2(sz, {1.0, 1.0});
                std::complex<TestType> expected_result(
                    std::size_t{1U} << (i + 1), 0);
                std::complex<TestType> result = Util::innerProdC<TestType, 1>(
                    data1.data(), data2.data(), sz);
                CAPTURE(result);
                CAPTURE(expected_result);
                CHECK(isApproxEqual(result, expected_result));
            }
        }
        SECTION("Random complex") {
            std::vector<std::complex<TestType>> data2{
                {0, -0.479426}, {0, 0}, {2.77556e-17, 0}, {0, 0},
                {0.877583, 0},  {0, 0}, {0, 0},           {0, 0}};
            std::vector<std::complex<TestType>> data1{
                {0.326417, 0},  {-0, 0.343918}, {0, 0.508364}, {-0.53562, -0},
                {0, -0.178322}, {0.187883, -0}, {0.277721, 0}, {-0, 0.292611}};
            std::complex<TestType> expected_result(0, -4.40916e-7);
            std::complex<TestType> result =
                Util::innerProdC<TestType, 1>(data1.data(), data2.data(), 8);
            CAPTURE(result);
            CAPTURE(expected_result);
            CHECK(real(result) == Approx(real(expected_result)).margin(1e-7));
            CHECK(imag(result) == Approx(imag(expected_result)).margin(1e-7));
        }
    }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEMPLATE_TEST_CASE("Transpose", "[Util][LinearAlgebra]", float, double) {
    SECTION("CFTranspose") {
        SECTION("Simple Matrix") {
            for (std::size_t m = 2; m < 10; m++) {
                std::vector<TestType> mat(m * m, {0});
                for (std::size_t i = 0; i < m; i++) {
                    mat[i * m + i] = 1.0;
                }
                std::vector<TestType> mat_t(m * m);
                Util::CFTranspose<TestType, 4>(mat.data(), mat_t.data(), m, m,
                                               0, m, 0, m);

                CAPTURE(mat_t);
                CAPTURE(mat);

                CHECK(mat_t == approx(mat).margin(1e-7));
            }
        }
        SECTION("Random Complex") {
            std::vector<TestType> mat{
                0.417876, 0.27448,   0.601209, 0.723548, 0.781624,
                0.538222, 0.0597232, 0.27755,  0.836569,
            };
            std::vector<TestType> mat_t_exp{
                0.417876, 0.723548, 0.0597232, 0.27448,  0.781624,
                0.27755,  0.601209, 0.538222,  0.836569,
            };
            std::vector<TestType> mat_t(9);
            Util::CFTranspose<TestType, 2>(mat.data(), mat_t.data(), 3, 3, 0, 3,
                                           0, 3);

            CAPTURE(mat_t);
            CAPTURE(mat_t_exp);

            CHECK(mat_t == approx(mat_t_exp));
        }
        SECTION("Random Complex non-square") {
            std::vector<TestType> mat{
                0.417876, 0.27448,  0.601209,  0.723548,
                0.781624, 0.538222, 0.0597232, 0.27755,
            };
            std::vector<TestType> mat_t_exp{0.417876, 0.781624, 0.27448,
                                            0.538222, 0.601209, 0.0597232,
                                            0.723548, 0.27755};
            std::vector<TestType> mat_t(8);
            Util::CFTranspose<TestType, 2>(mat.data(), mat_t.data(), 2, 4, 0, 2,
                                           0, 4);

            CAPTURE(mat_t);
            CAPTURE(mat_t_exp);

            CHECK(mat_t == approx(mat_t_exp));
        }
    }
    SECTION("Transpose") {
        SECTION("Simple Matrix") {
            for (std::size_t m = 2; m < 8; m++) {
                std::vector<std::complex<TestType>> mat(m * m, {0, 0});
                for (std::size_t i = 0; i < m; i++) {
                    mat[i * m + i] = {1, 1};
                }
                std::vector<std::complex<TestType>> mat_t =
                    Util::Transpose(mat, m, m);

                CAPTURE(mat_t);
                CAPTURE(mat);

                CHECK(mat_t == approx(mat).margin(1e-7));
            }
        }
        SECTION("Random Complex") {
            std::vector<std::complex<TestType>> mat{
                {0.417876, 0.27448},   {0.601209, 0.723548},
                {0.781624, 0.538222},  {0.0597232, 0.27755},
                {0.0431741, 0.593319}, {0.224124, 0.130335},
                {0.237877, 0.01557},   {0.931634, 0.786367},
                {0.378397, 0.894381},  {0.840747, 0.889789},
                {0.530623, 0.463644},  {0.868736, 0.760685},
                {0.258175, 0.836569},  {0.495012, 0.667726},
                {0.298962, 0.384992},  {0.659472, 0.232696}};
            std::vector<std::complex<TestType>> mat_t_exp{
                {0.417876, 0.27448},  {0.0431741, 0.593319},
                {0.378397, 0.894381}, {0.258175, 0.836569},
                {0.601209, 0.723548}, {0.224124, 0.130335},
                {0.840747, 0.889789}, {0.495012, 0.667726},
                {0.781624, 0.538222}, {0.237877, 0.01557},
                {0.530623, 0.463644}, {0.298962, 0.384992},
                {0.0597232, 0.27755}, {0.931634, 0.786367},
                {0.868736, 0.760685}, {0.659472, 0.232696}};
            std::vector<std::complex<TestType>> mat_t =
                Util::Transpose(mat, 4, 4);

            CAPTURE(mat_t);
            CAPTURE(mat_t_exp);

            CHECK(mat_t == approx(mat_t_exp));
        }
        SECTION("Invalid Arguments") {
            using namespace Catch::Matchers;
            std::vector<TestType> mat(2 * 3, {1.0});
            CHECK_THROWS_AS(
                Util::Transpose(std::span<const TestType>{mat}, 2, 2),
                std::invalid_argument);
            CHECK_THROWS_WITH(
                Util::Transpose(mat, 2, 2),
                Contains(
                    "Invalid number of rows and columns for the input matrix"));
        }
    }
    SECTION("Transpose<complex<T>>") {
        SECTION("Simple Matrix") {
            for (std::size_t m = 2; m < 8; m++) {
                std::vector<std::complex<TestType>> mat(m * m, {0, 0});
                for (std::size_t i = 0; i < m; i++) {
                    mat[i * m + i] = {1.0, 1.0};
                }
                std::vector<std::complex<TestType>> mat_t =
                    Util::Transpose(mat, m, m);

                CAPTURE(mat_t);
                CAPTURE(mat);

                CHECK(mat_t == approx(mat).margin(1e-7));
            }
        }
        SECTION("Random Complex") {
            std::vector<std::complex<TestType>> mat{
                {0.417876, 0.27448},   {0.601209, 0.723548},
                {0.781624, 0.538222},  {0.0597232, 0.27755},
                {0.0431741, 0.593319}, {0.224124, 0.130335},
                {0.237877, 0.01557},   {0.931634, 0.786367},
                {0.378397, 0.894381},  {0.840747, 0.889789},
                {0.530623, 0.463644},  {0.868736, 0.760685},
                {0.258175, 0.836569},  {0.495012, 0.667726},
                {0.298962, 0.384992},  {0.659472, 0.232696}};
            std::vector<std::complex<TestType>> mat_t_exp{
                {0.417876, 0.27448},  {0.0431741, 0.593319},
                {0.378397, 0.894381}, {0.258175, 0.836569},
                {0.601209, 0.723548}, {0.224124, 0.130335},
                {0.840747, 0.889789}, {0.495012, 0.667726},
                {0.781624, 0.538222}, {0.237877, 0.01557},
                {0.530623, 0.463644}, {0.298962, 0.384992},
                {0.0597232, 0.27755}, {0.931634, 0.786367},
                {0.868736, 0.760685}, {0.659472, 0.232696}};
            std::vector<std::complex<TestType>> mat_t =
                Util::Transpose(mat, 4, 4);

            CAPTURE(mat_t);
            CAPTURE(mat_t_exp);

            CHECK(mat_t == approx(mat_t_exp));
        }
        SECTION("Invalid Arguments") {
            using namespace Catch::Matchers;
            std::vector<std::complex<TestType>> mat(2 * 3, {1.0, 1.0});
            CHECK_THROWS_AS(Util::Transpose(mat, 2, 2), std::invalid_argument);
            CHECK_THROWS_WITH(
                Util::Transpose(mat, 2, 2),
                Contains(
                    "Invalid number of rows and columns for the input matrix"));
        }
    }
}

TEMPLATE_TEST_CASE("Util::scaleAndAdd", "[Util][LinearAlgebra]", float,
                   double) {
    using PrecisionT = TestType;
    using ComplexT = std::complex<PrecisionT>;

    SECTION("Test result is correct") {
        auto a = ComplexT{0.36572644485147254, 0.4729529811649217};
        std::vector<ComplexT> x{
            ComplexT{0.481941495077, 0.734106237571},
            ComplexT{0.960470937496, 0.880529982024},
            ComplexT{0.135982489400, 0.049663856666},
            ComplexT{0.589227566883, 0.646648171030},
            ComplexT{0.051294350194, 0.013730433456},
            ComplexT{0.716464613724, 0.296251370128},
            ComplexT{0.820197028755, 0.199230854010},
            ComplexT{0.100767632907, 0.745810000609},
            ComplexT{0.603122469037, 0.437680494447},
            ComplexT{0.815084269631, 0.501486284044},
            ComplexT{0.554633849948, 0.437321144284},
            ComplexT{0.822295519809, 0.810051588437},
            ComplexT{0.217638951648, 0.663920104700},
            ComplexT{0.289819402719, 0.839919161595},
            ComplexT{0.498496405040, 0.906874924446},
            ComplexT{0.365971064862, 0.230694150520},
        };
        std::vector<ComplexT> y{
            ComplexT{0.516438479285, 0.970319841313},
            ComplexT{0.085702308539, 0.005302125762},
            ComplexT{0.591955559108, 0.945946312721},
            ComplexT{0.710102120659, 0.410003006045},
            ComplexT{0.171020364152, 0.020935262021},
            ComplexT{0.904267565256, 0.235752839391},
            ComplexT{0.715111137847, 0.402137049186},
            ComplexT{0.590485707389, 0.550485111898},
            ComplexT{0.830734963458, 0.777755725832},
            ComplexT{0.988885576027, 0.541038298049},
            ComplexT{0.375479099161, 0.275849441779},
            ComplexT{0.441329976617, 0.825285998539},
            ComplexT{0.376823807696, 0.896094272876},
            ComplexT{0.558768533750, 0.963077088666},
            ComplexT{0.402000571969, 0.344065008137},
            ComplexT{0.805773653517, 0.316132703093},
        };
        std::vector<ComplexT> expected{
            ComplexT{0.345499495355, 1.466737572567},
            ComplexT{0.020522649889, 0.781592818884},
            ComplexT{0.618199282452, 1.028423022205},
            ComplexT{0.619764043650, 0.925176277047},
            ComplexT{0.183286215053, 0.050216660476},
            ComplexT{1.026184652619, 0.682953874730},
            ComplexT{0.920852054907, 0.862915671020},
            ComplexT{0.274606032358, 0.870905904344},
            ComplexT{0.844310505222, 1.223075626786},
            ComplexT{1.049804015161, 1.109941629077},
            ComplexT{0.371491026381, 0.698105081924},
            ComplexT{0.358948880046, 1.510450403616},
            ComplexT{0.142417134970, 1.241840403433},
            ComplexT{0.267520882141, 1.407328688114},
            ComplexT{0.155404690895, 0.911498511043},
            ComplexT{0.830511463762, 0.573590760757},
        };
        Util::scaleAndAdd(a, x, y);
        REQUIRE(y == approx(expected));
    }
    SECTION("Throws exception when the size mismatches") {
        std::vector<ComplexT> x(8, ComplexT{});
        std::vector<ComplexT> y(4, ComplexT{});

        PL_REQUIRE_THROWS_MATCHES(Util::scaleAndAdd(ComplexT{0.5, 0.4}, x, y),
                                  std::invalid_argument,
                                  "Dimensions of vectors mismatch");
    }
    SECTION("omp_scaleAndAdd uses STD_CROSSOVER") {
        std::vector<ComplexT> x(32);
        std::vector<ComplexT> y(32);
        REQUIRE_NOTHROW(Util::omp_scaleAndAdd<PrecisionT, 16>(
            32, {1.0, 0.0}, x.data(), y.data()));
    }
}
