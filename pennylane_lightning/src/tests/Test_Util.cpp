#include <algorithm>
#include <complex>

#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "BitUtil.hpp"
#include "Error.hpp"
#include "LinearAlgebra.hpp"
#include "Util.hpp"

#include "TestHelpers.hpp"

#if defined(_MSC_VER)
#pragma warning(disable : 4305)
#endif

using namespace Pennylane;

/**
 * @brief This tests the compile-time calculation of a given scalar
 * multiplication.
 */
TEMPLATE_TEST_CASE("Util::ConstMult", "[Util]", float, double) {
    constexpr TestType r_val{0.679};
    constexpr std::complex<TestType> c0_val{TestType{1.321}, TestType{-0.175}};
    constexpr std::complex<TestType> c1_val{TestType{0.579}, TestType{1.334}};

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
            std::modf(sqrt(static_cast<TestType>(i)), &rem);
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
}

/**
 * @brief Count number of 1s in the binary representation of x
 *
 * This is a slow version of countBit1 defined in Util.hpp
 */
size_t popcount_slow(uint64_t x) {
    size_t c = 0;
    for (; x != 0U; x >>= 1U) {
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
size_t ctz_slow(uint64_t x) {
    size_t c = 0;
    while ((x & 1U) == 0) {
        x >>= 1U;
        c++;
    }
    return c;
}
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("Utility bit operations", "[Util][BitUtil]") {
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
                CHECK(Util::Internal::countTrailing0(
                          n | (uint64_t{1U} << 63U)) == c);
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

TEST_CASE("Utility array and tuples", "[Util]") {
    std::array<std::pair<int, std::string_view>, 5> test_pairs{
        std::pair(0, "Zero"),  std::pair(1, "One"),  std::pair(2, "Two"),
        std::pair(3, "Three"), std::pair(4, "Four"),
    };

    REQUIRE(Util::reverse_pairs(test_pairs) ==
            std::array{
                std::pair<std::string_view, int>("Zero", 0),
                std::pair<std::string_view, int>("One", 1),
                std::pair<std::string_view, int>("Two", 2),
                std::pair<std::string_view, int>("Three", 3),
                std::pair<std::string_view, int>("Four", 4),
            });

    REQUIRE(Util::reverse_pairs(test_pairs) !=
            std::array{
                std::pair<std::string_view, int>("Zero", 0),
                std::pair<std::string_view, int>("One", 1),
                std::pair<std::string_view, int>("Two", 0),
                std::pair<std::string_view, int>("Three", 3),
                std::pair<std::string_view, int>("Four", 4),
            });
}

/**
 * @brief Test randomUnitary is correct
 */
TEMPLATE_TEST_CASE("randomUnitary", "[Util]", float, double) {
    using PrecisionT = TestType;

    std::mt19937 re{1337};

    for (size_t num_qubits = 1; num_qubits <= 5; num_qubits++) {
        const size_t dim = (1U << num_qubits);
        const auto unitary = Util::randomUnitary<PrecisionT>(re, num_qubits);

        auto unitary_dagger = Util::Transpose(unitary, dim, dim);
        std::transform(
            unitary_dagger.begin(), unitary_dagger.end(),
            unitary_dagger.begin(),
            [](const std::complex<PrecisionT> &v) { return std::conj(v); });

        std::vector<std::complex<PrecisionT>> mat(dim * dim);
        Util::matrixMatProd(unitary.data(), unitary_dagger.data(), mat.data(),
                            dim, dim, dim);

        std::vector<std::complex<PrecisionT>> identity(
            dim * dim, std::complex<PrecisionT>{});
        for (size_t i = 0; i < dim; i++) {
            identity[i * dim + i] = std::complex<PrecisionT>{1.0, 0.0};
        }

        REQUIRE(mat == approx(identity).margin(1e-5));
    }
}

enum class TestEnum { One, Two, Many };

TEST_CASE("Test utility functions for constants", "[Util][ConstantUtil]") {
    using namespace std::literals;

    SECTION("lookup") {
        constexpr std::array test_pairs = {
            std::pair{"Pennylane"sv, "-"sv},
            std::pair{"Lightning"sv, "is"sv},
            std::pair{"the"sv, "best"sv},
            std::pair{"QML"sv, "library"sv},
        };

        REQUIRE(Util::lookup(test_pairs, "Pennylane"sv) == "-"sv);
        REQUIRE(Util::lookup(test_pairs, "Lightning"sv) == "is"sv);
        REQUIRE(Util::lookup(test_pairs, "the"sv) == "best"sv);
        REQUIRE(Util::lookup(test_pairs, "QML"sv) == "library"sv);
        REQUIRE_THROWS(Util::lookup(test_pairs, "bad"sv));
    }

    SECTION("count_unique") {
        constexpr std::array test_arr1 = {"This"sv, "is"sv, "a"sv, "test"sv,
                                          "arr"sv};
        constexpr std::array test_arr2 = {"This"sv, "is"sv,  "a"sv,
                                          "test"sv, "arr"sv, "is"sv};

        REQUIRE(Util::count_unique(test_arr1) == 5);
        REQUIRE(Util::count_unique(test_arr2) == 5);
    }

    SECTION("static_lookup") {
        std::array test_pairs = {
            std::pair{TestEnum::One, uint32_t{1U}},
            std::pair{TestEnum::Two, uint32_t{2U}},
        };

        REQUIRE(Util::static_lookup<TestEnum::One>(test_pairs) == 1U);
        REQUIRE(Util::static_lookup<TestEnum::Two>(test_pairs) == 2U);
        REQUIRE(Util::static_lookup<TestEnum::Many>(test_pairs) == uint32_t{});
    }
}
