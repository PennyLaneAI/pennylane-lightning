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
#include "Memory.hpp"
#include "Util.hpp"

#include "TestHelpers.hpp"

#if defined(_MSC_VER)
#pragma warning(disable : 4305)
#endif

using namespace Pennylane;
using namespace Pennylane::Util;

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

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("Utility bit operations", "[Util][BitUtil]") {
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

    SECTION("Bitswap") {
        CHECK(Util::bitswap(0B001101, 0, 1) == 0B001110);
        CHECK(Util::bitswap(0B001101, 0, 2) == 0B001101);
        CHECK(Util::bitswap(0B001101, 0, 3) == 0B001101);
        CHECK(Util::bitswap(0B001101, 0, 4) == 0B011100);
    }

    SECTION("fillTrailingOnes") {
        CHECK(Util::fillTrailingOnes<uint8_t>(4) == 0B1111);
        CHECK(Util::fillTrailingOnes<uint8_t>(6) == 0B111111);
        CHECK(Util::fillTrailingOnes<uint32_t>(17) == 0B1'1111'1111'1111'1111);
        CHECK(Util::fillTrailingOnes<uint64_t>(54) ==
              0x3F'FFFF'FFFF'FFFF); // 54 == 4*13 + 2
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

        REQUIRE(Util::count_unique(std::array{nullptr, nullptr, nullptr}) == 1);
        REQUIRE(Util::count_unique(std::array{0, 0, 0}) == 1);
        REQUIRE(Util::count_unique(std::array{0, 1, 1}) == 2);
        REQUIRE(Util::count_unique(std::array{0, 1, 2}) == 3);
    }

    SECTION("lookup (constexpr context)") {
        enum class TestEnum { One, Two, Many };

        constexpr std::array test_pairs = {
            std::pair{TestEnum::One, uint32_t{1U}},
            std::pair{TestEnum::Two, uint32_t{2U}},
        };

        static_assert(Util::lookup(test_pairs, TestEnum::One) == 1U);
        static_assert(Util::lookup(test_pairs, TestEnum::Two) == 2U);
        // The following line must not be compiled
        // static_assert(Util::lookup(test_pairs, TestEnum::Many) == 2U);
    }
}

TEST_CASE("Test AlignedAllocator", "[Util][Memory]") {
    AlignedAllocator<double> allocator(8);
    REQUIRE(allocator.allocate(0) == nullptr);
    /* Allocate 1 PiB */
    REQUIRE_THROWS_AS(
        std::unique_ptr<double>(allocator.allocate(
            size_t{1024} * size_t{1024 * 1024} * size_t{1024 * 1024})),
        std::bad_alloc);
}

TEST_CASE("Test tensor transposition", "[Util]") {
    // Transposition axes and expected result.
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> input = {
        {{0, 1, 2}, {0, 1, 2, 3, 4, 5, 6, 7}},
        {{0, 2, 1}, {0, 2, 1, 3, 4, 6, 5, 7}},
        {{1, 0, 2}, {0, 1, 4, 5, 2, 3, 6, 7}},
        {{1, 2, 0}, {0, 4, 1, 5, 2, 6, 3, 7}},
        {{2, 0, 1}, {0, 2, 4, 6, 1, 3, 5, 7}},
        {{2, 1, 0}, {0, 4, 2, 6, 1, 5, 3, 7}},
        {{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
        {{0, 1, 3, 2}, {0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15}},
        {{0, 2, 1, 3}, {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15}},
        {{0, 2, 3, 1}, {0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15}},
        {{0, 3, 1, 2}, {0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15}},
        {{0, 3, 2, 1}, {0, 4, 2, 6, 1, 5, 3, 7, 8, 12, 10, 14, 9, 13, 11, 15}},
        {{1, 0, 2, 3}, {0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15}},
        {{1, 0, 3, 2}, {0, 2, 1, 3, 8, 10, 9, 11, 4, 6, 5, 7, 12, 14, 13, 15}},
        {{1, 2, 0, 3}, {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15}},
        {{1, 2, 3, 0}, {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15}},
        {{1, 3, 0, 2}, {0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15}},
        {{1, 3, 2, 0}, {0, 8, 2, 10, 1, 9, 3, 11, 4, 12, 6, 14, 5, 13, 7, 15}},
        {{2, 0, 1, 3}, {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15}},
        {{2, 0, 3, 1}, {0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15}},
        {{2, 1, 0, 3}, {0, 1, 8, 9, 4, 5, 12, 13, 2, 3, 10, 11, 6, 7, 14, 15}},
        {{2, 1, 3, 0}, {0, 8, 1, 9, 4, 12, 5, 13, 2, 10, 3, 11, 6, 14, 7, 15}},
        {{2, 3, 0, 1}, {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}},
        {{2, 3, 1, 0}, {0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15}},
        {{3, 0, 1, 2}, {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15}},
        {{3, 0, 2, 1}, {0, 4, 2, 6, 8, 12, 10, 14, 1, 5, 3, 7, 9, 13, 11, 15}},
        {{3, 1, 0, 2}, {0, 2, 8, 10, 4, 6, 12, 14, 1, 3, 9, 11, 5, 7, 13, 15}},
        {{3, 1, 2, 0}, {0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15}},
        {{3, 2, 0, 1}, {0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15}},
        {{3, 2, 1, 0}, {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15}}};

    SECTION("Looping over different wire configurations:") {
        for (const auto &term : input) {
            // Defining a tensor to be transposed.
            std::vector<size_t> indices(1U << term.first.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::vector<size_t> result_transposed_indices =
                transpose_state_tensor(indices, term.first);
            REQUIRE(term.second == result_transposed_indices);
        }
    }
}