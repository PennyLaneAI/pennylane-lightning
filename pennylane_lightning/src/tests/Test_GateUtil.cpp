#include "GateUtil.hpp"
#include "Gates.hpp"
#include "Util.hpp"

#include "TestHelpers.hpp"
#include "TestKernels.hpp"

#include <catch2/catch.hpp>

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

using namespace Pennylane;
using namespace Pennylane::Gates;

TEST_CASE("generateBitPatterns", "[GateUtil]") {
    const size_t num_qubits = 4;
    SECTION("Qubit indices {}") {
        auto bit_pattern = generateBitPatterns({}, num_qubits);
        CHECK(bit_pattern == std::vector<size_t>{0});
    }
    SECTION("Qubit indices {i}") {
        for (size_t i = 0; i < num_qubits; i++) {
            std::vector<size_t> expected{0, size_t{1U} << (num_qubits - i - 1)};
            auto bit_pattern = generateBitPatterns({i}, num_qubits);
            CHECK(bit_pattern == expected);
        }
    }
    SECTION("Qubit indices {i,i+1,i+2}") {
        std::vector<size_t> expected_123{0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<size_t> expected_012{0, 2, 4, 6, 8, 10, 12, 14};
        auto bit_pattern_123 = generateBitPatterns({1, 2, 3}, num_qubits);
        auto bit_pattern_012 = generateBitPatterns({0, 1, 2}, num_qubits);

        CHECK(bit_pattern_123 == expected_123);
        CHECK(bit_pattern_012 == expected_012);
    }
    SECTION("Qubit indices {0,2,3}") {
        std::vector<size_t> expected{0, 1, 2, 3, 8, 9, 10, 11};
        auto bit_pattern = generateBitPatterns({0, 2, 3}, num_qubits);

        CHECK(bit_pattern == expected);
    }
    SECTION("Qubit indices {3,1,0}") {
        std::vector<size_t> expected{0, 8, 4, 12, 1, 9, 5, 13};
        auto bit_pattern = generateBitPatterns({3, 1, 0}, num_qubits);
        CHECK(bit_pattern == expected);
    }
}

TEST_CASE("getIndicesAfterExclusion", "[GateUtil]") {
    const size_t num_qubits = 4;
    SECTION("Qubit indices {}") {
        std::vector<size_t> expected{0, 1, 2, 3};
        auto indices = getIndicesAfterExclusion({}, num_qubits);
        CHECK(indices == expected);
    }
    SECTION("Qubit indices {i}") {
        for (size_t i = 0; i < num_qubits; i++) {
            std::vector<size_t> expected{0, 1, 2, 3};
            expected.erase(expected.begin() + i);

            auto indices = getIndicesAfterExclusion({i}, num_qubits);
            CHECK(indices == expected);
        }
    }
    SECTION("Qubit indices {i,i+1,i+2}") {
        std::vector<size_t> expected_123{0};
        std::vector<size_t> expected_012{3};
        auto indices_123 = getIndicesAfterExclusion({1, 2, 3}, num_qubits);
        auto indices_012 = getIndicesAfterExclusion({0, 1, 2}, num_qubits);

        CHECK(indices_123 == expected_123);
        CHECK(indices_012 == expected_012);
    }
    SECTION("Qubit indices {0,2,3}") {
        std::vector<size_t> expected{1};
        auto indices = getIndicesAfterExclusion({0, 2, 3}, num_qubits);

        CHECK(indices == expected);
    }
    SECTION("Qubit indices {3,1,0}") {
        std::vector<size_t> expected{2};
        auto indices = getIndicesAfterExclusion({3, 1, 0}, num_qubits);

        CHECK(indices == expected);
    }
}
