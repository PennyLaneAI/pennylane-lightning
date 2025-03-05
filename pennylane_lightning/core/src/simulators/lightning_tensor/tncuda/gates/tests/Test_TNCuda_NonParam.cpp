// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <complex>
#include <vector>

#include <catch2/catch.hpp>

#include "DevTag.hpp"
#include "ExactTNCuda.hpp"
#include "MPSTNCuda.hpp"
#include "TNCudaGateCache.hpp"

#include "TestHelpers.hpp"
#include "TestHelpersTNCuda.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::LightningTensor::TNCuda::Gates;
using namespace Pennylane::Util;
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningTensor::TNCuda::Util;
} // namespace
/// @endcond

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::Identity", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);

        tn_state->applyOperation("Hadamard", {index}, inverse);

        tn_state->applyOperation("Identity", {index}, inverse);
        auto expected = cuUtil::INVSQRT2<cp_t>();

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(expected == Pennylane::Util::approx(
                              results[0b1 << ((num_qubits - 1 - index))]));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::Hadamard", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);

        tn_state_append_mps_final_state(tn_state);

        tn_state->applyOperation("Hadamard", {index}, inverse);

        tn_state_append_mps_final_state(tn_state);

        tn_state->applyOperation("Identity", {index}, inverse);

        // Test for multiple final states appendings
        tn_state_append_mps_final_state(tn_state);

        cp_t expected(1.0 / std::sqrt(2), 0);

        auto results = tn_state->getDataVector();

        CHECK(expected == Pennylane::Util::approx(
                              results[0b1 << ((num_qubits - 1 - index))]));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::PauliX", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;

    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);

        tn_state->applyOperation("PauliX", {index}, inverse);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results[0] == cuUtil::ZERO<cp_t>());
        CHECK(results[0b1 << (num_qubits - index - 1)] == cuUtil::ONE<cp_t>());
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::applyOperation-gatematrix",
                        "[TNCuda_Nonparam]", TestTNBackends) {
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;

    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);

        std::vector<cp_t> gate_matrix = {
            cuUtil::ZERO<cp_t>(), cuUtil::ONE<cp_t>(), cuUtil::ONE<cp_t>(),
            cuUtil::ZERO<cp_t>()};

        tn_state->applyOperation("applyMatrix", {index}, false, {},
                                 gate_matrix);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results[0] == cuUtil::ZERO<cp_t>());
        CHECK(results[0b1 << (num_qubits - index - 1)] == cuUtil::ONE<cp_t>());
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::PauliY", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const cp_t p = cuUtil::ConstMult(
        cp_t(0.5, 0.0),
        cuUtil::ConstMult(cuUtil::INVSQRT2<cp_t>(), cuUtil::IMAG<cp_t>()));
    const cp_t m = cuUtil::ConstMult(cp_t(-1, 0), p);

    const std::vector<std::vector<cp_t>> expected_results = {
        {m, m, m, m, p, p, p, p},
        {m, m, p, p, m, m, p, p},
        {m, p, m, p, m, p, m, p}};

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("PauliY", {index}, inverse);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[index]));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::PauliZ", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const cp_t p(cp_t(0.5, 0.0) * cuUtil::INVSQRT2<cp_t>());
    const cp_t m(cuUtil::ConstMult(cp_t{-1.0, 0.0}, p));

    const std::vector<std::vector<cp_t>> expected_results = {
        {p, p, p, p, m, m, m, m},
        {p, p, m, m, p, p, m, m},
        {p, m, p, m, p, m, p, m}};

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("PauliZ", {index}, inverse);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[index]));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::S", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    cp_t r(cp_t(0.5, 0.0) * cuUtil::INVSQRT2<cp_t>());
    cp_t i(cuUtil::ConstMult(r, cuUtil::IMAG<cp_t>()));

    if (inverse) {
        i = std::conj(i);
    }

    const std::vector<std::vector<cp_t>> expected_results = {
        {r, r, r, r, i, i, i, i},
        {r, r, i, i, r, r, i, i},
        {r, i, r, i, r, i, r, i}};

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("S", {index}, inverse);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[index]));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::SX", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    const cp_t z(0.0, 0.0);
    cp_t p(0.5, 0.5);
    cp_t m(0.5, -0.5);

    if (inverse) {
        p = conj(p);
        m = conj(m);
    }

    const std::vector<std::vector<cp_t>> expected_results = {
        {p, z, z, z, m, z, z, z},
        {p, z, m, z, z, z, z, z},
        {p, m, z, z, z, z, z, z}};

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);

        tn_state->applyOperation("SX", {index}, inverse);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[index]));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::T", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    cp_t r(1.0 / (2.0 * std::sqrt(2)), 0);
    cp_t i(1.0 / 4, 1.0 / 4);

    if (inverse) {
        i = conj(i);
    }

    const std::vector<std::vector<cp_t>> expected_results = {
        {r, r, r, r, i, i, i, i},
        {r, r, i, i, r, r, i, i},
        {r, i, r, i, r, i, r, i}};

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("T", {index}, inverse);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results ==
              Pennylane::Util::approx(expected_results[index]).margin(1e-8));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::CNOT", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;

    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    SECTION("Apply adjacent wire indices") {
        tn_state->applyOperations({"Hadamard", "CNOT", "CNOT"},
                                  {{0}, {0, 1}, {1, 2}},
                                  {false, inverse, inverse});

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results.front() ==
              Pennylane::Util::approx(cuUtil::INVSQRT2<cp_t>()).epsilon(1e-5));
        CHECK(results.back() ==
              Pennylane::Util::approx(cuUtil::INVSQRT2<cp_t>()).epsilon(1e-5));
    }

    SECTION("Apply non-adjacent wire indices") {
        tn_state->applyOperation("Hadamard", {0}, false);
        tn_state->applyOperation("CNOT", {0, 2}, inverse);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results[0] ==
              Pennylane::Util::approx(cuUtil::INVSQRT2<cp_t>()).epsilon(1e-5));
        CHECK(results[5] ==
              Pennylane::Util::approx(cuUtil::INVSQRT2<cp_t>()).epsilon(1e-5));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::SWAP", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;

    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    SECTION("Apply adjacent wire indices") {
        std::vector<cp_t> expected{
            cuUtil::ZERO<cp_t>(),   cuUtil::ZERO<cp_t>(),
            cuUtil::ZERO<cp_t>(),   cuUtil::ZERO<cp_t>(),
            cp_t(1.0 / sqrt(2), 0), cuUtil::ZERO<cp_t>(),
            cp_t(1.0 / sqrt(2), 0), cuUtil::ZERO<cp_t>()};

        tn_state->applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                  {false, false});

        tn_state->applyOperation("SWAP", {0, 1}, inverse);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected));
    }

    SECTION("Apply non-adjacent wire indices") {
        std::vector<cp_t> expected{
            cuUtil::ZERO<cp_t>(),     cuUtil::ZERO<cp_t>(),
            cuUtil::INVSQRT2<cp_t>(), cuUtil::INVSQRT2<cp_t>(),
            cuUtil::ZERO<cp_t>(),     cuUtil::ZERO<cp_t>(),
            cuUtil::ZERO<cp_t>(),     cuUtil::ZERO<cp_t>()};

        tn_state->applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                  {false, false});

        tn_state->applyOperation("SWAP", {0, 2}, inverse);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected).margin(1e-5));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::CY", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    SECTION("Apply adjacent wire indices") {
        std::vector<cp_t> expected_results{
            cuUtil::ZERO<cp_t>(),          cuUtil::ZERO<cp_t>(),
            cuUtil::INVSQRT2<cp_t>(),      cuUtil::ZERO<cp_t>(),
            -cuUtil::INVSQRT2IMAG<cp_t>(), cuUtil::ZERO<cp_t>(),
            cuUtil::ZERO<cp_t>(),          cuUtil::ZERO<cp_t>()};

        tn_state->applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                  {false, false});

        tn_state->applyOperation("CY", {0, 1}, inverse);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results ==
              Pennylane::Util::approx(expected_results).margin(1e-5));
    }

    SECTION("Apply non-adjacent wire indices") {
        std::vector<cp_t> expected_results{
            cuUtil::ZERO<cp_t>(),     cuUtil::ZERO<cp_t>(),
            cuUtil::INVSQRT2<cp_t>(), cuUtil::ZERO<cp_t>(),
            cuUtil::ZERO<cp_t>(),     cuUtil::ZERO<cp_t>(),
            cuUtil::ZERO<cp_t>(),     cuUtil::INVSQRT2IMAG<cp_t>()};

        tn_state->applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                  {false, false});

        tn_state->applyOperation("CY", {0, 2}, inverse);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::CZ", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    SECTION("Apply adjacent wire indices") {
        std::vector<cp_t> expected_results{
            cuUtil::ZERO<cp_t>(),      cuUtil::ZERO<cp_t>(),
            cuUtil::INVSQRT2<cp_t>(),  cuUtil::ZERO<cp_t>(),
            cuUtil::ZERO<cp_t>(),      cuUtil::ZERO<cp_t>(),
            -cuUtil::INVSQRT2<cp_t>(), cuUtil::ZERO<cp_t>()};

        tn_state->applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                  {false, false});

        tn_state->applyOperation("CZ", {0, 1}, inverse);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        std::for_each(results.begin(), results.end(),
                      [](cp_t &val) { val += cuUtil::ONE<cp_t>(); });
        std::for_each(expected_results.begin(), expected_results.end(),
                      [](cp_t &val) { val += cuUtil::ONE<cp_t>(); });

        CHECK(expected_results ==
              Pennylane::Util::approx(results).margin(1e-8));
    }

    SECTION("Apply non-adjacent wire indices") {
        std::vector<cp_t> expected_results{
            cuUtil::ZERO<cp_t>(),     cuUtil::ZERO<cp_t>(),
            cuUtil::INVSQRT2<cp_t>(), cuUtil::ZERO<cp_t>(),
            cuUtil::ZERO<cp_t>(),     cuUtil::ZERO<cp_t>(),
            cuUtil::INVSQRT2<cp_t>(), cuUtil::ZERO<cp_t>()};

        tn_state->applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                  {false, false});

        tn_state->applyOperation("CZ", {0, 2}, inverse);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results));
    }
}

TEMPLATE_LIST_TEST_CASE("ExactTNCuda::Gates::CSWAP", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;

    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state;

    if constexpr (std::is_same_v<TestType, MPSTNCuda<double>> ||
                  std::is_same_v<TestType, MPSTNCuda<float>>) {
        SECTION("CSWAP gate") {
            // Create the object for MPSTNCuda
            tn_state =
                std::make_unique<TNDevice_T>(num_qubits, maxExtent, dev_tag);
            REQUIRE_THROWS_AS(
                tn_state->applyOperation("CSWAP", {0, 1, 2}, inverse),
                LightningException);
        }
    } else {
        // Create the object for ExactTNCuda
        tn_state = std::make_unique<TNDevice_T>(num_qubits, dev_tag);

        SECTION("Apply adjacent wire indices") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<cp_t>(),     cuUtil::ZERO<cp_t>(),
                cuUtil::INVSQRT2<cp_t>(), cuUtil::ZERO<cp_t>(),
                cuUtil::ZERO<cp_t>(),     cuUtil::INVSQRT2<cp_t>(),
                cuUtil::ZERO<cp_t>(),     cuUtil::ZERO<cp_t>()};

            tn_state->applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                      {false, false});

            tn_state->applyOperation("CSWAP", {0, 1, 2}, inverse);

            auto results = tn_state->getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }

        SECTION("Apply non-adjacent wire indices") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<cp_t>(),   cuUtil::ZERO<cp_t>(),
                cp_t(1.0 / sqrt(2), 0), cp_t(1.0 / sqrt(2), 0),
                cuUtil::ZERO<cp_t>(),   cuUtil::ZERO<cp_t>(),
                cuUtil::ZERO<cp_t>(),   cuUtil::ZERO<cp_t>()};

            tn_state->applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                      {false, false});

            tn_state->applyOperation("CSWAP", {1, 0, 2}, inverse);

            auto results = tn_state->getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("ExactTNCuda::Gates::Toffoli", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;

    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state;

    if constexpr (std::is_same_v<TestType, MPSTNCuda<double>> ||
                  std::is_same_v<TestType, MPSTNCuda<float>>) {
        SECTION("Toffoli gate") {
            std::size_t num_qubits = 3;
            // Create the object for MPSTNCuda
            tn_state = std::make_unique<TNDevice_T>(num_qubits, maxExtent);

            REQUIRE_THROWS_AS(
                tn_state->applyOperation("Toffoli", {0, 1, 2}, inverse),
                LightningException);
        }
    } else {
        // Create the object for ExactTNCuda
        tn_state = std::make_unique<TNDevice_T>(num_qubits, dev_tag);

        SECTION("Apply adjacent wire indices") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<cp_t>(),     cuUtil::ZERO<cp_t>(),
                cuUtil::INVSQRT2<cp_t>(), cuUtil::ZERO<cp_t>(),
                cuUtil::ZERO<cp_t>(),     cuUtil::ZERO<cp_t>(),
                cuUtil::ZERO<cp_t>(),     cuUtil::INVSQRT2<cp_t>()};

            tn_state->applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                      {false, false});

            tn_state->applyOperation("Toffoli", {0, 1, 2}, inverse);

            auto results = tn_state->getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }

        SECTION("Apply non-adjacent wire indices") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<cp_t>(),   cuUtil::ZERO<cp_t>(),
                cp_t(1.0 / sqrt(2), 0), cuUtil::ZERO<cp_t>(),
                cuUtil::ZERO<cp_t>(),   cuUtil::ZERO<cp_t>(),
                cuUtil::ZERO<cp_t>(),   cp_t(1.0 / sqrt(2), 0)};

            tn_state->applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                      {false, false});

            tn_state->applyOperation("Toffoli", {1, 0, 2}, inverse);

            auto results = tn_state->getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::applyControlledOperation non-param "
                        "one-qubit with controls",
                        "[TNCuda]", TestTNBackends) {
    using TNDevice_T = TestType;
    using ComplexT = typename TNDevice_T::ComplexT;
    using PrecisionT = typename TNDevice_T::PrecisionT;

    constexpr int num_qubits = 4;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state0 =
        createTNState<TNDevice_T>(num_qubits, maxExtent);
    std::unique_ptr<TNDevice_T> tn_state1 =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const auto margin = PrecisionT{1e-5};
    const std::size_t control = GENERATE(0, 1, 2, 3);
    const std::size_t wire = GENERATE(0, 1, 2, 3);

    DYNAMIC_SECTION("Controlled gates with base operation - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire) {
            tn_state0->applyControlledOperation(
                "PauliX", std::vector<std::size_t>{control},
                std::vector<bool>{true}, std::vector<std::size_t>{wire});

            tn_state1->applyOperation(
                "CNOT", std::vector<std::size_t>{control, wire}, false);

            REQUIRE(tn_state0->getDataVector() ==
                    approx(tn_state1->getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("Controlled gates with a target matrix - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire) {
            std::vector<ComplexT> gate_matrix = {
                ComplexT{0.0, 0.0}, ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0},
                ComplexT{0.0, 0.0}};
            tn_state0->applyControlledOperation(
                "applyControlledGates", std::vector<std::size_t>{control},
                std::vector<bool>{true}, std::vector<std::size_t>{wire}, false,
                {}, gate_matrix);

            tn_state1->applyOperation(
                "CNOT", std::vector<std::size_t>{control, wire}, false);

            REQUIRE(tn_state0->getDataVector() ==
                    approx(tn_state1->getDataVector()).margin(margin));
        }
    }
}
