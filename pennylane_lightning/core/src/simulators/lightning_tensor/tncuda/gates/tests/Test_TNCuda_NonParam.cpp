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
#include "ExaTNCuda.hpp"
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
    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent, dev_tag);

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);

        tn_state->applyOperation("Hadamard", {index}, inverse);

        tn_state->applyOperation("Identity", {index}, inverse);
        cp_t expected(1.0 / std::sqrt(2), 0);

        if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

        auto results = tn_state->getDataVector();

        CHECK(expected.real() ==
              Approx(results[0b1 << ((num_qubits - 1 - index))].real()));
        CHECK(expected.imag() ==
              Approx(results[0b1 << ((num_qubits - index - 1))].imag()));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::Hadamard", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent, dev_tag);

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);

        if constexpr (std::is_same_v<TestType, MPSTNCuda<double>> ||
                      std::is_same_v<TestType, MPSTNCuda<float>>) {
            tn_state->append_mps_final_state();
        }

        tn_state->applyOperation("Hadamard", {index}, inverse);

        if constexpr (std::is_same_v<TestType, MPSTNCuda<double>> ||
                      std::is_same_v<TestType, MPSTNCuda<float>>) {
            tn_state->append_mps_final_state();
        }

        tn_state->applyOperation("Identity", {index}, inverse);

        // Test for multiple final states appendings
        if constexpr (std::is_same_v<TestType, MPSTNCuda<double>> ||
                      std::is_same_v<TestType, MPSTNCuda<float>>) {
            tn_state->append_mps_final_state();
        }

        cp_t expected(1.0 / std::sqrt(2), 0);

        if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

        auto results = tn_state->getDataVector();

        CHECK(expected.real() ==
              Approx(results[0b1 << ((num_qubits - 1 - index))].real()));
        CHECK(expected.imag() ==
              Approx(results[0b1 << ((num_qubits - index - 1))].imag()));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::PauliX", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;

    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent, dev_tag);

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);

        tn_state->applyOperation("PauliX", {index}, inverse);

        if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

        auto results = tn_state->getDataVector();

        CHECK(results[0] == cuUtil::ZERO<cp_t>());
        CHECK(results[0b1 << (num_qubits - index - 1)] == cuUtil::ONE<cp_t>());
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::applyOperation-gatematrix",
                        "[TNCuda_Nonparam]", TestTNBackends) {

    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    // using Precision_T = typename TNDevice_T::PrecisionT;

    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent, dev_tag);

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);

        std::vector<cp_t> gate_matrix = {
            cuUtil::ZERO<cp_t>(), cuUtil::ONE<cp_t>(), cuUtil::ONE<cp_t>(),
            cuUtil::ZERO<cp_t>()};

        tn_state->applyOperation("applyMatrix", {index}, false, {},
                                 gate_matrix);

        if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

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
    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent, dev_tag);

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

        if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[index]));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::PauliZ", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent, dev_tag);

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

        if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[index]));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::S", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent, dev_tag);

    cp_t r(cp_t(0.5, 0.0) * cuUtil::INVSQRT2<cp_t>());
    cp_t i(cuUtil::ConstMult(r, cuUtil::IMAG<cp_t>()));

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

        tn_state->applyOperation("S", {index}, inverse);

        if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

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
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent, dev_tag);

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

        if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[index]).margin(1e-8));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::CNOT", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;

    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent, dev_tag);

    SECTION("Apply adjacent wire indices") {

        tn_state->applyOperations({"Hadamard", "CNOT", "CNOT"},
                                  {{0}, {0, 1}, {1, 2}},
                                  {false, inverse, inverse});

        if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

        auto results = tn_state->getDataVector();

        CHECK(results.front() == Pennylane::Util::approx(cuUtil::INVSQRT2<cp_t>()).epsilon(1e-5));
        CHECK(results.back()  == Pennylane::Util::approx(cuUtil::INVSQRT2<cp_t>()).epsilon(1e-5));
    }

    SECTION("Apply non-adjacent wire indices") {

        tn_state->applyOperation("Hadamard", {0}, false);
        tn_state->applyOperation("CNOT", {0, 2}, inverse);

        if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

        auto results = tn_state->getDataVector();

        CHECK(results[0] == Pennylane::Util::approx(cuUtil::INVSQRT2<cp_t>()).epsilon(1e-5));
        CHECK(results[5] == Pennylane::Util::approx(cuUtil::INVSQRT2<cp_t>()).epsilon(1e-5));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::SWAP", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;

    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent, dev_tag);

    SECTION("Apply adjacent wire indices") {
        std::vector<cp_t> expected{
            cuUtil::ZERO<cp_t>(),   cuUtil::ZERO<cp_t>(),
            cuUtil::ZERO<cp_t>(),   cuUtil::ZERO<cp_t>(),
            cp_t(1.0 / sqrt(2), 0), cuUtil::ZERO<cp_t>(),
            cp_t(1.0 / sqrt(2), 0), cuUtil::ZERO<cp_t>()};

        tn_state->applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                  {false, false});

        tn_state->applyOperation("SWAP", {0, 1}, inverse);

        if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected));
    }

    SECTION("Apply non-adjacent wire indices") {
        std::vector<cp_t> expected{
            cuUtil::ZERO<cp_t>(), cuUtil::ZERO<cp_t>(),
            cuUtil::INVSQRT2<cp_t>(), cuUtil::INVSQRT2<cp_t>(),
            cuUtil::ZERO<cp_t>(), cuUtil::ZERO<cp_t>(),
            cuUtil::ZERO<cp_t>(), cuUtil::ZERO<cp_t>()};

        tn_state->applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                  {false, false});

        tn_state->applyOperation("SWAP", {0, 2}, inverse);

        if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected).margin(1e-5));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::CY", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent, dev_tag);

    SECTION("Apply adjacent wire indices") {
        std::vector<cp_t> expected_results{
            cuUtil::ZERO<cp_t>(), cuUtil::ZERO<cp_t>(), cuUtil::INVSQRT2<cp_t>(),
            cuUtil::ZERO<cp_t>(), -cuUtil::INVSQRT2IMAG<cp_t>(), cuUtil::ZERO<cp_t>(),
            cuUtil::ZERO<cp_t>(), cuUtil::ZERO<cp_t>()};

        tn_state->applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                  {false, false});

        tn_state->applyOperation("CY", {0, 1}, inverse);

        if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results).margin(1e-5));
    }

    SECTION("Apply non-adjacent wire indices") {
        std::vector<cp_t> expected_results{
            cuUtil::ZERO<cp_t>(), cuUtil::ZERO<cp_t>(),
            cuUtil::INVSQRT2<cp_t>(), cuUtil::ZERO<cp_t>(),
            cuUtil::ZERO<cp_t>(), cuUtil::ZERO<cp_t>(),
            cuUtil::ZERO<cp_t>(), cuUtil::INVSQRT2IMAG<cp_t>()};

        tn_state->applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                  {false, false});

        tn_state->applyOperation("CY", {0, 2}, inverse);

        if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::CZ", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent, dev_tag);

    SECTION("Apply adjacent wire indices") {
        std::vector<cp_t> expected_results{
            cuUtil::ZERO<cp_t>(), cuUtil::ZERO<cp_t>(), cuUtil::INVSQRT2<cp_t>(),
            cuUtil::ZERO<cp_t>(), cuUtil::ZERO<cp_t>(), cuUtil::ZERO<cp_t>(),
            -cuUtil::INVSQRT2<cp_t>(), cuUtil::ZERO<cp_t>()};

        tn_state->applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                  {false, false});

        tn_state->applyOperation("CZ", {0, 1}, inverse);

        if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

        auto results = tn_state->getDataVector();

        std::for_each(results.begin(), results.end(), [](cp_t &val)
                      { val += cuUtil::ONE<cp_t>(); });
        std::for_each(expected_results.begin(), expected_results.end(), [](cp_t &val)
                      { val += cuUtil::ONE<cp_t>(); });

        CHECK(expected_results == Pennylane::Util::approx(results).margin(1e-8));
    }

    SECTION("Apply non-adjacent wire indices") {
        std::vector<cp_t> expected_results{
            cuUtil::ZERO<cp_t>()    , cuUtil::ZERO<cp_t>(),
            cuUtil::INVSQRT2<cp_t>(), cuUtil::ZERO<cp_t>(),
            cuUtil::ZERO<cp_t>()    , cuUtil::ZERO<cp_t>(),
            cuUtil::INVSQRT2<cp_t>(), cuUtil::ZERO<cp_t>()};

        tn_state->applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                  {false, false});

        tn_state->applyOperation("CZ", {0, 2}, inverse);

        if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results));
    }
}

TEMPLATE_LIST_TEST_CASE("ExaTNCuda::Gates::CSWAP", "[ExaTNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;

    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state;

    if constexpr (std::is_same_v<TestType, MPSTNCuda<double>> ||
                  std::is_same_v<TestType, MPSTNCuda<float>>) {
        // Create the object for MPSTNCuda
        tn_state = std::make_unique<TNDevice_T>(num_qubits, maxExtent, dev_tag);
        SECTION("CSWAP gate") {
            // Create the object for MPSTNCuda
            tn_state =
                std::make_unique<TNDevice_T>(num_qubits, maxExtent, dev_tag);
            REQUIRE_THROWS_AS(
                tn_state->applyOperation("CSWAP", {0, 1, 2}, inverse),
                LightningException);
        }
    } else {
        // Create the object for ExaTNCuda
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

            if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

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

            if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

        auto results = tn_state->getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("ExaTNCuda::Gates::Toffoli", "[ExaTNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;

    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state;

    if constexpr (std::is_same_v<TestType, MPSTNCuda<double>> ||
                  std::is_same_v<TestType, MPSTNCuda<float>>) {
        // Create the object for MPSTNCuda
        tn_state = std::make_unique<TNDevice_T>(num_qubits, maxExtent, dev_tag);
        SECTION("Toffoli gate") {
            std::size_t num_qubits = 3;
            // Create the object for MPSTNCuda
            tn_state =
                std::make_unique<TNDevice_T>(num_qubits, maxExtent, dev_tag);

            REQUIRE_THROWS_AS(
                tn_state->applyOperation("Toffoli", {0, 1, 2}, inverse),
                LightningException);
        }
    } else {
        // Create the object for ExaTNCuda
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

            if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

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

            if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                      std::is_same_v<TNDevice_T, MPSTNCuda<float>>)
        {
            tn_state->append_mps_final_state();
        }

        auto results = tn_state->getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::applyControlledOperation non-param "
                        "one-qubit with controls",
                        "[MPSTNCuda]", TestTNBackends) {

    using TNDevice_T = TestType;
    using ComplexT = typename TNDevice_T::ComplexT;
    using PrecisionT = typename TNDevice_T::PrecisionT;

    const int num_qubits = 4;
    std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state0;
    std::unique_ptr<TNDevice_T> tn_state1;

    if constexpr (std::is_same_v<TestType, MPSTNCuda<double>> ||
                  std::is_same_v<TestType, MPSTNCuda<float>>) {
        // Create the object for MPSTNCuda
        tn_state0 =
            std::make_unique<TNDevice_T>(num_qubits, maxExtent, dev_tag);
        tn_state1 =
            std::make_unique<TNDevice_T>(num_qubits, maxExtent, dev_tag);
    } else {
        // Create the object for ExaTNCuda
        tn_state0 = std::make_unique<TNDevice_T>(num_qubits, dev_tag);
        tn_state1 = std::make_unique<TNDevice_T>(num_qubits, dev_tag);
    }

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

    // SECTION("Throw exception for 1+ target wires gates") {
    //     REQUIRE_THROWS_AS(tn_state0->applyControlledOperation(
    //                           "CSWAP", {0}, {true, true}, {1, 2}),
    //                       LightningException);
    // }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::applyMPO::2+_wires", "[TNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using PrecisionT = typename TNDevice_T::PrecisionT;

    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    std::size_t max_mpo_bond = 16;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent, dev_tag);

    std::vector<std::vector<cp_t>> mpo_cnot(2,
                                            std::vector<cp_t>(16, {0.0, 0.0}));

    // in-order decomposition of the cnot operator
    // data from scipy decompose in the lightning.tensor python layer
    mpo_cnot[0][0] = {1.0, 0.0};
    mpo_cnot[0][3] = {-1.0, 0.0};
    mpo_cnot[0][9] = {1.0, 0.0};
    mpo_cnot[0][10] = {-1.0, 0.0};

    mpo_cnot[1][0] = {1.0, 0.0};
    mpo_cnot[1][7] = {-1.0, 0.0};
    mpo_cnot[1][10] = {1.0, 0.0};
    mpo_cnot[1][13] = {-1.0, 0.0};

    std::vector<std::vector<cp_t>> mpo_cswap;
    mpo_cswap.emplace_back(std::vector<cp_t>(16, {0.0, 0.0}));
    mpo_cswap.emplace_back(std::vector<cp_t>(64, {0.0, 0.0}));
    mpo_cswap.emplace_back(std::vector<cp_t>(16, {0.0, 0.0}));

    mpo_cswap[0][0] = {-1.5811388300841898, 0.0};
    mpo_cswap[0][2] = {0.7071067811865475, 0.0};
    mpo_cswap[0][5] = {-1.0, 0.0};
    mpo_cswap[0][9] = mpo_cswap[0][0];
    mpo_cswap[0][11] = -mpo_cswap[0][2];
    mpo_cswap[0][14] = {1.0, 0.0};

    mpo_cswap[1][0] = {-0.413452607315265, 0.0};
    mpo_cswap[1][1] = {0.6979762349196628, 0.0};
    mpo_cswap[1][7] = {0.9870874576374964, 0.0};
    mpo_cswap[1][8] = {0.5736348503222318, 0.0};
    mpo_cswap[1][9] = {0.11326595025589799, 0.0};
    mpo_cswap[1][15] = {0.16018224300696726, 0.0};
    mpo_cswap[1][34] = -mpo_cswap[1][7];
    mpo_cswap[1][36] = mpo_cswap[1][0];
    mpo_cswap[1][37] = -mpo_cswap[1][1];
    mpo_cswap[1][42] = -mpo_cswap[1][15];
    mpo_cswap[1][44] = mpo_cswap[1][8];
    mpo_cswap[1][45] = -mpo_cswap[1][9];

    mpo_cswap[2][0] = mpo_cswap[1][15];
    mpo_cswap[2][1] = -mpo_cswap[1][7];
    mpo_cswap[2][7] = {1.0, 0.0};
    mpo_cswap[2][10] = {-1.0, 0.0};
    mpo_cswap[2][12] = -mpo_cswap[2][1];
    mpo_cswap[2][13] = mpo_cswap[2][0];

    SECTION("Target at wire indices") {

        MPSTNCuda<PrecisionT> mps_state_mpo{num_qubits, maxExtent, dev_tag};

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        mps_state_mpo.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("CNOT", {0, 1}, inverse);

        mps_state_mpo.applyMPOOperation(mpo_cnot, {0, 1}, max_mpo_bond);

        auto ref = tn_state->getDataVector();
        auto res = mps_state_mpo.getDataVector();

        CHECK(res == Pennylane::Util::approx(ref));
    }

    SECTION("Target at non-adjacent wire indices") {

        MPSTNCuda<PrecisionT> mps_state_mpo{num_qubits, maxExtent, dev_tag};

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        mps_state_mpo.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("CNOT", {0, 2}, inverse);

        mps_state_mpo.applyMPOOperation(mpo_cnot, {0, 2}, max_mpo_bond);

        auto ref = tn_state->getDataVector();
        auto res = mps_state_mpo.getDataVector();

        CHECK(res == Pennylane::Util::approx(ref));
    }

    SECTION("Tests for 3-wire MPOs") {

        MPSTNCuda<PrecisionT> mps_state_mpo{num_qubits, maxExtent, dev_tag};

        mps_state_mpo.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});
        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        mps_state_mpo.applyMPOOperation(mpo_cswap, {0, 1, 2}, max_mpo_bond);

        if constexpr (std::is_same_v<TestType, ExaTNCuda<double>> ||
                      std::is_same_v<TestType, ExaTNCuda<float>>) {
            tn_state->applyOperation("CSWAP", {0, 1, 2}, inverse);
        }

        auto res = mps_state_mpo.getDataVector();
        auto ref = tn_state->getDataVector();

        CHECK(res == Pennylane::Util::approx(ref));
    }
}
