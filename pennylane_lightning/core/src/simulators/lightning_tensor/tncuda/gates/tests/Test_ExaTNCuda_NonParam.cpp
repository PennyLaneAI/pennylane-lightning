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
#include "TNCudaGateCache.hpp"

#include "TestHelpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::LightningTensor::TNCuda::Gates;
using namespace Pennylane::Util;
namespace cuUtil = Pennylane::LightningGPU::Util;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("ExaTNCuda::Gates::Identity", "[ExaTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply different wire indices") {
            const std::size_t index = GENERATE(0, 1, 2);
            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperation("Hadamard", {index}, inverse);

            exatn_state.applyOperation("Identity", {index}, inverse);
            cp_t expected(1.0 / std::sqrt(2), 0);

            auto results = exatn_state.getDataVector();

            CHECK(expected.real() ==
                  Approx(results[0b1 << ((num_qubits - 1 - index))].real()));
            CHECK(expected.imag() ==
                  Approx(results[0b1 << ((num_qubits - index - 1))].imag()));
        }
    }
}

TEMPLATE_TEST_CASE("ExaTNCuda::Gates::Hadamard", "[ExaTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply different wire indices") {
            const std::size_t index = GENERATE(0, 1, 2);
            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperation("Hadamard", {index}, inverse);

            exatn_state.applyOperation("Identity", {index}, inverse);

            // Test for multiple final states appendings

            cp_t expected(1.0 / std::sqrt(2), 0);

            auto results = exatn_state.getDataVector();

            CHECK(expected.real() ==
                  Approx(results[0b1 << ((num_qubits - 1 - index))].real()));
            CHECK(expected.imag() ==
                  Approx(results[0b1 << ((num_qubits - index - 1))].imag()));
        }
    }
}

TEMPLATE_TEST_CASE("ExaTNCuda::Gates::PauliX", "[ExaTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        std::size_t num_qubits = 3;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply different wire indices") {
            const std::size_t index = GENERATE(0, 1, 2);
            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperation("PauliX", {index}, inverse);

            auto results = exatn_state.getDataVector();

            CHECK(results[0] == cuUtil::ZERO<std::complex<TestType>>());
            CHECK(results[0b1 << (num_qubits - index - 1)] ==
                  cuUtil::ONE<std::complex<TestType>>());
        }
    }
}

TEMPLATE_TEST_CASE("ExaTNCuda::Gates::PauliY", "[ExaTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        DevTag<int> dev_tag{0, 0};

        const cp_t p = cuUtil::ConstMult(
            std::complex<TestType>(0.5, 0.0),
            cuUtil::ConstMult(cuUtil::INVSQRT2<std::complex<TestType>>(),
                              cuUtil::IMAG<std::complex<TestType>>()));
        const cp_t m = cuUtil::ConstMult(std::complex<TestType>(-1, 0), p);

        const std::vector<std::vector<cp_t>> expected_results = {
            {m, m, m, m, p, p, p, p},
            {m, m, p, p, m, m, p, p},
            {m, p, m, p, m, p, m, p}};

        SECTION("Apply different wire indices") {
            const std::size_t index = GENERATE(0, 1, 2);
            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                        {{0}, {1}, {2}}, {false, false, false});

            exatn_state.applyOperation("PauliY", {index}, inverse);

            auto results = exatn_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("ExaTNCuda::Gates::PauliZ", "[ExaTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        DevTag<int> dev_tag{0, 0};

        const cp_t p(static_cast<TestType>(0.5) *
                     cuUtil::INVSQRT2<std::complex<TestType>>());
        const cp_t m(cuUtil::ConstMult(cp_t{-1.0, 0.0}, p));

        const std::vector<std::vector<cp_t>> expected_results = {
            {p, p, p, p, m, m, m, m},
            {p, p, m, m, p, p, m, m},
            {p, m, p, m, p, m, p, m}};

        SECTION("Apply different wire indices") {
            const std::size_t index = GENERATE(0, 1, 2);
            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                        {{0}, {1}, {2}}, {false, false, false});

            exatn_state.applyOperation("PauliZ", {index}, inverse);

            auto results = exatn_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("ExaTNCuda::Gates::S", "[ExaTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        DevTag<int> dev_tag{0, 0};

        cp_t r(std::complex<TestType>(0.5, 0.0) *
               cuUtil::INVSQRT2<std::complex<TestType>>());
        cp_t i(cuUtil::ConstMult(r, cuUtil::IMAG<std::complex<TestType>>()));

        if (inverse) {
            i = conj(i);
        }

        const std::vector<std::vector<cp_t>> expected_results = {
            {r, r, r, r, i, i, i, i},
            {r, r, i, i, r, r, i, i},
            {r, i, r, i, r, i, r, i}};

        SECTION("Apply different wire indices") {
            const std::size_t index = GENERATE(0, 1, 2);
            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                        {{0}, {1}, {2}}, {false, false, false});

            exatn_state.applyOperation("S", {index}, inverse);

            auto results = exatn_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("ExaTNCuda::Gates::T", "[ExaTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        DevTag<int> dev_tag{0, 0};

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
            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                        {{0}, {1}, {2}}, {false, false, false});

            exatn_state.applyOperation("T", {index}, inverse);

            auto results = exatn_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("ExaTNCuda::Gates::CNOT", "[ExaTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        std::size_t num_qubits = 3;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply adjacent wire indices") {
            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperations({"Hadamard", "CNOT", "CNOT"},
                                        {{0}, {0, 1}, {1, 2}},
                                        {false, inverse, inverse});

            auto results = exatn_state.getDataVector();

            CHECK(results.front() ==
                  cuUtil::INVSQRT2<std::complex<TestType>>());
            CHECK(results.back() == cuUtil::INVSQRT2<std::complex<TestType>>());
        }

        SECTION("Apply non-adjacent wire indices") {
            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperation("Hadamard", {0}, false);
            exatn_state.applyOperation("CNOT", {0, 2}, inverse);

            auto results = exatn_state.getDataVector();

            CHECK(results[0] == cuUtil::INVSQRT2<std::complex<TestType>>());
            CHECK(results[5] == cuUtil::INVSQRT2<std::complex<TestType>>());
        }
    }
}

TEMPLATE_TEST_CASE("ExaTNCuda::Gates::SWAP", "[ExaTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply adjacent wire indices") {
            std::vector<cp_t> expected{cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>()};

            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                        {false, false});

            exatn_state.applyOperation("SWAP", {0, 1}, inverse);

            auto results = exatn_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected));
        }

        SECTION("Apply non-adjacent wire indices") {
            std::vector<cp_t> expected{cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>()};

            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                        {false, false});

            exatn_state.applyOperation("SWAP", {0, 2}, inverse);

            auto results = exatn_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected));
        }
    }
}

TEMPLATE_TEST_CASE("ExaTNCuda::Gates::CY", "[ExaTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply adjacent wire indices") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(0, -1 / sqrt(2)),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>()};

            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                        {false, false});

            exatn_state.applyOperation("CY", {0, 1}, inverse);

            auto results = exatn_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }

        SECTION("Apply non-adjacent wire indices") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0.0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(0.0, 1.0 / sqrt(2))};

            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                        {false, false});

            exatn_state.applyOperation("CY", {0, 2}, inverse);

            auto results = exatn_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("ExaTNCuda::Gates::CZ", "[ExaTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply adjacent wire indices") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(-1 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>()};

            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                        {false, false});

            exatn_state.applyOperation("CZ", {0, 1}, inverse);

            auto results = exatn_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }

        SECTION("Apply non-adjacent wire indices") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>()};

            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                        {false, false});

            exatn_state.applyOperation("CZ", {0, 2}, inverse);

            auto results = exatn_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("ExaTNCuda::Gates::Toffoli", "[ExaTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply adjacent wire indices") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::INVSQRT2<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::INVSQRT2<std::complex<TestType>>()};

            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                        {false, false});

            exatn_state.applyOperation("Toffoli", {0, 1, 2}, inverse);

            auto results = exatn_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }

        SECTION("Apply non-adjacent wire indices") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0)};

            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                        {false, false});

            exatn_state.applyOperation("Toffoli", {1, 0, 2}, inverse);

            auto results = exatn_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("ExaTNCuda::Gates::CSWAP", "[ExaTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply adjacent wire indices") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::INVSQRT2<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::INVSQRT2<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>()};

            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                        {false, false});

            exatn_state.applyOperation("CSWAP", {0, 1, 2}, inverse);

            auto results = exatn_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }

        SECTION("Apply non-adjacent wire indices") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>()};

            ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

            exatn_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                        {false, false});

            exatn_state.applyOperation("CSWAP", {1, 0, 2}, inverse);

            auto results = exatn_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("ExaTNCuda::Gates::applyOperation-gatematrix",
                   "[ExaTNCuda_Nonparam]", float, double) {
    std::size_t num_qubits = 3;
    DevTag<int> dev_tag{0, 0};

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);
        ExaTNCuda<TestType> exatn_state{num_qubits, dev_tag};

        std::vector<std::complex<TestType>> gate_matrix = {
            cuUtil::ZERO<std::complex<TestType>>(),
            cuUtil::ONE<std::complex<TestType>>(),
            cuUtil::ONE<std::complex<TestType>>(),
            cuUtil::ZERO<std::complex<TestType>>()};

        exatn_state.applyOperation("applyMatrix", {index}, false, {},
                                   gate_matrix);

        auto results = exatn_state.getDataVector();

        CHECK(results[0] == cuUtil::ZERO<std::complex<TestType>>());
        CHECK(results[0b1 << (num_qubits - index - 1)] ==
              cuUtil::ONE<std::complex<TestType>>());
    }
}

TEMPLATE_TEST_CASE("ExaTNCuda::applyControlledOperation non-param "
                   "one-qubit with controls",
                   "[ExaTNCuda]", float, double) {
    using PrecisionT = TestType;
    using ComplexT = std::complex<PrecisionT>;
    const int num_qubits = 4;
    DevTag<int> dev_tag{0, 0};

    const auto margin = PrecisionT{1e-5};
    const std::size_t control = GENERATE(0, 1, 2, 3);
    const std::size_t wire = GENERATE(0, 1, 2, 3);

    ExaTNCuda<PrecisionT> exatn_state0{num_qubits, dev_tag};
    ExaTNCuda<PrecisionT> exatn_state1{num_qubits, dev_tag};

    DYNAMIC_SECTION("Controlled gates with base operation - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire) {
            exatn_state0.applyControlledOperation(
                "PauliX", std::vector<std::size_t>{control},
                std::vector<bool>{true}, std::vector<std::size_t>{wire});

            exatn_state1.applyOperation(
                "CNOT", std::vector<std::size_t>{control, wire}, false);

            REQUIRE(exatn_state0.getDataVector() ==
                    approx(exatn_state1.getDataVector()).margin(margin));
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
            exatn_state0.applyControlledOperation(
                "applyControlledGates", std::vector<std::size_t>{control},
                std::vector<bool>{true}, std::vector<std::size_t>{wire}, false,
                {}, gate_matrix);

            exatn_state1.applyOperation(
                "CNOT", std::vector<std::size_t>{control, wire}, false);

            REQUIRE(exatn_state0.getDataVector() ==
                    approx(exatn_state1.getDataVector()).margin(margin));
        }
    }

    SECTION("Throw exception for 1+ target wires gates") {
        REQUIRE_THROWS_AS(exatn_state0.applyControlledOperation(
                              "CSWAP", {0}, {true, true}, {1, 2}),
                          LightningException);
    }
}
