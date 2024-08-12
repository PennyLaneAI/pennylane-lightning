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
#include "MPSTNCuda.hpp"
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

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::Identity", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply different wire indices") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperation("Hadamard", {index}, inverse);

            mps_state.applyOperation("Identity", {index}, inverse);
            cp_t expected(1.0 / std::sqrt(2), 0);

            auto results = mps_state.getDataVector();

            CHECK(expected.real() ==
                  Approx(results[0b1 << ((num_qubits - 1 - index))].real()));
            CHECK(expected.imag() ==
                  Approx(results[0b1 << ((num_qubits - index - 1))].imag()));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::Hadamard", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply different wire indices") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperation("Hadamard", {index}, inverse);
            cp_t expected(1.0 / std::sqrt(2), 0);

            auto results = mps_state.getDataVector();

            CHECK(expected.real() ==
                  Approx(results[0b1 << ((num_qubits - 1 - index))].real()));
            CHECK(expected.imag() ==
                  Approx(results[0b1 << ((num_qubits - index - 1))].imag()));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::PauliX", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply different wire indices") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperation("PauliX", {index}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results[0] == cuUtil::ZERO<std::complex<TestType>>());
            CHECK(results[0b1 << (num_qubits - index - 1)] ==
                  cuUtil::ONE<std::complex<TestType>>());
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::PauliY", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
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
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});

            mps_state.applyOperation("PauliY", {index}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::PauliZ", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
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
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});

            mps_state.applyOperation("PauliZ", {index}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::S", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
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
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});

            mps_state.applyOperation("S", {index}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::T", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
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
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                      {{0}, {1}, {2}}, {false, false, false});

            mps_state.applyOperation("T", {index}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::CNOT", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply adjacent wire indices") {
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "CNOT", "CNOT"},
                                      {{0}, {0, 1}, {1, 2}},
                                      {false, inverse, inverse});

            auto results = mps_state.getDataVector();

            CHECK(results.front() ==
                  cuUtil::INVSQRT2<std::complex<TestType>>());
            CHECK(results.back() == cuUtil::INVSQRT2<std::complex<TestType>>());
        }

        SECTION("Apply non-adjacent wire indices") {
            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperation("Hadamard", {0}, false);
            mps_state.applyOperation("CNOT", {0, 2}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results[0] == cuUtil::INVSQRT2<std::complex<TestType>>());
            CHECK(results[5] == cuUtil::INVSQRT2<std::complex<TestType>>());
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::SWAP", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
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

            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                      {false, false});

            mps_state.applyOperation("SWAP", {0, 1}, inverse);

            auto results = mps_state.getDataVector();

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

            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                      {false, false});

            mps_state.applyOperation("SWAP", {0, 2}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::CY", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
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

            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                      {false, false});

            mps_state.applyOperation("CY", {0, 1}, inverse);

            auto results = mps_state.getDataVector();

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

            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                      {false, false});

            mps_state.applyOperation("CY", {0, 2}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Gates::CZ", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false, true);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
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

            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                      {false, false});

            mps_state.applyOperation("CZ", {0, 1}, inverse);

            auto results = mps_state.getDataVector();

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

            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            mps_state.applyOperations({"Hadamard", "PauliX"}, {{0}, {1}},
                                      {false, false});

            mps_state.applyOperation("CZ", {0, 2}, inverse);

            auto results = mps_state.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::Non_Param_Gates::2+_wires",
                   "[MPSTNCuda_Nonparam]", float, double) {
    const bool inverse = GENERATE(false, true);
    {
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Toffoli gate") {
            std::size_t num_qubits = 3;

            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};

            REQUIRE_THROWS_AS(
                mps_state.applyOperation("Toffoli", {0, 1, 2}, inverse),
                LightningException);
        }

        SECTION("CSWAP gate") {
            std::size_t num_qubits = 4;

            MPSTNCuda<TestType> mps_state{num_qubits, maxExtent, dev_tag};
            REQUIRE_THROWS_AS(
                mps_state.applyOperation("CSWAP", {0, 1, 2}, inverse),
                LightningException);
        }
    }
}
