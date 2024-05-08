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

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "DevTag.hpp"
#include "MPSTNCuda.hpp"
#include "TNCudaGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::Util;

namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
} // namespace

TEMPLATE_TEST_CASE("MPSTNCuda::applyHadamard", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply different wire indices using dispatcher") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {index}, inverse);
            cp_t expected(1.0 / std::sqrt(2), 0);

            auto results = sv.getDataVector();

            CHECK(expected.real() ==
                  Approx(results[0b1 << ((num_qubits - 1 - index))].real()));
            CHECK(expected.imag() ==
                  Approx(results[0b1 << ((num_qubits - index - 1))].imag()));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyPauliX", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(true, false);
    {
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply different wire indices using dispatcher") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("PauliX", {index}, inverse);

            auto results = sv.getDataVector();

            CHECK(results[0] == cuUtil::ZERO<std::complex<TestType>>());
            CHECK(results[0b1 << (num_qubits - index - 1)] ==
                  cuUtil::ONE<std::complex<TestType>>());
        }
    }
}
// TODO IMPORTANT NOTE: inverse doesn't work for PauliY here
TEMPLATE_TEST_CASE("MPSTNCuda::applyPauliY", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false);
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

        SECTION("Apply different wire indices using dispatcher") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);
            sv.appendGateTensorOperator("PauliY", {index}, inverse);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyPauliZ", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(true, false);
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

        SECTION("Apply different wire indices using dispatcher") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("PauliZ", {index}, inverse);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyS", "[MPSTNCuda_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
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

        SECTION("Apply different wire indices using dispatcher") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("S", {index}, inverse);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyT", "[MPSTNCuda_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
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

        SECTION("Apply different wire indices using dispatcher") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("T", {index}, inverse);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyCNOT", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(true, false);
    {
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply adjacent wire indices using dispatcher") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("CNOT", {0, 1}, inverse);
            sv.appendGateTensorOperator("CNOT", {1, 2}, inverse);

            auto results = sv.getDataVector();

            CHECK(results.front() ==
                  cuUtil::INVSQRT2<std::complex<TestType>>());
            CHECK(results.back() == cuUtil::INVSQRT2<std::complex<TestType>>());
        }

        SECTION("Apply non-adjacent wire indices using dispatcher") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("CNOT", {0, 2}, inverse);

            auto results = sv.getDataVector();

            CHECK(results[0] == cuUtil::INVSQRT2<std::complex<TestType>>());
            CHECK(results[5] == cuUtil::INVSQRT2<std::complex<TestType>>());
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applySWAP", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply adjacent wire indices using dispatcher") {
            std::vector<cp_t> expected{cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>()};

            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("PauliX", {1}, false);
            sv.appendGateTensorOperator("SWAP", {0, 1}, inverse);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected));
        }

        SECTION("Apply non-adjacent wire indices using dispatcher") {
            std::vector<cp_t> expected{cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>(),
                                       cuUtil::ZERO<std::complex<TestType>>()};

            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("PauliX", {1}, false);
            sv.appendGateTensorOperator("SWAP", {0, 2}, inverse);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyCY", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply adjacent wire indices using dispatcher") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(0, -1 / sqrt(2)),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>()};

            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("PauliX", {1}, false);
            sv.appendGateTensorOperator("CY", {0, 1}, inverse);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }

        SECTION("Apply non-adjacent wire indices using dispatcher") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0.0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(0.0, 1.0 / sqrt(2))};

            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("PauliX", {1}, false);
            sv.appendGateTensorOperator("CY", {0, 2}, inverse);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyCZ", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(true, false);
    {

        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply adjacent wire indices using dispatcher") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(-1 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>()};

            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("PauliX", {1}, false);
            sv.appendGateTensorOperator("CZ", {0, 1}, inverse);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }

        SECTION("Apply non-adjacent wire indices using dispatcher") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>()};

            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("PauliX", {1}, false);
            sv.appendGateTensorOperator("CZ", {0, 2}, inverse);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyToffoli", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(true, false);
    {

        using cp_t = std::complex<TestType>;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply adjacent wire indices using dispatcher") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0)};

            std::size_t num_qubits = 3;

            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("PauliX", {1}, false);
            sv.appendGateTensorOperator("Toffoli", {0, 1, 2}, inverse);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }

        SECTION("Apply non-adjacent wire indices using dispatcher") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>()};

            std::size_t num_qubits = 4;

            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("PauliX", {1}, false);
            sv.appendGateTensorOperator("Toffoli", {0, 2, 3}, inverse);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyCSWAP", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(true, false);
    {

        using cp_t = std::complex<TestType>;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply adjacent wire indices using dispatcher") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>()};

            std::size_t num_qubits = 3;

            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("PauliX", {1}, false);
            sv.appendGateTensorOperator("CSWAP", {0, 1, 2}, inverse);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
        }

        SECTION("Apply non-adjacent wire indices using dispatcher") {
            std::vector<cp_t> expected_results{
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>(),
                cuUtil::ZERO<std::complex<TestType>>()};

            std::size_t num_qubits = 4;

            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("PauliX", {1}, false);
            sv.appendGateTensorOperator("CSWAP", {0, 2, 3}, inverse);

            auto results = sv.getDataVector();
            // TODO remove the following line later. This is a test if we can
            // call getDataVector() multiple times
            auto results_test = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results));
            CHECK(results_test == Pennylane::Util::approx(expected_results));
        }
    }
}