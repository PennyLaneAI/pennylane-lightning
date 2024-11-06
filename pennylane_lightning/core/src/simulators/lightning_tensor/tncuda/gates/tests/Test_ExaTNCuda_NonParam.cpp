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