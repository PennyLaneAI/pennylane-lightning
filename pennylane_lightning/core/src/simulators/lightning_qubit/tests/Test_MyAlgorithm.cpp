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
#include <limits> // numeric_limits
#include <random>
#include <type_traits>
#include <vector>

#include <catch2/catch.hpp>

#include "LinearAlgebra.hpp" //randomUnitary
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"
#include "TestHelpers.hpp" // createRandomStateVectorData
#include "TestHelpersWires.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"

/**
 * @file
 *  Draft test to demonstrate how to write an algorithm with the C++ API.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using Pennylane::Util::randomUnitary;
} // namespace
/// @endcond

TEST_CASE("MyAlgorithm::runAlgorithm", "[runAlgorithm]") {
    SECTION("runAlgorithm<>") {

        std::cout << "Hello" << std::endl;

        using StateVectorT = StateVectorLQubitManaged<double>;
        using ComplexT = typename StateVectorT::ComplexT;

        const ComplexT one{1.0};
        const ComplexT zero{0.0};

        size_t num_qubits = 4;
        auto statevector_data = createZeroState<ComplexT>(num_qubits);

        StateVectorT sv(statevector_data.data(), statevector_data.size());

        for (std::size_t i = 0; i < num_qubits; i++) {
            sv.applyOperation("Hadamard", {i}, false);
        }

        // REQUIRE(sv.getDataVector() == approx(expected_state_000));
    }
}
