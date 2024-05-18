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
#include "MPSTNCuda.hpp"
#include "ObservablesTNCuda_host.hpp"

#include "TestHelpers.hpp"

#include <catch2/catch.hpp>

/// @cond DEV
namespace {
using namespace Pennylane::LightningTensor::TNCuda::Observables;
using Pennylane::Util::LightningException;
} // namespace
/// @endcond

TEMPLATE_PRODUCT_TEST_CASE("NamedObs", "[Observables]", (MPSTNCuda),
                           (float, double)) {
    using StateTensorT = TestType;
    using NamedObsT = NamedObs<StateTensorT>;

    std::size_t bondDim = GENERATE(2, 3, 4, 5);
    std::size_t num_qubits = 3;
    std::size_t maxBondDim = bondDim;

    StateTensorT mps_state{num_qubits, maxBondDim};

    SECTION("Test get obs name") {
        auto obs = NamedObsT("PauliX", {0});

        CHECK(obs.getObsName() == "PauliX[0]");
    }
}
