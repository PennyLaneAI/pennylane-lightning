// Copyright 2022-2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "LightningKokkosSimulator.hpp"
#include "catch2/catch.hpp"

/// @cond DEV
namespace {
using namespace Catalyst::Runtime::Simulator;
using LKSimulator = LightningKokkosSimulator;
} // namespace
/// @endcond

TEST_CASE("Zero qubits. Zero parameters", "[Gradient]") {
    std::unique_ptr<LKSimulator> LKsim = std::make_unique<LKSimulator>();

    std::vector<DataView<double, 1>> gradients;
    std::vector<intptr_t> Qs = LKsim->AllocateQubits(0);
    REQUIRE_NOTHROW(LKsim->Gradient(gradients, {}));
}
