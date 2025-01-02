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
#include <tuple>

#include <catch2/catch.hpp>

#include "MPSTNCuda.hpp"
#include "ObservablesTNCuda.hpp"
#include "ObservablesTNCudaOperator.hpp"

#include "TestHelpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningTensor::TNCuda::Observables;
using Pennylane::Util::LightningException;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("[CTOR]", "[ObservablesTNCudaOperator]", float, double) {
    {
        using TensorNetT = MPSTNCuda<TestType>;
        using PrecisionT = typename TensorNetT::PrecisionT;
        // using ComplexT = typename TensorNetT::ComplexT;
        using NamedObsT = NamedObsTNCuda<TensorNetT>;
        using HamiltonianT = HamiltonianTNCuda<TensorNetT>;

        auto cnot = std::make_shared<NamedObsT>(
            "CNOT", std::vector<std::size_t>{0, 1}); // codecov test only

        auto x0 =
            std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{0});
        auto x1 =
            std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{1});

        auto obs = HamiltonianT::create(
            {PrecisionT{1.0}, PrecisionT{1.0}, PrecisionT{1.0}},
            {cnot, x0, x1});

        SECTION("Test TNCudaOperator ctor failures") {
            std::size_t num_qubits = 3;
            std::size_t maxExtent = 2;
            DevTag<int> dev_tag{0, 0};

            TensorNetT mps_state{num_qubits, maxExtent, dev_tag};

            const bool val_cal = true;

            REQUIRE_THROWS_AS(
                ObservableTNCudaOperator<TensorNetT>(mps_state, *obs, val_cal),
                LightningException);
        }
    }
}