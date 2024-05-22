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
#include "ObservablesTNCuda.hpp"

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

    SECTION("Test get obs name") {
        auto obs = NamedObsT("PauliX", {0});

        CHECK(obs.getObsName() == "PauliX[0]");
        CHECK(obs.getWires() == std::vector<std::size_t>{0});
    }
}

TEMPLATE_TEST_CASE("[Hermitian]", "[Observables]", float, double) {
    {
        using StateTensorT = MPSTNCuda<TestType>;
        using ComplexT = typename StateTensorT::ComplexT;
        using HermitianObsT = HermitianObs<StateTensorT>;

        std::vector<ComplexT> mat = {
            {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

        SECTION("Test get obs name") {
            auto obs = HermitianObsT(mat, std::vector<std::size_t>{0});
            CHECK(obs.getObsName() == "Hermitian");
            CHECK(obs.getWires() == std::vector<std::size_t>{0});
        }
    }
}

TEMPLATE_TEST_CASE("[TensorProd]", "[Observables]", float, double) {
    {
        using StateTensorT = MPSTNCuda<TestType>;

        SECTION("Test get obs name") {
            auto H0 = std::make_shared<NamedObs<StateTensorT>>(
                "Hadamard", std::vector<std::size_t>{0});
            auto H1 = std::make_shared<NamedObs<StateTensorT>>(
                "Hadamard", std::vector<std::size_t>{1});
            auto H2 = std::make_shared<NamedObs<StateTensorT>>(
                "Hadamard", std::vector<std::size_t>{2});

            auto obs = TensorProdObs<StateTensorT>::create({H0, H1});

            CHECK(obs->getObsName() == "Hadamard[0] @ Hadamard[1]");
            CHECK(obs->getWires() == std::vector<std::size_t>{0, 1});
            CHECK(obs->getNumTensors() == std::vector<size_t>{2});

            REQUIRE_THROWS_WITH(TensorProdObs<StateTensorT>::create({obs}),
                                Catch::Matchers::Contains(
                                    "A new TensorProdObs observable cannot be "
                                    "created from a single TensorProdObs."));

            REQUIRE_THROWS_WITH(
                TensorProdObs<StateTensorT>::create({obs, H2}),
                Catch::Matchers::Contains("A TensorProdObs observable cannot "
                                          "be created from a TensorProdObs"));

            auto ham_obs = Hamiltonian<StateTensorT>::create(
                {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}}, {H0, H1, H2});

            REQUIRE_THROWS_WITH(
                TensorProdObs<StateTensorT>::create({ham_obs, H2}),
                Catch::Matchers::Contains("A TensorProdObs observable cannot "
                                          "be created from a Hamiltonian"));
        }
    }
}

TEMPLATE_TEST_CASE("[Hamiltonian]", "[Observables]", float, double) {
    {
        using StateTensorT = MPSTNCuda<TestType>;

        SECTION("Test get obs name") {
            auto H0 = std::make_shared<NamedObs<StateTensorT>>(
                "Hadamard", std::vector<std::size_t>{0});
            auto H1 = std::make_shared<NamedObs<StateTensorT>>(
                "Hadamard", std::vector<std::size_t>{1});

            auto ham_obs = Hamiltonian<StateTensorT>::create(
                {{1.0, 0.0}, {1.0, 0.0}}, {H0, H1});

            CHECK(ham_obs->getWires() == std::vector<std::size_t>{0, 1});

            REQUIRE(
                ham_obs->getObsName() ==
                "Hamiltonian: { 'coeffs' : real part [1, 1], imag part[0, 0], "
                "'observables' : [Hadamard[0], Hadamard[1]]}");
        }

        SECTION("Throw error") {
            auto H0 = std::make_shared<NamedObs<StateTensorT>>(
                "Hadamard", std::vector<std::size_t>{0});
            auto H1 = std::make_shared<NamedObs<StateTensorT>>(
                "Hadamard", std::vector<std::size_t>{1});
            auto H2 = std::make_shared<NamedObs<StateTensorT>>(
                "Hadamard", std::vector<std::size_t>{2});

            auto ham_obs = Hamiltonian<StateTensorT>::create(
                {{1.0, 0.0}, {1.0, 0.0}}, {H0, H1});

            CHECK(ham_obs->getNumTensors() == std::vector<size_t>{1, 1});

            REQUIRE_THROWS_WITH(
                Hamiltonian<StateTensorT>::create({{1.0, 0.0}, {1.0, 0.0}},
                                                  {ham_obs, H2}),
                Catch::Matchers::Contains("A Hamiltonian observable cannot be "
                                          "created from a Hamiltonian."));
        }
    }
}
