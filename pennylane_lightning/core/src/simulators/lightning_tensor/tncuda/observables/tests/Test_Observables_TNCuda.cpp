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

#include "TestHelpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningTensor::TNCuda::Observables;
using Pennylane::Util::LightningException;
} // namespace
/// @endcond

TEMPLATE_PRODUCT_TEST_CASE("NamedObs", "[Observables]", (MPSTNCuda),
                           (float, double)) {
    using TensorNetT = TestType;
    using PrecisionT = typename TensorNetT::PrecisionT;
    using ComplexT = typename TensorNetT::ComplexT;
    using NamedObsT = NamedObsTNCuda<TensorNetT>;

    SECTION("Test get obs name") {
        auto obs = NamedObsT("PauliX", {0});
        auto obs_metaData = obs.getMetaData()[0][0];
        CHECK(obs.getObsName() == "PauliX[0]");
        CHECK(obs.getWires() == std::vector<std::size_t>{0});
        CHECK(obs.getNumTensors() == std::vector<std::size_t>{1});
        CHECK(obs.getNumStateModes()[0] == std::vector<std::size_t>{1});
        CHECK(obs.getStateModes()[0][0] == std::vector<std::size_t>{0});
        CHECK(obs.getCoeffs() == std::vector<PrecisionT>{1.0});
        CHECK(std::get<0>(obs_metaData) == "PauliX");
        CHECK(std::get<1>(obs_metaData) == std::vector<PrecisionT>{});
        CHECK(std::get<2>(obs_metaData) == std::vector<ComplexT>{});
    }

    SECTION("Comparing objects names") {
        auto ob1 = NamedObsT("PauliX", {0});
        auto ob2 = NamedObsT("PauliX", {0});
        auto ob3 = NamedObsT("PauliZ", {0});

        REQUIRE(ob1 == ob2);
        REQUIRE(ob2 != ob3);
        REQUIRE(ob1 != ob3);
    }

    SECTION("Comparing objects wires") {
        auto ob1 = NamedObsT("PauliY", {0});
        auto ob2 = NamedObsT("PauliY", {0});
        auto ob3 = NamedObsT("PauliY", {1});

        REQUIRE(ob1 == ob2);
        REQUIRE(ob2 != ob3);
        REQUIRE(ob1 != ob3);
    }

    SECTION("Comparing objects parameters") {
        auto ob1 = NamedObsT("RZ", {0}, {0.4});
        auto ob2 = NamedObsT("RZ", {0}, {0.4});
        auto ob3 = NamedObsT("RZ", {0}, {0.1});

        REQUIRE(ob1 == ob2);
        REQUIRE(ob2 != ob3);
        REQUIRE(ob1 != ob3);
    }
}

TEMPLATE_TEST_CASE("[Hermitian]", "[Observables]", float, double) {
    using TensorNetT = MPSTNCuda<TestType>;
    using PrecisionT = typename TensorNetT::PrecisionT;
    using ComplexT = typename TensorNetT::ComplexT;
    using HermitianObsT = HermitianObsTNCuda<TensorNetT>;

    SECTION("HermitianObs only accepts correct arguments") {
        REQUIRE_THROWS_AS(
            HermitianObsT(std::vector<ComplexT>{0.0, 0.0, 0.0}, {0}),
            LightningException);
        REQUIRE_THROWS_AS(
            HermitianObsT(std::vector<ComplexT>{0.0, 0.0, 0.0, 0.0, 0.0},
                          {0, 1}),
            LightningException);
        REQUIRE_THROWS_WITH(
            HermitianObsT(std::vector<ComplexT>(16, ComplexT{0.0, 0.0}),
                          {0, 1}),
            Catch::Matchers::Contains("The number of Hermitian target wires "
                                      "must be 1 for Lightning-Tensor."));
    }

    SECTION("Test get obs name") {
        std::vector<ComplexT> mat = {
            {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
        auto obs = HermitianObsT(mat, std::vector<std::size_t>{0});
        auto obs_metaData = obs.getMetaData()[0][0];
        std::ostringstream res;
        res << "Hermitian" << MatrixHasher()(mat);
        CHECK(obs.getObsName() == res.str());
        CHECK(obs.getWires() == std::vector<std::size_t>{0});

        CHECK(obs.getNumTensors() == std::vector<std::size_t>{1});
        CHECK(obs.getNumStateModes()[0] == std::vector<std::size_t>{1});
        CHECK(obs.getStateModes()[0][0] == std::vector<std::size_t>{0});
        CHECK(obs.getCoeffs() == std::vector<PrecisionT>{1.0});
        CHECK(std::get<0>(obs_metaData) == "Hermitian");
        CHECK(std::get<1>(obs_metaData) == std::vector<PrecisionT>{});
        CHECK(std::get<2>(obs_metaData) == mat);
    }

    SECTION("Comparing objects matrices") {
        auto ob1 =
            HermitianObsT{std::vector<ComplexT>{1.0, 0.0, 0.0, 0.0}, {0}};
        auto ob2 =
            HermitianObsT{std::vector<ComplexT>{1.0, 0.0, 0.0, 0.0}, {0}};
        auto ob3 =
            HermitianObsT{std::vector<ComplexT>{0.0, 1.0, 0.0, 0.0}, {0}};
        REQUIRE(ob1 == ob2);
        REQUIRE(ob1 != ob3);
        REQUIRE(ob2 != ob3);
    }

    SECTION("Comparing objects wires") {
        auto ob1 =
            HermitianObsT{std::vector<ComplexT>{1.0, 0.0, -1.0, 0.0}, {0}};
        auto ob2 =
            HermitianObsT{std::vector<ComplexT>{1.0, 0.0, -1.0, 0.0}, {0}};
        auto ob3 =
            HermitianObsT{std::vector<ComplexT>{1.0, 0.0, -1.0, 0.0}, {1}};
        REQUIRE(ob1 == ob2);
        REQUIRE(ob1 != ob3);
        REQUIRE(ob2 != ob3);
    }
}

TEMPLATE_TEST_CASE("[TensorProd]", "[Observables]", float, double) {
    {
        using TensorNetT = MPSTNCuda<TestType>;
        using PrecisionT = typename TensorNetT::PrecisionT;
        using ComplexT = typename TensorNetT::ComplexT;
        using NamedObsT = NamedObsTNCuda<TensorNetT>;
        using TensorProdObsT = TensorProdObsTNCuda<TensorNetT>;
        using HermitianObsT = HermitianObsTNCuda<TensorNetT>;

        SECTION("Overlapping wires throw an exception") {
            auto ob1 = std::make_shared<HermitianObsT>(
                std::vector<ComplexT>(4, ComplexT{0.0, 0.0}),
                std::vector<std::size_t>{1});
            auto ob2_1 = std::make_shared<NamedObsT>(
                "PauliX", std::vector<std::size_t>{1});
            auto ob2_2 = std::make_shared<NamedObsT>(
                "PauliZ", std::vector<std::size_t>{2});
            auto ob2 = TensorProdObsT::create({ob2_1, ob2_2});

            REQUIRE_THROWS_AS(TensorProdObsT::create({ob1, ob2}),
                              LightningException);
        }

        SECTION("Constructing an observable with non-overlapping wires ") {
            auto ob1 = std::make_shared<HermitianObsT>(
                std::vector<ComplexT>(4, ComplexT{0.0, 0.0}),
                std::vector<std::size_t>{1});
            auto ob2_1 = std::make_shared<NamedObsT>(
                "PauliX", std::vector<std::size_t>{2});
            auto ob2_2 = std::make_shared<NamedObsT>(
                "PauliZ", std::vector<std::size_t>{3});
            auto ob2 = TensorProdObsT::create({ob2_1, ob2_2});

            REQUIRE_NOTHROW(TensorProdObsT::create({ob1, ob2}));
        }

        SECTION("Constructing an invalid TensorProd(TensorProd)") {
            auto ob2_1 = std::make_shared<NamedObsT>(
                "PauliX", std::vector<std::size_t>{2});
            auto ob2_2 = std::make_shared<NamedObsT>(
                "PauliZ", std::vector<std::size_t>{3});
            auto ob2 = TensorProdObsT::create({ob2_1, ob2_2});

            REQUIRE_THROWS_AS(TensorProdObsT::create({ob2}),
                              LightningException);
        }

        SECTION("getObsName") {
            auto ob =
                TensorProdObsT(std::make_shared<NamedObsT>(
                                   "PauliX", std::vector<std::size_t>{0}),
                               std::make_shared<NamedObsT>(
                                   "PauliZ", std::vector<std::size_t>{1}));
            REQUIRE(ob.getObsName() == "PauliX[0] @ PauliZ[1]");
            CHECK(ob.getNumTensors() == std::vector<std::size_t>{2});
            CHECK(ob.getNumStateModes()[0] == std::vector<std::size_t>{1, 1});
            CHECK(ob.getStateModes()[0][0] == std::vector<std::size_t>{0});
            CHECK(ob.getStateModes()[0][1] == std::vector<std::size_t>{1});
            CHECK(ob.getCoeffs() == std::vector<PrecisionT>{1.0});

            CHECK(std::get<0>(ob.getMetaData()[0][0]) == "PauliX");
            CHECK(std::get<0>(ob.getMetaData()[0][1]) == "PauliZ");
            CHECK(std::get<1>(ob.getMetaData()[0][0]) ==
                  std::vector<PrecisionT>{});
            CHECK(std::get<1>(ob.getMetaData()[0][1]) ==
                  std::vector<PrecisionT>{});
            CHECK(std::get<2>(ob.getMetaData()[0][0]) ==
                  std::vector<ComplexT>{});
            CHECK(std::get<2>(ob.getMetaData()[0][1]) ==
                  std::vector<ComplexT>{});
        }

        SECTION("Compare tensor product observables") {
            auto ob1 =
                TensorProdObsT{std::make_shared<NamedObsT>(
                                   "PauliX", std::vector<std::size_t>{0}),
                               std::make_shared<NamedObsT>(
                                   "PauliZ", std::vector<std::size_t>{1})};
            auto ob2 =
                TensorProdObsT{std::make_shared<NamedObsT>(
                                   "PauliX", std::vector<std::size_t>{0}),
                               std::make_shared<NamedObsT>(
                                   "PauliZ", std::vector<std::size_t>{1})};
            auto ob3 =
                TensorProdObsT{std::make_shared<NamedObsT>(
                                   "PauliX", std::vector<std::size_t>{0}),
                               std::make_shared<NamedObsT>(
                                   "PauliZ", std::vector<std::size_t>{2})};
            auto ob4 =
                TensorProdObsT{std::make_shared<NamedObsT>(
                                   "PauliZ", std::vector<std::size_t>{0}),
                               std::make_shared<NamedObsT>(
                                   "PauliZ", std::vector<std::size_t>{1})};

            auto ob5 = TensorProdObsT{std::make_shared<NamedObsT>(
                "PauliZ", std::vector<std::size_t>{0})};

            REQUIRE(ob1 == ob2);
            REQUIRE(ob1 != ob3);
            REQUIRE(ob1 != ob4);
            REQUIRE(ob1 != ob5);
        }
    }
}

TEMPLATE_TEST_CASE("[Hamiltonian]", "[Observables]", float, double) {
    {
        using TensorNetT = MPSTNCuda<TestType>;
        using PrecisionT = typename TensorNetT::PrecisionT;
        using ComplexT = typename TensorNetT::ComplexT;
        using NamedObsT = NamedObsTNCuda<TensorNetT>;
        using TensorProdObsT = TensorProdObsTNCuda<TensorNetT>;
        using HamiltonianT = HamiltonianTNCuda<TensorNetT>;

        const auto h = PrecisionT{0.809}; // half of the golden ratio

        auto zz = std::make_shared<TensorProdObsT>(
            std::make_shared<NamedObsT>("PauliZ", std::vector<std::size_t>{0}),
            std::make_shared<NamedObsT>("PauliZ", std::vector<std::size_t>{1}));

        auto x1 =
            std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{0});
        auto x2 =
            std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{1});

        SECTION("Hamiltonian constructor only accepts valid arguments") {
            REQUIRE_NOTHROW(
                HamiltonianT::create({PrecisionT{1.0}, h, h}, {zz, x1, x2}));

            REQUIRE_THROWS_AS(
                HamiltonianT::create({PrecisionT{1.0}, h}, {zz, x1, x2}),
                LightningException);

            SECTION("getObsName") {
                auto X0 = std::make_shared<NamedObsT>(
                    "PauliX", std::vector<std::size_t>{0});
                auto Z2 = std::make_shared<NamedObsT>(
                    "PauliZ", std::vector<std::size_t>{2});

                REQUIRE(
                    HamiltonianT::create({0.3, 0.5}, {X0, Z2})->getObsName() ==
                    "Hamiltonian: { 'coeffs' : [0.3, 0.5], "
                    "'observables' : [PauliX[0], PauliZ[2]]}");
            }

            SECTION("Compare Hamiltonians") {
                auto X0 = std::make_shared<NamedObsT>(
                    "PauliX", std::vector<std::size_t>{0});
                auto X1 = std::make_shared<NamedObsT>(
                    "PauliX", std::vector<std::size_t>{1});
                auto X2 = std::make_shared<NamedObsT>(
                    "PauliX", std::vector<std::size_t>{2});

                auto Y0 = std::make_shared<NamedObsT>(
                    "PauliY", std::vector<std::size_t>{0});
                auto Y1 = std::make_shared<NamedObsT>(
                    "PauliY", std::vector<std::size_t>{1});
                auto Y2 = std::make_shared<NamedObsT>(
                    "PauliY", std::vector<std::size_t>{2});

                auto Z0 = std::make_shared<NamedObsT>(
                    "PauliZ", std::vector<std::size_t>{0});
                auto Z1 = std::make_shared<NamedObsT>(
                    "PauliZ", std::vector<std::size_t>{1});
                auto Z2 = std::make_shared<NamedObsT>(
                    "PauliZ", std::vector<std::size_t>{2});

                auto ham1 = HamiltonianT::create(
                    {0.8, 0.5, 0.7},
                    {
                        std::make_shared<TensorProdObsT>(X0, Y1, Z2),
                        std::make_shared<TensorProdObsT>(Z0, X1, Y2),
                        std::make_shared<TensorProdObsT>(Y0, Z1, X2),
                    });

                auto ham2 = HamiltonianT::create(
                    {0.8, 0.5, 0.7},
                    {
                        std::make_shared<TensorProdObsT>(X0, Y1, Z2),
                        std::make_shared<TensorProdObsT>(Z0, X1, Y2),
                        std::make_shared<TensorProdObsT>(Y0, Z1, X2),
                    });

                auto ham3 = HamiltonianT::create(
                    {0.8, 0.5, 0.642},
                    {
                        std::make_shared<TensorProdObsT>(X0, Y1, Z2),
                        std::make_shared<TensorProdObsT>(Z0, X1, Y2),
                        std::make_shared<TensorProdObsT>(Y0, Z1, X2),
                    });

                auto ham4 = HamiltonianT::create(
                    {0.8, 0.5},
                    {
                        std::make_shared<TensorProdObsT>(X0, Y1, Z2),
                        std::make_shared<TensorProdObsT>(Z0, X1, Y2),
                    });

                auto ham5 = HamiltonianT::create(
                    {0.8, 0.5, 0.7},
                    {
                        std::make_shared<TensorProdObsT>(X0, Y1, Z2),
                        std::make_shared<TensorProdObsT>(Z0, X1, Y2),
                        std::make_shared<TensorProdObsT>(Y0, Z1, Y2),
                    });

                REQUIRE(*ham1 == *ham2);
                REQUIRE(*ham1 != *ham3);
                REQUIRE(*ham2 != *ham3);
                REQUIRE(*ham2 != *ham4);
                REQUIRE(*ham1 != *ham5);

                REQUIRE(ham5->getWires() == std::vector<std::size_t>{0, 1, 2});
                REQUIRE(ham5->getCoeffs() ==
                        std::vector<PrecisionT>{0.8, 0.5, 0.7});
                CHECK(ham5->getNumTensors() ==
                      std::vector<std::size_t>{3, 3, 3});
                CHECK(ham5->getNumStateModes()[0] ==
                      std::vector<std::size_t>{1, 1, 1});
                CHECK(ham5->getNumStateModes()[1] ==
                      std::vector<std::size_t>{1, 1, 1});
                CHECK(ham5->getNumStateModes()[2] ==
                      std::vector<std::size_t>{1, 1, 1});

                CHECK(ham5->getStateModes()[0][0] ==
                      std::vector<std::size_t>{0});
                CHECK(ham5->getStateModes()[0][1] ==
                      std::vector<std::size_t>{1});
                CHECK(ham5->getStateModes()[0][2] ==
                      std::vector<std::size_t>{2});

                CHECK(ham5->getStateModes()[1][0] ==
                      std::vector<std::size_t>{0});
                CHECK(ham5->getStateModes()[1][1] ==
                      std::vector<std::size_t>{1});
                CHECK(ham5->getStateModes()[1][2] ==
                      std::vector<std::size_t>{2});

                CHECK(ham5->getStateModes()[2][0] ==
                      std::vector<std::size_t>{0});
                CHECK(ham5->getStateModes()[2][1] ==
                      std::vector<std::size_t>{1});
                CHECK(ham5->getStateModes()[2][2] ==
                      std::vector<std::size_t>{2});

                CHECK(std::get<0>(ham5->getMetaData()[0][0]) == "PauliX");
                CHECK(std::get<0>(ham5->getMetaData()[1][0]) == "PauliZ");
                CHECK(std::get<0>(ham5->getMetaData()[2][0]) == "PauliY");

                CHECK(std::get<0>(ham5->getMetaData()[0][1]) == "PauliY");
                CHECK(std::get<0>(ham5->getMetaData()[1][1]) == "PauliX");
                CHECK(std::get<0>(ham5->getMetaData()[2][1]) == "PauliZ");

                CHECK(std::get<0>(ham5->getMetaData()[0][2]) == "PauliZ");
                CHECK(std::get<0>(ham5->getMetaData()[1][2]) == "PauliY");
                CHECK(std::get<0>(ham5->getMetaData()[2][2]) == "PauliY");
            }

            SECTION("getWires") {
                auto Z0 = std::make_shared<NamedObsT>(
                    "PauliZ", std::vector<std::size_t>{0});
                auto Z5 = std::make_shared<NamedObsT>(
                    "PauliZ", std::vector<std::size_t>{5});
                auto Z9 = std::make_shared<NamedObsT>(
                    "PauliZ", std::vector<std::size_t>{9});

                auto ham = HamiltonianT::create({0.8, 0.5, 0.7}, {Z0, Z5, Z9});

                REQUIRE(ham->getWires() == std::vector<std::size_t>{0, 5, 9});

                CHECK(ham->getCoeffs() ==
                      std::vector<PrecisionT>{0.8, 0.5, 0.7});

                CHECK(ham->getNumTensors() ==
                      std::vector<std::size_t>{1, 1, 1});
                CHECK(ham->getNumStateModes()[0] ==
                      std::vector<std::size_t>{1});
                CHECK(ham->getNumStateModes()[1] ==
                      std::vector<std::size_t>{1});
                CHECK(ham->getNumStateModes()[2] ==
                      std::vector<std::size_t>{1});
                CHECK(ham->getStateModes()[0][0] ==
                      std::vector<std::size_t>{0});
                CHECK(ham->getStateModes()[1][0] ==
                      std::vector<std::size_t>{5});
                CHECK(ham->getStateModes()[2][0] ==
                      std::vector<std::size_t>{9});

                CHECK(std::get<0>(ham->getMetaData()[0][0]) == "PauliZ");
                CHECK(std::get<0>(ham->getMetaData()[1][0]) == "PauliZ");
                CHECK(std::get<0>(ham->getMetaData()[2][0]) == "PauliZ");
                CHECK(std::get<1>(ham->getMetaData()[0][0]) ==
                      std::vector<PrecisionT>{});
                CHECK(std::get<1>(ham->getMetaData()[1][0]) ==
                      std::vector<PrecisionT>{});
                CHECK(std::get<1>(ham->getMetaData()[2][0]) ==
                      std::vector<PrecisionT>{});
                CHECK(std::get<2>(ham->getMetaData()[0][0]) ==
                      std::vector<ComplexT>{});
                CHECK(std::get<2>(ham->getMetaData()[1][0]) ==
                      std::vector<ComplexT>{});
                CHECK(std::get<2>(ham->getMetaData()[2][0]) ==
                      std::vector<ComplexT>{});
            }

            SECTION("Throw Errors") {
                auto X0 = std::make_shared<NamedObsT>(
                    "PauliX", std::vector<std::size_t>{0});
                auto Z1 = std::make_shared<NamedObsT>(
                    "PauliZ", std::vector<std::size_t>{1});

                auto ob0 = HamiltonianT::create({TestType(0.5)}, {X0});

                auto ob1 = HamiltonianT::create({TestType(0.5)}, {Z1});

                REQUIRE_THROWS_AS(
                    HamiltonianT::create({TestType(0.5), TestType(0.5)},
                                         {ob0, ob1}),
                    LightningException);
            }
        }
    }
}
