// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <catch2/catch.hpp>

#include "MPIManagerKokkos.hpp"
#include "ObservablesKokkos.hpp"
#include "ObservablesKokkosMPI.hpp"
#include "StateVectorKokkosMPI.hpp"
#include "TestHelpers.hpp"
#include "TestHelpersStateVectors.hpp" // initializeLKTestSV

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos::Observables;
using Pennylane::Util::LightningException;
} // namespace
/// @endcond

TEMPLATE_PRODUCT_TEST_CASE("NamedObsMPI", "[Observables]",
                           (StateVectorKokkosMPI), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using NamedObsT = NamedObsMPI<StateVectorT>;

    SECTION("Non-Default constructibility") {
        REQUIRE(!std::is_constructible_v<NamedObsT>);
    }

    SECTION("Constructibility") {
        REQUIRE(std::is_constructible_v<NamedObsT, std::string,
                                        std::vector<std::size_t>>);
    }

    SECTION("Constructibility - optional parameters") {
        REQUIRE(std::is_constructible_v<NamedObsT, std::string,
                                        std::vector<std::size_t>,
                                        std::vector<PrecisionT>>);
    }

    SECTION("Copy constructibility") {
        REQUIRE(std::is_copy_constructible_v<NamedObsT>);
    }

    SECTION("Move constructibility") {
        REQUIRE(std::is_move_constructible_v<NamedObsT>);
    }

    SECTION("NamedObs only accepts correct arguments") {
        REQUIRE_THROWS_AS(NamedObsT("PauliX", {}), LightningException);
        REQUIRE_THROWS_AS(NamedObsT("PauliX", {0, 3}), LightningException);

        REQUIRE_THROWS_AS(NamedObsT("RX", {0}), LightningException);
        REQUIRE_THROWS_AS(NamedObsT("RX", {0, 1, 2, 3}), LightningException);
        REQUIRE_THROWS_AS(
            NamedObsT("RX", {0}, std::vector<PrecisionT>{0.3, 0.4}),
            LightningException);
        REQUIRE_NOTHROW(
            NamedObsT("Rot", {0}, std::vector<PrecisionT>{0.3, 0.4, 0.5}));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("HermitianObsMPI", "[Observables]",
                           (StateVectorKokkosMPI), (float, double)) {
    using StateVectorT = TestType;
    using ComplexT = typename StateVectorT::ComplexT;
    using MatrixT = std::vector<ComplexT>;
    using HermitianObsT = HermitianObsMPI<StateVectorT>;

    SECTION("Non-Default constructibility") {
        REQUIRE(!std::is_constructible_v<HermitianObsT>);
    }

    SECTION("Constructibility") {
        REQUIRE(std::is_constructible_v<HermitianObsT, MatrixT,
                                        std::vector<std::size_t>>);
    }

    SECTION("Copy constructibility") {
        REQUIRE(std::is_copy_constructible_v<HermitianObsT>);
    }

    SECTION("Move constructibility") {
        REQUIRE(std::is_move_constructible_v<HermitianObsT>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("TensorProdObs", "[Observables]",
                           (StateVectorKokkosMPI), (float, double)) {
    using StateVectorT = TestType;
    using TensorProdObsT = TensorProdObsMPI<StateVectorT>;
    using NamedObsT = NamedObsMPI<StateVectorT>;
    using HermitianObsT = HermitianObsMPI<StateVectorT>;

    SECTION("Constructibility - NamedObs") {
        REQUIRE(
            std::is_constructible_v<TensorProdObsT,
                                    std::vector<std::shared_ptr<NamedObsT>>>);
    }

    SECTION("Constructibility - HermitianObs") {
        REQUIRE(std::is_constructible_v<
                TensorProdObsT, std::vector<std::shared_ptr<HermitianObsT>>>);
    }

    SECTION("Copy constructibility") {
        REQUIRE(std::is_copy_constructible_v<TensorProdObsT>);
    }

    SECTION("Move constructibility") {
        REQUIRE(std::is_move_constructible_v<TensorProdObsT>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("HamiltonianMPI", "[Observables]",
                           (StateVectorKokkosMPI), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using TensorProdObsT = TensorProdObsMPI<StateVectorT>;
    using NamedObsT = NamedObsMPI<StateVectorT>;
    using HermitianObsT = HermitianObsMPI<StateVectorT>;
    using HamiltonianT = HamiltonianMPI<StateVectorT>;

    SECTION("Constructibility - NamedObs") {
        REQUIRE(
            std::is_constructible_v<HamiltonianT, std::vector<PrecisionT>,
                                    std::vector<std::shared_ptr<NamedObsT>>>);
    }

    SECTION("Constructibility - HermitianObs") {
        REQUIRE(std::is_constructible_v<
                HamiltonianT, std::vector<PrecisionT>,
                std::vector<std::shared_ptr<HermitianObsT>>>);
    }

    SECTION("Constructibility - TensorProdObsT") {
        REQUIRE(std::is_constructible_v<
                HamiltonianT, std::vector<PrecisionT>,
                std::vector<std::shared_ptr<TensorProdObsT>>>);
    }

    SECTION("Copy constructibility") {
        REQUIRE(std::is_copy_constructible_v<HamiltonianT>);
    }

    SECTION("Move constructibility") {
        REQUIRE(std::is_move_constructible_v<HamiltonianT>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("HamiltonianMPI::ApplyInPlace", "[Observables]",
                           (StateVectorKokkosMPI), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using TensorProdObsT = TensorProdObsMPI<StateVectorT>;
    using NamedObsT = NamedObsMPI<StateVectorT>;
    using HamiltonianT = HamiltonianMPI<StateVectorT>;

    using TensorProdObs = TensorProdObs<StateVectorKokkos<PrecisionT>>;
    using NamedObs = NamedObs<StateVectorKokkos<PrecisionT>>;
    using Hamiltonian = Hamiltonian<StateVectorKokkos<PrecisionT>>;

    std::size_t num_qubits = 8;

    const auto h = PrecisionT{0.809}; // half of the golden ratio

    auto zz = std::make_shared<TensorProdObsT>(
        std::make_shared<NamedObsT>("PauliZ", std::vector<std::size_t>{0}),
        std::make_shared<NamedObsT>("PauliZ", std::vector<std::size_t>{1}));

    auto x1 =
        std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{0});
    auto x2 =
        std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{1});

    auto zz0 = std::make_shared<TensorProdObs>(
        std::make_shared<NamedObs>("PauliZ", std::vector<std::size_t>{0}),
        std::make_shared<NamedObs>("PauliZ", std::vector<std::size_t>{1}));

    auto x10 =
        std::make_shared<NamedObs>("PauliX", std::vector<std::size_t>{0});
    auto x20 =
        std::make_shared<NamedObs>("PauliX", std::vector<std::size_t>{1});

    auto ham = HamiltonianT::create({PrecisionT{1.0}, h, h}, {zz, x1, x2});
    auto ham0 = Hamiltonian::create({PrecisionT{1.0}, h, h}, {zz0, x10, x20});

    SECTION("Hamiltonian applyInPlace") {
        const PrecisionT EP = 1e-5;
        MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
        REQUIRE(mpi_manager.getSize() == 4);

        auto [sv, sv_ref] = initializeLKTestSV<PrecisionT>(num_qubits);

        ham->applyInPlace(sv);
        ham0->applyInPlace(sv_ref);

        auto data = getFullDataVector(sv, 0);
        auto reference = sv_ref.getDataVector();

        if (sv.getMPIManager().getRank() == 0) {
            for (std::size_t j = 0; j < data.size(); j++) {
                CHECK_THAT(real(data[j]),
                           Catch::Matchers::WithinAbs(real(reference[j]), EP));
            }
        }
    }
}
