#include "Observables.hpp"
#include "TestHelpers.hpp"

#include <catch2/catch.hpp>

using namespace Pennylane;
using namespace Pennylane::Simulators;
using Pennylane::Util::LightningException;

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEMPLATE_TEST_CASE("NamedObs", "[Observables]", float, double) {
    using PrecisionT = TestType;
    SECTION("NamedObs only accepts correct arguments") {
        REQUIRE_THROWS_AS(NamedObs<TestType>("PauliX", {}), LightningException);
        REQUIRE_THROWS_AS(NamedObs<TestType>("PauliX", {0, 3}),
                          LightningException);

        REQUIRE_THROWS_AS(NamedObs<TestType>("RX", {0}), LightningException);
        REQUIRE_THROWS_AS(NamedObs<TestType>("RX", {0, 1, 2, 3}),
                          LightningException);
        REQUIRE_THROWS_AS(
            NamedObs<TestType>("RX", {0}, std::vector<PrecisionT>{0.3, 0.4}),
            LightningException);
        REQUIRE_NOTHROW(NamedObs<TestType>(
            "Rot", {0}, std::vector<PrecisionT>{0.3, 0.4, 0.5}));
    }

    SECTION("Named of the Observable must be correct") {
        REQUIRE(NamedObs<TestType>("PauliZ", {0}).getObsName() == "PauliZ[0]");
    }

    SECTION("Objects with different names") {
        auto ob1 = NamedObs<TestType>("PauliX", {0});
        auto ob2 = NamedObs<TestType>("PauliX", {0});
        auto ob3 = NamedObs<TestType>("PauliZ", {0});

        REQUIRE(ob1 == ob2);
        REQUIRE(ob2 != ob3);
        REQUIRE(ob1 != ob3);
    }

    SECTION("Objects with different wires") {
        auto ob1 = NamedObs<TestType>("PauliY", {0});
        auto ob2 = NamedObs<TestType>("PauliY", {0});
        auto ob3 = NamedObs<TestType>("PauliY", {1});

        REQUIRE(ob1 == ob2);
        REQUIRE(ob2 != ob3);
        REQUIRE(ob1 != ob3);
    }

    SECTION("Objects with different parameters") {
        auto ob1 = NamedObs<TestType>("RZ", {0}, {0.4});
        auto ob2 = NamedObs<TestType>("RZ", {0}, {0.4});
        auto ob3 = NamedObs<TestType>("RZ", {0}, {0.1});

        REQUIRE(ob1 == ob2);
        REQUIRE(ob2 != ob3);
        REQUIRE(ob1 != ob3);
    }
}

TEMPLATE_TEST_CASE("HermitianObs", "[Observables]", float, double) {
    using PrecisionT = TestType;
    using ComplexPrecisionT = std::complex<TestType>;
    SECTION("HermitianObs only accepts correct arguments") {
        auto ob1 = HermitianObs<TestType>{
            std::vector<ComplexPrecisionT>{0.0, 0.0, 0.0, 0.0}, {0}};
        auto ob2 = HermitianObs<TestType>{
            std::vector<ComplexPrecisionT>(16, ComplexPrecisionT{}), {0, 1}};
        REQUIRE_THROWS_AS(
            HermitianObs<TestType>(
                std::vector<ComplexPrecisionT>{0.0, 0.0, 0.0}, {0}),
            LightningException);
        REQUIRE_THROWS_AS(
            HermitianObs<TestType>(
                std::vector<ComplexPrecisionT>{0.0, 0.0, 0.0, 0.0, 0.0},
                {0, 1}),
            LightningException);
    }
    SECTION("getObsName") {
        REQUIRE(HermitianObs<TestType>(
                    std::vector<ComplexPrecisionT>{1.0, 0.0, 2.0, 0.0}, {0})
                    .getObsName() == "Hermitian");
    }
    SECTION("Objects with different matrices") {
        auto ob1 = HermitianObs<PrecisionT>{
            std::vector<ComplexPrecisionT>{1.0, 0.0, 0.0, 0.0}, {0}};
        auto ob2 = HermitianObs<PrecisionT>{
            std::vector<ComplexPrecisionT>{1.0, 0.0, 0.0, 0.0}, {0}};
        auto ob3 = HermitianObs<PrecisionT>{
            std::vector<ComplexPrecisionT>{0.0, 1.0, 0.0, 0.0}, {0}};
        REQUIRE(ob1 == ob2);
        REQUIRE(ob1 != ob3);
        REQUIRE(ob2 != ob3);
    }
    SECTION("Objects with different wires") {
        auto ob1 = HermitianObs<PrecisionT>{
            std::vector<ComplexPrecisionT>{1.0, 0.0, -1.0, 0.0}, {0}};
        auto ob2 = HermitianObs<PrecisionT>{
            std::vector<ComplexPrecisionT>{1.0, 0.0, -1.0, 0.0}, {0}};
        auto ob3 = HermitianObs<PrecisionT>{
            std::vector<ComplexPrecisionT>{1.0, 0.0, -1.0, 0.0}, {1}};
        REQUIRE(ob1 == ob2);
        REQUIRE(ob1 != ob3);
        REQUIRE(ob2 != ob3);
    }
}

TEMPLATE_TEST_CASE("TensorProdObs", "[Observables]", float, double) {
    using PrecisionT = TestType;
    using ComplexPrecisionT = std::complex<TestType>;

    SECTION("Overlapping wires throw an exception") {
        auto ob1 = std::make_shared<HermitianObs<PrecisionT>>(
            std::vector<ComplexPrecisionT>(16, ComplexPrecisionT{0.0, 0.0}),
            std::vector<size_t>{0, 1});
        auto ob2_1 = std::make_shared<NamedObs<PrecisionT>>(
            "PauliX", std::vector<size_t>{1});
        auto ob2_2 = std::make_shared<NamedObs<PrecisionT>>(
            "PauliZ", std::vector<size_t>{2});
        auto ob2 = TensorProdObs<PrecisionT>::create({ob2_1, ob2_2});

        REQUIRE_THROWS_AS(TensorProdObs<PrecisionT>::create({ob1, ob2}),
                          LightningException);
    }

    SECTION("Can construct an observable with non-overlapping wires") {
        auto ob1 = std::make_shared<HermitianObs<PrecisionT>>(
            std::vector<ComplexPrecisionT>(16, ComplexPrecisionT{0.0, 0.0}),
            std::vector<size_t>{0, 1});
        auto ob2_1 = std::make_shared<NamedObs<PrecisionT>>(
            "PauliX", std::vector<size_t>{2});
        auto ob2_2 = std::make_shared<NamedObs<PrecisionT>>(
            "PauliZ", std::vector<size_t>{3});
        auto ob2 = TensorProdObs<PrecisionT>::create({ob2_1, ob2_2});

        REQUIRE_NOTHROW(TensorProdObs<PrecisionT>::create({ob1, ob2}));
    }

    SECTION("getObsName") {
        auto ob =
            TensorProdObs<PrecisionT>(std::make_shared<NamedObs<PrecisionT>>(
                                          "PauliX", std::vector<size_t>{0}),
                                      std::make_shared<NamedObs<PrecisionT>>(
                                          "PauliZ", std::vector<size_t>{1}));
        REQUIRE(ob.getObsName() == "PauliX[0] @ PauliZ[1]");
    }

    SECTION("Compare two tensor product observables") {
        auto ob1 =
            TensorProdObs<PrecisionT>{std::make_shared<NamedObs<PrecisionT>>(
                                          "PauliX", std::vector<size_t>{0}),
                                      std::make_shared<NamedObs<PrecisionT>>(
                                          "PauliZ", std::vector<size_t>{1})};
        auto ob2 =
            TensorProdObs<PrecisionT>{std::make_shared<NamedObs<PrecisionT>>(
                                          "PauliX", std::vector<size_t>{0}),
                                      std::make_shared<NamedObs<PrecisionT>>(
                                          "PauliZ", std::vector<size_t>{1})};
        auto ob3 =
            TensorProdObs<PrecisionT>{std::make_shared<NamedObs<PrecisionT>>(
                                          "PauliX", std::vector<size_t>{0}),
                                      std::make_shared<NamedObs<PrecisionT>>(
                                          "PauliZ", std::vector<size_t>{2})};
        auto ob4 =
            TensorProdObs<PrecisionT>{std::make_shared<NamedObs<PrecisionT>>(
                                          "PauliZ", std::vector<size_t>{0}),
                                      std::make_shared<NamedObs<PrecisionT>>(
                                          "PauliZ", std::vector<size_t>{1})};

        auto ob5 =
            TensorProdObs<PrecisionT>{std::make_shared<NamedObs<PrecisionT>>(
                "PauliZ", std::vector<size_t>{0})};

        REQUIRE(ob1 == ob2);
        REQUIRE(ob1 != ob3);
        REQUIRE(ob1 != ob4);
        REQUIRE(ob1 != ob5);
    }

    SECTION("Tensor product applies to a statevector correctly") {
        auto ob = TensorProdObs<PrecisionT>{
            std::make_shared<NamedObs<PrecisionT>>("PauliX",
                                                   std::vector<size_t>{0}),
            std::make_shared<NamedObs<PrecisionT>>("PauliX",
                                                   std::vector<size_t>{2}),
        };

        SECTION("Test using |1+0>") {
            constexpr auto num_qubits = size_t{3};
            StateVectorManagedCPU<PrecisionT> sv(num_qubits);

            sv.updateData(createProductState<PrecisionT>("1+0"));
            ob.applyInPlace(sv);

            REQUIRE(sv.getDataVector() ==
                    PLApprox(createProductState<PrecisionT>("0+1")));
        }

        SECTION("Test using |+-01>") {
            constexpr auto num_qubits = size_t{4};
            StateVectorManagedCPU<PrecisionT> sv(num_qubits);

            sv.updateData(createProductState<PrecisionT>("+-01"));
            ob.applyInPlace(sv);

            REQUIRE(sv.getDataVector() ==
                    PLApprox(createProductState<PrecisionT>("+-11")));
        }
    }
}

TEMPLATE_TEST_CASE("Hamiltonian", "[Observables]", float, double) {
    using PrecisionT = TestType;
    using ComplexPrecisionT = std::complex<TestType>;

    const auto h = PrecisionT{0.809}; // half of the golden ratio

    auto zz = std::make_shared<TensorProdObs<PrecisionT>>(
        std::make_shared<NamedObs<PrecisionT>>("PauliZ",
                                               std::vector<size_t>{0}),
        std::make_shared<NamedObs<PrecisionT>>("PauliZ",
                                               std::vector<size_t>{1}));

    auto x1 = std::make_shared<NamedObs<PrecisionT>>("PauliX",
                                                     std::vector<size_t>{0});
    auto x2 = std::make_shared<NamedObs<PrecisionT>>("PauliX",
                                                     std::vector<size_t>{1});

    SECTION("Hamiltonian constructor only accepts valid arguments") {
        REQUIRE_NOTHROW(Hamiltonian<PrecisionT>::create({PrecisionT{1.0}, h, h},
                                                        {zz, x1, x2}));

        REQUIRE_THROWS_AS(
            Hamiltonian<PrecisionT>::create({PrecisionT{1.0}, h}, {zz, x1, x2}),
            LightningException);
    }

    SECTION("getObsName") {
        auto X0 = std::make_shared<NamedObs<PrecisionT>>(
            "PauliX", std::vector<size_t>{0});
        auto Z2 = std::make_shared<NamedObs<PrecisionT>>(
            "PauliZ", std::vector<size_t>{2});

        REQUIRE(Hamiltonian<PrecisionT>::create({0.3, 0.5}, {X0, Z2})
                    ->getObsName() ==
                "Hamiltonian: { 'coeffs' : [0.3, 0.5], "
                "'observables' : [PauliX[0], PauliZ[2]]}");
    }

    SECTION("Compare Hamiltonians") {
        auto X0 = std::make_shared<NamedObs<PrecisionT>>(
            "PauliX", std::vector<size_t>{0});
        auto X1 = std::make_shared<NamedObs<PrecisionT>>(
            "PauliX", std::vector<size_t>{1});
        auto X2 = std::make_shared<NamedObs<PrecisionT>>(
            "PauliX", std::vector<size_t>{2});

        auto Y0 = std::make_shared<NamedObs<PrecisionT>>(
            "PauliY", std::vector<size_t>{0});
        auto Y1 = std::make_shared<NamedObs<PrecisionT>>(
            "PauliY", std::vector<size_t>{1});
        auto Y2 = std::make_shared<NamedObs<PrecisionT>>(
            "PauliY", std::vector<size_t>{2});

        auto Z0 = std::make_shared<NamedObs<PrecisionT>>(
            "PauliZ", std::vector<size_t>{0});
        auto Z1 = std::make_shared<NamedObs<PrecisionT>>(
            "PauliZ", std::vector<size_t>{1});
        auto Z2 = std::make_shared<NamedObs<PrecisionT>>(
            "PauliZ", std::vector<size_t>{2});

        auto ham1 = Hamiltonian<PrecisionT>::create(
            {0.8, 0.5, 0.7},
            {
                std::make_shared<TensorProdObs<PrecisionT>>(X0, Y1, Z2),
                std::make_shared<TensorProdObs<PrecisionT>>(Z0, X1, Y2),
                std::make_shared<TensorProdObs<PrecisionT>>(Y0, Z1, X2),
            });

        auto ham2 = Hamiltonian<PrecisionT>::create(
            {0.8, 0.5, 0.7},
            {
                std::make_shared<TensorProdObs<PrecisionT>>(X0, Y1, Z2),
                std::make_shared<TensorProdObs<PrecisionT>>(Z0, X1, Y2),
                std::make_shared<TensorProdObs<PrecisionT>>(Y0, Z1, X2),
            });

        auto ham3 = Hamiltonian<PrecisionT>::create(
            {0.8, 0.5, 0.642},
            {
                std::make_shared<TensorProdObs<PrecisionT>>(X0, Y1, Z2),
                std::make_shared<TensorProdObs<PrecisionT>>(Z0, X1, Y2),
                std::make_shared<TensorProdObs<PrecisionT>>(Y0, Z1, X2),
            });

        auto ham4 = Hamiltonian<PrecisionT>::create(
            {0.8, 0.5},
            {
                std::make_shared<TensorProdObs<PrecisionT>>(X0, Y1, Z2),
                std::make_shared<TensorProdObs<PrecisionT>>(Z0, X1, Y2),
            });

        auto ham5 = Hamiltonian<PrecisionT>::create(
            {0.8, 0.5, 0.7},
            {
                std::make_shared<TensorProdObs<PrecisionT>>(X0, Y1, Z2),
                std::make_shared<TensorProdObs<PrecisionT>>(Z0, X1, Y2),
                std::make_shared<TensorProdObs<PrecisionT>>(Y0, Z1, Y2),
            });

        REQUIRE(*ham1 == *ham2);
        REQUIRE(*ham1 != *ham3);
        REQUIRE(*ham2 != *ham3);
        REQUIRE(*ham2 != *ham4);
        REQUIRE(*ham1 != *ham5);
    }

    SECTION("Hamiltonian::applyInPlace") {
        auto ham = Hamiltonian<PrecisionT>::create({PrecisionT{1.0}, h, h},
                                                   {zz, x1, x2});

        SECTION(" to |+->") {
            constexpr auto num_qubits = size_t{2};
            StateVectorManagedCPU<PrecisionT> sv(num_qubits);

            sv.updateData(createProductState<PrecisionT>("+-"));

            ham->applyInPlace(sv);

            auto expected = std::vector<ComplexPrecisionT>{
                ComplexPrecisionT{0.5, 0.0},
                ComplexPrecisionT{0.5, 0.0},
                ComplexPrecisionT{-0.5, 0.0},
                ComplexPrecisionT{-0.5, 0.0},
            };

            REQUIRE(sv.getDataVector() == PLApprox(expected));
        }

        SECTION("Hamiltonian applies correctly to |01>") {
            constexpr auto num_qubits = size_t{2};
            StateVectorManagedCPU<PrecisionT> sv(num_qubits);

            sv.updateData(createProductState<PrecisionT>("01"));

            ham->applyInPlace(sv);

            auto expected = std::vector<ComplexPrecisionT>{
                ComplexPrecisionT{h, 0.0},
                ComplexPrecisionT{-1.0, 0.0},
                ComplexPrecisionT{0.0, 0.0},
                ComplexPrecisionT{h, 0.0},
            };

            REQUIRE(sv.getDataVector() == PLApprox(expected));
        }
    }

    SECTION("getWires") {
        auto Z0 = std::make_shared<NamedObs<PrecisionT>>(
            "PauliZ", std::vector<size_t>{0});
        auto Z5 = std::make_shared<NamedObs<PrecisionT>>(
            "PauliZ", std::vector<size_t>{5});
        auto Z9 = std::make_shared<NamedObs<PrecisionT>>(
            "PauliZ", std::vector<size_t>{9});

        auto ham1 =
            Hamiltonian<PrecisionT>::create({0.8, 0.5, 0.7}, {Z0, Z5, Z9});

        REQUIRE(ham1->getWires() == std::vector<size_t>{0, 5, 9});
    }
}
