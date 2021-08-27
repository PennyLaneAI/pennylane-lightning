#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "Gates.hpp"
#include "StateVector.hpp"
#include "Util.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane;

/**
 * @brief Tests the constructability of the StateVector class.
 *
 */
TEMPLATE_TEST_CASE("StateVectorManaged::StateVectorManaged",
                   "[StateVectorManaged_Nonparam]", float, double) {
    SECTION("StateVectorManaged") {
        REQUIRE(std::is_constructible<StateVectorManaged<>>::value);
    }
    SECTION("StateVectorManaged<TestType> {}") {
        REQUIRE(std::is_constructible<StateVectorManaged<TestType>>::value);
    }
    SECTION("StateVectorManaged<TestType> {const "
            "std::vector<std::complex<TestType>>&}") {
        REQUIRE(std::is_constructible<
                StateVectorManaged<TestType>,
                const std::vector<std::complex<TestType>> &>::value);
    }
    SECTION("StateVectorManaged<TestType> {std::complex<TestType>*, size_t}") {
        REQUIRE(std::is_constructible<StateVectorManaged<TestType>,
                                      std::complex<TestType> *, size_t>::value);
    }
    SECTION(
        "StateVectorManaged<TestType> {const StateVectorManaged<TestType>&}") {
        REQUIRE(
            std::is_constructible<StateVectorManaged<TestType>,
                                  const StateVectorManaged<TestType> &>::value);
    }

    SECTION("StateVectorManaged<TestType> {const StateVector<TestType>&}") {
        REQUIRE(std::is_constructible<StateVectorManaged<TestType>,
                                      const StateVector<TestType> &>::value);
    }
}

namespace {} // namespace

TEMPLATE_TEST_CASE("StateVectorManaged::applyHadamard",
                   "[StateVectorManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat(num_qubits);
            INFO("GOT HERE 1");
            auto int_idx = svdat.getInternalIndices({index});
            auto ext_idx = svdat.getExternalIndices({index});
            INFO("GOT HERE 2");

            CHECK(svdat.getDataVector()[0] == cp_t{1, 0});
            svdat.applyHadamard(int_idx, ext_idx, false);

            cp_t expected(1 / std::sqrt(2), 0);
            CHECK(expected.real() == Approx(svdat.getDataVector()[0].real()));
            CHECK(expected.imag() == Approx(svdat.getDataVector()[0].imag()));

            CHECK(expected.real() ==
                  Approx(svdat
                             .getDataVector()[0b1 << (svdat.getNumQubits() -
                                                      index - 1)]
                             .real()));
            CHECK(expected.imag() ==
                  Approx(svdat
                             .getDataVector()[0b1 << (svdat.getNumQubits() -
                                                      index - 1)]
                             .imag()));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat{num_qubits};
            CHECK(svdat.getDataVector()[0] == cp_t{1, 0});
            svdat.applyOperation("Hadamard", {index}, false);

            cp_t expected(1.0 / std::sqrt(2), 0);

            CHECK(expected.real() == Approx(svdat.getDataVector()[0].real()));
            CHECK(expected.imag() == Approx(svdat.getDataVector()[0].imag()));

            CHECK(expected.real() ==
                  Approx(svdat
                             .getDataVector()[0b1 << (svdat.getNumQubits() -
                                                      index - 1)]
                             .real()));
            CHECK(expected.imag() ==
                  Approx(svdat
                             .getDataVector()[0b1 << (svdat.getNumQubits() -
                                                      index - 1)]
                             .imag()));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyPauliX",
                   "[StateVectorManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat{num_qubits};
            auto int_idx = svdat.getInternalIndices({index});
            auto ext_idx = svdat.getExternalIndices({index});
            CHECK(svdat.getDataVector()[0] == Util::ONE<TestType>());
            svdat.applyPauliX(int_idx, ext_idx, false);
            CHECK(svdat.getDataVector()[0] == Util::ZERO<TestType>());
            CHECK(svdat.getDataVector()[0b1 << (svdat.getNumQubits() - index -
                                                1)] == Util::ONE<TestType>());
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat{num_qubits};
            CHECK(svdat.getDataVector()[0] == Util::ONE<TestType>());
            svdat.applyOperation("PauliX", {index}, false);
            CHECK(svdat.getDataVector()[0] == Util::ZERO<TestType>());
            CHECK(svdat.getDataVector()[0b1 << (svdat.getNumQubits() - index -
                                                1)] == Util::ONE<TestType>());
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyPauliY",
                   "[StateVectorManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorManaged<TestType> svdat{num_qubits};
    // Test using |+++> state
    svdat.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                          {{0}, {1}, {2}}, {{false}, {false}, {false}});

    constexpr cp_t p = Util::ConstMult(
        static_cast<TestType>(0.5),
        Util::ConstMult(Util::INVSQRT2<TestType>(), Util::IMAG<TestType>()));
    constexpr cp_t m = Util::ConstMult(-1, p);

    const std::vector<std::vector<cp_t>> expected_results = {
        {m, m, m, m, p, p, p, p},
        {m, m, p, p, m, m, p, p},
        {m, p, m, p, m, p, m, p}};

    const auto init_state = svdat.getDataVector();
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat_direct(init_state);
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});

            CHECK(svdat_direct.getDataVector() == init_state);
            svdat_direct.applyPauliY(int_idx, ext_idx, false);

            CHECK(isApproxEqual(svdat_direct.getDataVector(),
                                expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat_dispatch{init_state};
            CHECK(svdat_dispatch.getDataVector() == init_state);
            svdat_dispatch.applyOperation("PauliY", {index}, false);
            CHECK(isApproxEqual(svdat_dispatch.getDataVector(),
                                expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyPauliZ",
                   "[StateVectorManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorManaged<TestType> svdat{num_qubits};
    // Test using |+++> state
    svdat.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                          {{0}, {1}, {2}}, {{false}, {false}, {false}});

    constexpr cp_t p(static_cast<TestType>(0.5) * Util::INVSQRT2<TestType>());
    constexpr cp_t m(Util::ConstMult(-1, p));

    const std::vector<std::vector<cp_t>> expected_results = {
        {p, p, p, p, m, m, m, m},
        {p, p, m, m, p, p, m, m},
        {p, m, p, m, p, m, p, m}};

    const auto init_state = svdat.getDataVector();
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat_direct{init_state};
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});

            CHECK(svdat_direct.getDataVector() == init_state);
            svdat_direct.applyPauliZ(int_idx, ext_idx, false);

            CHECK(isApproxEqual(svdat_direct.getDataVector(),
                                expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat_dispatch{init_state};
            CHECK(svdat_dispatch.getDataVector() == init_state);
            svdat_dispatch.applyOperation("PauliZ", {index}, false);
            CHECK(isApproxEqual(svdat_dispatch.getDataVector(),
                                expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyS",
                   "[StateVectorManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorManaged<TestType> svdat{num_qubits};
    // Test using |+++> state
    svdat.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                          {{0}, {1}, {2}}, {{false}, {false}, {false}});

    constexpr cp_t r(static_cast<TestType>(0.5) * Util::INVSQRT2<TestType>());
    constexpr cp_t i(Util::ConstMult(r, Util::IMAG<TestType>()));

    const std::vector<std::vector<cp_t>> expected_results = {
        {r, r, r, r, i, i, i, i},
        {r, r, i, i, r, r, i, i},
        {r, i, r, i, r, i, r, i}};

    const auto init_state = svdat.getDataVector();
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat_direct{init_state};
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});

            CHECK(svdat_direct.getDataVector() == init_state);
            svdat_direct.applyS(int_idx, ext_idx, false);

            CHECK(isApproxEqual(svdat_direct.getDataVector(),
                                expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat_dispatch{init_state};
            CHECK(svdat_dispatch.getDataVector() == init_state);
            svdat_dispatch.applyOperation("S", {index}, false);
            CHECK(isApproxEqual(svdat_dispatch.getDataVector(),
                                expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyT",
                   "[StateVectorManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorManaged<TestType> svdat{num_qubits};
    // Test using |+++> state
    svdat.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                          {{0}, {1}, {2}}, {{false}, {false}, {false}});

    cp_t r(1.0 / (2.0 * std::sqrt(2)), 0);
    cp_t i(1.0 / 4, 1.0 / 4);

    const std::vector<std::vector<cp_t>> expected_results = {
        {r, r, r, r, i, i, i, i},
        {r, r, i, i, r, r, i, i},
        {r, i, r, i, r, i, r, i}};

    const auto init_state = svdat.getDataVector();
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat_direct{init_state};
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});

            CHECK(svdat_direct.getDataVector() == init_state);
            svdat_direct.applyT(int_idx, ext_idx, false);

            CHECK(isApproxEqual(svdat_direct.getDataVector(),
                                expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat_dispatch{init_state};
            CHECK(svdat_dispatch.getDataVector() == init_state);
            svdat_dispatch.applyOperation("T", {index}, false);
            CHECK(isApproxEqual(svdat_dispatch.getDataVector(),
                                expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyCNOT",
                   "[StateVectorManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorManaged<TestType> svdat{num_qubits};

    // Test using |+00> state to generate 3-qubit GHZ state
    svdat.applyOperation("Hadamard", {0});
    const auto init_state = svdat.getDataVector();

    SECTION("Apply directly") {
        StateVectorManaged<TestType> svdat_direct{init_state};

        for (size_t index = 1; index < num_qubits; index++) {
            auto int_idx = svdat_direct.getInternalIndices({index - 1, index});
            auto ext_idx = svdat_direct.getExternalIndices({index - 1, index});

            svdat_direct.applyCNOT(int_idx, ext_idx, false);
        }
        CHECK(svdat_direct.getDataVector().front() ==
              Util::INVSQRT2<TestType>());
        CHECK(svdat_direct.getDataVector().back() ==
              Util::INVSQRT2<TestType>());
    }

    SECTION("Apply using dispatcher") {
        StateVectorManaged<TestType> svdat_dispatch{init_state};

        for (size_t index = 1; index < num_qubits; index++) {
            svdat_dispatch.applyOperation("CNOT", {index - 1, index}, false);
        }
        CHECK(svdat_dispatch.getDataVector().front() ==
              Util::INVSQRT2<TestType>());
        CHECK(svdat_dispatch.getDataVector().back() ==
              Util::INVSQRT2<TestType>());
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applySWAP",
                   "[StateVectorManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorManaged<TestType> svdat{num_qubits};

    // Test using |+10> state
    svdat.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                          {false, false});
    const auto init_state = svdat.getDataVector();

    SECTION("Apply directly") {
        CHECK(svdat.getDataVector() ==
              std::vector<cp_t>{
                  Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                  Util::INVSQRT2<TestType>(), Util::ZERO<TestType>(),
                  Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                  Util::INVSQRT2<TestType>(), Util::ZERO<TestType>()});

        SECTION("SWAP0,1 |+10> -> |1+0>") {
            std::vector<cp_t> expected{Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       Util::ZERO<TestType>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       Util::ZERO<TestType>()};

            StateVectorManaged<TestType> svdat01{init_state};
            StateVectorManaged<TestType> svdat10{init_state};

            svdat01.applySWAP(svdat.getInternalIndices({0, 1}),
                              svdat.getExternalIndices({0, 1}), false);
            svdat10.applySWAP(svdat.getInternalIndices({1, 0}),
                              svdat.getExternalIndices({1, 0}), false);

            CHECK(svdat01.getDataVector() == expected);
            CHECK(svdat10.getDataVector() == expected);
        }

        SECTION("SWAP0,2 |+10> -> |01+>") {
            std::vector<cp_t> expected{Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>()};

            StateVectorManaged<TestType> svdat02{init_state};
            StateVectorManaged<TestType> svdat20{init_state};

            svdat02.applySWAP(svdat.getInternalIndices({0, 2}),
                              svdat.getExternalIndices({0, 2}), false);
            svdat20.applySWAP(svdat.getInternalIndices({2, 0}),
                              svdat.getExternalIndices({2, 0}), false);
            CHECK(svdat02.getDataVector() == expected);
            CHECK(svdat20.getDataVector() == expected);
        }
        SECTION("SWAP1,2 |+10> -> |+01>") {
            std::vector<cp_t> expected{Util::ZERO<TestType>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>()};

            StateVectorManaged<TestType> svdat12{init_state};
            StateVectorManaged<TestType> svdat21{init_state};

            svdat12.applySWAP(svdat.getInternalIndices({1, 2}),
                              svdat.getExternalIndices({1, 2}), false);
            svdat21.applySWAP(svdat.getInternalIndices({2, 1}),
                              svdat.getExternalIndices({2, 1}), false);
            CHECK(svdat12.getDataVector() == expected);
            CHECK(svdat21.getDataVector() == expected);
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("SWAP0,1 |+10> -> |1+0>") {
            std::vector<cp_t> expected{Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       Util::ZERO<TestType>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       Util::ZERO<TestType>()};

            StateVectorManaged<TestType> svdat01{init_state};
            StateVectorManaged<TestType> svdat10{init_state};

            svdat01.applyOperation("SWAP", {0, 1});
            svdat10.applyOperation("SWAP", {1, 0});

            CHECK(svdat01.getDataVector() == expected);
            CHECK(svdat10.getDataVector() == expected);
        }

        SECTION("SWAP0,2 |+10> -> |01+>") {
            std::vector<cp_t> expected{Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>()};

            StateVectorManaged<TestType> svdat02{init_state};
            StateVectorManaged<TestType> svdat20{init_state};

            svdat02.applyOperation("SWAP", {0, 2});
            svdat20.applyOperation("SWAP", {2, 0});

            CHECK(svdat02.getDataVector() == expected);
            CHECK(svdat20.getDataVector() == expected);
        }
        SECTION("SWAP1,2 |+10> -> |+01>") {
            std::vector<cp_t> expected{Util::ZERO<TestType>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>()};

            StateVectorManaged<TestType> svdat12{init_state};
            StateVectorManaged<TestType> svdat21{init_state};

            svdat12.applyOperation("SWAP", {1, 2});
            svdat21.applyOperation("SWAP", {2, 1});

            CHECK(svdat12.getDataVector() == expected);
            CHECK(svdat21.getDataVector() == expected);
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyCZ",
                   "[StateVectorManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorManaged<TestType> svdat{num_qubits};

    // Test using |+10> state
    svdat.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                          {false, false});
    const auto init_state = svdat.getDataVector();

    SECTION("Apply directly") {
        CHECK(svdat.getDataVector() ==
              std::vector<cp_t>{Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                                std::complex<TestType>(1.0 / sqrt(2), 0),
                                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                                Util::ZERO<TestType>(),
                                std::complex<TestType>(1.0 / sqrt(2), 0),
                                Util::ZERO<TestType>()});

        SECTION("CZ0,1 |+10> -> |-10>") {
            std::vector<cp_t> expected{Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       std::complex<TestType>(-1 / sqrt(2), 0),
                                       Util::ZERO<TestType>()};

            StateVectorManaged<TestType> svdat01{init_state};
            StateVectorManaged<TestType> svdat10{init_state};

            svdat01.applyCZ(svdat.getInternalIndices({0, 1}),
                            svdat.getExternalIndices({0, 1}), false);
            svdat10.applyCZ(svdat.getInternalIndices({1, 0}),
                            svdat.getExternalIndices({1, 0}), false);

            CHECK(svdat01.getDataVector() == expected);
            CHECK(svdat10.getDataVector() == expected);
        }

        SECTION("CZ0,2 |+10> -> |+10>") {
            std::vector<cp_t> expected{init_state};

            StateVectorManaged<TestType> svdat02{init_state};
            StateVectorManaged<TestType> svdat20{init_state};

            svdat02.applyCZ(svdat.getInternalIndices({0, 2}),
                            svdat.getExternalIndices({0, 2}), false);
            svdat20.applyCZ(svdat.getInternalIndices({2, 0}),
                            svdat.getExternalIndices({2, 0}), false);
            CHECK(svdat02.getDataVector() == expected);
            CHECK(svdat20.getDataVector() == expected);
        }
        SECTION("CZ1,2 |+10> -> |+10>") {
            std::vector<cp_t> expected{init_state};

            StateVectorManaged<TestType> svdat12{init_state};
            StateVectorManaged<TestType> svdat21{init_state};

            svdat12.applyCZ(svdat.getInternalIndices({1, 2}),
                            svdat.getExternalIndices({1, 2}), false);
            svdat21.applyCZ(svdat.getInternalIndices({2, 1}),
                            svdat.getExternalIndices({2, 1}), false);

            CHECK(svdat12.getDataVector() == expected);
            CHECK(svdat21.getDataVector() == expected);
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("CZ0,1 |+10> -> |1+0>") {
            std::vector<cp_t> expected{Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       std::complex<TestType>(-1 / sqrt(2), 0),
                                       Util::ZERO<TestType>()};

            StateVectorManaged<TestType> svdat01{init_state};
            StateVectorManaged<TestType> svdat10{init_state};

            svdat01.applyOperation("CZ", {0, 1});
            svdat10.applyOperation("CZ", {1, 0});

            CHECK(svdat01.getDataVector() == expected);
            CHECK(svdat10.getDataVector() == expected);
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyToffoli",
                   "[StateVectorManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorManaged<TestType> svdat{num_qubits};

    // Test using |+10> state
    svdat.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                          {false, false});
    const auto init_state = svdat.getDataVector();

    SECTION("Apply directly") {
        SECTION("Toffoli 0,1,2 |+10> -> |010> + |111>") {
            std::vector<cp_t> expected{
                Util::ZERO<TestType>(),
                Util::ZERO<TestType>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                Util::ZERO<TestType>(),
                Util::ZERO<TestType>(),
                Util::ZERO<TestType>(),
                Util::ZERO<TestType>(),
                std::complex<TestType>(1.0 / sqrt(2), 0)};

            StateVectorManaged<TestType> svdat012{init_state};

            svdat012.applyToffoli(svdat.getInternalIndices({0, 1, 2}),
                                  svdat.getExternalIndices({0, 1, 2}), false);

            CHECK(svdat012.getDataVector() == expected);
        }

        SECTION("Toffoli 1,0,2 |+10> -> |010> + |111>") {
            std::vector<cp_t> expected{
                Util::ZERO<TestType>(),
                Util::ZERO<TestType>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                Util::ZERO<TestType>(),
                Util::ZERO<TestType>(),
                Util::ZERO<TestType>(),
                Util::ZERO<TestType>(),
                std::complex<TestType>(1.0 / sqrt(2), 0)};

            StateVectorManaged<TestType> svdat102{init_state};

            svdat102.applyToffoli(svdat.getInternalIndices({1, 0, 2}),
                                  svdat.getExternalIndices({1, 0, 2}), false);

            CHECK(svdat102.getDataVector() == expected);
        }
        SECTION("Toffoli 0,2,1 |+10> -> |+10>") {
            std::vector<cp_t> expected{init_state};

            StateVectorManaged<TestType> svdat021{init_state};

            svdat021.applyToffoli(svdat.getInternalIndices({0, 2, 1}),
                                  svdat.getExternalIndices({0, 2, 1}), false);

            CHECK(svdat021.getDataVector() == expected);
        }
        SECTION("Toffoli 1,2,0 |+10> -> |+10>") {
            std::vector<cp_t> expected{init_state};

            StateVectorManaged<TestType> svdat120{init_state};

            svdat120.applyToffoli(svdat.getInternalIndices({1, 2, 0}),
                                  svdat.getExternalIndices({1, 2, 0}), false);

            CHECK(svdat120.getDataVector() == expected);
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("Toffoli [0,1,2], [1,0,2] |+10> -> |+1+>") {
            std::vector<cp_t> expected{
                Util::ZERO<TestType>(),
                Util::ZERO<TestType>(),
                std::complex<TestType>(1.0 / sqrt(2), 0),
                Util::ZERO<TestType>(),
                Util::ZERO<TestType>(),
                Util::ZERO<TestType>(),
                Util::ZERO<TestType>(),
                std::complex<TestType>(1.0 / sqrt(2), 0)};

            StateVectorManaged<TestType> svdat012{init_state};
            StateVectorManaged<TestType> svdat102{init_state};

            svdat012.applyOperation("Toffoli", {0, 1, 2});
            svdat102.applyOperation("Toffoli", {1, 0, 2});

            CHECK(svdat012.getDataVector() == expected);
            CHECK(svdat102.getDataVector() == expected);
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyCSWAP",
                   "[StateVectorManaged_Nonparam]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorManaged<TestType> svdat{num_qubits};

    // Test using |+10> state
    svdat.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                          {false, false});
    const auto init_state = svdat.getDataVector();

    SECTION("Apply directly") {
        SECTION("CSWAP 0,1,2 |+10> -> |010> + |101>") {
            std::vector<cp_t> expected{Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>()};
            StateVectorManaged<TestType> svdat012{init_state};

            svdat012.applyCSWAP(svdat.getInternalIndices({0, 1, 2}),
                                svdat.getExternalIndices({0, 1, 2}), false);

            CHECK(svdat012.getDataVector() == expected);
        }

        SECTION("CSWAP 1,0,2 |+10> -> |01+>") {
            std::vector<cp_t> expected{Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>()};

            StateVectorManaged<TestType> svdat102{init_state};

            svdat102.applyCSWAP(svdat.getInternalIndices({1, 0, 2}),
                                svdat.getExternalIndices({1, 0, 2}), false);

            CHECK(svdat102.getDataVector() == expected);
        }
        SECTION("CSWAP 2,1,0 |+10> -> |+10>") {
            std::vector<cp_t> expected{init_state};

            StateVectorManaged<TestType> svdat021{init_state};

            svdat021.applyCSWAP(svdat.getInternalIndices({2, 1, 0}),
                                svdat.getExternalIndices({2, 1, 0}), false);

            CHECK(svdat021.getDataVector() == expected);
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("CSWAP 0,1,2 |+10> -> |010> + |101>") {
            std::vector<cp_t> expected{Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>(),
                                       std::complex<TestType>(1.0 / sqrt(2), 0),
                                       Util::ZERO<TestType>(),
                                       Util::ZERO<TestType>()};
            StateVectorManaged<TestType> svdat012{init_state};

            svdat012.applyOperation("CSWAP", {0, 1, 2});

            CHECK(svdat012.getDataVector() == expected);
        }
    }
}