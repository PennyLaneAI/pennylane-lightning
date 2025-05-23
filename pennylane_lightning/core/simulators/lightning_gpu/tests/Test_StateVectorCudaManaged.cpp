// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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

#include "StateVectorCudaManaged.hpp"
#include "TestHelpers.hpp" // createRandomStateVectorData

/**
 * @file
 *  Tests for functionality for the class StateVectorCudaManaged.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::Util;

using Pennylane::Util::isApproxEqual;
using Pennylane::Util::randomUnitary;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("StateVectorCudaManaged::Constructibility",
                   "[Default Constructibility]", StateVectorCudaManaged<>) {
    SECTION("StateVectorBackend<>") {
        REQUIRE(!std::is_constructible_v<TestType>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorCudaManaged::Constructibility",
                           "[General Constructibility]",
                           (StateVectorCudaManaged), (float, double)) {
    using StateVectorT = TestType;
    using ComplexT = typename StateVectorT::ComplexT;

    SECTION("StateVectorBackend<TestType>") {
        REQUIRE(!std::is_constructible_v<StateVectorT>);
    }
    SECTION("StateVectorBackend<TestType> {ComplexT*, std::size_t}") {
        REQUIRE(std::is_constructible_v<StateVectorT, ComplexT *, std::size_t>);
    }
    SECTION(
        "StateVectorBackend<TestType> {const StateVectorBackend<TestType>&}") {
        REQUIRE(std::is_copy_constructible_v<StateVectorT>);
    }
    SECTION("StateVectorBackend<TestType> {StateVectorBackend<TestType>&&}") {
        REQUIRE(std::is_move_constructible_v<StateVectorT>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorCudaManaged::getDataVector",
                           "[getDataVector]", (StateVectorCudaManaged),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using VectorT = TestVector<std::complex<PrecisionT>>;
    std::mt19937_64 re{1337};

    const std::size_t num_qubits = 4;
    VectorT st_data = createRandomStateVectorData<PrecisionT>(re, num_qubits);
    StateVectorT state_vector(reinterpret_cast<ComplexT *>(st_data.data()),
                              st_data.size());

    auto host_data = state_vector.getDataVector();

    CHECK(host_data == Pennylane::Util::approx(st_data));
}

TEMPLATE_PRODUCT_TEST_CASE(
    "StateVectorCudaManaged::applyMatrix with a std::vector", "[applyMatrix]",
    (StateVectorCudaManaged), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using VectorT = TestVector<std::complex<PrecisionT>>;
    std::mt19937_64 re{1337};

    SECTION("Test wrong matrix size") {
        std::vector<ComplexT> m(7, 0.0);
        const std::size_t num_qubits = 4;
        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);
        StateVectorT state_vector(reinterpret_cast<ComplexT *>(st_data.data()),
                                  st_data.size());
        REQUIRE_THROWS_WITH(
            state_vector.applyMatrix(m, {0, 1}),
            Catch::Contains(
                "The size of matrix does not match with the given"));
    }

    SECTION("Test wrong number of wires") {
        std::vector<ComplexT> m(8, 0.0);
        const std::size_t num_qubits = 4;
        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(reinterpret_cast<ComplexT *>(st_data.data()),
                                  st_data.size());
        REQUIRE_THROWS_WITH(
            state_vector.applyMatrix(m, {0}),
            Catch::Contains(
                "The size of matrix does not match with the given"));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorCudaManaged::applyMatrix with a pointer",
                           "[applyMatrix]", (StateVectorCudaManaged),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using VectorT = TestVector<std::complex<PrecisionT>>;
    std::mt19937_64 re{1337};

    SECTION("Test wrong matrix") {
        std::vector<ComplexT> m(8, 0.0);
        const std::size_t num_qubits = 4;
        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(reinterpret_cast<ComplexT *>(st_data.data()),
                                  st_data.size());
        REQUIRE_THROWS_WITH(state_vector.applyMatrix(m.data(), {}),
                            Catch::Contains("must be larger than 0"));
    }

    SECTION("Test a matrix represent PauliX") {
        std::vector<ComplexT> m = {
            {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
        const std::size_t num_qubits = 4;
        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(reinterpret_cast<ComplexT *>(st_data.data()),
                                  st_data.size());
        StateVectorT state_vector_ref(
            reinterpret_cast<ComplexT *>(st_data.data()), st_data.size());
        state_vector.applyMatrix(m.data(), {1});
        state_vector_ref.applyPauliX({1}, false);

        CHECK(state_vector.getDataVector() ==
              Pennylane::Util::approx(state_vector_ref.getDataVector()));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorCudaManaged::applyOperations",
                           "[applyOperations invalid arguments]",
                           (StateVectorCudaManaged), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using VectorT = TestVector<std::complex<PrecisionT>>;
    std::mt19937_64 re{1337};

    SECTION("Test invalid arguments without parameters") {
        const std::size_t num_qubits = 4;
        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(reinterpret_cast<ComplexT *>(st_data.data()),
                                  st_data.size());

        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"PauliX", "PauliY"}, {{0}},
                                         {false, false}),
            LightningException, "must all be equal"); // invalid wires
        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"PauliX", "PauliY"}, {{0}, {1}},
                                         {false}),
            LightningException, "must all be equal"); // invalid inverse
        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperation("PauliX", {}, std::vector<bool>{false},
                                        std::vector<std::size_t>{1}, false,
                                        {0.0}, std::vector<ComplexT>{}),
            LightningException,
            "`controlled_wires` and `controlled_values` must have the "
            "same size."); // invalid controlled_wires
    }

    SECTION("Test invalid arguments with parameters") {
        const std::size_t num_qubits = 4;

        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(reinterpret_cast<ComplexT *>(st_data.data()),
                                  st_data.size());

        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"RX", "RY"}, {{0}}, {false, false},
                                         {{0.0}, {0.0}}),
            LightningException, "must all be equal"); // invalid wires

        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"RX", "RY"}, {{0}, {1}}, {false},
                                         {{0.0}, {0.0}}),
            LightningException, "must all be equal"); // invalid wires

        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"RX", "RY"}, {{0}, {1}},
                                         {false, false}, {{0.0}}),
            LightningException, "must all be equal"); // invalid parameters
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::StateVectorCudaManaged",
                   "[StateVectorCudaManaged]", float, double) {
    using StateVectorT = StateVectorCudaManaged<TestType>;
    using PrecisionT = TestType;
    using ComplexT = typename StateVectorT::ComplexT;
    std::mt19937_64 re{1337};

    SECTION("StateVectorCudaManaged<TestType> {std::size_t}") {
        REQUIRE(std::is_constructible_v<StateVectorT, std::size_t>);
        const std::size_t num_qubits = 4;
        StateVectorT sv(num_qubits);

        REQUIRE(sv.getNumQubits() == 4);
        REQUIRE(sv.getLength() == 16);
        REQUIRE(sv.getDataVector().size() == 16);
    }

    SECTION("StateVectorCudaManaged<TestType> {ComplexT *, std::size_t}") {
        using TestVectorT = TestVector<std::complex<PrecisionT>>;
        REQUIRE(std::is_constructible_v<StateVectorT, ComplexT *, std::size_t>);
        const std::size_t num_qubits = 5;
        TestVectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);
        StateVectorT sv(reinterpret_cast<ComplexT *>(st_data.data()),
                        st_data.size());

        REQUIRE(sv.getNumQubits() == 5);
        REQUIRE(sv.getLength() == 32);
        REQUIRE(sv.getDataVector().size() == 32);
    }

    SECTION("StateVectorCudaManaged<TestType> {const "
            "StateVectorCudaManaged<TestType>&}") {
        REQUIRE(std::is_constructible_v<StateVectorT, const StateVectorT &>);
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::collapse",
                   "[StateVectorCudaManaged]", float, double) {
    using PrecisionT = TestType;
    using ComplexT = typename StateVectorCudaManaged<PrecisionT>::ComplexT;
    using CFP_t = typename StateVectorCudaManaged<PrecisionT>::CFP_t;
    using TestVectorT = TestVector<ComplexT>;

    std::size_t wire = GENERATE(0, 1, 2);
    std::size_t branch = GENERATE(0, 1);
    constexpr std::size_t num_qubits = 3;

    // TODO @tomlqc use same template for testing all Lightning flavours?

    SECTION("Collapse the state vector after having measured one of the "
            "qubits.") {
        TestVectorT init_state = createPlusState_<ComplexT>(num_qubits);

        const ComplexT coef{0.5, PrecisionT{0.0}};
        const ComplexT zero{PrecisionT{0.0}, PrecisionT{0.0}};

        std::vector<std::vector<std::vector<ComplexT>>> expected_state = {
            {{coef, coef, coef, coef, zero, zero, zero, zero},
             {coef, coef, zero, zero, coef, coef, zero, zero},
             {coef, zero, coef, zero, coef, zero, coef, zero}},
            {{zero, zero, zero, zero, coef, coef, coef, coef},
             {zero, zero, coef, coef, zero, zero, coef, coef},
             {zero, coef, zero, coef, zero, coef, zero, coef}},
        };

        StateVectorCudaManaged<PrecisionT> sv(
            reinterpret_cast<CFP_t *>(init_state.data()), init_state.size());

        sv.collapse(wire, branch);

        PrecisionT eps = std::numeric_limits<PrecisionT>::epsilon() * 1e2;
        REQUIRE(isApproxEqual(sv.getDataVector().data(),
                              sv.getDataVector().size(),
                              expected_state[branch][wire].data(),
                              expected_state[branch][wire].size(), eps));
    }
}
