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

std::mt19937_64 re{1337};
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
    SECTION("StateVectorBackend<TestType> {ComplexT*, size_t}") {
        REQUIRE(std::is_constructible_v<StateVectorT, ComplexT *, size_t>);
    }
    SECTION(
        "StateVectorBackend<TestType> {const StateVectorBackend<TestType>&}") {
        REQUIRE(std::is_copy_constructible_v<StateVectorT>);
    }
    SECTION("StateVectorBackend<TestType> {StateVectorBackend<TestType>&&}") {
        REQUIRE(std::is_move_constructible_v<StateVectorT>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE(
    "StateVectorCudaManaged::applyMatrix with a std::vector", "[applyMatrix]",
    (StateVectorCudaManaged), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using VectorT = TestVector<std::complex<PrecisionT>>;

    SECTION("Test wrong matrix size") {
        std::vector<ComplexT> m(7, 0.0);
        const size_t num_qubits = 4;
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
        const size_t num_qubits = 4;
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

    SECTION("Test wrong matrix") {
        std::vector<ComplexT> m(8, 0.0);
        const size_t num_qubits = 4;
        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(reinterpret_cast<ComplexT *>(st_data.data()),
                                  st_data.size());
        REQUIRE_THROWS_WITH(state_vector.applyMatrix(m.data(), {}),
                            Catch::Contains("must be larger than 0"));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorCudaManaged::applyOperations",
                           "[applyOperations invalid arguments]",
                           (StateVectorCudaManaged), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using VectorT = TestVector<std::complex<PrecisionT>>;

    SECTION("Test invalid arguments without parameters") {
        const size_t num_qubits = 4;
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
    }

    SECTION("Test invalid arguments with parameters") {
        const size_t num_qubits = 4;

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

    SECTION("StateVectorCudaManaged<TestType> {size_t}") {
        REQUIRE(std::is_constructible_v<StateVectorT, size_t>);
        const size_t num_qubits = 4;
        StateVectorT sv(num_qubits);

        REQUIRE(sv.getNumQubits() == 4);
        REQUIRE(sv.getLength() == 16);
        REQUIRE(sv.getDataVector().size() == 16);
    }

    SECTION("StateVectorCudaManaged<TestType> {ComplexT *, size_t}") {
        using TestVectorT = TestVector<std::complex<PrecisionT>>;
        REQUIRE(std::is_constructible_v<StateVectorT, ComplexT *, size_t>);
        const size_t num_qubits = 5;
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
