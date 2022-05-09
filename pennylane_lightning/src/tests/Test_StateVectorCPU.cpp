#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "StateVectorManagedCPU.hpp"
#include "StateVectorRawCPU.hpp"
#include "Util.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane;

TEMPLATE_TEST_CASE("StateVectorManagedCPU::StateVectorManagedCPU",
                   "[StateVectorRaw]", float, double) {
    using PrecisionT = TestType;

    SECTION("StateVectorManagedCPU") {
        REQUIRE(!std::is_constructible_v<StateVectorManagedCPU<>>);
    }
    SECTION("StateVectorManagedCPU<TestType>") {
        REQUIRE(!std::is_constructible_v<StateVectorManagedCPU<TestType>>);
    }
    SECTION("StateVectorManagedCPU<TestType> {size_t}") {
        REQUIRE(
            std::is_constructible_v<StateVectorManagedCPU<TestType>, size_t>);
        const size_t num_qubits = 4;
        StateVectorManagedCPU<PrecisionT> sv(num_qubits);

        REQUIRE(sv.getNumQubits() == 4);
        REQUIRE(sv.getLength() == 16);
        REQUIRE(sv.getDataVector().size() == 16);
    }
    SECTION("StateVectorManagedCPU<TestType> {const "
            "StateVectorRawCPU<TestType>&}") {
        REQUIRE(std::is_constructible_v<StateVectorManagedCPU<TestType>,
                                        const StateVectorRawCPU<TestType> &>);
    }
    SECTION("StateVectorManagedCPU<TestType> {const "
            "StateVectorManagedCPU<TestType>&}") {
        REQUIRE(std::is_copy_constructible_v<StateVectorManagedCPU<TestType>>);
    }
    SECTION(
        "StateVectorManagedCPU<TestType> {StateVectorManagedCPU<TestType>&&}") {
        REQUIRE(std::is_move_constructible_v<StateVectorManagedCPU<TestType>>);
    }

    SECTION("Aligned 256bit statevector") {
        const auto memory_model = CPUMemoryModel::Aligned256;
        StateVectorManagedCPU<PrecisionT> sv(4, Threading::SingleThread,
                                             memory_model);
        /* Even when we allocate 256 bit aligend memory, it is possible that the
         * alignment happens to be 512 bit */
        REQUIRE(((getMemoryModel(sv.getDataVector().data()) ==
                  CPUMemoryModel::Aligned256) ||
                 (getMemoryModel(sv.getDataVector().data()) ==
                  CPUMemoryModel::Aligned512)));
    }

    SECTION("Aligned 512bit statevector") {
        const auto memory_model = CPUMemoryModel::Aligned512;
        StateVectorManagedCPU<PrecisionT> sv(4, Threading::SingleThread,
                                             memory_model);
        REQUIRE((getMemoryModel(sv.getDataVector().data()) ==
                 CPUMemoryModel::Aligned512));
    }
}

std::mt19937_64 re{1337};

TEMPLATE_TEST_CASE("StateVectorRawCPU::StateVectorRawCPU",
                   "[StateVectorRawCPU]", float, double) {
    using PrecisionT = TestType;

    SECTION("StateVectorRawCPU<TestType> {std::complex<TestType>*, size_t}") {
        const size_t num_qubits = 4;
        auto st_data = createRandomState<PrecisionT>(re, num_qubits);
        StateVectorRawCPU<PrecisionT> sv(st_data.data(), st_data.size());

        REQUIRE(sv.getNumQubits() == 4);
        REQUIRE(sv.getData() == st_data.data());
        REQUIRE(sv.getLength() == 16);
    }
    SECTION("StateVectorRawCPU<TestType> {std::complex<TestType>*, size_t}") {
        std::vector<std::complex<TestType>> st_data(14, 0.0);
        REQUIRE_THROWS(
            StateVectorRawCPU<PrecisionT>(st_data.data(), st_data.size()));
    }
}

TEMPLATE_TEST_CASE("StateVectorRawCPU::setData", "[StateVectorRawCPU]", float,
                   double) {
    using PrecisionT = TestType;

    auto st_data = createRandomState<PrecisionT>(re, 4);
    StateVectorRawCPU<PrecisionT> sv(st_data.data(), st_data.size());

    auto st_data2 = createRandomState<PrecisionT>(re, 8);
    sv.setData(st_data2.data(), st_data2.size());

    REQUIRE(sv.getNumQubits() == 8);
    REQUIRE(sv.getData() == st_data2.data());
    REQUIRE(sv.getLength() == (1U << 8U));
}
