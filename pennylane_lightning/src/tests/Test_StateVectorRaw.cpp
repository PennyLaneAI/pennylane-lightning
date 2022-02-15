#include <complex>
#include <numeric>
#include <vector>

#include "StateVectorRaw.hpp"
#include "TestHelpers.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

using namespace Pennylane;

std::mt19937_64 re{1337};

TEMPLATE_TEST_CASE("StateVectorRaw::StateVectorRaw", "[StateVectorRaw]", float,
                   double) {
    using fp_t = TestType;

    SECTION("StateVectorRaw<TestType> {std::complex<TestType>*, size_t}") {
        const size_t num_qubits = 4;
        auto st_data = createRandomState<fp_t>(re, num_qubits);
        StateVectorRaw<fp_t> sv(st_data.data(), st_data.size());

        REQUIRE(sv.getNumQubits() == 4);
        REQUIRE(sv.getData() == st_data.data());
        REQUIRE(sv.getLength() == 16);
    }
    SECTION("StateVectorRaw<TestType> {std::complex<TestType>*, size_t}") {
        std::vector<std::complex<TestType>> st_data(14, 0.0);
        REQUIRE_THROWS(StateVectorRaw<fp_t>(st_data.data(), st_data.size()));
    }
}

TEMPLATE_TEST_CASE("StateVectorRaw::setData", "[StateVectorRaw]", float,
                   double) {
    using fp_t = TestType;

    auto st_data = createRandomState<fp_t>(re, 4);
    StateVectorRaw<fp_t> sv(st_data.data(), st_data.size());

    auto st_data2 = createRandomState<fp_t>(re, 8);
    sv.setData(st_data2.data(), st_data2.size());

    REQUIRE(sv.getNumQubits() == 8);
    REQUIRE(sv.getData() == st_data2.data());
    REQUIRE(sv.getLength() == (1U << 8));
}
