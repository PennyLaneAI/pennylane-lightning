#include <complex>
#include <numeric>
#include <vector>

#include "StateVectorRawCPU.hpp"
#include "TestHelpers.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

using namespace Pennylane;

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

    SECTION("changeDataPtr correctly update data") {
        auto st_data = createRandomState<PrecisionT>(re, 4);
        StateVectorRawCPU<PrecisionT> sv(st_data.data(), st_data.size());

        auto st_data2 = createRandomState<PrecisionT>(re, 8);
        sv.changeDataPtr(st_data2.data(), st_data2.size());

        REQUIRE(sv.getNumQubits() == 8);
        REQUIRE(sv.getData() == st_data2.data());
        REQUIRE(sv.getLength() == (1U << 8U));
    }

    SECTION("changeDataPtr throws an exception when the data is incorrect") {
        auto st_data = createRandomState<PrecisionT>(re, 4);
        StateVectorRawCPU<PrecisionT> sv(st_data.data(), st_data.size());

        std::vector<std::complex<PrecisionT>> new_data(7, PrecisionT{0.0});

        REQUIRE_THROWS_AS(sv.changeDataPtr(new_data.data(), new_data.size()),
                          Util::LightningException);
    }

    SECTION("setDataFrom correctly update data") {
        auto st_data1 = createRandomState<PrecisionT>(re, 4);
        auto st_data2 = createRandomState<PrecisionT>(re, 4);
        StateVectorRawCPU<PrecisionT> sv(st_data1.data(), st_data1.size());

        sv.setDataFrom(st_data2.data(),
                       st_data2.size()); // Should update st_data1
        REQUIRE(st_data1 == st_data2);
    }

    SECTION("setDataFrom throws an exception when the data is incorrect") {
        auto st_data1 = createRandomState<PrecisionT>(re, 4);
        auto st_data2 = createRandomState<PrecisionT>(re, 8);
        StateVectorRawCPU<PrecisionT> sv(st_data1.data(), st_data1.size());

        REQUIRE_THROWS_AS(sv.setDataFrom(st_data2.data(), st_data2.size()),
                          Util::LightningException);
    }
}
