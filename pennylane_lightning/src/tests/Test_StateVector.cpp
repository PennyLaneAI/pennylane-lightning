#include <algorithm>
#include <complex>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "StateVector.hpp"
#include "Util.hpp"

using namespace Pennylane;

TEMPLATE_TEST_CASE("StateVector::StateVector", "[StateVector]", float, double) {
    SECTION("StateVector") {
        REQUIRE(std::is_constructible<StateVector<>>::value);
    }
    SECTION("StateVector<TestType> {}") {
        REQUIRE(std::is_constructible<StateVector<TestType>>::value);
    }
    SECTION("StateVector<TestType> {std::complex<TestType>, size_t}") {
        REQUIRE(std::is_constructible<StateVector<TestType>,
                                      std::complex<TestType>, size_t>::value);
    }
    SECTION("StateVector<TestType> cross types") {
        if constexpr (!std::is_same_v<TestType, double>) {
            REQUIRE_FALSE(
                std::is_constructible<StateVector<TestType>,
                                      std::complex<double>, size_t>::value);
            REQUIRE_FALSE(
                std::is_constructible<StateVector<double>,
                                      std::complex<TestType>, size_t>::value);
        } else if constexpr (!std::is_same_v<TestType, float>) {
            REQUIRE_FALSE(
                std::is_constructible<StateVector<TestType>,
                                      std::complex<float>, size_t>::value);
            REQUIRE_FALSE(
                std::is_constructible<StateVector<float>,
                                      std::complex<TestType>, size_t>::value);
        }
    }
}

template <typename fp_t> struct SVData {
    size_t num_qubits;
    std::vector<size_t> qubit_indices;
    std::vector<std::complex<fp_t>> cdata;
    StateVector<fp_t> sv;

    SVData(size_t num_qubits)
        : num_qubits{num_qubits}, qubit_indices{num_qubits},
          cdata(0b1 << num_qubits), sv{cdata.data(), cdata.size()} {
        std::iota(qubit_indices.begin(), qubit_indices.end(), 0);
        cdata[0] = std::complex<fp_t>{1, 0};
    }
};

TEMPLATE_TEST_CASE("StateVector::applyPauliX", "[StateVector]", float, double) {

    SVData<TestType> svdat{3};

    vector<size_t> internalIndices =
        svdat.sv.generateBitPatterns(svdat.qubit_indices);
    vector<size_t> externalWires =
        svdat.sv.getIndicesAfterExclusion(svdat.qubit_indices);
    vector<size_t> externalIndices =
        svdat.sv.generateBitPatterns(externalWires);

    SECTION("XII|000> -> |100>") {}
}
TEMPLATE_TEST_CASE("StateVector::applyPauliY", "[StateVector]", float, double) {

}
TEMPLATE_TEST_CASE("StateVector::applyPauliZ", "[StateVector]", float, double) {

}
TEMPLATE_TEST_CASE("StateVector::applyHadamard", "[StateVector]", float,
                   double) {}
