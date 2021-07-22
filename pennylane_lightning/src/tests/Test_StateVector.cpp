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

/**
 * @brief Utility function to compare complex statevector data.
 *
 * @tparam Data_t Floating point data-type.
 * @param data1 StateVector data 1.
 * @param data2 StateVector data 2.
 * @return true Data are approximately equal.
 * @return false Data are not approximately equal.
 */
template <class Data_t>
inline bool isApproxEqual(const std::vector<Data_t> &data1,
                          const std::vector<Data_t> &data2) {
    if (data1.size() != data2.size())
        return false;

    for (size_t i = 0; i < data1.size(); i++) {
        if (data1[i].real() != Approx(data2[i].real()) ||
            data1[i].imag() != Approx(data2[i].imag())) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Tests the constructability of the StateVector class.
 *
 */
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

/**
 * @brief Utility data-structure to assist with testing StateVector class
 *
 * @tparam fp_t Floating-point type. Supported options: float, double
 */
template <typename fp_t> struct SVData {
    size_t num_qubits;
    std::vector<std::complex<fp_t>> cdata;
    StateVector<fp_t> sv;

    SVData(size_t num_qubits)
        : num_qubits{num_qubits}, // qubit_indices{num_qubits},
          cdata(0b1 << num_qubits), sv{cdata.data(), cdata.size()} {
        cdata[0] = std::complex<fp_t>{1, 0};
    }
    SVData(size_t num_qubits,
           const std::vector<std::complex<fp_t>> &cdata_input)
        : num_qubits{num_qubits}, // qubit_indices{num_qubits},
          cdata(cdata_input), sv{cdata.data(), cdata.size()} {}
    vector<size_t>
    getInternalIndices(const std::vector<size_t> &qubit_indices) {
        return sv.generateBitPatterns(qubit_indices);
    }
    vector<size_t>
    getExternalIndices(const std::vector<size_t> &qubit_indices) {
        vector<size_t> externalWires =
            sv.getIndicesAfterExclusion(qubit_indices);
        vector<size_t> externalIndices = sv.generateBitPatterns(externalWires);
        return externalIndices;
    }
};
TEMPLATE_TEST_CASE("StateVector::applyHadamard", "[StateVector]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits};

            auto int_idx = svdat.getInternalIndices({index});
            auto ext_idx = svdat.getExternalIndices({index});
            CHECK(svdat.cdata[0] == cp_t{1, 0});
            svdat.sv.applyHadamard(int_idx, ext_idx, false);

            cp_t expected = {1 / std::sqrt(2), 0};
            CHECK(expected.real() == Approx(svdat.cdata[0].real()));
            CHECK(expected.imag() == Approx(svdat.cdata[0].imag()));

            CHECK(
                expected.real() ==
                Approx(
                    svdat.cdata[0b1 << (svdat.num_qubits - index - 1)].real()));
            CHECK(
                expected.imag() ==
                Approx(
                    svdat.cdata[0b1 << (svdat.num_qubits - index - 1)].imag()));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits};
            CHECK(svdat.cdata[0] == cp_t{1, 0});
            svdat.sv.applyOperation("Hadamard", {index}, false);

            cp_t expected = {1 / std::sqrt(2), 0};

            CHECK(expected.real() == Approx(svdat.cdata[0].real()));
            CHECK(expected.imag() == Approx(svdat.cdata[0].imag()));

            CHECK(
                expected.real() ==
                Approx(
                    svdat.cdata[0b1 << (svdat.num_qubits - index - 1)].real()));
            CHECK(
                expected.imag() ==
                Approx(
                    svdat.cdata[0b1 << (svdat.num_qubits - index - 1)].imag()));
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyPauliX", "[StateVector]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits};
            auto int_idx = svdat.getInternalIndices({index});
            auto ext_idx = svdat.getExternalIndices({index});
            CHECK(svdat.cdata[0] == cp_t{1, 0});
            svdat.sv.applyPauliX(int_idx, ext_idx, false);
            CHECK(svdat.cdata[0] == cp_t{0, 0});
            CHECK(svdat.cdata[0b1 << (svdat.num_qubits - index - 1)] ==
                  cp_t{1, 0});
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits};
            CHECK(svdat.cdata[0] == cp_t{1, 0});
            svdat.sv.applyOperation("PauliX", {index}, false);
            CHECK(svdat.cdata[0] == cp_t{0, 0});
            CHECK(svdat.cdata[0b1 << (svdat.num_qubits - index - 1)] ==
                  cp_t{1, 0});
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyPauliY", "[StateVector]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};
    // Test using |+++> state
    svdat.sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                             {{0}, {1}, {2}}, {{false}, {false}, {false}});

    cp_t p = {0, 1 / (2 * std::sqrt(2))};
    cp_t m = {0, -1 / (2 * std::sqrt(2))};

    const std::vector<std::vector<cp_t>> expected_results = {
        {m, m, m, m, p, p, p, p},
        {m, m, p, p, m, m, p, p},
        {m, p, m, p, m, p, m, p}};

    const auto init_state = svdat.cdata;
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits, init_state};
            auto int_idx = svdat.getInternalIndices({index});
            auto ext_idx = svdat.getExternalIndices({index});

            CHECK(svdat.cdata == init_state);
            svdat.sv.applyPauliY(int_idx, ext_idx, false);

            CHECK(isApproxEqual(svdat.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits, init_state};
            CHECK(svdat.cdata == init_state);
            svdat.sv.applyOperation("PauliY", {index}, false);
            CHECK(isApproxEqual(svdat.cdata, expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyPauliZ", "[StateVector]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};
    // Test using |+++> state
    svdat.sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                             {{0}, {1}, {2}}, {{false}, {false}, {false}});

    cp_t p = {1 / (2 * std::sqrt(2)), 0};
    cp_t m = {-1 / (2 * std::sqrt(2)), 0};

    const std::vector<std::vector<cp_t>> expected_results = {
        {p, p, p, p, m, m, m, m},
        {p, p, m, m, p, p, m, m},
        {p, m, p, m, p, m, p, m}};

    const auto init_state = svdat.cdata;
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits, init_state};
            auto int_idx = svdat.getInternalIndices({index});
            auto ext_idx = svdat.getExternalIndices({index});

            CHECK(svdat.cdata == init_state);
            svdat.sv.applyPauliZ(int_idx, ext_idx, false);

            CHECK(isApproxEqual(svdat.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits, init_state};
            CHECK(svdat.cdata == init_state);
            svdat.sv.applyOperation("PauliZ", {index}, false);
            CHECK(isApproxEqual(svdat.cdata, expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyS", "[StateVector]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};
    // Test using |+++> state
    svdat.sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                             {{0}, {1}, {2}}, {{false}, {false}, {false}});

    cp_t r = {1 / (2 * std::sqrt(2)), 0};
    cp_t i = {0, 1 / (2 * std::sqrt(2))};

    const std::vector<std::vector<cp_t>> expected_results = {
        {r, r, r, r, i, i, i, i},
        {r, r, i, i, r, r, i, i},
        {r, i, r, i, r, i, r, i}};

    const auto init_state = svdat.cdata;
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits, init_state};
            auto int_idx = svdat.getInternalIndices({index});
            auto ext_idx = svdat.getExternalIndices({index});

            CHECK(svdat.cdata == init_state);
            svdat.sv.applyS(int_idx, ext_idx, false);

            CHECK(isApproxEqual(svdat.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits, init_state};
            CHECK(svdat.cdata == init_state);
            svdat.sv.applyOperation("S", {index}, false);
            CHECK(isApproxEqual(svdat.cdata, expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyT", "[StateVector]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};
    // Test using |+++> state
    svdat.sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                             {{0}, {1}, {2}}, {{false}, {false}, {false}});

    cp_t r = {1 / (2 * std::sqrt(2)), 0};
    cp_t i = {1.0 / 4, 1.0 / 4};

    const std::vector<std::vector<cp_t>> expected_results = {
        {r, r, r, r, i, i, i, i},
        {r, r, i, i, r, r, i, i},
        {r, i, r, i, r, i, r, i}};

    const auto init_state = svdat.cdata;
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits, init_state};
            auto int_idx = svdat.getInternalIndices({index});
            auto ext_idx = svdat.getExternalIndices({index});

            CHECK(svdat.cdata == init_state);
            svdat.sv.applyT(int_idx, ext_idx, false);

            CHECK(isApproxEqual(svdat.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits, init_state};
            CHECK(svdat.cdata == init_state);
            svdat.sv.applyOperation("T", {index}, false);
            CHECK(isApproxEqual(svdat.cdata, expected_results[index]));
        }
    }
}