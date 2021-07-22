#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
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
inline bool isApproxEqual(
    const std::vector<Data_t> &data1, const std::vector<Data_t> &data2,
    const typename Data_t::value_type eps =
        std::numeric_limits<typename Data_t::value_type>::epsilon() * 100) {
    if (data1.size() != data2.size())
        return false;

    for (size_t i = 0; i < data1.size(); i++) {
        if (data1[i].real() != Approx(data2[i].real()).epsilon(eps) ||
            data1[i].imag() != Approx(data2[i].imag()).epsilon(eps)) {
            return false;
        }
    }
    return true;
}

template <class Data_t> vector<std::complex<Data_t>> RX(Data_t parameter) {
    const std::complex<Data_t> c{std::cos(parameter / 2), 0};
    const std::complex<Data_t> js{0, std::sin(-parameter / 2)};
    return {c, js, js, c};
}

template <class Data_t> vector<std::complex<Data_t>> RY(Data_t parameter) {
    const std::complex<Data_t> c{std::cos(parameter / 2), 0};
    const std::complex<Data_t> s{std::sin(parameter / 2), 0};
    return {c, s, s, c};
}

template <class Data_t> vector<std::complex<Data_t>> RZ(Data_t parameter) {
    return {std::exp(std::complex<Data_t>{0, -parameter / 2}),
            {0, 0},
            {0, 0},
            std::exp(std::complex<Data_t>{0, parameter / 2})};
}

template <class Data_t> vector<std::complex<Data_t>> Phase(Data_t parameter) {
    return {
        {1, 0}, {0, 0}, {0, 0}, std::exp(std::complex<Data_t>{0, parameter})};
}

template <class Data_t>
vector<std::complex<Data_t>> Rot(Data_t phi, Data_t theta, Data_t omega) {
    const std::complex<Data_t> e00{0, (-phi - omega) / 2};
    const std::complex<Data_t> e10{0, (-phi + omega) / 2};
    const std::complex<Data_t> e01{0, (phi - omega) / 2};
    const std::complex<Data_t> e11{0, (phi + omega) / 2};

    const std::complex<Data_t> exp00{std::pow(M_E, e00)};
    const std::complex<Data_t> exp10{std::pow(M_E, e10)};
    const std::complex<Data_t> exp01{std::pow(M_E, e01)};
    const std::complex<Data_t> exp11{std::pow(M_E, e11)};

    const Data_t c{std::cos(theta / 2)};
    const Data_t s{std::sin(theta / 2)};

    return {exp00 * c, -exp01 * s, exp10 * s, exp11 * c};
}

template <class Data_t>
void scaleVector(std::vector<std::complex<Data_t>> &data,
                 std::complex<Data_t> scalar) {
    for (auto &a : data)
        a *= scalar;
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

TEMPLATE_TEST_CASE("StateVector::applyRX", "[StateVector]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.1, 0.6, 2.1};
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(8), std::vector<cp_t>(8), std::vector<cp_t>(8)};

    for (size_t i = 0; i < angles.size(); i++) {
        const auto rx_mat = RX(angles[i]);
        expected_results[i][0] = rx_mat[0];
        expected_results[i][0b1 << (num_qubits - i - 1)] = rx_mat[1];
    }

    const auto init_state = svdat.cdata;
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits};
            auto int_idx = svdat.getInternalIndices({index});
            auto ext_idx = svdat.getExternalIndices({index});

            svdat.sv.applyRX(int_idx, ext_idx, false, {angles[index]});

            CHECK(isApproxEqual(svdat.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits};
            svdat.sv.applyOperation("RX", {index}, false, {angles[index]});
            CHECK(isApproxEqual(svdat.cdata, expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyRY", "[StateVector]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.2, 0.7, 2.9};
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(8), std::vector<cp_t>(8), std::vector<cp_t>(8)};

    for (size_t i = 0; i < angles.size(); i++) {
        const auto ry_mat = RY(angles[i]);
        expected_results[i][0] = ry_mat[0];
        expected_results[i][0b1 << (num_qubits - i - 1)] = ry_mat[2];
    }

    const auto init_state = svdat.cdata;
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits};
            auto int_idx = svdat.getInternalIndices({index});
            auto ext_idx = svdat.getExternalIndices({index});

            svdat.sv.applyRY(int_idx, ext_idx, false, {angles[index]});

            CHECK(isApproxEqual(svdat.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits};
            svdat.sv.applyOperation("RY", {index}, false, {angles[index]});
            CHECK(isApproxEqual(svdat.cdata, expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyRZ", "[StateVector]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};

    // Test using |+++> state
    svdat.sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                             {{0}, {1}, {2}}, {{false}, {false}, {false}});

    const std::vector<TestType> angles{0.2, 0.7, 2.9};
    const cp_t coef = {1 / (2 * std::sqrt(2)), 0};

    std::vector<std::vector<cp_t>> rz_data;
    for (auto &a : angles) {
        rz_data.push_back(RZ(a));
    }

    std::vector<std::vector<cp_t>> expected_results = {
        {rz_data[0][0], rz_data[0][0], rz_data[0][0], rz_data[0][0],
         rz_data[0][3], rz_data[0][3], rz_data[0][3], rz_data[0][3]},
        {
            rz_data[1][0],
            rz_data[1][0],
            rz_data[1][3],
            rz_data[1][3],
            rz_data[1][0],
            rz_data[1][0],
            rz_data[1][3],
            rz_data[1][3],
        },
        {rz_data[2][0], rz_data[2][3], rz_data[2][0], rz_data[2][3],
         rz_data[2][0], rz_data[2][3], rz_data[2][0], rz_data[2][3]}};

    for (auto &vec : expected_results) {
        scaleVector(vec, coef);
    }

    const auto init_state = svdat.cdata;
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits, init_state};
            auto int_idx = svdat.getInternalIndices({index});
            auto ext_idx = svdat.getExternalIndices({index});

            svdat.sv.applyRZ(int_idx, ext_idx, false, {angles[index]});

            CHECK(isApproxEqual(svdat.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits, init_state};
            svdat.sv.applyOperation("RZ", {index}, false, {angles[index]});
            CHECK(isApproxEqual(svdat.cdata, expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyPhaseShift", "[StateVector]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};

    // Test using |+++> state
    svdat.sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                             {{0}, {1}, {2}}, {{false}, {false}, {false}});

    const std::vector<TestType> angles{0.3, 0.8, 2.4};
    const cp_t coef = {1 / (2 * std::sqrt(2)), 0};

    std::vector<std::vector<cp_t>> ps_data;
    for (auto &a : angles) {
        ps_data.push_back(Phase(a));
    }

    std::vector<std::vector<cp_t>> expected_results = {
        {ps_data[0][0], ps_data[0][0], ps_data[0][0], ps_data[0][0],
         ps_data[0][3], ps_data[0][3], ps_data[0][3], ps_data[0][3]},
        {
            ps_data[1][0],
            ps_data[1][0],
            ps_data[1][3],
            ps_data[1][3],
            ps_data[1][0],
            ps_data[1][0],
            ps_data[1][3],
            ps_data[1][3],
        },
        {ps_data[2][0], ps_data[2][3], ps_data[2][0], ps_data[2][3],
         ps_data[2][0], ps_data[2][3], ps_data[2][0], ps_data[2][3]}};

    for (auto &vec : expected_results) {
        scaleVector(vec, coef);
    }

    const auto init_state = svdat.cdata;
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits, init_state};
            auto int_idx = svdat.getInternalIndices({index});
            auto ext_idx = svdat.getExternalIndices({index});

            svdat.sv.applyPhaseShift(int_idx, ext_idx, false, {angles[index]});

            CHECK(isApproxEqual(svdat.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits, init_state};
            svdat.sv.applyOperation("PhaseShift", {index}, false,
                                    {angles[index]});
            CHECK(isApproxEqual(svdat.cdata, expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyRot", "[StateVector]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};

    const std::vector<std::vector<TestType>> angles{
        std::vector<TestType>{0.3, 0.8, 2.4},
        std::vector<TestType>{0.5, 1.1, 3.0},
        std::vector<TestType>{2.3, 0.1, 0.4}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(0b1 << num_qubits),
        std::vector<cp_t>(0b1 << num_qubits),
        std::vector<cp_t>(0b1 << num_qubits)};

    for (size_t i = 0; i < angles.size(); i++) {
        const auto rot_mat = Rot(angles[i][0], angles[i][1], angles[i][2]);
        expected_results[i][0] = rot_mat[0];
        expected_results[i][0b1 << (num_qubits - i - 1)] = rot_mat[2];
    }

    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits};
            auto int_idx = svdat.getInternalIndices({index});
            auto ext_idx = svdat.getExternalIndices({index});
            svdat.sv.applyRot(int_idx, ext_idx, false, angles[index][0],
                              angles[index][1], angles[index][2]);

            CHECK(isApproxEqual(svdat.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits};
            svdat.sv.applyOperation("Rot", {index}, false, angles[index]);
            CHECK(isApproxEqual(svdat.cdata, expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyCNOT", "[StateVector]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};

    // Test using |+00> state to generate 3-qubit GHZ state
    svdat.sv.applyOperation("Hadamard", {0});
    const auto init_state = svdat.cdata;

    SECTION("Apply directly") {
        SVData<TestType> svdat{num_qubits, init_state};

        for (size_t index = 1; index < num_qubits; index++) {
            auto int_idx = svdat.getInternalIndices({index - 1, index});
            auto ext_idx = svdat.getExternalIndices({index - 1, index});

            svdat.sv.applyCNOT(int_idx, ext_idx, false);
        }
        CHECK(svdat.cdata.front() == cp_t{1 / sqrt(2), 0});
        CHECK(svdat.cdata.back() == cp_t{1 / sqrt(2), 0});
    }

    SECTION("Apply using dispatcher") {
        SVData<TestType> svdat{num_qubits, init_state};

        for (size_t index = 1; index < num_qubits; index++) {
            svdat.sv.applyOperation("CNOT", {index - 1, index}, false);
        }
        CHECK(svdat.cdata.front() == cp_t{1 / sqrt(2), 0});
        CHECK(svdat.cdata.back() == cp_t{1 / sqrt(2), 0});
    }
}

TEMPLATE_TEST_CASE("StateVector::applySWAP", "[StateVector]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};

    // Test using |+10> state
    svdat.sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                             {false, false});
    const auto init_state = svdat.cdata;

    SECTION("Apply directly") {
        CHECK(svdat.cdata == std::vector<cp_t>{{0, 0},
                                               {0, 0},
                                               {1 / sqrt(2), 0},
                                               {0, 0},
                                               {0, 0},
                                               {0, 0},
                                               {1 / sqrt(2), 0},
                                               {0, 0}});

        SECTION("SWAP0,1 |+10> -> |1+0>") {
            std::vector<cp_t> expected{
                {0, 0},           {0, 0}, {0, 0},           {0, 0},
                {1 / sqrt(2), 0}, {0, 0}, {1 / sqrt(2), 0}, {0, 0}};

            SVData<TestType> svdat01{num_qubits, init_state};
            SVData<TestType> svdat10{num_qubits, init_state};

            svdat01.sv.applySWAP(svdat.getInternalIndices({0, 1}),
                                 svdat.getExternalIndices({0, 1}), false);
            svdat10.sv.applySWAP(svdat.getInternalIndices({1, 0}),
                                 svdat.getExternalIndices({1, 0}), false);

            CHECK(svdat01.cdata == expected);
            CHECK(svdat10.cdata == expected);
        }

        SECTION("SWAP0,2 |+10> -> |01+>") {
            std::vector<cp_t> expected{
                {0, 0}, {0, 0}, {1 / sqrt(2), 0}, {1 / sqrt(2), 0},
                {0, 0}, {0, 0}, {0, 0},           {0, 0}};

            SVData<TestType> svdat02{num_qubits, init_state};
            SVData<TestType> svdat20{num_qubits, init_state};

            svdat02.sv.applySWAP(svdat.getInternalIndices({0, 2}),
                                 svdat.getExternalIndices({0, 2}), false);
            svdat20.sv.applySWAP(svdat.getInternalIndices({2, 0}),
                                 svdat.getExternalIndices({2, 0}), false);
            CHECK(svdat02.cdata == expected);
            CHECK(svdat20.cdata == expected);
        }
        SECTION("SWAP1,2 |+10> -> |+01>") {
            std::vector<cp_t> expected{
                {0, 0}, {1 / sqrt(2), 0}, {0, 0}, {0, 0},
                {0, 0}, {1 / sqrt(2), 0}, {0, 0}, {0, 0}};

            SVData<TestType> svdat12{num_qubits, init_state};
            SVData<TestType> svdat21{num_qubits, init_state};

            svdat12.sv.applySWAP(svdat.getInternalIndices({1, 2}),
                                 svdat.getExternalIndices({1, 2}), false);
            svdat21.sv.applySWAP(svdat.getInternalIndices({2, 1}),
                                 svdat.getExternalIndices({2, 1}), false);
            CHECK(svdat12.cdata == expected);
            CHECK(svdat21.cdata == expected);
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("SWAP0,1 |+10> -> |1+0>") {

            std::vector<cp_t> expected{
                {0, 0},           {0, 0}, {0, 0},           {0, 0},
                {1 / sqrt(2), 0}, {0, 0}, {1 / sqrt(2), 0}, {0, 0}};

            SVData<TestType> svdat01{num_qubits, init_state};
            SVData<TestType> svdat10{num_qubits, init_state};

            svdat01.sv.applyOperation("SWAP", {0, 1});
            svdat10.sv.applyOperation("SWAP", {1, 0});

            CHECK(svdat01.cdata == expected);
            CHECK(svdat10.cdata == expected);
        }

        SECTION("SWAP0,2 |+10> -> |01+>") {
            std::vector<cp_t> expected{
                {0, 0}, {0, 0}, {1 / sqrt(2), 0}, {1 / sqrt(2), 0},
                {0, 0}, {0, 0}, {0, 0},           {0, 0}};

            SVData<TestType> svdat02{num_qubits, init_state};
            SVData<TestType> svdat20{num_qubits, init_state};

            svdat02.sv.applyOperation("SWAP", {0, 2});
            svdat20.sv.applyOperation("SWAP", {2, 0});

            CHECK(svdat02.cdata == expected);
            CHECK(svdat20.cdata == expected);
        }
        SECTION("SWAP1,2 |+10> -> |+01>") {
            std::vector<cp_t> expected{
                {0, 0}, {1 / sqrt(2), 0}, {0, 0}, {0, 0},
                {0, 0}, {1 / sqrt(2), 0}, {0, 0}, {0, 0}};

            SVData<TestType> svdat12{num_qubits, init_state};
            SVData<TestType> svdat21{num_qubits, init_state};

            svdat12.sv.applyOperation("SWAP", {1, 2});
            svdat21.sv.applyOperation("SWAP", {2, 1});

            CHECK(svdat12.cdata == expected);
            CHECK(svdat21.cdata == expected);
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyCZ", "[StateVector]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};

    // Test using |+10> state
    svdat.sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                             {false, false});
    const auto init_state = svdat.cdata;

    SECTION("Apply directly") {
        CHECK(svdat.cdata == std::vector<cp_t>{{0, 0},
                                               {0, 0},
                                               {1 / sqrt(2), 0},
                                               {0, 0},
                                               {0, 0},
                                               {0, 0},
                                               {1 / sqrt(2), 0},
                                               {0, 0}});

        SECTION("CZ0,1 |+10> -> |-10>") {
            std::vector<cp_t> expected{
                {0, 0}, {0, 0}, {1 / sqrt(2), 0},  {0, 0},
                {0, 0}, {0, 0}, {-1 / sqrt(2), 0}, {0, 0}};

            SVData<TestType> svdat01{num_qubits, init_state};
            SVData<TestType> svdat10{num_qubits, init_state};

            svdat01.sv.applyCZ(svdat.getInternalIndices({0, 1}),
                               svdat.getExternalIndices({0, 1}), false);
            svdat10.sv.applyCZ(svdat.getInternalIndices({1, 0}),
                               svdat.getExternalIndices({1, 0}), false);

            CHECK(svdat01.cdata == expected);
            CHECK(svdat10.cdata == expected);
        }

        SECTION("CZ0,2 |+10> -> |+10>") {
            std::vector<cp_t> expected{init_state};

            SVData<TestType> svdat02{num_qubits, init_state};
            SVData<TestType> svdat20{num_qubits, init_state};

            svdat02.sv.applyCZ(svdat.getInternalIndices({0, 2}),
                               svdat.getExternalIndices({0, 2}), false);
            svdat20.sv.applyCZ(svdat.getInternalIndices({2, 0}),
                               svdat.getExternalIndices({2, 0}), false);
            CHECK(svdat02.cdata == expected);
            CHECK(svdat20.cdata == expected);
        }
        SECTION("CZ1,2 |+10> -> |+10>") {
            std::vector<cp_t> expected{init_state};

            SVData<TestType> svdat12{num_qubits, init_state};
            SVData<TestType> svdat21{num_qubits, init_state};

            svdat12.sv.applyCZ(svdat.getInternalIndices({1, 2}),
                               svdat.getExternalIndices({1, 2}), false);
            svdat21.sv.applyCZ(svdat.getInternalIndices({2, 1}),
                               svdat.getExternalIndices({2, 1}), false);

            CHECK(svdat12.cdata == expected);
            CHECK(svdat21.cdata == expected);
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("CZ0,1 |+10> -> |1+0>") {

            std::vector<cp_t> expected{
                {0, 0}, {0, 0}, {1 / sqrt(2), 0},  {0, 0},
                {0, 0}, {0, 0}, {-1 / sqrt(2), 0}, {0, 0}};

            SVData<TestType> svdat01{num_qubits, init_state};
            SVData<TestType> svdat10{num_qubits, init_state};

            svdat01.sv.applyOperation("CZ", {0, 1});
            svdat10.sv.applyOperation("CZ", {1, 0});

            CHECK(svdat01.cdata == expected);
            CHECK(svdat10.cdata == expected);
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyCRot", "[StateVector]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.3, 0.8, 2.4};

    std::vector<cp_t> expected_results(8);
    const auto rot_mat = Rot(angles[0], angles[1], angles[2]);
    expected_results[0b1 << (num_qubits - 1)] = rot_mat[0];
    expected_results[(0b1 << num_qubits) - 2] = rot_mat[2];

    const auto init_state = svdat.cdata;

    SECTION("Apply directly") {
        SECTION("CRot0,1 |000> -> |000>") {
            SVData<TestType> svdat{num_qubits};
            auto int_idx = svdat.getInternalIndices({0, 1});
            auto ext_idx = svdat.getExternalIndices({0, 1});
            svdat.sv.applyCRot(int_idx, ext_idx, false, angles[0], angles[1],
                               angles[2]);

            CHECK(isApproxEqual(svdat.cdata, init_state));
        }
        SECTION("CRot0,1 |100> -> |1>(a|0>+b|1>)|0>") {
            SVData<TestType> svdat{num_qubits};
            svdat.sv.applyOperation("PauliX", {0});

            auto int_idx = svdat.getInternalIndices({0, 1});
            auto ext_idx = svdat.getExternalIndices({0, 1});

            svdat.sv.applyCRot(int_idx, ext_idx, false, angles[0], angles[1],
                               angles[2]);

            CHECK(isApproxEqual(svdat.cdata, expected_results));
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("CRot0,1 |100> -> |1>(a|0>+b|1>)|0>") {
            SVData<TestType> svdat{num_qubits};
            svdat.sv.applyOperation("PauliX", {0});

            svdat.sv.applyOperation("CRot", {0, 1}, false, angles);
            CHECK(isApproxEqual(svdat.cdata, expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyToffoli", "[StateVector]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};

    // Test using |+10> state
    svdat.sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                             {false, false});
    const auto init_state = svdat.cdata;

    SECTION("Apply directly") {
        SECTION("Toffoli 0,1,2 |+10> -> |010> + |111>") {
            std::vector<cp_t> expected{
                {0, 0}, {0, 0}, {1 / sqrt(2), 0}, {0, 0},
                {0, 0}, {0, 0}, {0, 0},           {1 / sqrt(2), 0}};

            SVData<TestType> svdat012{num_qubits, init_state};

            svdat012.sv.applyToffoli(svdat.getInternalIndices({0, 1, 2}),
                                     svdat.getExternalIndices({0, 1, 2}),
                                     false);

            CHECK(svdat012.cdata == expected);
        }

        SECTION("Toffoli 1,0,2 |+10> -> |010> + |111>") {
            std::vector<cp_t> expected{
                {0, 0}, {0, 0}, {1 / sqrt(2), 0}, {0, 0},
                {0, 0}, {0, 0}, {0, 0},           {1 / sqrt(2), 0}};

            SVData<TestType> svdat102{num_qubits, init_state};

            svdat102.sv.applyToffoli(svdat.getInternalIndices({1, 0, 2}),
                                     svdat.getExternalIndices({1, 0, 2}),
                                     false);

            CHECK(svdat102.cdata == expected);
        }
        SECTION("Toffoli 0,2,1 |+10> -> |+10>") {
            std::vector<cp_t> expected{init_state};

            SVData<TestType> svdat021{num_qubits, init_state};

            svdat021.sv.applyToffoli(svdat.getInternalIndices({0, 2, 1}),
                                     svdat.getExternalIndices({0, 2, 1}),
                                     false);

            CHECK(svdat021.cdata == expected);
        }
        SECTION("Toffoli 1,2,0 |+10> -> |+10>") {
            std::vector<cp_t> expected{init_state};

            SVData<TestType> svdat120{num_qubits, init_state};

            svdat120.sv.applyToffoli(svdat.getInternalIndices({1, 2, 0}),
                                     svdat.getExternalIndices({1, 2, 0}),
                                     false);

            CHECK(svdat120.cdata == expected);
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("Toffoli [0,1,2], [1,0,2] |+10> -> |+1+>") {

            std::vector<cp_t> expected{
                {0, 0}, {0, 0}, {1 / sqrt(2), 0}, {0, 0},
                {0, 0}, {0, 0}, {0, 0},           {1 / sqrt(2), 0}};

            SVData<TestType> svdat012{num_qubits, init_state};
            SVData<TestType> svdat102{num_qubits, init_state};

            svdat012.sv.applyOperation("Toffoli", {0, 1, 2});
            svdat102.sv.applyOperation("Toffoli", {1, 0, 2});

            CHECK(svdat012.cdata == expected);
            CHECK(svdat102.cdata == expected);
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyCSWAP", "[StateVector]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};

    // Test using |+10> state
    svdat.sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                             {false, false});
    const auto init_state = svdat.cdata;

    SECTION("Apply directly") {
        SECTION("CSWAP 0,1,2 |+10> -> |010> + |101>") {
            std::vector<cp_t> expected{{0, 0}, {0, 0}, {1 / sqrt(2), 0},
                                       {0, 0}, {0, 0}, {1 / sqrt(2), 0},
                                       {0, 0}, {0, 0}};

            SVData<TestType> svdat012{num_qubits, init_state};

            svdat012.sv.applyCSWAP(svdat.getInternalIndices({0, 1, 2}),
                                   svdat.getExternalIndices({0, 1, 2}), false);

            CHECK(svdat012.cdata == expected);
        }

        SECTION("CSWAP 1,0,2 |+10> -> |01+>") {
            std::vector<cp_t> expected{
                {0, 0}, {0, 0}, {1 / sqrt(2), 0}, {1 / sqrt(2), 0},
                {0, 0}, {0, 0}, {0, 0},           {0, 0}};

            SVData<TestType> svdat102{num_qubits, init_state};

            svdat102.sv.applyCSWAP(svdat.getInternalIndices({1, 0, 2}),
                                   svdat.getExternalIndices({1, 0, 2}), false);

            CHECK(svdat102.cdata == expected);
        }
        SECTION("CSWAP 2,1,0 |+10> -> |+10>") {
            std::vector<cp_t> expected{init_state};

            SVData<TestType> svdat021{num_qubits, init_state};

            svdat021.sv.applyCSWAP(svdat.getInternalIndices({2, 1, 0}),
                                   svdat.getExternalIndices({2, 1, 0}), false);

            CHECK(svdat021.cdata == expected);
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("CSWAP 0,1,2 |+10> -> |010> + |101>") {

            std::vector<cp_t> expected{{0, 0}, {0, 0}, {1 / sqrt(2), 0},
                                       {0, 0}, {0, 0}, {1 / sqrt(2), 0},
                                       {0, 0}, {0, 0}};

            SVData<TestType> svdat012{num_qubits, init_state};

            svdat012.sv.applyOperation("CSWAP", {0, 1, 2});

            CHECK(svdat012.cdata == expected);
        }
    }
}