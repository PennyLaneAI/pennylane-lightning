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
TEMPLATE_TEST_CASE("StateVector::StateVector", "[StateVector]", float, double) {
    SECTION("StateVector") {
        REQUIRE(std::is_constructible<StateVector<>>::value);
    }
    SECTION("StateVector<TestType> {}") {
        REQUIRE(std::is_constructible<StateVector<TestType>>::value);
    }
    SECTION("StateVector<TestType> {std::complex<TestType>, size_t}") {
        REQUIRE(std::is_constructible<StateVector<TestType>,
                                      std::complex<TestType> *, size_t>::value);
    }
    SECTION("StateVector<TestType> cross types") {
        if constexpr (!std::is_same_v<TestType, double>) {
            REQUIRE_FALSE(
                std::is_constructible<StateVector<TestType>,
                                      std::complex<double> *, size_t>::value);
            REQUIRE_FALSE(
                std::is_constructible<StateVector<double>,
                                      std::complex<TestType> *, size_t>::value);
        } else if constexpr (!std::is_same_v<TestType, float>) {
            REQUIRE_FALSE(
                std::is_constructible<StateVector<TestType>,
                                      std::complex<float> *, size_t>::value);
            REQUIRE_FALSE(
                std::is_constructible<StateVector<float>,
                                      std::complex<TestType> *, size_t>::value);
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

    explicit SVData(size_t num_qubits)
        : num_qubits{num_qubits}, // qubit_indices{num_qubits},
          cdata(0b1 << num_qubits), sv{cdata.data(), cdata.size()} {
        cdata[0] = std::complex<fp_t>{1, 0};
    }
    explicit SVData(size_t num_qubits,
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

TEST_CASE("StateVector::generateBitPatterns", "[StateVector]") {
    const size_t num_qubits = 4;
    SECTION("Qubit indices {}") {
        auto bit_pattern = StateVector<>::generateBitPatterns({}, num_qubits);
        CHECK(bit_pattern == std::vector<size_t>{0});
    }
    SECTION("Qubit indices {i}") {
        for (size_t i = 0; i < num_qubits; i++) {
            std::vector<size_t> expected{0, 0b1UL << (num_qubits - i - 1)};
            auto bit_pattern =
                StateVector<>::generateBitPatterns({i}, num_qubits);
            CHECK(bit_pattern == expected);
        }
    }
    SECTION("Qubit indices {i,i+1,i+2}") {
        std::vector<size_t> expected_123{0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<size_t> expected_012{0, 2, 4, 6, 8, 10, 12, 14};
        auto bit_pattern_123 =
            StateVector<>::generateBitPatterns({1, 2, 3}, num_qubits);
        auto bit_pattern_012 =
            StateVector<>::generateBitPatterns({0, 1, 2}, num_qubits);

        CHECK(bit_pattern_123 == expected_123);
        CHECK(bit_pattern_012 == expected_012);
    }
    SECTION("Qubit indices {0,2,3}") {
        std::vector<size_t> expected{0, 1, 2, 3, 8, 9, 10, 11};
        auto bit_pattern =
            StateVector<>::generateBitPatterns({0, 2, 3}, num_qubits);

        CHECK(bit_pattern == expected);
    }
    SECTION("Qubit indices {3,1,0}") {
        std::vector<size_t> expected{0, 8, 4, 12, 1, 9, 5, 13};
        auto bit_pattern =
            StateVector<>::generateBitPatterns({3, 1, 0}, num_qubits);
        CHECK(bit_pattern == expected);
    }
}

TEST_CASE("StateVector::getIndicesAfterExclusion", "[StateVector]") {
    const size_t num_qubits = 4;
    SECTION("Qubit indices {}") {
        std::vector<size_t> expected{0, 1, 2, 3};
        auto indices = StateVector<>::getIndicesAfterExclusion({}, num_qubits);
        CHECK(indices == expected);
    }
    SECTION("Qubit indices {i}") {
        for (size_t i = 0; i < num_qubits; i++) {
            std::vector<size_t> expected{0, 1, 2, 3};
            expected.erase(expected.begin() + i);

            auto indices =
                StateVector<>::getIndicesAfterExclusion({i}, num_qubits);
            CHECK(indices == expected);
        }
    }
    SECTION("Qubit indices {i,i+1,i+2}") {
        std::vector<size_t> expected_123{0};
        std::vector<size_t> expected_012{3};
        auto indices_123 =
            StateVector<>::getIndicesAfterExclusion({1, 2, 3}, num_qubits);
        auto indices_012 =
            StateVector<>::getIndicesAfterExclusion({0, 1, 2}, num_qubits);

        CHECK(indices_123 == expected_123);
        CHECK(indices_012 == expected_012);
    }
    SECTION("Qubit indices {0,2,3}") {
        std::vector<size_t> expected{1};
        auto indices =
            StateVector<>::getIndicesAfterExclusion({0, 2, 3}, num_qubits);

        CHECK(indices == expected);
    }
    SECTION("Qubit indices {3,1,0}") {
        std::vector<size_t> expected{2};
        auto indices =
            StateVector<>::getIndicesAfterExclusion({3, 1, 0}, num_qubits);

        CHECK(indices == expected);
    }
}

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
            CHECK(svdat.cdata[0] == Util::ONE<TestType>());
            svdat.sv.applyPauliX(int_idx, ext_idx, false);
            CHECK(svdat.cdata[0] == Util::ZERO<TestType>());
            CHECK(svdat.cdata[0b1 << (svdat.num_qubits - index - 1)] ==
                  Util::ONE<TestType>());
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat{num_qubits};
            CHECK(svdat.cdata[0] == Util::ONE<TestType>());
            svdat.sv.applyOperation("PauliX", {index}, false);
            CHECK(svdat.cdata[0] == Util::ZERO<TestType>());
            CHECK(svdat.cdata[0b1 << (svdat.num_qubits - index - 1)] ==
                  Util::ONE<TestType>());
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

    constexpr cp_t p = Util::ConstMult(
        static_cast<TestType>(0.5),
        Util::ConstMult(Util::INVSQRT2<TestType>(), Util::IMAG<TestType>()));
    constexpr cp_t m = Util::ConstMult(-1, p);

    const std::vector<std::vector<cp_t>> expected_results = {
        {m, m, m, m, p, p, p, p},
        {m, m, p, p, m, m, p, p},
        {m, p, m, p, m, p, m, p}};

    const auto init_state = svdat.cdata;
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat_direct{num_qubits, init_state};
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});

            CHECK(svdat_direct.cdata == init_state);
            svdat_direct.sv.applyPauliY(int_idx, ext_idx, false);

            CHECK(isApproxEqual(svdat_direct.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat_dispatch{num_qubits, init_state};
            CHECK(svdat_dispatch.cdata == init_state);
            svdat_dispatch.sv.applyOperation("PauliY", {index}, false);
            CHECK(isApproxEqual(svdat_dispatch.cdata, expected_results[index]));
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

    constexpr cp_t p = static_cast<TestType>(0.5) * Util::INVSQRT2<TestType>();
    constexpr cp_t m = Util::ConstMult(-1, p);

    const std::vector<std::vector<cp_t>> expected_results = {
        {p, p, p, p, m, m, m, m},
        {p, p, m, m, p, p, m, m},
        {p, m, p, m, p, m, p, m}};

    const auto init_state = svdat.cdata;
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat_direct{num_qubits, init_state};
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});

            CHECK(svdat_direct.cdata == init_state);
            svdat_direct.sv.applyPauliZ(int_idx, ext_idx, false);

            CHECK(isApproxEqual(svdat_direct.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat_dispatch{num_qubits, init_state};
            CHECK(svdat_dispatch.cdata == init_state);
            svdat_dispatch.sv.applyOperation("PauliZ", {index}, false);
            CHECK(isApproxEqual(svdat_dispatch.cdata, expected_results[index]));
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

    constexpr cp_t r = static_cast<TestType>(0.5) * Util::INVSQRT2<TestType>();
    constexpr cp_t i = Util::ConstMult(r, Util::IMAG<TestType>());

    const std::vector<std::vector<cp_t>> expected_results = {
        {r, r, r, r, i, i, i, i},
        {r, r, i, i, r, r, i, i},
        {r, i, r, i, r, i, r, i}};

    const auto init_state = svdat.cdata;
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat_direct{num_qubits, init_state};
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});

            CHECK(svdat_direct.cdata == init_state);
            svdat_direct.sv.applyS(int_idx, ext_idx, false);

            CHECK(isApproxEqual(svdat_direct.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat_dispatch{num_qubits, init_state};
            CHECK(svdat_dispatch.cdata == init_state);
            svdat_dispatch.sv.applyOperation("S", {index}, false);
            CHECK(isApproxEqual(svdat_dispatch.cdata, expected_results[index]));
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
            SVData<TestType> svdat_direct{num_qubits, init_state};
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});

            CHECK(svdat_direct.cdata == init_state);
            svdat_direct.sv.applyT(int_idx, ext_idx, false);

            CHECK(isApproxEqual(svdat_direct.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat_dispatch{num_qubits, init_state};
            CHECK(svdat_dispatch.cdata == init_state);
            svdat_dispatch.sv.applyOperation("T", {index}, false);
            CHECK(isApproxEqual(svdat_dispatch.cdata, expected_results[index]));
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
        const auto rx_mat = Gates::getRX<TestType>(angles[i]);
        expected_results[i][0] = rx_mat[0];
        expected_results[i][0b1 << (num_qubits - i - 1)] = rx_mat[1];
    }

    const auto init_state = svdat.cdata;
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat_direct{num_qubits};
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});

            svdat_direct.sv.applyRX(int_idx, ext_idx, false, {angles[index]});

            CHECK(isApproxEqual(svdat_direct.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat_dispatch{num_qubits};
            svdat_dispatch.sv.applyOperation("RX", {index}, false,
                                             {angles[index]});
            CHECK(isApproxEqual(svdat_dispatch.cdata, expected_results[index]));
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
        const auto ry_mat = Gates::getRY<TestType>(angles[i]);
        expected_results[i][0] = ry_mat[0];
        expected_results[i][0b1 << (num_qubits - i - 1)] = ry_mat[2];
    }

    const auto init_state = svdat.cdata;
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat_direct{num_qubits};
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});

            svdat_direct.sv.applyRY(int_idx, ext_idx, false, {angles[index]});

            CHECK(isApproxEqual(svdat_direct.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat_dispatch{num_qubits};
            svdat_dispatch.sv.applyOperation("RY", {index}, false,
                                             {angles[index]});
            CHECK(isApproxEqual(svdat_dispatch.cdata, expected_results[index]));
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
        rz_data.push_back(Gates::getRZ<TestType>(a));
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
            SVData<TestType> svdat_direct{num_qubits, init_state};
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});

            svdat_direct.sv.applyRZ(int_idx, ext_idx, false, {angles[index]});

            CHECK(isApproxEqual(svdat_direct.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat_dispatch{num_qubits, init_state};
            svdat_dispatch.sv.applyOperation("RZ", {index}, false,
                                             {angles[index]});
            CHECK(isApproxEqual(svdat_dispatch.cdata, expected_results[index]));
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
        ps_data.push_back(Gates::getPhaseShift<TestType>(a));
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
            SVData<TestType> svdat_direct{num_qubits, init_state};
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});

            svdat_direct.sv.applyPhaseShift(int_idx, ext_idx, false,
                                            {angles[index]});

            CHECK(isApproxEqual(svdat_direct.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat_dispatch{num_qubits, init_state};
            svdat_dispatch.sv.applyOperation("PhaseShift", {index}, false,
                                             {angles[index]});
            CHECK(isApproxEqual(svdat_dispatch.cdata, expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyControlledPhaseShift", "[StateVector]",
                   float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};

    // Test using |+++> state
    svdat.sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                             {{0}, {1}, {2}}, {{false}, {false}, {false}});

    const std::vector<TestType> angles{0.3, 2.4};
    const cp_t coef = {1 / (2 * std::sqrt(2)), 0};

    std::vector<std::vector<cp_t>> ps_data;
    for (auto &a : angles) {
        ps_data.push_back(Gates::getPhaseShift<TestType>(a));
    }

    std::vector<std::vector<cp_t>> expected_results = {
        {ps_data[0][0], ps_data[0][0], ps_data[0][0], ps_data[0][0],
         ps_data[0][0], ps_data[0][0], ps_data[0][3], ps_data[0][3]},
        {ps_data[1][0], ps_data[1][0], ps_data[1][0], ps_data[1][3],
         ps_data[1][0], ps_data[1][0], ps_data[1][0], ps_data[1][3]}};

    for (auto &vec : expected_results) {
        scaleVector(vec, coef);
    }

    const auto init_state = svdat.cdata;
    SECTION("Apply directly") {
        SVData<TestType> svdat_direct{num_qubits, init_state};
        auto int_idx = svdat_direct.getInternalIndices({0, 1});
        auto ext_idx = svdat_direct.getExternalIndices({0, 1});

        svdat_direct.sv.applyControlledPhaseShift(int_idx, ext_idx, false,
                                                  {angles[0]});
        CAPTURE(svdat_direct.cdata);
        CHECK(isApproxEqual(svdat_direct.cdata, expected_results[0]));
    }
    SECTION("Apply using dispatcher") {
        SVData<TestType> svdat_dispatch{num_qubits, init_state};
        svdat_dispatch.sv.applyOperation("ControlledPhaseShift", {1, 2}, false,
                                         {angles[1]});
        CAPTURE(svdat_dispatch.cdata);
        CHECK(isApproxEqual(svdat_dispatch.cdata, expected_results[1]));
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
        const auto rot_mat =
            Gates::getRot<TestType>(angles[i][0], angles[i][1], angles[i][2]);
        expected_results[i][0] = rot_mat[0];
        expected_results[i][0b1 << (num_qubits - i - 1)] = rot_mat[2];
    }

    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat_direct{num_qubits};
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});
            svdat_direct.sv.applyRot(int_idx, ext_idx, false, angles[index][0],
                                     angles[index][1], angles[index][2]);

            CHECK(isApproxEqual(svdat_direct.cdata, expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            SVData<TestType> svdat_dispatch{num_qubits};
            svdat_dispatch.sv.applyOperation("Rot", {index}, false,
                                             angles[index]);
            CHECK(isApproxEqual(svdat_dispatch.cdata, expected_results[index]));
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
        SVData<TestType> svdat_direct{num_qubits, init_state};

        for (size_t index = 1; index < num_qubits; index++) {
            auto int_idx = svdat_direct.getInternalIndices({index - 1, index});
            auto ext_idx = svdat_direct.getExternalIndices({index - 1, index});

            svdat_direct.sv.applyCNOT(int_idx, ext_idx, false);
        }
        CHECK(svdat_direct.cdata.front() == Util::INVSQRT2<TestType>());
        CHECK(svdat_direct.cdata.back() == Util::INVSQRT2<TestType>());
    }

    SECTION("Apply using dispatcher") {
        SVData<TestType> svdat_dispatch{num_qubits, init_state};

        for (size_t index = 1; index < num_qubits; index++) {
            svdat_dispatch.sv.applyOperation("CNOT", {index - 1, index}, false);
        }
        CHECK(svdat_dispatch.cdata.front() == Util::INVSQRT2<TestType>());
        CHECK(svdat_dispatch.cdata.back() == Util::INVSQRT2<TestType>());
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
        CHECK(svdat.cdata ==
              std::vector<cp_t>{
                  Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                  Util::INVSQRT2<TestType>(), Util::ZERO<TestType>(),
                  Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                  Util::INVSQRT2<TestType>(), Util::ZERO<TestType>()});

        SECTION("SWAP0,1 |+10> -> |1+0>") {
            std::vector<cp_t> expected{
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                {1 / sqrt(2), 0},       Util::ZERO<TestType>(),
                {1 / sqrt(2), 0},       Util::ZERO<TestType>()};

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
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                {1 / sqrt(2), 0},       {1 / sqrt(2), 0},
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                Util::ZERO<TestType>(), Util::ZERO<TestType>()};

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
                Util::ZERO<TestType>(), {1 / sqrt(2), 0},
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                Util::ZERO<TestType>(), {1 / sqrt(2), 0},
                Util::ZERO<TestType>(), Util::ZERO<TestType>()};

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
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                {1 / sqrt(2), 0},       Util::ZERO<TestType>(),
                {1 / sqrt(2), 0},       Util::ZERO<TestType>()};

            SVData<TestType> svdat01{num_qubits, init_state};
            SVData<TestType> svdat10{num_qubits, init_state};

            svdat01.sv.applyOperation("SWAP", {0, 1});
            svdat10.sv.applyOperation("SWAP", {1, 0});

            CHECK(svdat01.cdata == expected);
            CHECK(svdat10.cdata == expected);
        }

        SECTION("SWAP0,2 |+10> -> |01+>") {
            std::vector<cp_t> expected{
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                {1 / sqrt(2), 0},       {1 / sqrt(2), 0},
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                Util::ZERO<TestType>(), Util::ZERO<TestType>()};

            SVData<TestType> svdat02{num_qubits, init_state};
            SVData<TestType> svdat20{num_qubits, init_state};

            svdat02.sv.applyOperation("SWAP", {0, 2});
            svdat20.sv.applyOperation("SWAP", {2, 0});

            CHECK(svdat02.cdata == expected);
            CHECK(svdat20.cdata == expected);
        }
        SECTION("SWAP1,2 |+10> -> |+01>") {
            std::vector<cp_t> expected{
                Util::ZERO<TestType>(), {1 / sqrt(2), 0},
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                Util::ZERO<TestType>(), {1 / sqrt(2), 0},
                Util::ZERO<TestType>(), Util::ZERO<TestType>()};

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
        CHECK(svdat.cdata == std::vector<cp_t>{Util::ZERO<TestType>(),
                                               Util::ZERO<TestType>(),
                                               {1 / sqrt(2), 0},
                                               Util::ZERO<TestType>(),
                                               Util::ZERO<TestType>(),
                                               Util::ZERO<TestType>(),
                                               {1 / sqrt(2), 0},
                                               Util::ZERO<TestType>()});

        SECTION("CZ0,1 |+10> -> |-10>") {
            std::vector<cp_t> expected{
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                {1 / sqrt(2), 0},       Util::ZERO<TestType>(),
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                {-1 / sqrt(2), 0},      Util::ZERO<TestType>()};

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
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                {1 / sqrt(2), 0},       Util::ZERO<TestType>(),
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                {-1 / sqrt(2), 0},      Util::ZERO<TestType>()};

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
    const auto rot_mat =
        Gates::getRot<TestType>(angles[0], angles[1], angles[2]);
    expected_results[0b1 << (num_qubits - 1)] = rot_mat[0];
    expected_results[(0b1 << num_qubits) - 2] = rot_mat[2];

    const auto init_state = svdat.cdata;

    SECTION("Apply directly") {
        SECTION("CRot0,1 |000> -> |000>") {
            SVData<TestType> svdat_direct{num_qubits};
            auto int_idx = svdat_direct.getInternalIndices({0, 1});
            auto ext_idx = svdat_direct.getExternalIndices({0, 1});
            svdat_direct.sv.applyCRot(int_idx, ext_idx, false, angles[0],
                                      angles[1], angles[2]);

            CHECK(isApproxEqual(svdat_direct.cdata, init_state));
        }
        SECTION("CRot0,1 |100> -> |1>(a|0>+b|1>)|0>") {
            SVData<TestType> svdat_direct{num_qubits};
            svdat_direct.sv.applyOperation("PauliX", {0});

            auto int_idx = svdat_direct.getInternalIndices({0, 1});
            auto ext_idx = svdat_direct.getExternalIndices({0, 1});

            svdat_direct.sv.applyCRot(int_idx, ext_idx, false, angles[0],
                                      angles[1], angles[2]);

            CHECK(isApproxEqual(svdat_direct.cdata, expected_results));
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("CRot0,1 |100> -> |1>(a|0>+b|1>)|0>") {
            SVData<TestType> svdat_direct{num_qubits};
            svdat_direct.sv.applyOperation("PauliX", {0});

            svdat_direct.sv.applyOperation("CRot", {0, 1}, false, angles);
            CHECK(isApproxEqual(svdat_direct.cdata, expected_results));
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
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                {1 / sqrt(2), 0},       Util::ZERO<TestType>(),
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                Util::ZERO<TestType>(), {1 / sqrt(2), 0}};

            SVData<TestType> svdat012{num_qubits, init_state};

            svdat012.sv.applyToffoli(svdat.getInternalIndices({0, 1, 2}),
                                     svdat.getExternalIndices({0, 1, 2}),
                                     false);

            CHECK(svdat012.cdata == expected);
        }

        SECTION("Toffoli 1,0,2 |+10> -> |010> + |111>") {
            std::vector<cp_t> expected{
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                {1 / sqrt(2), 0},       Util::ZERO<TestType>(),
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                Util::ZERO<TestType>(), {1 / sqrt(2), 0}};

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
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                {1 / sqrt(2), 0},       Util::ZERO<TestType>(),
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                Util::ZERO<TestType>(), {1 / sqrt(2), 0}};

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
            std::vector<cp_t> expected{
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                {1 / sqrt(2), 0},       Util::ZERO<TestType>(),
                Util::ZERO<TestType>(), {1 / sqrt(2), 0},
                Util::ZERO<TestType>(), Util::ZERO<TestType>()};
            SVData<TestType> svdat012{num_qubits, init_state};

            svdat012.sv.applyCSWAP(svdat.getInternalIndices({0, 1, 2}),
                                   svdat.getExternalIndices({0, 1, 2}), false);

            CHECK(svdat012.cdata == expected);
        }

        SECTION("CSWAP 1,0,2 |+10> -> |01+>") {
            std::vector<cp_t> expected{
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                {1 / sqrt(2), 0},       {1 / sqrt(2), 0},
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                Util::ZERO<TestType>(), Util::ZERO<TestType>()};

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
            std::vector<cp_t> expected{
                Util::ZERO<TestType>(), Util::ZERO<TestType>(),
                {1 / sqrt(2), 0},       Util::ZERO<TestType>(),
                Util::ZERO<TestType>(), {1 / sqrt(2), 0},
                Util::ZERO<TestType>(), Util::ZERO<TestType>()};
            SVData<TestType> svdat012{num_qubits, init_state};

            svdat012.sv.applyOperation("CSWAP", {0, 1, 2});

            CHECK(svdat012.cdata == expected);
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyMatrix 1 wire", "[StateVector]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 5;

    // Note: gates are defined as right-to-left order

    SECTION("Apply XZ gate") {
        const std::vector<cp_t> xz_gate{
            Util::ZERO<TestType>(), Util::ONE<TestType>(),
            -Util::ONE<TestType>(), Util::ZERO<TestType>()};

        SECTION("Apply directly") {
            SVData<TestType> svdat{num_qubits};
            SVData<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                auto int_idx = svdat.getInternalIndices({index});
                auto ext_idx = svdat.getExternalIndices({index});
                svdat.sv.applyMatrix(xz_gate, int_idx, ext_idx, false);

                svdat_expected.sv.applyPauliX(int_idx, ext_idx, false);
                svdat_expected.sv.applyPauliZ(int_idx, ext_idx, false);
            }

            CHECK(isApproxEqual(svdat.cdata, svdat_expected.cdata));
        }
        SECTION("Apply using dispatcher") {
            SVData<TestType> svdat{num_qubits};
            SVData<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                svdat_expected.sv.applyOperations({{"PauliX"}, {"PauliZ"}},
                                                  {{index}, {index}},
                                                  {false, false});
                svdat.sv.applyOperation(xz_gate, {index}, false);
            }

            CHECK(isApproxEqual(svdat.cdata, svdat_expected.cdata));
        }
    }
    SECTION("Apply ZX gate") {
        const std::vector<cp_t> zx_gate{
            Util::ZERO<TestType>(), -Util::ONE<TestType>(),
            Util::ONE<TestType>(), Util::ZERO<TestType>()};
        SECTION("Apply directly") {
            SVData<TestType> svdat{num_qubits};
            SVData<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                auto int_idx = svdat.getInternalIndices({index});
                auto ext_idx = svdat.getExternalIndices({index});
                svdat.sv.applyMatrix(zx_gate, int_idx, ext_idx, false);

                svdat_expected.sv.applyPauliZ(int_idx, ext_idx, false);
                svdat_expected.sv.applyPauliX(int_idx, ext_idx, false);
            }
            CHECK(isApproxEqual(svdat.cdata, svdat_expected.cdata));
        }
        SECTION("Apply using dispatcher") {
            SVData<TestType> svdat{num_qubits};
            SVData<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                svdat_expected.sv.applyOperations({{"PauliZ"}, {"PauliX"}},
                                                  {{index}, {index}},
                                                  {false, false});
                svdat.sv.applyOperation(zx_gate, {index}, false);
            }
            CHECK(isApproxEqual(svdat.cdata, svdat_expected.cdata));
        }
    }
    SECTION("Apply XY gate") {
        const std::vector<cp_t> xy_gate{
            -Util::IMAG<TestType>(), Util::ZERO<TestType>(),
            Util::ZERO<TestType>(), Util::IMAG<TestType>()};
        SECTION("Apply directly") {
            SVData<TestType> svdat{num_qubits};
            SVData<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                auto int_idx = svdat.getInternalIndices({index});
                auto ext_idx = svdat.getExternalIndices({index});
                svdat.sv.applyMatrix(xy_gate, int_idx, ext_idx, false);

                svdat_expected.sv.applyPauliX(int_idx, ext_idx, false);
                svdat_expected.sv.applyPauliY(int_idx, ext_idx, false);
            }
            CHECK(isApproxEqual(svdat.cdata, svdat_expected.cdata));
        }
        SECTION("Apply using dispatcher") {
            SVData<TestType> svdat{num_qubits};
            SVData<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                svdat_expected.sv.applyOperations({{"PauliX"}, {"PauliY"}},
                                                  {{index}, {index}},
                                                  {false, false});
                svdat.sv.applyOperation(xy_gate, {index}, false);
            }
            CHECK(isApproxEqual(svdat.cdata, svdat_expected.cdata));
        }
    }
    SECTION("Apply YX gate") {
        const std::vector<cp_t> yx_gate{
            Util::IMAG<TestType>(), Util::ZERO<TestType>(),
            Util::ZERO<TestType>(), -Util::IMAG<TestType>()};
        SECTION("Apply directly") {
            SVData<TestType> svdat{num_qubits};
            SVData<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                auto int_idx = svdat.getInternalIndices({index});
                auto ext_idx = svdat.getExternalIndices({index});
                svdat.sv.applyMatrix(yx_gate, int_idx, ext_idx, false);

                svdat_expected.sv.applyPauliY(int_idx, ext_idx, false);
                svdat_expected.sv.applyPauliX(int_idx, ext_idx, false);
            }
            CHECK(isApproxEqual(svdat.cdata, svdat_expected.cdata));
        }
        SECTION("Apply using dispatcher") {
            SVData<TestType> svdat{num_qubits};
            SVData<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                svdat_expected.sv.applyOperations({{"PauliY"}, {"PauliX"}},
                                                  {{index}, {index}},
                                                  {false, false});
                svdat.sv.applyOperation(yx_gate, {index}, false);
            }
            CHECK(isApproxEqual(svdat.cdata, svdat_expected.cdata));
        }
    }
    SECTION("Apply YZ gate") {
        const std::vector<cp_t> yz_gate{
            Util::ZERO<TestType>(), -Util::IMAG<TestType>(),
            -Util::IMAG<TestType>(), Util::ZERO<TestType>()};
        SECTION("Apply directly") {
            SVData<TestType> svdat{num_qubits};
            SVData<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                auto int_idx = svdat.getInternalIndices({index});
                auto ext_idx = svdat.getExternalIndices({index});
                svdat.sv.applyMatrix(yz_gate, int_idx, ext_idx, false);

                svdat_expected.sv.applyPauliY(int_idx, ext_idx, false);
                svdat_expected.sv.applyPauliZ(int_idx, ext_idx, false);
            }
            CHECK(isApproxEqual(svdat.cdata, svdat_expected.cdata));
        }
        SECTION("Apply using dispatcher") {
            SVData<TestType> svdat{num_qubits};
            SVData<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                svdat_expected.sv.applyOperations({{"PauliY"}, {"PauliZ"}},
                                                  {{index}, {index}},
                                                  {false, false});
                svdat.sv.applyOperation(yz_gate, {index}, false);
            }
            CHECK(isApproxEqual(svdat.cdata, svdat_expected.cdata));
        }
    }
    SECTION("Apply ZY gate") {
        const std::vector<cp_t> zy_gate{
            Util::ZERO<TestType>(), Util::IMAG<TestType>(),
            Util::IMAG<TestType>(), Util::ZERO<TestType>()};
        SECTION("Apply directly") {
            SVData<TestType> svdat{num_qubits};
            SVData<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                auto int_idx = svdat.getInternalIndices({index});
                auto ext_idx = svdat.getExternalIndices({index});
                svdat.sv.applyMatrix(zy_gate, int_idx, ext_idx, false);

                svdat_expected.sv.applyPauliZ(int_idx, ext_idx, false);
                svdat_expected.sv.applyPauliY(int_idx, ext_idx, false);
            }
            CHECK(isApproxEqual(svdat.cdata, svdat_expected.cdata));
        }
        SECTION("Apply using dispatcher") {
            SVData<TestType> svdat{num_qubits};
            SVData<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                svdat_expected.sv.applyOperations({{"PauliZ"}, {"PauliY"}},
                                                  {{index}, {index}},
                                                  {false, false});
                svdat.sv.applyOperation(zy_gate, {index}, false);
            }
            CHECK(isApproxEqual(svdat.cdata, svdat_expected.cdata));
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyMatrix multiple wires", "[StateVector]",
                   float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;

    SVData<TestType> svdat_init{num_qubits};
    svdat_init.sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                  {{0}, {1}, {2}}, {false, false, false});

    const auto cz_gate = Gates::getCZ<TestType>();
    const auto tof_gate = Gates::getToffoli<TestType>();
    const auto arb_gate = Gates::getToffoli<TestType>();

    SECTION("Apply CZ gate") {
        SVData<TestType> svdat{num_qubits, svdat_init.cdata};
        SVData<TestType> svdat_expected{num_qubits, svdat_init.cdata};

        svdat_expected.sv.applyOperations(
            {{"Hadamard"}, {"CNOT"}, {"Hadamard"}}, {{1}, {0, 1}, {1}},
            {false, false, false});
        svdat.sv.applyOperation(cz_gate, {0, 1}, false);

        CHECK(isApproxEqual(svdat.cdata, svdat_expected.cdata));
    }
}
