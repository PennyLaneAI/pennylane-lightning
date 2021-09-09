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

namespace {

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
        : num_qubits{num_qubits},
          cdata(0b1 << num_qubits, std::complex<fp_t>{0, 0}),
          sv{StateVector<fp_t>(nullptr,
                               static_cast<size_t>(Util::exp2(num_qubits)))} {
        cdata[0] = std::complex<fp_t>{1, 0};
        sv.setData(cdata.data());
    }
    explicit SVData(size_t num_qubits,
                    const std::vector<std::complex<fp_t>> &cdata_input)
        : num_qubits{num_qubits}, cdata{cdata_input.begin(), cdata_input.end()},
          sv{StateVector<fp_t>(nullptr,
                               static_cast<size_t>(Util::exp2(num_qubits)))} {
        sv.setData(cdata.data());
    }
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
    void setCData() {}
};
} // namespace

TEMPLATE_TEST_CASE("StateVector::applyRX", "[StateVector_Param]", double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 1;
    SVData<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{{0.1}, {0.6}};
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.9987502603949663, 0.0},
                          {0.0, -0.04997916927067834}},
        std::vector<cp_t>{{0.9553364891256061, 0.0}, {0, -0.2955202066613395}},
        std::vector<cp_t>{{0.49757104789172696, 0.0}, {0, -0.867423225594017}}};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>{{0.9987502603949663, 0.0},
                          {0.0, 0.04997916927067834}},
        std::vector<cp_t>{{0.9553364891256061, 0.0}, {0, 0.2955202066613395}},
        std::vector<cp_t>{{0.49757104789172696, 0.0}, {0, 0.867423225594017}}};

    const auto init_state = svdat.cdata;
    SECTION("adj = false") {
        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVData<TestType> svdat_direct{num_qubits, init_state};

                auto int_idx = svdat_direct.getInternalIndices({0});
                auto ext_idx = svdat_direct.getExternalIndices({0});

                svdat_direct.sv.applyRX(int_idx, ext_idx, false, angles[index]);

                CHECK(isApproxEqual(svdat_direct.cdata, expected_results[index],
                                    1e-7));
            }
        }
        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVData<TestType> svdat_dispatch{num_qubits, init_state};
                svdat_dispatch.sv.applyOperation("RX", {0}, false,
                                                 {angles[index]});

                CHECK(isApproxEqual(svdat_dispatch.cdata,
                                    expected_results[index], 1e-7));
            }
        }
    }
    SECTION("adj = true") {
        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVData<TestType> svdat_direct{num_qubits, init_state};
                auto int_idx = svdat_direct.getInternalIndices({0});
                auto ext_idx = svdat_direct.getExternalIndices({0});

                svdat_direct.sv.applyRX(int_idx, ext_idx, true,
                                        {angles[index]});
                CHECK(isApproxEqual(svdat_direct.cdata,
                                    expected_results_adj[index], 1e-7));
            }
        }
        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVData<TestType> svdat_dispatch{num_qubits, init_state};
                svdat_dispatch.sv.applyOperation("RX", {0}, true,
                                                 {angles[index]});
                CHECK(isApproxEqual(svdat_dispatch.cdata,
                                    expected_results_adj[index], 1e-7));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyRY", "[StateVector_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 1;
    SVData<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.2, 0.7, 2.9};
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.8731983044562817, 0.04786268954660339},
                          {0.0876120655431924, -0.47703040785184303}},
        std::vector<cp_t>{{0.8243771119105122, 0.16439396602553008},
                          {0.3009211363333468, -0.45035926880694604}},
        std::vector<cp_t>{{0.10575112905629831, 0.47593196040758534},
                          {0.8711876098966215, -0.0577721051072477}}};
    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>{{0.8731983044562817, -0.04786268954660339},
                          {-0.0876120655431924, -0.47703040785184303}},
        std::vector<cp_t>{{0.8243771119105122, -0.16439396602553008},
                          {-0.3009211363333468, -0.45035926880694604}},
        std::vector<cp_t>{{0.10575112905629831, -0.47593196040758534},
                          {-0.8711876098966215, -0.0577721051072477}}};

    const std::vector<cp_t> init_state{{0.8775825618903728, 0.0},
                                       {0.0, -0.47942553860420306}};
    SECTION("adj = false") {
        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVData<TestType> svdat_direct{num_qubits, init_state};
                auto int_idx = svdat_direct.getInternalIndices({0});
                auto ext_idx = svdat_direct.getExternalIndices({0});

                svdat_direct.sv.applyRY(int_idx, ext_idx, false,
                                        {angles[index]});
                CHECK(isApproxEqual(svdat_direct.cdata, expected_results[index],
                                    1e-5));
            }
        }
        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVData<TestType> svdat_dispatch{num_qubits, init_state};
                svdat_dispatch.sv.applyOperation("RY", {0}, false,
                                                 {angles[index]});
                CHECK(isApproxEqual(svdat_dispatch.cdata,
                                    expected_results[index], 1e-5));
            }
        }
    }
    SECTION("adj = true") {
        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVData<TestType> svdat_direct{num_qubits, init_state};
                auto int_idx = svdat_direct.getInternalIndices({0});
                auto ext_idx = svdat_direct.getExternalIndices({0});

                svdat_direct.sv.applyRY(int_idx, ext_idx, true,
                                        {angles[index]});
                CHECK(isApproxEqual(svdat_direct.cdata,
                                    expected_results_adj[index], 1e-5));
            }
        }
        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < angles.size(); index++) {
                SVData<TestType> svdat_dispatch{num_qubits, init_state};
                svdat_dispatch.sv.applyOperation("RY", {0}, true,
                                                 {angles[index]});
                CHECK(isApproxEqual(svdat_dispatch.cdata,
                                    expected_results_adj[index], 1e-5));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVector::applyRZ", "[StateVector_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};

    // Test using |+++> state
    svdat.sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                             {{0}, {1}, {2}}, {{false}, {false}, {false}});

    const std::vector<TestType> angles{0.2, 0.7, 2.9};
    const cp_t coef(1.0 / (2 * std::sqrt(2)), 0);

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

TEMPLATE_TEST_CASE("StateVector::applyPhaseShift", "[StateVector_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};

    // Test using |+++> state
    svdat.sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                             {{0}, {1}, {2}}, {{false}, {false}, {false}});

    const std::vector<TestType> angles{0.3, 0.8, 2.4};
    const cp_t coef(1.0 / (2 * std::sqrt(2)), 0);

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

TEMPLATE_TEST_CASE("StateVector::applyControlledPhaseShift",
                   "[StateVector_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    SVData<TestType> svdat{num_qubits};

    // Test using |+++> state
    svdat.sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                             {{0}, {1}, {2}}, {{false}, {false}, {false}});

    const std::vector<TestType> angles{0.3, 2.4};
    const cp_t coef(1.0 / (2 * std::sqrt(2)), 0);

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

TEMPLATE_TEST_CASE("StateVector::applyRot", "[StateVector_Param]", float,
                   double) {
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

TEMPLATE_TEST_CASE("StateVector::applyCRot", "[StateVector_Param]", float,
                   double) {
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

TEMPLATE_TEST_CASE("StateVector::applyMatrix 1 wire", "[StateVector_Param]",
                   float, double) {
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

TEMPLATE_TEST_CASE("StateVector::applyMatrix multiple wires",
                   "[StateVector_Param]", float, double) {
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
