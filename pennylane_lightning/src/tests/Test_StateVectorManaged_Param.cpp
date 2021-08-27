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

TEMPLATE_TEST_CASE("StateVectorManaged::applyRX", "[StateVectorManaged_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorManaged<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.1, 0.6, 2.1};
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(8), std::vector<cp_t>(8), std::vector<cp_t>(8)};

    for (size_t i = 0; i < angles.size(); i++) {
        const auto rx_mat = Gates::getRX<TestType>(angles[i]);
        expected_results[i][0] = rx_mat[0];
        expected_results[i][0b1 << (num_qubits - i - 1)] = rx_mat[1];
    }

    const auto init_state = svdat.getDataVector();
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat_direct{num_qubits};
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});

            svdat_direct.applyRX(int_idx, ext_idx, false, {angles[index]});

            CHECK(isApproxEqual(svdat_direct.getDataVector(), expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat_dispatch{num_qubits};
            svdat_dispatch.applyOperation("RX", {index}, false,
                                             {angles[index]});
            CHECK(isApproxEqual(svdat_dispatch.getDataVector(), expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyRY", "[StateVectorManaged_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorManaged<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.2, 0.7, 2.9};
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(8), std::vector<cp_t>(8), std::vector<cp_t>(8)};

    for (size_t i = 0; i < angles.size(); i++) {
        const auto ry_mat = Gates::getRY<TestType>(angles[i]);
        expected_results[i][0] = ry_mat[0];
        expected_results[i][0b1 << (num_qubits - i - 1)] = ry_mat[2];
    }

    const auto init_state = svdat.getDataVector();
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat_direct{num_qubits};
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});

            svdat_direct.applyRY(int_idx, ext_idx, false, {angles[index]});

            CHECK(isApproxEqual(svdat_direct.getDataVector(), expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat_dispatch{num_qubits};
            svdat_dispatch.applyOperation("RY", {index}, false,
                                             {angles[index]});
            CHECK(isApproxEqual(svdat_dispatch.getDataVector(), expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyRZ", "[StateVectorManaged_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorManaged<TestType> svdat{num_qubits};

    // Test using |+++> state
    svdat.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
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

    const auto init_state = svdat.getDataVector();
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat_direct{init_state};
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});

            svdat_direct.applyRZ(int_idx, ext_idx, false, {angles[index]});

            CHECK(isApproxEqual(svdat_direct.getDataVector(), expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat_dispatch{init_state};
            svdat_dispatch.applyOperation("RZ", {index}, false,
                                             {angles[index]});
            CHECK(isApproxEqual(svdat_dispatch.getDataVector(), expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyPhaseShift", "[StateVectorManaged_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorManaged<TestType> svdat{num_qubits};

    // Test using |+++> state
    svdat.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
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

    const auto init_state = svdat.getDataVector();
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat_direct{init_state};
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});

            svdat_direct.applyPhaseShift(int_idx, ext_idx, false,
                                            {angles[index]});

            CHECK(isApproxEqual(svdat_direct.getDataVector(), expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat_dispatch{init_state};
            svdat_dispatch.applyOperation("PhaseShift", {index}, false,
                                             {angles[index]});
            CHECK(isApproxEqual(svdat_dispatch.getDataVector(), expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyControlledPhaseShift",
                   "[StateVectorManaged_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorManaged<TestType> svdat{num_qubits};

    // Test using |+++> state
    svdat.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
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

    const auto init_state = svdat.getDataVector();
    SECTION("Apply directly") {
        StateVectorManaged<TestType> svdat_direct{init_state};
        auto int_idx = svdat_direct.getInternalIndices({0, 1});
        auto ext_idx = svdat_direct.getExternalIndices({0, 1});

        svdat_direct.applyControlledPhaseShift(int_idx, ext_idx, false,
                                                  {angles[0]});
        CAPTURE(svdat_direct.getDataVector());
        CHECK(isApproxEqual(svdat_direct.getDataVector(), expected_results[0]));
    }
    SECTION("Apply using dispatcher") {
        StateVectorManaged<TestType> svdat_dispatch{init_state};
        svdat_dispatch.applyOperation("ControlledPhaseShift", {1, 2}, false,
                                         {angles[1]});
        CAPTURE(svdat_dispatch.getDataVector());
        CHECK(isApproxEqual(svdat_dispatch.getDataVector(), expected_results[1]));
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyRot", "[StateVectorManaged_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorManaged<TestType> svdat{num_qubits};

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
            StateVectorManaged<TestType> svdat_direct{num_qubits};
            auto int_idx = svdat_direct.getInternalIndices({index});
            auto ext_idx = svdat_direct.getExternalIndices({index});
            svdat_direct.applyRot(int_idx, ext_idx, false, angles[index][0],
                                     angles[index][1], angles[index][2]);

            CHECK(isApproxEqual(svdat_direct.getDataVector(), expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorManaged<TestType> svdat_dispatch{num_qubits};
            svdat_dispatch.applyOperation("Rot", {index}, false,
                                             angles[index]);
            CHECK(isApproxEqual(svdat_dispatch.getDataVector(), expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyCRot", "[StateVectorManaged_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorManaged<TestType> svdat{num_qubits};

    const std::vector<TestType> angles{0.3, 0.8, 2.4};

    std::vector<cp_t> expected_results(8);
    const auto rot_mat =
        Gates::getRot<TestType>(angles[0], angles[1], angles[2]);
    expected_results[0b1 << (num_qubits - 1)] = rot_mat[0];
    expected_results[(0b1 << num_qubits) - 2] = rot_mat[2];

    const auto init_state = svdat.getDataVector();

    SECTION("Apply directly") {
        SECTION("CRot0,1 |000> -> |000>") {
            StateVectorManaged<TestType> svdat_direct{num_qubits};
            auto int_idx = svdat_direct.getInternalIndices({0, 1});
            auto ext_idx = svdat_direct.getExternalIndices({0, 1});
            svdat_direct.applyCRot(int_idx, ext_idx, false, angles[0],
                                      angles[1], angles[2]);

            CHECK(isApproxEqual(svdat_direct.getDataVector(), init_state));
        }
        SECTION("CRot0,1 |100> -> |1>(a|0>+b|1>)|0>") {
            StateVectorManaged<TestType> svdat_direct{num_qubits};
            svdat_direct.applyOperation("PauliX", {0});

            auto int_idx = svdat_direct.getInternalIndices({0, 1});
            auto ext_idx = svdat_direct.getExternalIndices({0, 1});

            svdat_direct.applyCRot(int_idx, ext_idx, false, angles[0],
                                      angles[1], angles[2]);

            CHECK(isApproxEqual(svdat_direct.getDataVector(), expected_results));
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("CRot0,1 |100> -> |1>(a|0>+b|1>)|0>") {
            StateVectorManaged<TestType> svdat_direct{num_qubits};
            svdat_direct.applyOperation("PauliX", {0});

            svdat_direct.applyOperation("CRot", {0, 1}, false, angles);
            CHECK(isApproxEqual(svdat_direct.getDataVector(), expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyMatrix 1 wire", "[StateVectorManaged_Param]",
                   float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 5;

    // Note: gates are defined as right-to-left order

    SECTION("Apply XZ gate") {
        const std::vector<cp_t> xz_gate{
            Util::ZERO<TestType>(), Util::ONE<TestType>(),
            -Util::ONE<TestType>(), Util::ZERO<TestType>()};

        SECTION("Apply directly") {
            StateVectorManaged<TestType> svdat{num_qubits};
            StateVectorManaged<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                auto int_idx = svdat.getInternalIndices({index});
                auto ext_idx = svdat.getExternalIndices({index});
                svdat.applyMatrix(xz_gate, int_idx, ext_idx, false);

                svdat_expected.applyPauliX(int_idx, ext_idx, false);
                svdat_expected.applyPauliZ(int_idx, ext_idx, false);
            }

            CHECK(isApproxEqual(svdat.getDataVector(), svdat_expected.getDataVector()));
        }
        SECTION("Apply using dispatcher") {
            StateVectorManaged<TestType> svdat{num_qubits};
            StateVectorManaged<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                svdat_expected.applyOperations({{"PauliX"}, {"PauliZ"}},
                                                  {{index}, {index}},
                                                  {false, false});
                svdat.applyOperation(xz_gate, {index}, false);
            }

            CHECK(isApproxEqual(svdat.getDataVector(), svdat_expected.getDataVector()));
        }
    }
    SECTION("Apply ZX gate") {
        const std::vector<cp_t> zx_gate{
            Util::ZERO<TestType>(), -Util::ONE<TestType>(),
            Util::ONE<TestType>(), Util::ZERO<TestType>()};
        SECTION("Apply directly") {
            StateVectorManaged<TestType> svdat{num_qubits};
            StateVectorManaged<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                auto int_idx = svdat.getInternalIndices({index});
                auto ext_idx = svdat.getExternalIndices({index});
                svdat.applyMatrix(zx_gate, int_idx, ext_idx, false);

                svdat_expected.applyPauliZ(int_idx, ext_idx, false);
                svdat_expected.applyPauliX(int_idx, ext_idx, false);
            }
            CHECK(isApproxEqual(svdat.getDataVector(), svdat_expected.getDataVector()));
        }
        SECTION("Apply using dispatcher") {
            StateVectorManaged<TestType> svdat{num_qubits};
            StateVectorManaged<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                svdat_expected.applyOperations({{"PauliZ"}, {"PauliX"}},
                                                  {{index}, {index}},
                                                  {false, false});
                svdat.applyOperation(zx_gate, {index}, false);
            }
            CHECK(isApproxEqual(svdat.getDataVector(), svdat_expected.getDataVector()));
        }
    }
    SECTION("Apply XY gate") {
        const std::vector<cp_t> xy_gate{
            -Util::IMAG<TestType>(), Util::ZERO<TestType>(),
            Util::ZERO<TestType>(), Util::IMAG<TestType>()};
        SECTION("Apply directly") {
            StateVectorManaged<TestType> svdat{num_qubits};
            StateVectorManaged<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                auto int_idx = svdat.getInternalIndices({index});
                auto ext_idx = svdat.getExternalIndices({index});
                svdat.applyMatrix(xy_gate, int_idx, ext_idx, false);

                svdat_expected.applyPauliX(int_idx, ext_idx, false);
                svdat_expected.applyPauliY(int_idx, ext_idx, false);
            }
            CHECK(isApproxEqual(svdat.getDataVector(), svdat_expected.getDataVector()));
        }
        SECTION("Apply using dispatcher") {
            StateVectorManaged<TestType> svdat{num_qubits};
            StateVectorManaged<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                svdat_expected.applyOperations({{"PauliX"}, {"PauliY"}},
                                                  {{index}, {index}},
                                                  {false, false});
                svdat.applyOperation(xy_gate, {index}, false);
            }
            CHECK(isApproxEqual(svdat.getDataVector(), svdat_expected.getDataVector()));
        }
    }
    SECTION("Apply YX gate") {
        const std::vector<cp_t> yx_gate{
            Util::IMAG<TestType>(), Util::ZERO<TestType>(),
            Util::ZERO<TestType>(), -Util::IMAG<TestType>()};
        SECTION("Apply directly") {
            StateVectorManaged<TestType> svdat{num_qubits};
            StateVectorManaged<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                auto int_idx = svdat.getInternalIndices({index});
                auto ext_idx = svdat.getExternalIndices({index});
                svdat.applyMatrix(yx_gate, int_idx, ext_idx, false);

                svdat_expected.applyPauliY(int_idx, ext_idx, false);
                svdat_expected.applyPauliX(int_idx, ext_idx, false);
            }
            CHECK(isApproxEqual(svdat.getDataVector(), svdat_expected.getDataVector()));
        }
        SECTION("Apply using dispatcher") {
            StateVectorManaged<TestType> svdat{num_qubits};
            StateVectorManaged<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                svdat_expected.applyOperations({{"PauliY"}, {"PauliX"}},
                                                  {{index}, {index}},
                                                  {false, false});
                svdat.applyOperation(yx_gate, {index}, false);
            }
            CHECK(isApproxEqual(svdat.getDataVector(), svdat_expected.getDataVector()));
        }
    }
    SECTION("Apply YZ gate") {
        const std::vector<cp_t> yz_gate{
            Util::ZERO<TestType>(), -Util::IMAG<TestType>(),
            -Util::IMAG<TestType>(), Util::ZERO<TestType>()};
        SECTION("Apply directly") {
            StateVectorManaged<TestType> svdat{num_qubits};
            StateVectorManaged<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                auto int_idx = svdat.getInternalIndices({index});
                auto ext_idx = svdat.getExternalIndices({index});
                svdat.applyMatrix(yz_gate, int_idx, ext_idx, false);

                svdat_expected.applyPauliY(int_idx, ext_idx, false);
                svdat_expected.applyPauliZ(int_idx, ext_idx, false);
            }
            CHECK(isApproxEqual(svdat.getDataVector(), svdat_expected.getDataVector()));
        }
        SECTION("Apply using dispatcher") {
            StateVectorManaged<TestType> svdat{num_qubits};
            StateVectorManaged<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                svdat_expected.applyOperations({{"PauliY"}, {"PauliZ"}},
                                                  {{index}, {index}},
                                                  {false, false});
                svdat.applyOperation(yz_gate, {index}, false);
            }
            CHECK(isApproxEqual(svdat.getDataVector(), svdat_expected.getDataVector()));
        }
    }
    SECTION("Apply ZY gate") {
        const std::vector<cp_t> zy_gate{
            Util::ZERO<TestType>(), Util::IMAG<TestType>(),
            Util::IMAG<TestType>(), Util::ZERO<TestType>()};
        SECTION("Apply directly") {
            StateVectorManaged<TestType> svdat{num_qubits};
            StateVectorManaged<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                auto int_idx = svdat.getInternalIndices({index});
                auto ext_idx = svdat.getExternalIndices({index});
                svdat.applyMatrix(zy_gate, int_idx, ext_idx, false);

                svdat_expected.applyPauliZ(int_idx, ext_idx, false);
                svdat_expected.applyPauliY(int_idx, ext_idx, false);
            }
            CHECK(isApproxEqual(svdat.getDataVector(), svdat_expected.getDataVector()));
        }
        SECTION("Apply using dispatcher") {
            StateVectorManaged<TestType> svdat{num_qubits};
            StateVectorManaged<TestType> svdat_expected{num_qubits};

            for (size_t index = 0; index < num_qubits; index++) {
                svdat_expected.applyOperations({{"PauliZ"}, {"PauliY"}},
                                                  {{index}, {index}},
                                                  {false, false});
                svdat.applyOperation(zy_gate, {index}, false);
            }
            CHECK(isApproxEqual(svdat.getDataVector(), svdat_expected.getDataVector()));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyMatrix multiple wires",
                   "[StateVectorManaged_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;

    StateVectorManaged<TestType> svdat_init{num_qubits};
    svdat_init.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                  {{0}, {1}, {2}}, {false, false, false});

    const auto cz_gate = Gates::getCZ<TestType>();
    const auto tof_gate = Gates::getToffoli<TestType>();
    const auto arb_gate = Gates::getToffoli<TestType>();

    SECTION("Apply CZ gate") {
        StateVectorManaged<TestType> svdat{svdat_init.getDataVector()};
        StateVectorManaged<TestType> svdat_expected{svdat_init.getDataVector()};

        svdat_expected.applyOperations(
            {{"Hadamard"}, {"CNOT"}, {"Hadamard"}}, {{1}, {0, 1}, {1}},
            {false, false, false});
        svdat.applyOperation(cz_gate, {0, 1}, false);

        CHECK(isApproxEqual(svdat.getDataVector(), svdat_expected.getDataVector()));
    }
}
