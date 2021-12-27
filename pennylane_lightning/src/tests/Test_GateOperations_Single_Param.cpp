#include "Gates.hpp"
#include "GateOperationsLM.hpp"
#include "GateOperationsPI.hpp"
#include "Util.hpp"
#include "TestHelpers.hpp"

#include <catch2/catch.hpp>

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file This file contains tests for single qubit parameterized gates. List of such gates is
 * [RX, RY, RZ, PhaseShift, Rot]
 */


using namespace Pennylane;

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyRX", "[GateOperations_Single_Param]",
                           (GateOperationsPI, GateOperationsLM), (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 1;

    const std::vector<fp_t> angles{{0.1}, {0.6}};
    std::vector<std::vector<CFP_t>> expected_results{
        std::vector<CFP_t>{{0.9987502603949663, 0.0},
                          {0.0, -0.04997916927067834}},
        std::vector<CFP_t>{{0.9553364891256061, 0.0}, {0, -0.2955202066613395}},
        std::vector<CFP_t>{{0.49757104789172696, 0.0}, {0, -0.867423225594017}}};

    std::vector<std::vector<CFP_t>> expected_results_adj{
        std::vector<CFP_t>{{0.9987502603949663, 0.0},
                          {0.0, 0.04997916927067834}},
        std::vector<CFP_t>{{0.9553364891256061, 0.0}, {0, 0.2955202066613395}},
        std::vector<CFP_t>{{0.49757104789172696, 0.0}, {0, 0.867423225594017}}};

    SECTION("adj = false") {
        for (size_t index = 0; index < angles.size(); index++) {
            auto st = create_zero_state<fp_t>(num_qubits);

            TestType::applyRX(st.data(), num_qubits, {0}, false, {angles[index]});

            CHECK(isApproxEqual(st, expected_results[index], 1e-7));
        }
    }
    SECTION("adj = true") {
        for (size_t index = 0; index < angles.size(); index++) {
            auto st = create_zero_state<fp_t>(num_qubits);

            TestType::applyRX(st.data(), num_qubits, {0}, true, {angles[index]});
            
            CHECK(isApproxEqual(st, expected_results_adj[index], 1e-7));
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyRY", "[GateOperations_Single_Param]",
                           (GateOperationsPI, GateOperationsLM), (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 1;

    const std::vector<fp_t> angles{0.2, 0.7, 2.9};
    std::vector<std::vector<CFP_t>> expected_results{
        std::vector<CFP_t>{{0.8731983044562817, 0.04786268954660339},
                          {0.0876120655431924, -0.47703040785184303}},
        std::vector<CFP_t>{{0.8243771119105122, 0.16439396602553008},
                          {0.3009211363333468, -0.45035926880694604}},
        std::vector<CFP_t>{{0.10575112905629831, 0.47593196040758534},
                          {0.8711876098966215, -0.0577721051072477}}};
    std::vector<std::vector<CFP_t>> expected_results_adj{
        std::vector<CFP_t>{{0.8731983044562817, -0.04786268954660339},
                          {-0.0876120655431924, -0.47703040785184303}},
        std::vector<CFP_t>{{0.8243771119105122, -0.16439396602553008},
                          {-0.3009211363333468, -0.45035926880694604}},
        std::vector<CFP_t>{{0.10575112905629831, -0.47593196040758534},
                          {-0.8711876098966215, -0.0577721051072477}}};

    const std::vector<CFP_t> init_state{{0.8775825618903728, 0.0},
                                       {0.0, -0.47942553860420306}};
    SECTION("adj = false") {
        for (size_t index = 0; index < angles.size(); index++) {
            auto st = init_state;
            TestType::applyRY(st.data(), num_qubits, {0}, false, {angles[index]});
            CHECK(isApproxEqual(st, expected_results[index], 1e-5));
        }
    }
    SECTION("adj = true") {
        for (size_t index = 0; index < angles.size(); index++) {
            auto st = init_state;

            TestType::applyRY(st.data(), num_qubits, {0}, true, {angles[index]});

            CHECK(isApproxEqual(st, expected_results_adj[index], 1e-5));
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyRZ", "[GateOperations_Single_Param]",
                           (GateOperationsPI, GateOperationsLM), (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;

    // Test using |+++> state

    const std::vector<fp_t> angles{0.2, 0.7, 2.9};
    const CFP_t coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<CFP_t>> rz_data;
    rz_data.reserve(angles.size());
    for (auto &a : angles) {
        rz_data.push_back(Gates::getRZ<fp_t>(a));
    }

    std::vector<std::vector<CFP_t>> expected_results = {
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

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = create_plus_state<fp_t>(num_qubits);

        TestType::applyRZ(st.data(), num_qubits, {index}, false, {angles[index]});

        CHECK(isApproxEqual(st, expected_results[index]));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyPhaseShift", "[GateOperations_Single_Param]",
                           (GateOperationsPI, GateOperationsLM), (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;

    // Test using |+++> state

    const std::vector<fp_t> angles{0.3, 0.8, 2.4};
    const CFP_t coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<CFP_t>> ps_data;
    ps_data.reserve(angles.size());
    for (auto &a : angles) {
        ps_data.push_back(Gates::getPhaseShift<fp_t>(a));
    }

    std::vector<std::vector<CFP_t>> expected_results = {
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

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = create_plus_state<fp_t>(num_qubits);

        TestType::applyPhaseShift(st.data(), num_qubits, {index}, false, {angles[index]});

        CHECK(isApproxEqual(st, expected_results[index]));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyRot", "[GateOperations_Single_Param]",
                           (GateOperationsPI, GateOperationsLM), (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;
    auto ini_st = create_zero_state<fp_t>(num_qubits);

    const std::vector<std::vector<fp_t>> angles{
        std::vector<fp_t>{0.3, 0.8, 2.4},
        std::vector<fp_t>{0.5, 1.1, 3.0},
        std::vector<fp_t>{2.3, 0.1, 0.4}};

    std::vector<std::vector<CFP_t>> expected_results{
        std::vector<CFP_t>(0b1 << num_qubits),
        std::vector<CFP_t>(0b1 << num_qubits),
        std::vector<CFP_t>(0b1 << num_qubits)};

    for (size_t i = 0; i < angles.size(); i++) {
        const auto rot_mat =
            Gates::getRot<fp_t>(angles[i][0], angles[i][1], angles[i][2]);
        expected_results[i][0] = rot_mat[0];
        expected_results[i][0b1 << (num_qubits - i - 1)] = rot_mat[2];
    }

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = create_zero_state<fp_t>(num_qubits);
        TestType::applyRot(st.data(), num_qubits, {index}, false, angles[index][0],
                           angles[index][1], angles[index][2]);

        CHECK(isApproxEqual(st, expected_results[index]));
    }
}
