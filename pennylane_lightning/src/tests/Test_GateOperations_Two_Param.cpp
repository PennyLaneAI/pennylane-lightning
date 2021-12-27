#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "GateOperationsLM.hpp"
#include "GateOperationsPI.hpp"
#include "Gates.hpp"
#include "Util.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane;

/**
 * @file This file contains tests for gate operations with parameteriezed two-qubits gates.
 * Such gates are [ControlledPhaseShift, CRX, CRY, CRZ, CRot].
 */


TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyControlledPhaseShift", "[GateOperations_Two_Param]",
        (GateOperationsPI, GateOperationsLM), (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;

    const size_t num_qubits = 3;

    // Test using |+++> state
    auto ini_st = create_plus_state<fp_t>(num_qubits);

    const std::vector<fp_t> angles{0.3, 2.4};
    const CFP_t coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<CFP_t>> ps_data;
    ps_data.reserve(angles.size());
    for (auto &a : angles) {
        ps_data.push_back(Gates::getPhaseShift<fp_t>(a));
    }

    std::vector<std::vector<CFP_t>> expected_results = {
        {ps_data[0][0], ps_data[0][0], ps_data[0][0], ps_data[0][0],
         ps_data[0][0], ps_data[0][0], ps_data[0][3], ps_data[0][3]},
        {ps_data[1][0], ps_data[1][0], ps_data[1][0], ps_data[1][3],
         ps_data[1][0], ps_data[1][0], ps_data[1][0], ps_data[1][3]}};

    for (auto &vec : expected_results) {
        scaleVector(vec, coef);
    }

    // correct this when LM kernels are fully developed
    if constexpr (TestType::kernel_id == KernelType::PI) {
        auto st = ini_st;

        TestType::applyControlledPhaseShift(st.data(), num_qubits, {0, 1}, false, {angles[0]});
        CAPTURE(st);
        CHECK(isApproxEqual(st, expected_results[0]));
    } else {
        auto st = ini_st;
        CHECK_THROWS(TestType::applyControlledPhaseShift(st.data(), num_qubits, {0, 1}, false, {angles[0]}));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyCRot", "[GateOperations_Two_Param]",
        (GateOperationsPI, GateOperationsLM), (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;

    auto ini_st = create_zero_state<fp_t>(num_qubits);

    const std::vector<fp_t> angles{0.3, 0.8, 2.4};

    std::vector<CFP_t> expected_results(8);
    const auto rot_mat =
        Gates::getRot<fp_t>(angles[0], angles[1], angles[2]);
    expected_results[0b1 << (num_qubits - 1)] = rot_mat[0];
    expected_results[(0b1 << num_qubits) - 2] = rot_mat[2];

    // correct this when LM kernels are fully developed
    if constexpr (TestType::kernel_id == KernelType::PI) {
        SECTION("CRot0,1 |000> -> |000>") {
            auto st = create_zero_state<fp_t>(num_qubits);
            TestType::applyCRot(st.data(), num_qubits, {0, 1}, false, 
                                angles[0], angles[1], angles[2]);

            CHECK(isApproxEqual(st, ini_st));
        }
        SECTION("CRot0,1 |100> -> |1>(a|0>+b|1>)|0>") {
            auto st = create_zero_state<fp_t>(num_qubits);
            TestType::applyPauliX(st.data(), num_qubits, {0}, false);

            TestType::applyCRot(st.data(), num_qubits, {0, 1}, false, angles[0],
                                      angles[1], angles[2]);

            CHECK(isApproxEqual(st, expected_results));
        }
    } else {
        auto st = ini_st;
        CHECK_THROWS(TestType::applyCRot(st.data(), num_qubits, {0, 1}, false, 
                    angles[0], angles[1], angles[2]));
    }
}
