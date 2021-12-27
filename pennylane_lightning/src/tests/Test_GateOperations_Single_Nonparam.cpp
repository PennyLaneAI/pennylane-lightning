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
 * @file This file contains tests for single qubit non-parameterized gates. List of such gates
 * are [PauliX, PauliY, PauliZ, Hadamard, S, T]
 */


using namespace Pennylane;

namespace {
    using std::vector;
}

/*******************************************************************************
 * Single-qubit gates
 ******************************************************************************/

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyPauliX", "[GateOperations_Single_Nonparam]",
                           (GateOperationsPI, GateOperationsLM), (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;
    for (size_t index = 0; index < num_qubits; index++) {
        auto st = create_zero_state<fp_t>(num_qubits);
        CHECK(st[0] == Util::ONE<fp_t>());

        TestType::applyPauliX(st.data(), num_qubits, {index}, false);
        CHECK(st[0] == Util::ZERO<fp_t>());
        CHECK(st[0b1 << (num_qubits - index - 1)] ==
                Util::ONE<fp_t>());
    }
}

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyPauliY", "[GateOperations_Single_Nonparam]",
                           (GateOperationsPI, GateOperationsLM), (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;

    constexpr CFP_t p = Util::ConstMult(static_cast<fp_t>(0.5),
        Util::ConstMult(Util::INVSQRT2<fp_t>(), Util::IMAG<fp_t>()));
    constexpr CFP_t m = Util::ConstMult(-1, p);

    const std::vector<std::vector<CFP_t>> expected_results = {
        {m, m, m, m, p, p, p, p},
        {m, m, p, p, m, m, p, p},
        {m, p, m, p, m, p, m, p}};

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = create_plus_state<fp_t>(num_qubits);

        TestType::applyPauliY(st.data(), num_qubits, {index}, false);

        CHECK(isApproxEqual(st, expected_results[index]));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyPauliZ", "[GateOperations_Single_Nonparam]",
                           (GateOperationsPI, GateOperationsLM), (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;

    constexpr CFP_t p(static_cast<fp_t>(0.5) * Util::INVSQRT2<fp_t>());
    constexpr CFP_t m(Util::ConstMult(-1, p));

    const std::vector<std::vector<CFP_t>> expected_results = {
        {p, p, p, p, m, m, m, m},
        {p, p, m, m, p, p, m, m},
        {p, m, p, m, p, m, p, m}};

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = create_plus_state<fp_t>(num_qubits);
        TestType::applyPauliZ(st.data(), num_qubits, {index}, false);

        CHECK(isApproxEqual(st, expected_results[index]));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyHadamard", "[GateOperations_Single_Nonparam]",
                           (GateOperationsLM, GateOperationsPI), (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;
    for (size_t index = 0; index < num_qubits; index++) {
        auto st = create_zero_state<fp_t>(num_qubits);

        CHECK(st[0] == CFP_t{1, 0});
        TestType::applyHadamard(st.data(), num_qubits, {index}, false);

        CFP_t expected(1 / std::sqrt(2), 0);
        CHECK(expected.real() == Approx(st[0].real()));
        CHECK(expected.imag() == Approx(st[0].imag()));

        CHECK(
            expected.real() ==
            Approx(st[0b1 << (num_qubits - index - 1)].real()));
        CHECK(
            expected.imag() ==
            Approx(st[0b1 << (num_qubits - index - 1)].imag()));
    }
}



TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyS", "[GateOperations_Single_Nonparam]",
                           (GateOperationsPI, GateOperationsLM), (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;

    constexpr CFP_t r(static_cast<fp_t>(0.5) * Util::INVSQRT2<fp_t>());
    constexpr CFP_t i(Util::ConstMult(r, Util::IMAG<fp_t>()));

    const std::vector<std::vector<CFP_t>> expected_results = {
        {r, r, r, r, i, i, i, i},
        {r, r, i, i, r, r, i, i},
        {r, i, r, i, r, i, r, i}};

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = create_plus_state<fp_t>(num_qubits);

        TestType::applyS(st.data(), num_qubits, {index}, false);

        CHECK(isApproxEqual(st, expected_results[index]));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("GateOperations::applyT", "[GateOperations_Single_Nonparam]",
                           (GateOperationsPI, GateOperationsLM), (float, double)) {
    using fp_t = typename TestType::scalar_type_t;
    using CFP_t = typename TestType::CFP_t;
    const size_t num_qubits = 3;
    // Test using |+++> state

    CFP_t r(1.0 / (2.0 * std::sqrt(2)), 0);
    CFP_t i(1.0 / 4, 1.0 / 4);

    const std::vector<std::vector<CFP_t>> expected_results = {
        {r, r, r, r, i, i, i, i},
        {r, r, i, i, r, r, i, i},
        {r, i, r, i, r, i, r, i}};

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = create_plus_state<fp_t>(num_qubits);

        TestType::applyT(st.data(), num_qubits, {index}, false);

        CHECK(isApproxEqual(st, expected_results[index]));
    }
}
