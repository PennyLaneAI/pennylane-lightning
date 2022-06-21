#include "CreateAllWires.hpp"
#include "TestHelpers.hpp"
#include "TestKernels.hpp"

#include "ConstantUtil.hpp"
#include "DynamicDispatcher.hpp"
#include "KernelType.hpp"
#include "OpToMemberFuncPtr.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

#include <algorithm>
#include <complex>
#include <iostream>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file Test_GateImplementations_Nonparam.cpp
 *
 * This file tests all gate operations (besides matrix) by comparing results
 * between different kernels (gate implementations).
 */
using namespace Pennylane;
using namespace Pennylane::Gates;
using namespace Pennylane::Util;

using std::vector;

template<typename PrecisionT>
auto kernelsImplementingGate(GateOperation gate_op) -> std::vector<KernelType> {
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
    auto kernels = dispatcher.registeredKernels();

    std::vector<KernelType> res;

    std::copy_if(
        kernels.begin(), kernels.end(), std::back_inserter(res),
        [&dispatcher, gate_op](KernelType kernel) {
            return dispatcher.registeredGatesForKernel(kernel).contains(gate_op);
        });
    return res;
}

/**
 * @brief Apply the given gate using all implementing kernels and compare
 * the results.
 */
template <typename PrecisionT, class RandomEngine>
void testApplyGate(RandomEngine &re, GateOperation gate_op, size_t num_qubits) {
    using Gates::Constant::gate_names;
    const auto implementing_kernels = kernelsImplementingGate<PrecisionT>(gate_op);
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    std::ostringstream ss;
    ss << "Kernels implementing " << lookup(gate_names, gate_op) << " are ";
    for (KernelType kernel : implementing_kernels) {
        ss << dispatcher.getKernelName(kernel) << ", ";
    }

    INFO(ss.str());
    INFO("PrecisionT = " << PrecisionToName<PrecisionT>::value << ", ");

    const auto ini = createRandomState<PrecisionT>(re, num_qubits);

    const auto all_wires = createAllWires(num_qubits, gate_op, true);
    for (const auto &wires : all_wires) {
        const auto params = createParams<PrecisionT>(gate_op);
        const auto gate_name = lookup(gate_names, gate_op);

        // Test with inverse = false
        DYNAMIC_SECTION("Test gate " << gate_name << " with inverse = false") {
            std::vector<TestVector<std::complex<PrecisionT>>> res;

            // Collect results from all implementing kernels
            for (auto kernel : implementing_kernels) {
                auto st = ini;
                dispatcher.applyOperation(kernel, st.data(), num_qubits,
                                          gate_op, wires, false, params);
                res.emplace_back(std::move(st));
            }

            // And compare them
            for (size_t i = 0; i < res.size() - 1; i++) {
                REQUIRE(
                    res[i] ==
                    approx(res[i + 1]).margin(static_cast<PrecisionT>(1e-5)));
            }
        }

         // Test with inverse = true
        DYNAMIC_SECTION("Test gate "
                        << gate_name
                        << " with inverse = true") {
            std::vector<TestVector<std::complex<PrecisionT>>> res;

            // Collect results from all implementing kernels
            for (auto kernel : implementing_kernels) {
                auto st = ini;
                dispatcher.applyOperation(kernel, st.data(), num_qubits,
                                          gate_op, wires, true, params);
                res.emplace_back(std::move(st));
            }

            // And compare them
            for (size_t i = 0; i < res.size() - 1; i++) {
                REQUIRE(
                    res[i] ==
                    approx(res[i + 1]).margin(static_cast<PrecisionT>(1e-5)));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Test all kernels give the same results for gates",
                   "[GateImplementations_CompareKernels]", float, double) {
    /* We test all gate operations up to the number of qubits we give */
    constexpr size_t max_num_qubits = 6;
    std::mt19937 re{1337};
    Util::for_each_enum<GateOperation>([&](GateOperation gate_op) {
        testApplyGate<TestType>(re, gate_op, max_num_qubits);
    });
}

/*
template <typename PrecisionT, class RandomEngine>
void testSingleQubitOp(RandomEngine &re, size_t num_qubits, bool inverse) {
    constexpr size_t num_wires = 1;

    const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);
    const auto matrix = randomUnitary<PrecisionT>(re, num_qubits);
    const auto all_kernels =
        kernels_implementing_matrix<MatrixOperation::SingleQubitOp>;

    std::ostringstream ss;
    ss << "Test SingleQubitOp with kernels: ";
    for (KernelType kernel : all_kernels) {
        ss << Util::lookup(kernel_id_name_pairs, kernel) << ", ";
    }
    ss << "inverse: " << inverse;

    DYNAMIC_SECTION(ss.str()) {
        const auto all_wires =
            CombinationGenerator(num_qubits, num_wires).all_perms();
        for (const auto &wires : all_wires) {
            std::vector<TestVector<std::complex<PrecisionT>>> res;
            for (KernelType kernel : all_kernels) {
                auto st = ini_st;
                DynamicDispatcher<PrecisionT>::getInstance().applyMatrix(
                    kernel, st.data(), num_qubits, matrix.data(), wires,
                    inverse);
                res.emplace_back(std::move(st));
            }
            for (size_t idx = 0; idx < all_kernels.size() - 1; idx++) {
                REQUIRE(res[idx] == approx(res[idx + 1]).margin(1e-7));
            }
        }
    }
}

template <typename PrecisionT, class RandomEngine>
void testTwoQubitOp(RandomEngine &re, size_t num_qubits, bool inverse) {
    constexpr size_t num_wires = 1;

    const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);
    const auto matrix = randomUnitary<PrecisionT>(re, num_qubits);
    const auto all_kernels =
        kernels_implementing_matrix<MatrixOperation::TwoQubitOp>;

    std::ostringstream ss;
    ss << "Test TwoQubitOp with kernels: ";
    for (KernelType kernel : all_kernels) {
        ss << Util::lookup(kernel_id_name_pairs, kernel) << ", ";
    }
    ss << "inverse: " << inverse;

    DYNAMIC_SECTION(ss.str()) {
        const auto all_wires =
            CombinationGenerator(num_qubits, num_wires).all_perms();
        for (const auto &wires : all_wires) {
            std::vector<TestVector<std::complex<PrecisionT>>> res;
            for (KernelType kernel : all_kernels) {
                auto st = ini_st;
                DynamicDispatcher<PrecisionT>::getInstance().applyMatrix(
                    kernel, st.data(), num_qubits, matrix.data(), wires,
                    inverse);
                res.emplace_back(std::move(st));
            }
            for (size_t idx = 0; idx < all_kernels.size() - 1; idx++) {
                REQUIRE(res[idx] == approx(res[idx + 1]).margin(1e-7));
            }
        }
    }
}

template <typename PrecisionT, class RandomEngine>
void testMultiQubitOp(RandomEngine &re, size_t num_qubits, size_t num_wires,
                      bool inverse) {
    assert(num_wires >= 2);
    const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);
    const auto matrix = randomUnitary<PrecisionT>(re, num_qubits);
    const auto all_kernels =
        kernels_implementing_matrix<MatrixOperation::MultiQubitOp>;

    std::ostringstream ss;
    ss << "Test MultiQubitOp with kernels: ";
    for (KernelType kernel : all_kernels) {
        ss << Util::lookup(kernel_id_name_pairs, kernel) << ", ";
    }
    ss << "inverse: " << inverse;

    DYNAMIC_SECTION(ss.str()) {
        const auto all_wires =
            PermutationGenerator(num_qubits, num_wires).all_perms();
        for (const auto &wires : all_wires) {
            std::vector<TestVector<std::complex<PrecisionT>>> res;
            for (KernelType kernel : all_kernels) {
                auto st = ini_st;
                DynamicDispatcher<PrecisionT>::getInstance().applyMatrix(
                    kernel, st.data(), num_qubits, matrix.data(), wires,
                    inverse);
                res.emplace_back(std::move(st));
            }
            for (size_t idx = 0; idx < all_kernels.size() - 1; idx++) {
                REQUIRE(res[idx] == approx(res[idx + 1]).margin(1e-7));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Test all kernels give the same results for matrices",
                   "[Test_GateImplementations_CompareKernels]", float, double) {
    std::mt19937 re{1337};

    const size_t num_qubits = 5;

    for (bool inverse : {true, false}) {
        testSingleQubitOp<TestType>(re, num_qubits, inverse);
        testTwoQubitOp<TestType>(re, num_qubits, inverse);
        testMultiQubitOp<TestType>(re, num_qubits, 2, inverse);
        testMultiQubitOp<TestType>(re, num_qubits, 3, inverse);
        testMultiQubitOp<TestType>(re, num_qubits, 4, inverse);
        testMultiQubitOp<TestType>(re, num_qubits, 5, inverse);
    }
}
*/
