#include "CreateAllWires.hpp"
#include "TestHelpers.hpp"
#include "TestKernels.hpp"

#include "ConstantUtil.hpp"
#include "DynamicDispatcher.hpp"
#include "OpToMemberFuncPtr.hpp"
#include "SelectKernel.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

#include <algorithm>
#include <complex>
#include <iostream>
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

namespace {
using namespace Pennylane::Gates::Constant;
} // namespace

using std::vector;

/**
 * @brief Change the given type list of kernels to string
 */
template <typename TypeList> std::string kernelsToString() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        return std::string(TypeList::Type::name) + ", " +
               kernelsToString<typename TypeList::Next>();
    }
    return "";
}

/* Type transformation */
/**
 * @brief Construct a type list of kernels implementing the given gate
 */
template <Gates::GateOperation gate_op, typename TypeList>
struct KernelsImplementingGateHelper {
    using Type = std::conditional_t<
        array_has_elt(TypeList::Type::implemented_gates, gate_op),
        typename PrependToTypeList<
            typename TypeList::Type,
            typename KernelsImplementingGateHelper<
                gate_op, typename TypeList::Next>::Type>::Type,
        typename KernelsImplementingGateHelper<gate_op,
                                               typename TypeList::Next>::Type>;
};
template <Gates::GateOperation gate_op>
struct KernelsImplementingGateHelper<gate_op, void> {
    using Type = void;
};

/**
 * @brief Construct a type list of kernels implementing the given matrix
 */
template <Gates::MatrixOperation mat_op, typename TypeList>
constexpr auto kernelsImplementingMatrixHelper() {
    if constexpr (std::is_same_v<TypeList, void>) {
        return std::tuple{};
    }
    else {
        constexpr auto r = kernelsImplementingMatrixHelper<mat_op, typename TypeList::Next>();
        if constexpr (Util::array_has_elt(TypeList::Type::implemented_matrices, mat_op)) {
            return Util::prepend_to_tuple(TypeList::Type::kernel_id, r);
        } else{
            return r;
        }
    }
}

/**
 * @brief Type list of kernels implementing the given gate operation.
 */
template <Gates::GateOperation gate_op> struct KernelsImplementingGate {
    using Type =
        typename KernelsImplementingGateHelper<gate_op, TestKernels>::Type;
};

/**
 * @brief Type list of kernels implementing the given matrix operation.
 */
template <Gates::MatrixOperation mat_op>
constexpr static auto kernels_implementing_matrix = 
    Util::tuple_to_array(kernelsImplementingMatrixHelper<mat_op, TestKernels>());

/**
 * @brief Apply the given gate operation with the given gate implementation.
 *
 * @tparam gate_op Gate operation to test
 * @tparam PrecisionT Floating point data type for statevector
 * @tparam ParamT Floating point data type for parameter
 * @tparam GateImplementation Gate implementation class
 * @param ini Initial statevector
 * @param num_qubits Number of qubits
 * @param wires Wires the gate applies to
 * @param inverse Whether to use inverse of gate
 * @param params Paramters for gate
 */
template <Gates::GateOperation gate_op, typename PrecisionT, typename ParamT,
          typename GateImplementation, class Alloc>
auto applyGate(std::vector<std::complex<PrecisionT>, Alloc> ini,
               size_t num_qubits, const std::vector<size_t> &wires,
               bool inverse, const std::vector<ParamT> &params)
    -> std::vector<std::complex<PrecisionT>, Alloc> {
    callGateOps(GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                      gate_op>::value,
                ini.data(), num_qubits, wires, inverse, params);
    return ini;
}

/**
 * @brief Apply the given gate using all implementing kernels and return
 * results in tuple.
 */
template <Gates::GateOperation gate_op, typename PrecisionT, typename ParamT,
          typename Kernels, class Alloc, size_t... I>
auto applyGateForImplemetingKernels(
    const std::vector<std::complex<PrecisionT>, Alloc> &ini, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse,
    const std::vector<ParamT> &params,
    [[maybe_unused]] std::index_sequence<I...> dummy) {
    return std::make_tuple(
        applyGate<gate_op, PrecisionT, ParamT, getNthType<Kernels, I>>(
            ini, num_qubits, wires, inverse, params)...);
}

/**
 * @brief Apply the given gate using all implementing kernels and compare
 * the results.
 */
template <Gates::GateOperation gate_op, typename PrecisionT, typename ParamT,
          class RandomEngine>
void testApplyGate(RandomEngine &re, size_t num_qubits) {
    const auto ini = createRandomState<PrecisionT>(re, num_qubits);

    using Kernels = typename KernelsImplementingGate<gate_op>::Type;

    INFO("Kernels implementing " << lookup(gate_names, gate_op) << " are "
                                 << kernelsToString<Kernels>());

    INFO("PrecisionT, ParamT = " << PrecisionToName<PrecisionT>::value << ", "
                                 << PrecisionToName<ParamT>::value);

    const auto all_wires = createAllWires(num_qubits, gate_op, true);
    for (const auto &wires : all_wires) {
        const auto params = createParams<ParamT>(gate_op);
        const auto gate_name = lookup(gate_names, gate_op);
        DYNAMIC_SECTION(
            "Test gate "
            << gate_name
            << " with inverse = false") { // Test with inverse = false
            const auto results = Util::tuple_to_array(
                applyGateForImplemetingKernels<gate_op, PrecisionT, ParamT,
                                               Kernels>(
                    ini, num_qubits, wires, false, params,
                    std::make_index_sequence<length<Kernels>()>()));

            for (size_t i = 0; i < results.size() - 1; i++) {
                REQUIRE(results[i] ==
                        approx(results[i + 1])
                            .margin(static_cast<PrecisionT>(1e-5)));
            }
        }

        DYNAMIC_SECTION("Test gate "
                        << gate_name
                        << " with inverse = true") { // Test with inverse = true
            const auto results = Util::tuple_to_array(
                applyGateForImplemetingKernels<gate_op, PrecisionT, ParamT,
                                               Kernels>(
                    ini, num_qubits, wires, true, params,
                    std::make_index_sequence<length<Kernels>()>()));

            for (size_t i = 0; i < results.size() - 1; i++) {
                REQUIRE(results[i] ==
                        approx(results[i + 1])
                            .margin(static_cast<PrecisionT>(1e-5)));
            }
        }
    }
}

template <size_t gate_idx, typename PrecisionT, typename ParamT,
          class RandomEngine>
void testAllGatesIter(RandomEngine &re, size_t max_num_qubits) {
    if constexpr (gate_idx < static_cast<size_t>(GateOperation::END)) {
        constexpr static auto gate_op = static_cast<GateOperation>(gate_idx);

        size_t min_num_qubits = array_has_elt(multi_qubit_gates, gate_op)
                                    ? 1
                                    : lookup(gate_wires, gate_op);
        for (size_t num_qubits = min_num_qubits; num_qubits < max_num_qubits;
             num_qubits++) {
            testApplyGate<gate_op, PrecisionT, ParamT>(re, num_qubits);
        }
        testAllGatesIter<gate_idx + 1, PrecisionT, ParamT>(re, max_num_qubits);
    }
}

template <typename PrecisionT, typename ParamT, class RandomEngine>
void testAllGates(RandomEngine &re, size_t max_num_qubits) {
    testAllGatesIter<0, PrecisionT, ParamT>(re, max_num_qubits);
}

TEMPLATE_TEST_CASE("Test all kernels give the same results for gates",
                   "[Test_GateImplementations_CompareKernels]", float, double) {
    std::mt19937 re{1337};
    testAllGates<TestType, TestType>(re, 6);
}

template <typename PrecisionT, class RandomEngine>
void testSingleQubitOp(RandomEngine& re, size_t num_qubits, bool inverse) {
    constexpr size_t num_wires = 1;

    const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);
    const auto matrix = randomUnitary<PrecisionT>(re, num_qubits);
    const auto all_kernels = kernels_implementing_matrix<MatrixOperation::SingleQubitOp>;

    std::ostringstream ss;
    ss << "Test SingleQubitOp with kernels: ";
    for (KernelType kernel: all_kernels) {
        ss << Util::lookup(kernel_id_name_pairs, kernel) << ", ";
    }
    ss << "inverse: " << inverse;

    DYNAMIC_SECTION(ss.str()) {
        const auto all_wires = CombinationGenerator(num_qubits, num_wires).all_perms();
        for(const auto wires: all_wires) {
            std::vector<std::vector<std::complex<PrecisionT>>> res;
            for (KernelType kernel: all_kernels) {
                auto st = ini_st;
                DynamicDispatcher<PrecisionT>::getInstance().applyMatrix(kernel, st.data(),
                        num_qubits, matrix.data(), wires, inverse);
                res.emplace_back(std::move(st));
            }
            for(size_t idx = 0; idx < all_kernels.size() - 1; idx++) {
                REQUIRE(res[idx] == approx(res[idx+1]).margin(1e-7));
            }
        }
    }
}

template <typename PrecisionT, class RandomEngine>
void testTwoQubitOp(RandomEngine& re, size_t num_qubits, bool inverse) {
    constexpr size_t num_wires = 1;

    const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);
    const auto matrix = randomUnitary<PrecisionT>(re, num_qubits);
    const auto all_kernels = kernels_implementing_matrix<MatrixOperation::TwoQubitOp>;

    std::ostringstream ss;
    ss << "Test TwoQubitOp with kernels: ";
    for (KernelType kernel: all_kernels) {
        ss << Util::lookup(kernel_id_name_pairs, kernel) << ", ";
    }
    ss << "inverse: " << inverse;

    DYNAMIC_SECTION(ss.str()) {
        const auto all_wires = CombinationGenerator(num_qubits, num_wires).all_perms();
        for(const auto wires: all_wires) {
            std::vector<std::vector<std::complex<PrecisionT>>> res;
            for (KernelType kernel: all_kernels) {
                auto st = ini_st;
                DynamicDispatcher<PrecisionT>::getInstance().applyMatrix(kernel, st.data(),
                        num_qubits, matrix.data(), wires, inverse);
                res.emplace_back(std::move(st));
            }
            for(size_t idx = 0; idx < all_kernels.size() - 1; idx++) {
                REQUIRE(res[idx] == approx(res[idx+1]).margin(1e-7));
            }
        }
    }
}

template <typename PrecisionT, class RandomEngine>
void testMultiQubitOp(RandomEngine& re, size_t num_qubits, size_t num_wires, bool inverse) {
    assert(num_wires >= 2);
    const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);
    const auto matrix = randomUnitary<PrecisionT>(re, num_qubits);
    const auto all_kernels = kernels_implementing_matrix<MatrixOperation::MultiQubitOp>;

    std::ostringstream ss;
    ss << "Test MultiQubitOp with kernels: ";
    for (KernelType kernel: all_kernels) {
        ss << Util::lookup(kernel_id_name_pairs, kernel) << ", ";
    }
    ss << "inverse: " << inverse;

    DYNAMIC_SECTION(ss.str()) {
        const auto all_wires = PermutationGenerator(num_qubits, num_wires).all_perms();
        for(const auto wires: all_wires) {
            std::vector<std::vector<std::complex<PrecisionT>>> res;
            for (KernelType kernel: all_kernels) {
                auto st = ini_st;
                DynamicDispatcher<PrecisionT>::getInstance().applyMatrix(kernel, st.data(),
                        num_qubits, matrix.data(), wires, inverse);
                res.emplace_back(std::move(st));
            }
            for(size_t idx = 0; idx < all_kernels.size() - 1; idx++) {
                REQUIRE(res[idx] == approx(res[idx+1]).margin(1e-7));
            }
        }
    }
}


TEMPLATE_TEST_CASE("Test all kernels give the same results for matrices",
                   "[Test_GateImplementations_CompareKernels]", float, double) {
    std::mt19937 re{1337};

    const size_t num_qubits = 5;

    for(bool inverse: {true, false}) {
        testSingleQubitOp<TestType>(re, num_qubits, inverse);
        testTwoQubitOp<TestType>(re, num_qubits, inverse);
        testMultiQubitOp<TestType>(re, num_qubits, 2, inverse);
        testMultiQubitOp<TestType>(re, num_qubits, 3, inverse);
        testMultiQubitOp<TestType>(re, num_qubits, 4, inverse);
        testMultiQubitOp<TestType>(re, num_qubits, 5, inverse);
    }
}
