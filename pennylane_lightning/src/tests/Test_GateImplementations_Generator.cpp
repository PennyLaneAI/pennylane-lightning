#include "OpToMemberFuncPtr.hpp"
#include "SelectKernel.hpp"
#include "TestHelpers.hpp"
#include "TestKernels.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file Test_GateImplementations_Generator.cpp
 *
 * This file tests gate generators. To be specific, we test whether each
 * generator satisfies
 * @rst
 * :math:`I*G |\psi> = \parital{U(\theta)}/\partial{\theta}_{\theta=0} |\psi>`
 * @endrst
 */
using namespace Pennylane;
using namespace Pennylane::Gates;

/**
 * @brief As clang does not support constexpr string_view::remove_prefix yet.
 */
constexpr std::string_view remove_prefix(const std::string_view &str,
                                         size_t len) {
    return {str.data() + len, str.length() - len};
}

constexpr auto gate_name_to_ops = Util::reverse_pairs(Constant::gate_names);

template <GeneratorOperation gntr_op>
constexpr auto findGateOpForGenerator() -> GateOperation {
    constexpr auto gntr_name =
        remove_prefix(static_lookup<gntr_op>(Constant::generator_names), 9);

    for (const auto &[gate_op, gate_name] : Constant::gate_names) {
        if (gate_name == gntr_name) {
            return gate_op;
        }
    }
    return GateOperation{};
}

template <size_t gntr_idx> constexpr auto generatorGatePairsIter() {
    if constexpr (gntr_idx < Constant::generator_names.size()) {
        constexpr auto gntr_op =
            std::get<0>(Constant::generator_names[gntr_idx]);
        constexpr auto gate_op = findGateOpForGenerator<gntr_op>();

        return Util::prepend_to_tuple(std::pair{gntr_op, gate_op},
                                      generatorGatePairsIter<gntr_idx + 1>());
    } else {
        return std::tuple{};
    }
}

/**
 * @brief Array of all generator operations with the corresponding gate
 * operations.
 */
constexpr static auto generator_gate_pairs =
    Util::tuple_to_array(generatorGatePairsIter<0>());

template <class PrecisionT, class ParamT, class GateImplementation,
          GeneratorOperation gntr_op, class RandomEngine>
void testGeneratorForGate(RandomEngine &re, size_t num_qubits) {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    constexpr auto I = Util::IMAG<PrecisionT>();

    constexpr ParamT eps = 1e-4; // For finite difference

    constexpr auto gate_op = static_lookup<gntr_op>(generator_gate_pairs);
    constexpr auto gate_name = static_lookup<gate_op>(Constant::gate_names);

    DYNAMIC_SECTION("Test generator of " << gate_name << " for kernel "
                                         << GateImplementation::name) {
        const auto wires = createWires(gate_op);
        const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);

        auto gntr_func =
            GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                       gntr_op>::value;
        auto gate_func =
            GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                  gate_op>::value;

        /* Apply generator to gntr_st */
        auto gntr_st = ini_st;
        PrecisionT scale = gntr_func(gntr_st.data(), num_qubits, wires, false);
        scaleVector(gntr_st, I * scale);

        /* Compute the derivative of the unitary gate applied to ini_st using
         * finite difference */

        auto diff_st_1 = ini_st;
        auto diff_st_2 = ini_st;

        gate_func(diff_st_1.data(), num_qubits, wires, false, eps);
        gate_func(diff_st_2.data(), num_qubits, wires, false, -eps);

        std::vector<ComplexPrecisionT> gate_der_st(1U << num_qubits);

        std::transform(
            diff_st_1.cbegin(), diff_st_1.cend(), diff_st_2.cbegin(),
            gate_der_st.begin(),
            [](ComplexPrecisionT a, ComplexPrecisionT b) { return a - b; });

        scaleVector(gate_der_st, static_cast<PrecisionT>(0.5) / eps);

        REQUIRE(gntr_st == PLApprox(gate_der_st).margin(1e-3));
    }
}
template <typename PrecisionT, typename ParamT, class GateImplementation,
          size_t gntr_idx, class RandomEngine>
void testAllGeneratorForKernel(RandomEngine &re, size_t num_qubits) {
    if constexpr (gntr_idx <
                  GateImplementation::implemented_generators.size()) {
        constexpr auto gntr_op =
            GateImplementation::implemented_generators[gntr_idx];
        testGeneratorForGate<PrecisionT, ParamT, GateImplementation, gntr_op>(
            re, num_qubits);
        testAllGeneratorForKernel<PrecisionT, ParamT, GateImplementation,
                                  gntr_idx + 1>(re, num_qubits);
    } else {
        static_cast<void>(re);
        static_cast<void>(num_qubits);
    }
}

template <typename PrecisionT, typename ParamT, class TypeList,
          class RandomEngine>
void testAllGeneratorsAndKernels(RandomEngine &re, size_t num_qubits) {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using GateImplementation = typename TypeList::Type;
        testAllGeneratorForKernel<PrecisionT, ParamT, GateImplementation, 0>(
            re, num_qubits);
        testAllGeneratorsAndKernels<PrecisionT, ParamT,
                                    typename TypeList::Next>(re, num_qubits);
    } else {
        static_cast<void>(re);
        static_cast<void>(num_qubits);
    }
}

TEMPLATE_TEST_CASE("Test all generators of all kernels",
                   "[GateImplementations_Generator]", float, double) {
    using PrecisionT = TestType;
    using ParamT = TestType;

    std::mt19937 re{1337};

    testAllGeneratorsAndKernels<PrecisionT, ParamT, TestKernels>(re, 4);
}
