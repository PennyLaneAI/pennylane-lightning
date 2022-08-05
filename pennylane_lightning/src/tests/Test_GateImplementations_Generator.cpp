#include "DynamicDispatcher.hpp"
#include "TestHelpers.hpp"
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

template <GeneratorOperation gntr_op>
constexpr auto findGateOpForGenerator() -> GateOperation {
    constexpr auto gntr_name =
        remove_prefix(Util::lookup(Constant::generator_names, gntr_op), 9);

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

constexpr auto minNumQubitsFor(GeneratorOperation gntr_op) -> size_t {
    if (Util::array_has_elt(Constant::multi_qubit_generators, gntr_op)) {
        return 1;
    }
    return Util::lookup(Constant::generator_wires, gntr_op);
}

/**
 * @brief Array of all generator operations with the corresponding gate
 * operations.
 */
constexpr static auto generator_gate_pairs =
    Util::tuple_to_array(generatorGatePairsIter<0>());

template <class PrecisionT, class RandomEngine>
void testGeneratorEqualsGateDerivativeForKernel(
    RandomEngine &re, Gates::KernelType kernel,
    Gates::GeneratorOperation gntr_op, bool inverse) {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    constexpr static auto I = Util::IMAG<PrecisionT>();

    constexpr static auto eps = PrecisionT{1e-3}; // For finite difference

    const auto gate_op = Util::lookup(generator_gate_pairs, gntr_op);
    const auto gate_name = Util::lookup(Constant::gate_names, gate_op);
    const auto min_num_qubits = minNumQubitsFor(gntr_op);
    constexpr static size_t max_num_qubits = 6;

    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    const auto kernel_name = dispatcher.getKernelName(kernel);

    DYNAMIC_SECTION("Test generator of " << gate_name << " for kernel "
                                         << kernel_name) {
        for (size_t num_qubits = min_num_qubits; num_qubits < max_num_qubits;
             num_qubits++) {
            const auto wires = createWires(gate_op, num_qubits);
            const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);

            /* Apply generator to gntr_st */
            auto gntr_st = ini_st;
            PrecisionT scale = dispatcher.applyGenerator(
                kernel, gntr_st.data(), num_qubits, gntr_op, wires, false);
            if (inverse) {
                scale *= -1;
            }
            scaleVector(gntr_st, I * scale);

            /* Compute the derivative of the unitary gate applied to ini_st
             * using finite difference */

            auto diff_st_1 = ini_st;
            auto diff_st_2 = ini_st;

            dispatcher.applyOperation(kernel, diff_st_1.data(), num_qubits,
                                      gate_op, wires, inverse, {eps});
            dispatcher.applyOperation(kernel, diff_st_2.data(), num_qubits,
                                      gate_op, wires, inverse, {-eps});

            std::vector<ComplexPrecisionT> gate_der_st(size_t{1U}
                                                       << num_qubits);

            std::transform(
                diff_st_1.cbegin(), diff_st_1.cend(), diff_st_2.cbegin(),
                gate_der_st.begin(),
                [](ComplexPrecisionT a, ComplexPrecisionT b) { return a - b; });

            scaleVector(gate_der_st, static_cast<PrecisionT>(0.5) / eps);

            REQUIRE(gntr_st == approx(gate_der_st).margin(PrecisionT{1e-3}));
        }
    }
}

template <typename PrecisionT, class RandomEngine>
void testAllGeneratorsForKernel(RandomEngine &re, Gates::KernelType kernel) {
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
    const auto all_gntr_ops = dispatcher.registeredGeneratorsForKernel(kernel);

    for (const auto gntr_op : all_gntr_ops) {
        testGeneratorEqualsGateDerivativeForKernel<PrecisionT>(re, kernel,
                                                               gntr_op, false);
        testGeneratorEqualsGateDerivativeForKernel<PrecisionT>(re, kernel,
                                                               gntr_op, true);
    }
}

TEMPLATE_TEST_CASE("Test all generators of all kernels",
                   "[GateImplementations_Generator]", float, double) {
    using PrecisionT = TestType;

    std::mt19937 re{1337};

    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    for (auto kernel : dispatcher.registeredKernels()) {
        testAllGeneratorsForKernel<PrecisionT>(re, kernel);
    }
}
