#include "OpToMemberFuncPtr.hpp"
#include "SelectGateOps.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

using namespace Pennylane;

template <typename EnumClass, uint32_t... I>
constexpr auto allGateOpsHelper(std::integer_sequence<uint32_t, I...>) {
    return std::make_tuple(static_cast<EnumClass>(I)...);
}

template <typename EnumClass> constexpr auto allGateOps() {
    return Util::tuple_to_array(allGateOpsHelper<EnumClass>(
        std::make_integer_sequence<uint32_t,
                                   static_cast<uint32_t>(EnumClass::END)>{}));
};

#define PENNYLANE_TESTS_DEFINE_GATE_OP_PARAM0(GATE_NAME)                       \
    template <class PrecisionT>                                                \
    static void apply##GATE_NAME(                                              \
        std::complex<PrecisionT> *arr, size_t num_qubits,                      \
        const std::vector<size_t> &wires, bool inverse) {                      \
        static_cast<void>(arr);                                                \
        static_cast<void>(num_qubits);                                         \
        static_cast<void>(wires);                                              \
        static_cast<void>(inverse);                                            \
    }
#define PENNYLANE_TESTS_DEFINE_GATE_OP_PARAM1(GATE_NAME)                       \
    template <class PrecisionT, class ParamT>                                  \
    static void apply##GATE_NAME(                                              \
        std::complex<PrecisionT> *arr, size_t num_qubits,                      \
        const std::vector<size_t> &wires, bool inverse, ParamT) {              \
        static_cast<void>(arr);                                                \
        static_cast<void>(num_qubits);                                         \
        static_cast<void>(wires);                                              \
        static_cast<void>(inverse);                                            \
    }
#define PENNYLANE_TESTS_DEFINE_GATE_OP_PARAM3(GATE_NAME)                       \
    template <class PrecisionT, class ParamT>                                  \
    static void apply##GATE_NAME(std::complex<PrecisionT> *arr,                \
                                 size_t num_qubits,                            \
                                 const std::vector<size_t> &wires,             \
                                 bool inverse, ParamT, ParamT, ParamT) {       \
        static_cast<void>(arr);                                                \
        static_cast<void>(num_qubits);                                         \
        static_cast<void>(wires);                                              \
        static_cast<void>(inverse);                                            \
    }
#define PENNYLANE_TESTS_DEFINE_GATE_OP(GATE_NAME, NUM_PARAMS)                  \
    PENNYLANE_TESTS_DEFINE_GATE_OP_PARAM##NUM_PARAMS(GATE_NAME)

#define PENNYLANE_TESTS_DEFINE_GENERATOR_OP(GENERATOR_NAME)                    \
    template <class PrecisionT>                                                \
    static PrecisionT applyGenerator##GENERATOR_NAME(                          \
        std::complex<PrecisionT> *arr, size_t num_qubits,                      \
        const std::vector<size_t> &wires, bool adj) {                          \
        static_cast<void>(arr);                                                \
        static_cast<void>(num_qubits);                                         \
        static_cast<void>(wires);                                              \
        static_cast<void>(adj);                                                \
        return PrecisionT{};                                                   \
    }

class DummyImplementation {
  public:
    constexpr static std::array implemented_gates = allGateOps<GateOperation>();
    constexpr static std::array implemented_generators =
        allGateOps<GeneratorOperation>();
    constexpr static std::string_view name = "Dummy";

    template <class PrecisionT>
    static void applyMatrix(std::complex<PrecisionT> *arr, size_t num_qubits,
                            const std::complex<PrecisionT> *matrix,
                            const std::vector<size_t> &wires, bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(matrix);
        static_cast<void>(inverse);
    }

    PENNYLANE_TESTS_DEFINE_GATE_OP(PauliX, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(PauliY, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(PauliZ, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(Hadamard, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(S, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(T, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(PhaseShift, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(RX, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(RY, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(RZ, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(Rot, 3)
    PENNYLANE_TESTS_DEFINE_GATE_OP(CNOT, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(CY, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(CZ, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(SWAP, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(ControlledPhaseShift, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(CRX, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(CRY, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(CRZ, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(CRot, 3)
    PENNYLANE_TESTS_DEFINE_GATE_OP(Toffoli, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(CSWAP, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(MultiRZ, 1)

    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(PhaseShift)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(RX)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(RY)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(RZ)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(CRX)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(CRY)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(CRZ)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(ControlledPhaseShift)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(MultiRZ)
};

template <class GateImplementation> struct ImplementedGates {
    constexpr static auto value = GateImplementation::implemented_gates;
    constexpr static std::array<GateOperation, 1> ignore_list = {
        GateOperation::Matrix};

    template <typename PrecisionT, typename ParamT, GateOperation op>
    constexpr static auto func_ptr =
        GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                              op>::value;
};
template <class GateImplementation> struct ImplementedGenerators {
    constexpr static auto value = GateImplementation::implemented_generators;
    constexpr static std::array<GeneratorOperation, 0> ignore_list = {};

    template <typename PrecisionT, typename ParamT, GeneratorOperation op>
    constexpr static auto func_ptr =
        GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation, op>::value;
};

template <typename PrecisionT, typename ParamT,
          template <typename> class ValueClass, size_t idx>
constexpr auto opFuncPtrPairsIter() {
    if constexpr (idx == ValueClass<DummyImplementation>::value.size()) {
        return std::tuple{};
    } else {
        constexpr auto op = ValueClass<DummyImplementation>::value[idx];
        if constexpr (array_has_elt(
                          ValueClass<DummyImplementation>::ignore_list, op)) {
            return opFuncPtrPairsIter<PrecisionT, ParamT, ValueClass,
                                      idx + 1>();
        } else {
            const auto elt = std::pair{
                op,
                ValueClass<DummyImplementation>::template func_ptr<PrecisionT,
                                                                   ParamT, op>};
            return Util::prepend_to_tuple(
                elt,
                opFuncPtrPairsIter<PrecisionT, ParamT, ValueClass, idx + 1>());
        }
    }
};

/**
 * @brief Pairs of all implemented gate operations and the corresponding
 * function pointers.
 */
template <typename PrecisionT, typename ParamT>
constexpr static auto gate_op_func_ptr_pairs =
    opFuncPtrPairsIter<PrecisionT, ParamT, ImplementedGates, 0>();

template <typename PrecisionT, typename ParamT>
constexpr static auto generator_op_func_ptr_pairs =
    opFuncPtrPairsIter<PrecisionT, ParamT, ImplementedGenerators, 0>();

template <typename PrecisionT, typename ParamT, size_t num_params,
          size_t tuple_idx>
constexpr auto gateOpFuncPtrPairsWithNumParamsIter() {
    if constexpr (tuple_idx <
                  std::tuple_size_v<decltype(
                      gate_op_func_ptr_pairs<PrecisionT, ParamT>)>) {
        constexpr auto elt =
            std::get<tuple_idx>(gate_op_func_ptr_pairs<PrecisionT, ParamT>);
        if constexpr (static_lookup<elt.first>(Constant::gate_num_params) ==
                      num_params) {
            return prepend_to_tuple(
                elt, gateOpFuncPtrPairsWithNumParamsIter<
                         PrecisionT, ParamT, num_params, tuple_idx + 1>());
        } else {
            return gateOpFuncPtrPairsWithNumParamsIter<
                PrecisionT, ParamT, num_params, tuple_idx + 1>();
        }
    } else {
        return std::tuple{};
    }
}

template <typename PrecisionT, typename ParamT, size_t num_params>
constexpr auto gate_op_func_ptr_with_params = Util::tuple_to_array(
    gateOpFuncPtrPairsWithNumParamsIter<PrecisionT, ParamT, num_params, 0>());

template <typename PrecisionT, typename ParamT>
constexpr auto generator_op_func_ptr =
    Util::tuple_to_array(generator_op_func_ptr_pairs<PrecisionT, PrecisionT>);

template <typename T, typename U, size_t size>
constexpr auto testUniqueness(const std::array<std::pair<T, U>, size> &pairs) {
    REQUIRE(Util::count_unique(Util::first_elts_of(pairs)) == pairs.size());
    REQUIRE(Util::count_unique(Util::second_elts_of(pairs)) == pairs.size());
}

TEMPLATE_TEST_CASE("GateOpToMemberFuncPtr", "[GateOpToMemberFuncPtr]", float,
                   double) {
    testUniqueness(gate_op_func_ptr_with_params<TestType, TestType, 0>);
    testUniqueness(gate_op_func_ptr_with_params<TestType, TestType, 1>);
    testUniqueness(gate_op_func_ptr_with_params<TestType, TestType, 3>);
    REQUIRE(true);
}
TEMPLATE_TEST_CASE("GeneratorOpToMemberFuncPtr", "[GeneratorOpToMemberFuncPtr]",
                   float, double) {
    testUniqueness(generator_op_func_ptr<TestType, TestType>);
    REQUIRE(true);
}
