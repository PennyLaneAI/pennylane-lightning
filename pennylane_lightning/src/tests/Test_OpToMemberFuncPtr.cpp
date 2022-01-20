#include "SelectGateOps.hpp"
#include "OpToMemberFuncPtr.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

using namespace Pennylane;

template <typename EnumClass, uint32_t... I>
constexpr auto allGateOpsHelper(std::integer_sequence<uint32_t, I...>) {
    return std::make_tuple(static_cast<EnumClass>(I)...);
}

template <typename EnumClass>
constexpr auto allGateOps() {
    return Util::tuple_to_array(
            allGateOpsHelper<EnumClass>(
                    std::make_integer_sequence<uint32_t, 
                            static_cast<uint32_t>(EnumClass::END)>{})
    );
};

#define PENNYLANE_TESTS_DEFINE_GATE_OP_PARAM0(GATE_NAME)\
    template <class PrecisionT>\
    static void apply##GATE_NAME(std::complex<PrecisionT> *arr, size_t num_qubits, \
                            const std::vector<size_t> &wires, \
                            bool inverse) { \
        static_cast<void>(arr); \
        static_cast<void>(num_qubits); \
        static_cast<void>(wires);\
        static_cast<void>(inverse);\
    }
#define PENNYLANE_TESTS_DEFINE_GATE_OP_PARAM1(GATE_NAME)\
    template <class PrecisionT, class ParamT>\
    static void apply##GATE_NAME(std::complex<PrecisionT> *arr, size_t num_qubits, \
                            const std::vector<size_t> &wires, \
                            bool inverse, ParamT ) { \
        static_cast<void>(arr); \
        static_cast<void>(num_qubits); \
        static_cast<void>(wires);\
        static_cast<void>(inverse);\
    }
#define PENNYLANE_TESTS_DEFINE_GATE_OP_PARAM3(GATE_NAME)\
    template <class PrecisionT, class ParamT>\
    static void apply##GATE_NAME(std::complex<PrecisionT> *arr, size_t num_qubits, \
                            const std::vector<size_t> &wires, \
                            bool inverse, ParamT, ParamT, ParamT) { \
        static_cast<void>(arr); \
        static_cast<void>(num_qubits); \
        static_cast<void>(wires);\
        static_cast<void>(inverse);\
    }
#define PENNYLANE_TESTS_DEFINE_GATE_OP(GATE_NAME, NUM_PARAMS) \
    PENNYLANE_TESTS_DEFINE_GATE_OP_PARAM##NUM_PARAMS(GATE_NAME)

#define PENNYLANE_TESTS_DEFINE_GENERATOR_OP(GENERATOR_NAME)\
    template <class PrecisionT>                                   \
    static PrecisionT applyGenerator##GENETATOR_NAME(\
            std::complex<PrecisionT> *arr, size_t num_qubits, \
            const std::vector<size_t> &wires, \
            bool adj) { \
        static_cast<void>(arr); \
        static_cast<void>(num_qubits); \
        static_cast<void>(wires);\
        static_cast<void>(adj);\
        return PrecisionT{}; \
    }

class DummyImplementation {
  public:
    constexpr static std::array implemented_gates = allGateOps<GateOperation>();
    constexpr static std::array implemented_generators = allGateOps<GeneratorOperation>();
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
};

template<typename PrecisionT, typename ParamT, size_t gate_idx>
constexpr auto gateOpFuncPtrPairsIter() {
    if constexpr (gate_idx == DummyImplementation::implemented_gates.size() ) {
        return std::tuple{};
    }
    else {
        constexpr auto gate_op = DummyImplementation::implemented_gates[gate_idx];
        if constexpr (gate_op == GateOperation::Matrix) {
            return gateOpFuncPtrPairsIter<PrecisionT, ParamT, gate_idx+1>();
        }
        else {
            const auto elt = std::pair{gate_op, GateOpToMemberFuncPtr<PrecisionT, ParamT,
                    DummyImplementation, gate_op>::value};
            return Util::prepend_to_tuple(elt,
                    gateOpFuncPtrPairsIter<PrecisionT, ParamT, gate_idx+1>());
        }
    }
};


template <typename PrecisionT, typename ParamT>
struct GateOpFuncPtrPairs {
    constexpr static auto value = gateOpFuncPtrPairsIter<PrecisionT, ParamT, 0>();
};

template <typename PrecisionT, typename ParamT, size_t num_params, size_t tuple_idx>
constexpr auto gateOpFuncPtrPairsWithNumParamsIter() {
    if constexpr (tuple_idx < std::tuple_size_v<decltype(GateOpFuncPtrPairs<PrecisionT, ParamT>::value)>) {
        constexpr auto elt = std::get<tuple_idx>(GateOpFuncPtrPairs<PrecisionT, ParamT>::value);
        if constexpr (static_lookup<elt.first>(Constant::gate_num_params) == num_params) {
            return prepend_to_tuple(elt, gateOpFuncPtrPairsWithNumParamsIter
                    <PrecisionT, ParamT, num_params, tuple_idx+1>());
        } else{
            return gateOpFuncPtrPairsWithNumParamsIter<PrecisionT, ParamT, num_params, tuple_idx+1>();
        }
    } else {
        return std::tuple{};
    }
}

template <typename PrecisionT, typename ParamT, size_t num_params>
constexpr auto gate_op_func_ptr_with_params = 
        Util::tuple_to_array(gateOpFuncPtrPairsWithNumParamsIter<PrecisionT, ParamT, num_params, 0>());

template <typename PrecisionT, typename ParamT, size_t num_params>
constexpr auto testUniqueness() {
    constexpr auto pairs = gate_op_func_ptr_with_params<PrecisionT, ParamT, num_params>;
    static_assert(
        Util::count_unique(
            Util::first_elts_of(pairs)) == pairs.size());
    static_assert(
        Util::count_unique(
            Util::second_elts_of(pairs)) == pairs.size());
}

TEMPLATE_TEST_CASE("GateOpToMemberFuncPtr", "[GateOpToMemberFuncPtr]",
        float, double) {
    testUniqueness<TestType, TestType, 0>();
    testUniqueness<TestType, TestType, 1>();
    testUniqueness<TestType, TestType, 3>();
}
