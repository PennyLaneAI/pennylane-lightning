#include "cpu_kernels/avx_common/TwoQubitGateHelper.hpp"

#include <catch2/catch.hpp>

using namespace Pennylane::Gates::AVXCommon;

template <typename PrecisionT, size_t packed_size>
struct MockSymmetricTwoQubitGateWithoutParam {
    using Precision = PrecisionT;
    constexpr static size_t packed_size_ = packed_size;
    constexpr static bool symmetric = true;

    template <size_t rev_wire0, size_t rev_wire1>
    static std::tuple<std::string, size_t, size_t, bool>
    applyInternalInternal(std::complex<PrecisionT> *arr,
                          const size_t num_qubits, bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(inverse);
        return {"applyInternal", rev_wire, inverse};
    }

    static std::tuple<std::string, size_t, bool>
    applyExternal(std::complex<PrecisionT> *arr, const size_t num_qubits,
                  const size_t rev_wire, bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(rev_wire);
        static_cast<void>(inverse);
        return {"applyExternal", rev_wire, inverse};
    }
};

TEMPLATE_TEST_CASE("Test SingleQubitGateHelper template functions",
                   "[SingleQubitGateHelper]", float, double) {
    STATIC_REQUIRE(HasInternalWithoutParam<
                   MockSingleQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasInternalWithParam<
                   MockSingleQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(HasExternalWithoutParam<
                   MockSingleQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasExternalWithParam<
                   MockSingleQubitGateWithoutParam<TestType, 4>>::value);

    STATIC_REQUIRE(!HasInternalWithoutParam<
                   MockSingleQubitGateWithParam<TestType, 4>>::value);
    STATIC_REQUIRE(
        HasInternalWithParam<MockSingleQubitGateWithParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasExternalWithoutParam<
                   MockSingleQubitGateWithParam<TestType, 4>>::value);
    STATIC_REQUIRE(
        HasExternalWithParam<MockSingleQubitGateWithParam<TestType, 4>>::value);

    STATIC_REQUIRE(HasInternalWithoutParam<
                   MockSingleQubitGateSomethingWrong<TestType, 4>>::value);
    STATIC_REQUIRE(!HasInternalWithParam<
                   MockSingleQubitGateSomethingWrong<TestType, 4>>::value);
    STATIC_REQUIRE(!HasExternalWithoutParam<
                   MockSingleQubitGateSomethingWrong<TestType, 4>>::value);
    STATIC_REQUIRE(HasExternalWithParam<
                   MockSingleQubitGateSomethingWrong<TestType, 4>>::value);
}
