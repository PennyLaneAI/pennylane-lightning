#include "Constant.hpp"
#include "ConstantUtil.hpp"
#include "DefaultKernelsForStateVector.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

using namespace Pennylane;

TEST_CASE("Test default kernels for gates are well defined",
          "[Test_DefaultKernelsForStateVector]") {
    auto &instance = DefaultKernelsForStateVector::getInstance();
    Util::for_each_enum<Threading, CPUMemoryModel>(
        [&instance](Threading threading, CPUMemoryModel memory_model) {
            for (size_t num_qubits = 1; num_qubits < 27; num_qubits++) {
                REQUIRE_NOTHROW(instance.getGateKernelMap(num_qubits, threading,
                                                          memory_model));
            }
        });
}

TEST_CASE("Test default kernels for generators are well defined",
          "[Test_DefaultKernelsForStateVector]") {
    auto &instance = DefaultKernelsForStateVector::getInstance();
    Util::for_each_enum<Threading, CPUMemoryModel>(
        [&instance](Threading threading, CPUMemoryModel memory_model) {
            for (size_t num_qubits = 1; num_qubits < 27; num_qubits++) {
                REQUIRE_NOTHROW(instance.getGeneratorKernelMap(
                    num_qubits, threading, memory_model));
            }
        });
}
