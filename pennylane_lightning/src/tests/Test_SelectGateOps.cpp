#include "SelectGateOps.hpp"
#include "TestHelpers.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

using namespace Pennylane;

TEST_CASE("SelectGateOps", "[SelectGateOps]") {
    REQUIRE(SelectGateOps<float, KernelType::PI>::kernel_id == KernelType::PI);
    REQUIRE(SelectGateOps<double, KernelType::PI>::kernel_id == KernelType::PI);

    REQUIRE(SelectGateOps<float, KernelType::LM>::kernel_id == KernelType::LM);
    REQUIRE(SelectGateOps<double, KernelType::LM>::kernel_id == KernelType::LM);
}
