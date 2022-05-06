#include "Macros.hpp"
#include "RuntimeInfo.hpp"

#include <catch2/catch.hpp>

using namespace Pennylane::Util;

TEST_CASE("Runtime information is correct", "[Test_RuntimeInfo]") {
    INFO("RuntimeInfo::AVX " << RuntimeInfo::AVX());
    INFO("RuntimeInfo::AVX2 " << RuntimeInfo::AVX2());
    INFO("RuntimeInfo::AVX512F " << RuntimeInfo::AVX512F());
    INFO("RuntimeInfo::vendor " << RuntimeInfo::vendor());
    INFO("RuntimeInfo::brand " << RuntimeInfo::brand());
    REQUIRE(true);
}
