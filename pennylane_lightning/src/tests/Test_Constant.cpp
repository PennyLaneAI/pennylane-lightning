#include "Constant.hpp"

#include <catch2/catch.hpp>

/**
 * @file Test_Constant.cpp
 * Do some runtime tests for constants. Note that these checks can be done in compile
 * time in C++20.
 */

using namespace Pennylane;

TEST_CASE("Test generator names start with \"Generator\"", "Constant") {
    for (const auto& [gntr_op, gntr_name]: Constant::generator_names) {
        REQUIRE(gntr_name.find("Generator") == 0);
    }
}
